"""Shared eval utilities: battery loading, greedy generation, answer-span
token masking for teacher-forced NLL. All sweep steps must call these with
IDENTICAL decoding params / prompt order / seed -- enforced by always
loading from configs/eval.yaml.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class BatteryItem:
    task: str
    id: str
    prompt: str
    gold: str
    eval_type: str
    tests: list | None = None
    metadata: dict | None = None


def load_battery(path: str | Path) -> list[BatteryItem]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(
                BatteryItem(
                    task=d["task"],
                    id=d["id"],
                    prompt=d["prompt"],
                    gold=d["gold"],
                    eval_type=d["eval_type"],
                    tests=d.get("tests"),
                    metadata=d.get("metadata"),
                )
            )
    return items


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def greedy_generate(bundle, prompt: str, max_new_tokens: int = 256) -> str:
    """Deterministic greedy decode. Returns only the newly generated text."""
    tok = bundle.tokenizer
    inputs = tok(prompt, return_tensors="pt").to(bundle.device)
    input_len = inputs["input_ids"].shape[1]
    out = bundle.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
    )
    new_tokens = out[0][input_len:]
    return tok.decode(new_tokens, skip_special_tokens=True)


@torch.no_grad()
def teacher_forced_nll(bundle, prompt: str, gold: str) -> dict:
    """Compute mean per-token NLL and perplexity over gold-answer tokens only
    (prompt tokens masked out of the loss). Returns dict with nll, ppl,
    n_gold_tokens, and also provides the mask needed by CaptureHook so
    activation capture aggregates only answer-token positions.
    """
    tok = bundle.tokenizer
    prompt_ids = tok(prompt, return_tensors="pt")["input_ids"][0]
    gold_ids = tok(gold, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if gold_ids.numel() == 0:
        # degenerate gold (e.g. tokenizer collapses to nothing) -- guard
        gold_ids = torch.tensor([tok.eos_token_id])

    full_ids = torch.cat([prompt_ids, gold_ids], dim=0).unsqueeze(0).to(bundle.device)
    attn_mask = torch.ones_like(full_ids)

    answer_mask = torch.zeros_like(full_ids, dtype=torch.bool)
    p_len = prompt_ids.shape[0]
    answer_mask[0, p_len:] = True  # answer-token positions in the FULL sequence

    out = bundle.model(input_ids=full_ids, attention_mask=attn_mask)
    logits = out.logits  # [1, T, V]

    # next-token prediction: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]
    shift_answer_mask = answer_mask[:, 1:]  # label at t+1 is an answer token

    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    nll_per_tok = -gathered

    sel = shift_answer_mask[0]
    n_gold = int(sel.sum().item())
    if n_gold == 0:
        raise RuntimeError("No gold tokens selected for NLL -- check tokenization")
    mean_nll = nll_per_tok[0][sel].mean().item()
    ppl = float(np.exp(mean_nll))

    # Full-sequence positional mask usable directly by CaptureHook
    # (aligned to logits/hidden-state positions, i.e. length T, True where
    # the CURRENT token's hidden state feeds the answer-token loss -- we use
    # answer_mask itself, shifted back by one, since hidden state at position
    # t-1 produces logits for token t; for capture purposes we mark the
    # hidden states at answer-token positions themselves, i.e. answer_mask).
    return {
        "nll": mean_nll,
        "ppl": ppl,
        "n_gold_tokens": n_gold,
        "full_ids": full_ids,
        "attn_mask": attn_mask,
        "answer_mask": answer_mask,  # [1, T] bool, True at answer-token positions
    }


@torch.no_grad()
def batched_teacher_forced_nll(bundle, pairs, max_batch=16) -> dict:
    """Batched teacher-forced NLL over gold tokens for a list of (prompt, gold)
    pairs. Left-padding not needed -- we right-pad and mask. Returns
    {mean_nll, mean_ppl, per_item_nll}. MUCH faster than per-item for sweeps.

    Each sequence = prompt_ids + gold_ids; loss computed only over gold-token
    positions; pad positions excluded via attention mask AND loss mask.
    """
    tok = bundle.tokenizer
    device = bundle.device
    per_nll = []
    for start in range(0, len(pairs), max_batch):
        chunk = pairs[start:start + max_batch]
        seqs = []
        gold_spans = []  # (prompt_len, total_len)
        for prompt, gold in chunk:
            pid = tok(prompt, return_tensors="pt")["input_ids"][0]
            gid = tok(gold, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            if gid.numel() == 0:
                gid = torch.tensor([tok.eos_token_id])
            full = torch.cat([pid, gid], dim=0)
            seqs.append(full)
            gold_spans.append((pid.shape[0], full.shape[0]))
        maxlen = max(s.shape[0] for s in seqs)
        pad_id = tok.pad_token_id
        batch = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for k, s in enumerate(seqs):
            batch[k, :s.shape[0]] = s
            attn[k, :s.shape[0]] = 1
        batch = batch.to(device); attn = attn.to(device)
        logits = bundle.model(input_ids=batch, attention_mask=attn).logits  # [B,T,V]
        shift_logits = logits[:, :-1, :].float()
        shift_labels = batch[:, 1:]
        logp = torch.log_softmax(shift_logits, dim=-1)
        gathered = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B,T-1]
        for k, (plen, tlen) in enumerate(gold_spans):
            # gold token positions in labels space: indices [plen-1 .. tlen-2]
            lo = plen - 1
            hi = tlen - 1
            span = gathered[k, lo:hi]
            nll = -span.mean().item()
            per_nll.append(nll)
    mean_nll = float(np.mean(per_nll)) if per_nll else None
    mean_ppl = float(np.exp(mean_nll)) if per_nll else None
    return {"mean_nll": mean_nll, "mean_ppl": mean_ppl, "per_item_nll": per_nll}
