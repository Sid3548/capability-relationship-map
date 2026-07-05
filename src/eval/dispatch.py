"""Unified per-task generation-accuracy dispatch across all eval_types used by
the 15-capability comprehensive battery:
  exec     -> coding unit tests (pass@1)
  numeric  -> numeric exact match
  mcq      -> single letter A-D
  factual  -> normalized exact / alias / token-F1
  tokenf1  -> token-F1 vs reference (threshold)
  nll_only -> no accuracy (soft task); returns None

The UNIVERSAL primary metric is teacher-forced NLL (see eval/loss.py +
batched_teacher_forced_nll); this module provides the SECONDARY accuracy
where a capability is cleanly gradeable.
"""
from __future__ import annotations

from src.eval.base import BatteryItem, greedy_generate
from src.eval.coding import extract_code, run_in_sandbox
from src.eval.math import extract_number
from src.eval.reasoning import extract_letter
from src.eval.factual import normalize, token_f1


def _acc_exec(bundle, items, max_new_tokens, exec_timeout):
    n = npass = npf = 0
    for it in items:
        raw = greedy_generate(bundle, it.prompt, max_new_tokens=max_new_tokens)
        ep = (it.metadata or {}).get("entry_point", "")
        code = extract_code(raw, it.prompt, ep)
        res = run_in_sandbox(code, (it.metadata or {}).get("tests", []), timeout_sec=exec_timeout)
        n += 1; npass += int(res["passed"]); npf += int(res["parse_failure"])
    return npass / n if n else 0.0, npf / n if n else 0.0


def _acc_numeric(bundle, items, max_new_tokens, tol):
    n = npass = npf = 0
    for it in items:
        raw = greedy_generate(bundle, it.prompt, max_new_tokens=min(max_new_tokens, 16))
        pred = extract_number(raw); gold = float(it.gold)
        pf = pred is None
        n += 1; npf += int(pf); npass += int((not pf) and abs(pred - gold) <= tol)
    return npass / n if n else 0.0, npf / n if n else 0.0


def _acc_mcq(bundle, items, max_new_tokens):
    n = npass = npf = 0
    for it in items:
        raw = greedy_generate(bundle, it.prompt, max_new_tokens=8)
        pred = extract_letter(raw); gold = it.gold.strip().upper()
        pf = pred is None
        n += 1; npf += int(pf); npass += int((not pf) and pred == gold)
    return npass / n if n else 0.0, npf / n if n else 0.0


def _acc_factual(bundle, items, max_new_tokens):
    n = npass = npf = 0
    for it in items:
        raw = greedy_generate(bundle, it.prompt, max_new_tokens=24)
        pred = raw.strip().split("\n")[0].strip()
        pf = len(pred) == 0
        gold = it.gold.strip()
        aliases = (it.metadata or {}).get("aliases", []) or []
        thr = (it.metadata or {}).get("f1_threshold", 0.5)
        npred = normalize(pred)
        match = False
        for cand in [gold] + aliases:
            nc = normalize(cand)
            if nc and (nc in npred or (npred and npred in nc)):
                match = True; break
        best_f1 = max((token_f1(pred, c) for c in [gold] + aliases), default=0.0)
        n += 1; npf += int(pf); npass += int((not pf) and (match or best_f1 >= thr))
    return npass / n if n else 0.0, npf / n if n else 0.0


def _acc_tokenf1(bundle, items, max_new_tokens):
    n = npass = npf = 0
    for it in items:
        raw = greedy_generate(bundle, it.prompt, max_new_tokens=24)
        pred = raw.strip().split("\n")[0].strip()
        pf = len(pred) == 0
        thr = (it.metadata or {}).get("f1_threshold", 0.5)
        f1 = token_f1(pred, it.gold.strip())
        n += 1; npf += int(pf); npass += int((not pf) and f1 >= thr)
    return npass / n if n else 0.0, npf / n if n else 0.0


def gen_accuracy(bundle, items, eval_type, max_new_tokens, exec_timeout, tol):
    """Return (accuracy, parse_fail_rate) or (None, None) for nll_only tasks."""
    if not items:
        return None, None
    if eval_type == "exec":
        return _acc_exec(bundle, items, max_new_tokens, exec_timeout)
    if eval_type == "numeric":
        return _acc_numeric(bundle, items, max_new_tokens, tol)
    if eval_type == "mcq":
        return _acc_mcq(bundle, items, max_new_tokens)
    if eval_type == "factual":
        return _acc_factual(bundle, items, max_new_tokens)
    if eval_type == "tokenf1":
        return _acc_tokenf1(bundle, items, max_new_tokens)
    if eval_type == "nll_only":
        return None, None
    raise ValueError(f"unknown eval_type {eval_type}")
