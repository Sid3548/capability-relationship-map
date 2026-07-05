"""Factual eval (history task): normalized exact match / alias match / token-F1.

A prediction counts as correct if, after normalization, EITHER the gold
answer or any alias is contained in the prediction (or vice-versa), OR the
token-F1 between prediction and gold exceeds the per-item threshold. Greedy
decode; deterministic. Tracks parse-failure rate (empty generation)."""
from __future__ import annotations

import re
import string

from src.eval.base import BatteryItem, greedy_generate

_ARTICLES = {"a", "an", "the"}


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    toks = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(toks)


def token_f1(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g:
        return 0.0
    common = {}
    for t in g:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in p:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def _first_line(text: str) -> str:
    # factual answers are short; cut at newline to avoid trailing rambling
    return text.strip().split("\n")[0].strip()


def eval_factual_item(bundle, item: BatteryItem, max_new_tokens: int) -> dict:
    raw = greedy_generate(bundle, item.prompt, max_new_tokens=min(max_new_tokens, 32))
    pred_line = _first_line(raw)
    gold = item.gold.strip()
    aliases = (item.metadata or {}).get("aliases", []) or []
    thr = (item.metadata or {}).get("f1_threshold", 0.5)

    parse_failure = len(pred_line) == 0
    npred = normalize(pred_line)
    candidates = [gold] + aliases
    match = False
    for cand in candidates:
        nc = normalize(cand)
        if nc and (nc in npred or (npred and npred in nc)):
            match = True
            break
    best_f1 = max((token_f1(pred_line, c) for c in candidates), default=0.0)
    correct = (not parse_failure) and (match or best_f1 >= thr)
    return {"id": item.id, "passed": correct, "parse_failure": parse_failure,
            "raw_completion": raw, "pred": pred_line, "gold": gold, "f1": best_f1}


def eval_factual_battery(bundle, items: list[BatteryItem], max_new_tokens: int, **_) -> dict:
    results = [eval_factual_item(bundle, it, max_new_tokens) for it in items if it.task == "history"]
    n = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    n_pf = sum(1 for r in results if r["parse_failure"])
    return {"accuracy": n_pass / n if n else 0.0, "n": n, "n_pass": n_pass,
            "parse_failure_rate": n_pf / n if n else 0.0, "results": results}
