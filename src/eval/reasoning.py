"""Reasoning eval: CONSTRAINED multiple-choice ONLY (single letter A-D).

Never free-form essays. The prompt presents 4 labelled options and asks for
one letter; we extract the first standalone A/B/C/D from the generation and
exact-match it to the gold letter. Greedy, deterministic. Tracks
parse-failure rate (no letter found)."""
from __future__ import annotations

import re

from src.eval.base import BatteryItem, greedy_generate

_LETTER_RE = re.compile(r"\b([ABCD])\b")


def extract_letter(text: str) -> str | None:
    # look at the first few tokens only; MCQ answer should be immediate
    head = text.strip()[:20]
    m = _LETTER_RE.search(head)
    if m:
        return m.group(1)
    # fallback: first A-D anywhere
    m = _LETTER_RE.search(text)
    return m.group(1) if m else None


def eval_reasoning_item(bundle, item: BatteryItem, max_new_tokens: int) -> dict:
    raw = greedy_generate(bundle, item.prompt, max_new_tokens=min(max_new_tokens, 8))
    pred = extract_letter(raw)
    gold = item.gold.strip().upper()
    parse_failure = pred is None
    correct = (not parse_failure) and pred == gold
    return {"id": item.id, "passed": correct, "parse_failure": parse_failure,
            "raw_completion": raw, "pred": pred, "gold": gold}


def eval_reasoning_battery(bundle, items: list[BatteryItem], max_new_tokens: int, **_) -> dict:
    results = [eval_reasoning_item(bundle, it, max_new_tokens) for it in items if it.task == "reasoning"]
    n = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    n_pf = sum(1 for r in results if r["parse_failure"])
    return {"accuracy": n_pass / n if n else 0.0, "n": n, "n_pass": n_pass,
            "parse_failure_rate": n_pf / n if n else 0.0, "results": results}
