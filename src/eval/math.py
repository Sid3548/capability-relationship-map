"""Math eval: numeric/exact match with tolerance. Also tracks parse-failure
rate (no numeric token found in the completion)."""
from __future__ import annotations

import re

from src.eval.base import BatteryItem, greedy_generate

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_number(text: str) -> float | None:
    m = _NUM_RE.search(text)
    if m is None:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def eval_math_item(bundle, item: BatteryItem, max_new_tokens: int, tolerance: float) -> dict:
    raw = greedy_generate(bundle, item.prompt, max_new_tokens=max_new_tokens)
    pred = extract_number(raw)
    gold = float(item.gold)
    parse_failure = pred is None
    correct = (not parse_failure) and abs(pred - gold) <= tolerance
    return {
        "id": item.id,
        "passed": correct,
        "parse_failure": parse_failure,
        "raw_completion": raw,
        "pred": pred,
        "gold": gold,
    }


def eval_math_battery(bundle, items: list[BatteryItem], max_new_tokens: int, tolerance: float) -> dict:
    results = [
        eval_math_item(bundle, it, max_new_tokens, tolerance)
        for it in items
        if it.task == "math"
    ]
    n = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    n_parse_fail = sum(1 for r in results if r["parse_failure"])
    return {
        "accuracy": n_pass / n if n else 0.0,
        "n": n,
        "n_pass": n_pass,
        "parse_failure_rate": n_parse_fail / n if n else 0.0,
        "results": results,
    }
