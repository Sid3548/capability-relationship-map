"""Smooth metric: teacher-forced NLL / perplexity over gold-answer tokens,
aggregated across a battery. Mandatory alongside the accuracy tracks --
this is what we check every 0.1% sweep step (cheap) even when generation
eval is run less frequently."""
from __future__ import annotations

from src.eval.base import BatteryItem, teacher_forced_nll


def eval_loss_battery(bundle, items: list[BatteryItem], task: str | None = None) -> dict:
    sel = [it for it in items if (task is None or it.task == task)]
    per_item = []
    for it in sel:
        r = teacher_forced_nll(bundle, it.prompt, it.gold)
        per_item.append({"id": it.id, "nll": r["nll"], "ppl": r["ppl"], "n_gold_tokens": r["n_gold_tokens"]})
    if not per_item:
        return {"mean_nll": None, "mean_ppl": None, "n": 0, "per_item": []}
    mean_nll = sum(p["nll"] for p in per_item) / len(per_item)
    mean_ppl = sum(p["ppl"] for p in per_item) / len(per_item)
    return {"mean_nll": mean_nll, "mean_ppl": mean_ppl, "n": len(per_item), "per_item": per_item}
