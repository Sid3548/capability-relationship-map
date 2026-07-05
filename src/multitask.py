"""Multi-task / interlink experiment on Qwen2.5-0.5B (0.5B ONLY -- do not scale).

Four tasks: coding, math, history (factual), reasoning (MCQ).
Pipeline:
  1. baselines (gen accuracy + teacher-forced NLL) for all 4 tasks
  2. capture per-task down_proj-input aggregates (mean_abs / max_abs)
  3. calibration means for mean-ablation
  4. per-task per-layer percentile scoring
  5. core mask (high on ALL tasks), A-specific-high per task, A-low envelope per task
  6. interlink N x N: for each target task remove (i) its low envelope and
     (ii) its specific-high set, eval ALL four tasks, store delta_acc + delta_loss
  7. characterize the coding low-envelope removed neurons (when do they fire?)

All guardrails honored: per-layer percentile thresholds, mean-ablate default
(+ zero-ablate available), reversible hooks, smooth NLL alongside accuracy,
A-low and A-specific-high kept as SEPARATE masks, deterministic greedy eval.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ablation import attach_ablation_hooks_from_mask, remove_ablation_hooks
from src.capture import capture_task_aggregates, compute_calibration_means
from src.eval.base import load_battery, set_all_seeds
from src.eval.coding import eval_coding_battery
from src.eval.factual import eval_factual_battery
from src.eval.loss import eval_loss_battery
from src.eval.math import eval_math_battery
from src.eval.reasoning import eval_reasoning_battery
from src.models import load_config, load_model
from src.plotting import plot_interlink_heatmap
from src.scoring import (
    ScoreBundle,
    compute_core_mask_multi,
    compute_low_envelope_mask,
    compute_specific_high_mask,
)
from src.storage import save_rows_parquet, write_manifest

TASKS = ["coding", "math", "history", "reasoning"]
BATTERY_FILES = {
    "coding": "data/batteries/coding.jsonl",
    "math": "data/batteries/math.jsonl",
    "history": "data/batteries/history.jsonl",
    "reasoning": "data/batteries/reasoning.jsonl",
}
SCORE_STAT = "mean_abs"
ABLATION_MODE = "mean"
ENVELOPE_BUDGET = 0.05     # A-low envelope removal budget = 5% of total neurons
CORE_PCT = 0.95
SPEC_HIGH = 0.85
SPEC_LOW = 0.60


def load_items():
    items = {}
    for t in TASKS:
        items[t] = load_battery(REPO_ROOT / BATTERY_FILES[t])
    return items


def gen_eval_task(bundle, items_t, task, max_new_tokens, exec_timeout, tol):
    if task == "coding":
        r = eval_coding_battery(bundle, items_t, max_new_tokens, exec_timeout)
        return r["pass_at_1"], r["parse_failure_rate"]
    if task == "math":
        r = eval_math_battery(bundle, items_t, max_new_tokens, tol)
        return r["accuracy"], r["parse_failure_rate"]
    if task == "history":
        r = eval_factual_battery(bundle, items_t, max_new_tokens)
        return r["accuracy"], r["parse_failure_rate"]
    if task == "reasoning":
        r = eval_reasoning_battery(bundle, items_t, max_new_tokens)
        return r["accuracy"], r["parse_failure_rate"]
    raise ValueError(task)


def eval_all_tasks(bundle, items, max_new_tokens, exec_timeout, tol):
    """Return dict task -> {acc, parse_fail, nll}."""
    out = {}
    for t in TASKS:
        acc, pf = gen_eval_task(bundle, items[t], t, max_new_tokens, exec_timeout, tol)
        loss = eval_loss_battery(bundle, items[t], task=t)
        out[t] = {"acc": acc, "parse_fail": pf, "nll": loss["mean_nll"], "ppl": loss["mean_ppl"]}
    return out


def main():
    model_cfg = load_config(REPO_ROOT / "configs/model.yaml")
    eval_cfg = load_config(REPO_ROOT / "configs/eval.yaml")
    set_all_seeds(model_cfg["seed"])
    max_new_tokens = eval_cfg["decoding"]["max_new_tokens"]
    exec_timeout = eval_cfg["exec_timeout_sec"]
    tol = eval_cfg["numeric_tolerance"]

    print("== Loading model ==")
    bundle = load_model(model_cfg)
    total_neurons = bundle.arch.total_mlp_neurons
    print(f"layers={bundle.num_layers()} intermediate={bundle.intermediate_size()} "
          f"total_mlp_neurons={total_neurons}")

    items = load_items()
    for t in TASKS:
        print(f"  battery {t}: {len(items[t])} prompts")

    all_items = [it for t in TASKS for it in items[t]]

    print("\n== Baselines (all 4 tasks) ==")
    base = eval_all_tasks(bundle, items, max_new_tokens, exec_timeout, tol)
    for t in TASKS:
        print(f"  {t:9s} acc={base[t]['acc']:.3f} parse_fail={base[t]['parse_fail']:.3f} "
              f"nll={base[t]['nll']:.4f} ppl={base[t]['ppl']:.3f}")

    print("\n== Capture per-task aggregates ==")
    aggs = {}
    for t in TASKS:
        aggs[t] = capture_task_aggregates(bundle, items[t], task=t, verbose=True)

    print("\n== Calibration means (mean-ablation replacement) ==")
    calib = compute_calibration_means(bundle, all_items)

    print("\n== Scoring (per-layer percentile per task) ==")
    scores = {t: ScoreBundle(t, aggs[t]["mean_abs"], aggs[t]["max_abs"]) for t in TASKS}
    ranks = {t: scores[t].rank(SCORE_STAT) for t in TASKS}

    core = compute_core_mask_multi(ranks, core_percentile=CORE_PCT)
    print(f"core (high on ALL 4 tasks, p>={CORE_PCT}): {int(core.sum())} "
          f"({100*core.sum()/total_neurons:.3f}%)")

    spec_high = {t: compute_specific_high_mask(t, ranks, core, SPEC_HIGH, SPEC_LOW) for t in TASKS}
    low_env = {t: compute_low_envelope_mask(t, ranks, core, ENVELOPE_BUDGET) for t in TASKS}

    print(f"\n== A-specific-high counts (rank>={SPEC_HIGH} on target, <={SPEC_LOW} on all others, minus core) ==")
    for t in TASKS:
        print(f"  {t:9s} specific-high={int(spec_high[t].sum()):5d}  "
              f"low-envelope={int(low_env[t].sum()):5d} ({100*low_env[t].sum()/total_neurons:.2f}%)")

    print("\n== Pairwise overlap of A-specific-high sets (Jaccard) ==")
    overlap_rows = []
    for a in TASKS:
        for b in TASKS:
            if a >= b:
                continue
            ma, mb = spec_high[a].reshape(-1), spec_high[b].reshape(-1)
            inter = int((ma & mb).sum())
            union = int((ma | mb).sum())
            jac = inter / union if union else 0.0
            print(f"  {a} & {b}: inter={inter} union={union} jaccard={jac:.4f}")
            overlap_rows.append({"a": a, "b": b, "inter": inter, "union": union, "jaccard": jac})

    # ---------------- INTERLINK N x N ----------------
    print("\n== INTERLINK: remove each target's env / specific-high, eval all 4 ==")
    N = len(TASKS)
    dacc_env = np.zeros((N, N)); dloss_env = np.zeros((N, N))
    dacc_hi = np.zeros((N, N)); dloss_hi = np.zeros((N, N))
    rows = []
    base_acc = {t: base[t]["acc"] for t in TASKS}
    base_nll = {t: base[t]["nll"] for t in TASKS}

    def run_condition(mask, cond_name, target):
        handles = attach_ablation_hooks_from_mask(bundle, mask, ABLATION_MODE, calib)
        try:
            res = eval_all_tasks(bundle, items, max_new_tokens, exec_timeout, tol)
        finally:
            remove_ablation_hooks(handles)
        for col in TASKS:
            rows.append({
                "condition": cond_name, "target": target, "measured": col,
                "n_removed": int(mask.sum()),
                "acc_post": res[col]["acc"], "acc_base": base_acc[col],
                "delta_acc": res[col]["acc"] - base_acc[col],
                "nll_post": res[col]["nll"], "nll_base": base_nll[col],
                "delta_nll": res[col]["nll"] - base_nll[col],
            })
        return res

    t0 = time.time()
    for i, target in enumerate(TASKS):
        r_env = run_condition(low_env[target], "low_envelope", target)
        for j, col in enumerate(TASKS):
            dacc_env[i, j] = r_env[col]["acc"] - base_acc[col]
            dloss_env[i, j] = r_env[col]["nll"] - base_nll[col]
        r_hi = run_condition(spec_high[target], "specific_high", target)
        for j, col in enumerate(TASKS):
            dacc_hi[i, j] = r_hi[col]["acc"] - base_acc[col]
            dloss_hi[i, j] = r_hi[col]["nll"] - base_nll[col]
        print(f"  target={target:9s} done ({time.time()-t0:.1f}s)  "
              f"env self-dacc={dacc_env[i,i]:+.2f} hi self-dacc={dacc_hi[i,i]:+.2f}")

    # ---------------- CHARACTERIZE coding low-envelope removed neurons ----------------
    print("\n== Characterize coding low-envelope removed set ==")
    cmask = low_env["coding"].reshape(-1)
    idx = np.nonzero(cmask)[0]
    n_removed = len(idx)
    char = {"n_removed": int(n_removed), "budget_fraction": ENVELOPE_BUDGET}
    per_task_mean = {}
    per_task_max = {}
    for t in TASKS:
        ma = aggs[t]["mean_abs"].reshape(-1)[idx]
        mx = aggs[t]["max_abs"].reshape(-1)[idx]
        per_task_mean[t] = float(ma.mean())
        per_task_max[t] = float(mx.mean())
        char[f"mean_abs_on_{t}"] = float(ma.mean())
        char[f"max_abs_on_{t}"] = float(mx.mean())
    # global reference: median mean_abs across all neurons per task (context)
    for t in TASKS:
        char[f"global_median_mean_abs_{t}"] = float(np.median(aggs[t]["mean_abs"]))
    # "effectively dead": near-zero mean_abs on EVERY task
    eps = 0.02
    dead = np.ones(n_removed, dtype=bool)
    for t in TASKS:
        dead &= (aggs[t]["mean_abs"].reshape(-1)[idx] < eps)
    char["frac_effectively_dead"] = float(dead.mean())
    char["dead_eps"] = eps
    # do they fire for OTHER tasks more than for coding? compare mean_abs coding vs best-other
    coding_ma = aggs["coding"]["mean_abs"].reshape(-1)[idx]
    other_ma = np.maximum.reduce([aggs[t]["mean_abs"].reshape(-1)[idx] for t in TASKS if t != "coding"])
    fires_more_elsewhere = float((other_ma > coding_ma).mean())
    char["frac_fires_more_on_other_than_coding"] = fires_more_elsewhere

    print(f"  removed {n_removed} neurons (coding envelope, {ENVELOPE_BUDGET*100:.0f}%)")
    for t in TASKS:
        print(f"    on {t:9s}: mean_abs={per_task_mean[t]:.4f} (global median {char['global_median_mean_abs_'+t]:.4f}) "
              f"max_abs={per_task_max[t]:.4f}")
    print(f"  fraction effectively dead (mean_abs<{eps} on ALL tasks): {char['frac_effectively_dead']:.3f}")
    print(f"  fraction firing more on some other task than coding: {fires_more_elsewhere:.3f}")

    # ---------------- SAVE ----------------
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    save_rows_parquet(results_dir / "interlink_rows.parquet", rows)

    plot_interlink_heatmap(dacc_env, TASKS, "Interlink delta-acc: remove target LOW-ENVELOPE",
                           results_dir / "interlink_env_dacc.png")
    plot_interlink_heatmap(dloss_env, TASKS, "Interlink delta-NLL: remove target LOW-ENVELOPE",
                           results_dir / "interlink_env_dloss.png")
    plot_interlink_heatmap(dacc_hi, TASKS, "Interlink delta-acc: remove target SPECIFIC-HIGH",
                           results_dir / "interlink_high_dacc.png")
    plot_interlink_heatmap(dloss_hi, TASKS, "Interlink delta-NLL: remove target SPECIFIC-HIGH",
                           results_dir / "interlink_high_dloss.png")

    summary = {
        "tasks": TASKS,
        "baselines": base,
        "core_count": int(core.sum()),
        "specific_high_counts": {t: int(spec_high[t].sum()) for t in TASKS},
        "low_envelope_counts": {t: int(low_env[t].sum()) for t in TASKS},
        "specific_high_overlap": overlap_rows,
        "interlink_dacc_env": dacc_env.tolist(),
        "interlink_dloss_env": dloss_env.tolist(),
        "interlink_dacc_high": dacc_hi.tolist(),
        "interlink_dloss_high": dloss_hi.tolist(),
        "coding_envelope_characterization": char,
        "params": {"score_stat": SCORE_STAT, "ablation_mode": ABLATION_MODE,
                   "envelope_budget": ENVELOPE_BUDGET, "core_pct": CORE_PCT,
                   "spec_high": SPEC_HIGH, "spec_low": SPEC_LOW},
    }
    with open(results_dir / "interlink_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_manifest(results_dir / "interlink_manifest.json", extra={
        "model_name": model_cfg["model_name"], "dtype": model_cfg["dtype"],
        "device": model_cfg["device"], "seed": model_cfg["seed"],
        "total_mlp_neurons": total_neurons,
    })
    print(f"\nSaved interlink results to {results_dir}")
    print("Matrices (rows=targeted, cols=measured):")
    print("delta-acc env:\n", np.round(dacc_env, 3))
    print("delta-acc specific-high:\n", np.round(dacc_hi, 3))


if __name__ == "__main__":
    main()
