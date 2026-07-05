"""Coding cliff sweep (Job 2e): push the low-activation envelope past 10% on
the cheap teacher-forced NLL track to locate where coding finally breaks.

Reuses the exact scoring + ablation + sweep machinery from the smoke run,
just with a higher max_fraction and coarser generation checkpoints. 0.5B only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ablation import build_control_orders
from src.capture import capture_task_aggregates, compute_calibration_means
from src.eval.base import load_battery, set_all_seeds
from src.models import load_config, load_model
from src.scoring import ScoreBundle, compute_core_mask
from src.storage import save_rows_parquet
from src.sweep_runner import run_sweep


def main():
    model_cfg = load_config(REPO_ROOT / "configs/model.yaml")
    eval_cfg = load_config(REPO_ROOT / "configs/eval.yaml")
    sweep_cfg = load_config(REPO_ROOT / "configs/sweep_cliff.yaml")
    set_all_seeds(model_cfg["seed"])

    bundle = load_model(model_cfg)
    print(f"loaded; total_mlp_neurons={bundle.arch.total_mlp_neurons}")

    items = load_battery(REPO_ROOT / "data/batteries/coding.jsonl") \
        + load_battery(REPO_ROOT / "data/batteries/math.jsonl")

    max_new_tokens = eval_cfg["decoding"]["max_new_tokens"]
    exec_timeout = eval_cfg["exec_timeout_sec"]
    tol = eval_cfg["numeric_tolerance"]
    score_stat = sweep_cfg["score_stat"]

    agg_c = capture_task_aggregates(bundle, items, task="coding", verbose=True)
    agg_m = capture_task_aggregates(bundle, items, task="math", verbose=True)
    calib = compute_calibration_means(bundle, items)

    sc = ScoreBundle("coding", agg_c["mean_abs"], agg_c["max_abs"])
    sm = ScoreBundle("math", agg_m["mean_abs"], agg_m["max_abs"])
    core = compute_core_mask(sc.rank(score_stat), [sm.rank(score_stat)],
                             core_percentile=sweep_cfg["core_percentile"])
    print(f"core neurons: {int(core.sum())}")

    orders = build_control_orders(sc.rank(score_stat), core, sweep_cfg["random_seeds"])

    rows = run_sweep(
        bundle=bundle, items=items, target_task="coding", other_tasks=["math"],
        orders=orders, calibration_means=calib,
        step_fraction=sweep_cfg["step_fraction"], max_fraction=sweep_cfg["max_fraction"],
        controls=sweep_cfg["controls"], random_seeds=sweep_cfg["random_seeds"],
        ablation_mode=sweep_cfg["ablation_mode"], score_stat=score_stat,
        loss_eval_every_steps=sweep_cfg["loss_eval_every_steps"],
        gen_eval_every_steps=sweep_cfg["gen_eval_every_steps"],
        max_new_tokens=max_new_tokens, exec_timeout_sec=exec_timeout, numeric_tolerance=tol,
    )
    df = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / "cliff_sweep_coding.parquet"
    save_rows_parquet(out, rows)
    print(f"saved {len(rows)} rows to {out}")

    # locate the cliff on the low envelope: NLL loss track
    low = df[(df.control == "low") & df.nll_target.notna()].sort_values("frac_removed")
    base_nll = low.iloc[0].nll_target
    print(f"\nbaseline coding NLL={base_nll:.3f}")
    print("frac% | low_NLL | low_acc(gen)")
    gen = df[(df.control == "low") & df.acc_target.notna()].set_index("frac_removed").acc_target.to_dict()
    cliff_frac = None
    for _, r in low.iterrows():
        acc = gen.get(r.frac_removed)
        acc_s = f"{acc:.2f}" if acc is not None else "  . "
        if r.frac_removed * 100 % 1 < 0.11 or acc is not None:  # print ~every 1% or at gen points
            print(f"{r.frac_removed*100:5.1f} | {r.nll_target:6.3f} | {acc_s}")
        if cliff_frac is None and r.nll_target > 2 * base_nll:
            cliff_frac = r.frac_removed
    print(f"\nCLIFF (low envelope, NLL first exceeds 2x baseline): "
          f"{cliff_frac*100:.1f}%" if cliff_frac else "\nNo 2x-NLL cliff reached within max_fraction")


if __name__ == "__main__":
    main()
