"""CLI: capture | score | sweep | plot | smoke (runs the whole pipeline).

Usage (from repo root, with venv active):
    python -m src.cli smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ablation import build_control_orders
from src.capture import capture_task_aggregates, compute_calibration_means
from src.eval.base import load_battery, set_all_seeds
from src.eval.coding import eval_coding_battery
from src.eval.loss import eval_loss_battery
from src.eval.math import eval_math_battery
from src.models import load_config, load_model
from src.plotting import plot_accuracy_curve, plot_loss_curve
from src.scoring import ScoreBundle, compute_core_mask, print_sanity_topbottom
from src.storage import save_rows_parquet, write_manifest
from src.sweep_runner import run_sweep


def load_all_items(eval_cfg: dict):
    items = []
    for path in eval_cfg["batteries"]:
        items.extend(load_battery(REPO_ROOT / path))
    return items


def cmd_smoke(args):
    model_cfg = load_config(REPO_ROOT / "configs/model.yaml")
    eval_cfg = load_config(REPO_ROOT / "configs/eval.yaml")
    sweep_cfg = load_config(REPO_ROOT / "configs/sweep.yaml")

    set_all_seeds(model_cfg["seed"])

    print("== Loading model ==")
    t0 = time.time()
    bundle = load_model(model_cfg)
    print(f"Loaded {model_cfg['model_name']} in {time.time()-t0:.1f}s")
    print(f"num_layers={bundle.num_layers()} intermediate_size={bundle.intermediate_size()} "
          f"hidden_size={bundle.arch.hidden_size} total_mlp_neurons={bundle.arch.total_mlp_neurons}")
    print(f"cuda available: {torch.cuda.is_available()}, device: {next(bundle.model.parameters()).device}, "
          f"dtype: {next(bundle.model.parameters()).dtype}")

    items = load_all_items(eval_cfg)
    max_new_tokens = eval_cfg["decoding"]["max_new_tokens"]
    exec_timeout = eval_cfg["exec_timeout_sec"]
    tol = eval_cfg["numeric_tolerance"]

    print("\n== Baseline eval (no ablation) ==")
    baseline_coding = eval_coding_battery(bundle, items, max_new_tokens, exec_timeout)
    baseline_math = eval_math_battery(bundle, items, max_new_tokens, tol)
    baseline_loss_coding = eval_loss_battery(bundle, items, task="coding")
    baseline_loss_math = eval_loss_battery(bundle, items, task="math")
    print(f"coding pass@1={baseline_coding['pass_at_1']:.3f} (n={baseline_coding['n']}) "
          f"parse_fail_rate={baseline_coding['parse_failure_rate']:.3f}")
    print(f"math accuracy={baseline_math['accuracy']:.3f} (n={baseline_math['n']}) "
          f"parse_fail_rate={baseline_math['parse_failure_rate']:.3f}")
    print(f"coding NLL={baseline_loss_coding['mean_nll']:.4f} ppl={baseline_loss_coding['mean_ppl']:.4f}")
    print(f"math NLL={baseline_loss_math['mean_nll']:.4f} ppl={baseline_loss_math['mean_ppl']:.4f}")

    for item in baseline_coding["results"]:
        print(f"  [coding {item['id']}] passed={item['passed']} parse_failure={item['parse_failure']}")
    for item in baseline_math["results"]:
        print(f"  [math {item['id']}] passed={item['passed']} pred={item['pred']} gold={item['gold']}")

    print("\n== Activation capture ==")
    agg_coding = capture_task_aggregates(bundle, items, task="coding", verbose=True)
    agg_math = capture_task_aggregates(bundle, items, task="math", verbose=True)
    print(f"coding mean_abs range: [{agg_coding['mean_abs'].min():.5f}, {agg_coding['mean_abs'].max():.5f}], "
          f"finite={np.isfinite(agg_coding['mean_abs']).all()}")
    print(f"math   mean_abs range: [{agg_math['mean_abs'].min():.5f}, {agg_math['mean_abs'].max():.5f}], "
          f"finite={np.isfinite(agg_math['mean_abs']).all()}")

    print("\n== Calibration (mean-ablation values) ==")
    calibration_means = compute_calibration_means(bundle, items)  # [L, I]
    print(f"calibration_means shape={calibration_means.shape}")

    print("\n== Scoring ==")
    score_stat = sweep_cfg["score_stat"]
    score_coding = ScoreBundle("coding", agg_coding["mean_abs"], agg_coding["max_abs"])
    score_math = ScoreBundle("math", agg_math["mean_abs"], agg_math["max_abs"])
    print_sanity_topbottom(agg_coding[score_stat], score_coding.rank(score_stat), "coding", score_stat)
    print_sanity_topbottom(agg_math[score_stat], score_math.rank(score_stat), "math", score_stat)

    target_task = sweep_cfg["target_task"]
    other_tasks = sweep_cfg["other_tasks"]
    target_score = score_coding if target_task == "coding" else score_math
    other_scores = [score_math] if target_task == "coding" else [score_coding]

    core_mask = compute_core_mask(
        target_score.rank(score_stat),
        [s.rank(score_stat) for s in other_scores],
        core_percentile=sweep_cfg["core_percentile"],
    )
    print(f"core neurons: {int(core_mask.sum())} / {core_mask.size} "
          f"({100*core_mask.sum()/core_mask.size:.3f}%)")

    print("\n== Verify single-neuron ablation changes logits ==")
    verify_single_neuron_ablation(bundle, items[0])

    print("\n== Building control orders (low/random/high) ==")
    orders = build_control_orders(
        target_score.rank(score_stat), core_mask, sweep_cfg["random_seeds"]
    )
    print(f"eligible (non-core) pool size: {len(orders.low_order)} / {bundle.arch.total_mlp_neurons}")

    print("\n== Sanity: mask-ablate bottom 0.1% of coding, confirm it runs ==")
    from src.ablation import channels_for_step, attach_ablation_hooks, remove_ablation_hooks
    step_size = max(1, round(sweep_cfg["step_fraction"] * bundle.arch.total_mlp_neurons))
    ch_by_layer = channels_for_step("low", step_size, orders, bundle.num_layers())
    handles = attach_ablation_hooks(bundle, ch_by_layer, sweep_cfg["ablation_mode"], calibration_means)
    smoke_coding = eval_coding_battery(bundle, items, max_new_tokens, exec_timeout)
    remove_ablation_hooks(handles)
    print(f"after removing bottom {step_size} neurons (~0.1%): coding pass@1={smoke_coding['pass_at_1']:.3f}")

    print("\n== Full sweep ==")
    rows = run_sweep(
        bundle=bundle,
        items=items,
        target_task=target_task,
        other_tasks=other_tasks,
        orders=orders,
        calibration_means=calibration_means,
        step_fraction=sweep_cfg["step_fraction"],
        max_fraction=sweep_cfg["max_fraction"],
        controls=sweep_cfg["controls"],
        random_seeds=sweep_cfg["random_seeds"],
        ablation_mode=sweep_cfg["ablation_mode"],
        score_stat=score_stat,
        loss_eval_every_steps=eval_cfg.get("loss_eval_every_steps", sweep_cfg["loss_eval_every_steps"]),
        gen_eval_every_steps=sweep_cfg["gen_eval_every_steps"],
        max_new_tokens=max_new_tokens,
        exec_timeout_sec=exec_timeout,
        numeric_tolerance=tol,
    )

    df = pd.DataFrame(rows)
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    parquet_path = results_dir / "sweep_coding.parquet"
    save_rows_parquet(parquet_path, rows)
    print(f"\nSaved {len(rows)} rows to {parquet_path}")

    manifest = write_manifest(
        results_dir / "run_manifest.json",
        extra={
            "model_name": model_cfg["model_name"],
            "dtype": model_cfg["dtype"],
            "device": model_cfg["device"],
            "seed": model_cfg["seed"],
            "num_layers": bundle.num_layers(),
            "intermediate_size": bundle.intermediate_size(),
            "total_mlp_neurons": bundle.arch.total_mlp_neurons,
            "target_task": target_task,
            "ablation_mode": sweep_cfg["ablation_mode"],
            "score_stat": score_stat,
            "core_neuron_count": int(core_mask.sum()),
            "baseline_coding_pass_at_1": baseline_coding["pass_at_1"],
            "baseline_math_accuracy": baseline_math["accuracy"],
            "baseline_coding_nll": baseline_loss_coding["mean_nll"],
            "baseline_math_nll": baseline_loss_math["mean_nll"],
        },
    )
    print(f"Manifest: {manifest}")

    loss_png = results_dir / "loss_retention_curve.png"
    acc_png = results_dir / "accuracy_retention_curve.png"
    plot_loss_curve(df, target_task, loss_png)
    plot_accuracy_curve(df, target_task, acc_png)
    print(f"Saved plots: {loss_png}, {acc_png}")


@torch.no_grad()
def verify_single_neuron_ablation(bundle, item):
    """Ablate ONE neuron in layer 0 and confirm logits change vs baseline."""
    from src.eval.base import teacher_forced_nll
    from src.ablation import attach_ablation_hooks, remove_ablation_hooks

    nll_info = teacher_forced_nll(bundle, item.prompt, item.gold)
    full_ids, attn_mask = nll_info["full_ids"], nll_info["attn_mask"]

    baseline_logits = bundle.model(input_ids=full_ids, attention_mask=attn_mask).logits.clone()

    dummy_means = np.zeros((bundle.num_layers(), bundle.intermediate_size()), dtype=np.float32)
    channels_by_layer = [np.array([], dtype=np.int64) for _ in range(bundle.num_layers())]
    channels_by_layer[0] = np.array([0], dtype=np.int64)  # neuron 0 of layer 0
    handles = attach_ablation_hooks(bundle, channels_by_layer, "zero", dummy_means)
    ablated_logits = bundle.model(input_ids=full_ids, attention_mask=attn_mask).logits.clone()
    remove_ablation_hooks(handles)

    diff = (baseline_logits - ablated_logits).abs().max().item()
    print(f"single-neuron (layer=0, ch=0) zero-ablation max logit diff: {diff:.6e} "
          f"({'CHANGED' if diff > 0 else 'NO CHANGE -- BUG'})")
    assert diff > 0, "Ablating one neuron did not change logits -- hook wiring is broken"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    sub.add_parser("capture")
    sub.add_parser("score")
    sub.add_parser("sweep")
    sub.add_parser("plot")
    args = parser.parse_args()

    if args.cmd == "smoke":
        cmd_smoke(args)
    else:
        print(f"'{args.cmd}' is available as a standalone step in later iterations; "
              f"for this smoke run use: python -m src.cli smoke")


if __name__ == "__main__":
    main()
