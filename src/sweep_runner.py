"""Loop target x percentile-step x control x ablation-mode -> parquet rows.
Identical eval (same prompts, decoding, max tokens, parser, seed) is used at
every single step -- eval config is loaded once from configs/eval.yaml and
never varies across the sweep.
"""
from __future__ import annotations

import time

import numpy as np

from src.ablation import (
    ControlOrders,
    attach_ablation_hooks,
    channels_for_step,
    remove_ablation_hooks,
)
from src.eval.base import BatteryItem
from src.eval.coding import eval_coding_battery
from src.eval.loss import eval_loss_battery
from src.eval.math import eval_math_battery


def run_sweep(
    bundle,
    items: list[BatteryItem],
    target_task: str,
    other_tasks: list[str],
    orders: ControlOrders,
    calibration_means: np.ndarray,
    step_fraction: float,
    max_fraction: float,
    controls: list[str],
    random_seeds: list[int],
    ablation_mode: str,
    score_stat: str,
    loss_eval_every_steps: int,
    gen_eval_every_steps: int,
    max_new_tokens: int,
    exec_timeout_sec: int,
    numeric_tolerance: float,
    verbose: bool = True,
) -> list[dict]:
    n_layers = bundle.num_layers()
    total_neurons = n_layers * bundle.intermediate_size()
    eligible_pool = len(orders.low_order)
    step_size = max(1, round(step_fraction * total_neurons))
    max_cumulative = min(eligible_pool, round(max_fraction * total_neurons))
    n_steps = max_cumulative // step_size

    rows = []

    def eval_at(cumulative_n: int, control: str, seed: int | None, step_idx: int) -> dict:
        if cumulative_n == 0:
            channels_by_layer = [np.array([], dtype=np.int64) for _ in range(n_layers)]
        else:
            channels_by_layer = channels_for_step(control, cumulative_n, orders, n_layers, seed=seed)
        n_ablated = sum(len(c) for c in channels_by_layer)

        handles = []
        if n_ablated > 0:
            handles = attach_ablation_hooks(bundle, channels_by_layer, ablation_mode, calibration_means)

        row = {
            "step_idx": step_idx,
            "cumulative_n": int(cumulative_n),
            "n_ablated": int(n_ablated),
            "frac_removed": cumulative_n / total_neurons,
            "control": control,
            "seed": -1 if seed is None else seed,
            "ablation_mode": ablation_mode,
            "score_stat": score_stat,
            "target_task": target_task,
        }

        do_loss = (step_idx % loss_eval_every_steps == 0)
        do_gen = (step_idx % gen_eval_every_steps == 0) or (step_idx == n_steps)

        try:
            if do_loss:
                loss_target = eval_loss_battery(bundle, items, task=target_task)
                row["nll_target"] = loss_target["mean_nll"]
                row["ppl_target"] = loss_target["mean_ppl"]
                for ot in other_tasks:
                    loss_other = eval_loss_battery(bundle, items, task=ot)
                    row[f"nll_{ot}"] = loss_other["mean_nll"]
                    row[f"ppl_{ot}"] = loss_other["mean_ppl"]

            if do_gen:
                if target_task == "coding":
                    gen_target = eval_coding_battery(bundle, items, max_new_tokens, exec_timeout_sec)
                    row["acc_target"] = gen_target["pass_at_1"]
                    row["parse_fail_target"] = gen_target["parse_failure_rate"]
                elif target_task == "math":
                    gen_target = eval_math_battery(bundle, items, max_new_tokens, numeric_tolerance)
                    row["acc_target"] = gen_target["accuracy"]
                    row["parse_fail_target"] = gen_target["parse_failure_rate"]

                # NOTE: collateral (other_tasks) generation-eval is deliberately
                # run only at step 0 and the final step (not every gen-eval
                # point) to keep the smoke run's wall-clock bounded. The
                # per-step SMOOTH loss/NLL track above still runs for
                # other_tasks at every step -- that's the mandatory
                # "which other skills survive vs die" signal for this run.
                if step_idx == 0 or step_idx == n_steps:
                    for ot in other_tasks:
                        if ot == "coding":
                            gen_other = eval_coding_battery(bundle, items, max_new_tokens, exec_timeout_sec)
                            row[f"acc_{ot}"] = gen_other["pass_at_1"]
                            row[f"parse_fail_{ot}"] = gen_other["parse_failure_rate"]
                        elif ot == "math":
                            gen_other = eval_math_battery(bundle, items, max_new_tokens, numeric_tolerance)
                            row[f"acc_{ot}"] = gen_other["accuracy"]
                            row[f"parse_fail_{ot}"] = gen_other["parse_failure_rate"]
        finally:
            remove_ablation_hooks(handles)

        return row

    # step 0 baseline (shared across controls -- record once per control for
    # plotting convenience, since seed doesn't matter with 0 ablated)
    t0 = time.time()
    for control in controls:
        seeds = random_seeds if control == "random" else [None]
        for seed in seeds:
            row = eval_at(0, control, seed, step_idx=0)
            rows.append(row)
            if verbose:
                print(f"[sweep] step=0 control={control} seed={seed} frac=0.0000 "
                      f"nll_target={row.get('nll_target')} acc_target={row.get('acc_target')}")

    for step_idx in range(1, n_steps + 1):
        cumulative_n = step_idx * step_size
        for control in controls:
            seeds = random_seeds if control == "random" else [None]
            for seed in seeds:
                row = eval_at(cumulative_n, control, seed, step_idx=step_idx)
                rows.append(row)
                if verbose:
                    print(
                        f"[sweep] step={step_idx}/{n_steps} control={control} seed={seed} "
                        f"frac={row['frac_removed']:.4f} nll_target={row.get('nll_target')} "
                        f"acc_target={row.get('acc_target')} elapsed={time.time()-t0:.1f}s"
                    )

    return rows
