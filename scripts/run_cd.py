"""Run ONLY deliverables C (global stress sweep) + D (15x15 interlink) for a
comprehensive run, reusing already-captured baselines.json + aggregates_mean_abs.npz.

Avoids pyarrow/matplotlib inside the torch process (pyarrow write_table segfaults
on Windows when a torch-CUDA context is live). Writes JSON / JSONL only.
PNGs + parquet are produced afterwards by a separate torch-free script.

Usage: python -m scripts.run_cd <model_name> <outdir> [max_fraction]
"""
from __future__ import annotations
import json, sys, time, subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ablation import attach_ablation_hooks_from_mask, remove_ablation_hooks
from src.capture import compute_calibration_means
from src.eval.base import load_battery, set_all_seeds, batched_teacher_forced_nll
from src.eval.dispatch import gen_accuracy
from src.models import load_config, load_model
from src.scoring import percentile_rank_per_layer

CAPS = ["coding", "math", "formal_logic", "grammar", "translation",
        "reading_comprehension", "history_facts", "philosophy", "science_facts",
        "commonsense", "problem_solving", "creative_writing", "summarization",
        "spatial_pattern", "ethics"]
THRESH_MAIN = 0.80
ENV_BUDGET = 0.05
SEED = 1234
MAXNEW_SWEEP = 128


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gitcommit(msg):
    try:
        subprocess.run(["git", "add", "-A"], cwd=REPO_ROOT, capture_output=True)
        subprocess.run(["git", "-c", "user.name=Claude", "-c", "user.email=noreply@anthropic.com",
                        "commit", "-q", "-m", msg], cwd=REPO_ROOT, capture_output=True)
    except Exception:
        pass


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    outdir = REPO_ROOT / (sys.argv[2] if len(sys.argv) > 2 else "results/comprehensive_qwen0.5b")
    max_fraction = float(sys.argv[3]) if len(sys.argv) > 3 else 0.70
    outdir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config(REPO_ROOT / "configs/model.yaml"); model_cfg["model_name"] = model_name
    eval_cfg = load_config(REPO_ROOT / "configs/eval.yaml")
    set_all_seeds(SEED)
    max_new = eval_cfg["decoding"]["max_new_tokens"]; exec_to = eval_cfg["exec_timeout_sec"]; tol = eval_cfg["numeric_tolerance"]

    log(f"loading {model_name}")
    bundle = load_model(model_cfg)
    L, I = bundle.num_layers(), bundle.intermediate_size(); total = L * I
    log(f"layers={L} intermediate={I} total={total}")

    bat = {c: load_battery(REPO_ROOT / f"data/batteries/comprehensive/{c}.jsonl") for c in CAPS}
    eval_types = {c: bat[c][0].eval_type for c in CAPS}
    pairs = {c: [(it.prompt, it.gold) for it in bat[c]] for c in CAPS}

    base = json.load(open(outdir / "baselines.json"))
    base_nll = {c: base[c]["nll"] for c in CAPS}
    base_acc = {c: base[c]["acc"] for c in CAPS}

    npz = np.load(outdir / "aggregates_mean_abs.npz")
    ranks = {c: percentile_rank_per_layer(npz[c]) for c in CAPS}
    log("computing calibration means")
    calib = compute_calibration_means(bundle, [it for c in CAPS for it in bat[c]])

    stack = np.stack([(ranks[c] >= THRESH_MAIN).reshape(-1) for c in CAPS], axis=0)
    active_count = stack.sum(axis=0)
    core_mask_flat = (active_count >= 13)
    global_rank = np.mean(np.stack([ranks[c].reshape(-1) for c in CAPS], axis=0), axis=0)

    def mask_from_flat(idx):
        m = np.zeros(total, dtype=bool); m[idx] = True; return m.reshape(L, I)

    def nll_all(mask):
        h = attach_ablation_hooks_from_mask(bundle, mask, "mean", calib)
        try:
            return {c: batched_teacher_forced_nll(bundle, pairs[c])["mean_nll"] for c in CAPS}
        finally:
            remove_ablation_hooks(h)

    def acc_all(mask, mn):
        h = attach_ablation_hooks_from_mask(bundle, mask, "mean", calib)
        try:
            out = {}
            for c in CAPS:
                a, _ = gen_accuracy(bundle, bat[c], eval_types[c], mn, exec_to, tol); out[c] = a
            return out
        finally:
            remove_ablation_hooks(h)

    # ============ C: GLOBAL STRESS SWEEP ============
    log(f"C: global stress sweep to {max_fraction*100:.0f}%")
    step = max(1, round(0.001 * total)); n_steps = int(max_fraction * total) // step
    seeds = [0, 1, 2, 3, 4]
    order_global = np.argsort(global_rank, kind="mergesort")
    rng_orders = {s: np.random.default_rng(s).permutation(total) for s in seeds}
    gen_every = 70; rand_every = 10
    fail_frac = {c: None for c in CAPS}
    jl = open(outdir / "C_stress_rows.jsonl", "w")
    t0 = time.time()
    for st in range(0, n_steps + 1):
        cum = st * step; frac = cum / total
        do_gen = (st % gen_every == 0) or (st == n_steps)
        gmask = mask_from_flat(order_global[:cum]) if cum > 0 else np.zeros((L, I), bool)
        gn = nll_all(gmask)
        row = {"step": st, "frac": frac, "control": "global", "seed": -1}
        for c in CAPS:
            row[f"nll_{c}"] = gn[c]; row[f"nllratio_{c}"] = gn[c] / base_nll[c]
        if do_gen:
            ga = acc_all(gmask, MAXNEW_SWEEP)
            for c in CAPS:
                row[f"acc_{c}"] = ga[c]
                if base_acc[c]:
                    row[f"ret_{c}"] = (ga[c] / base_acc[c]) if ga[c] is not None else None
        for c in CAPS:
            if fail_frac[c] is None and gn[c] / base_nll[c] > 2.0:
                fail_frac[c] = frac
        jl.write(json.dumps(row) + "\n")
        if st % rand_every == 0 or st == n_steps:
            for s in seeds:
                rmask = mask_from_flat(rng_orders[s][:cum]) if cum > 0 else np.zeros((L, I), bool)
                rn = nll_all(rmask)
                rr = {"step": st, "frac": frac, "control": "random", "seed": s}
                for c in CAPS:
                    rr[f"nll_{c}"] = rn[c]; rr[f"nllratio_{c}"] = rn[c] / base_nll[c]
                jl.write(json.dumps(rr) + "\n")
        if st % 20 == 0 or do_gen:
            jl.flush()
            worst = max(CAPS, key=lambda c: gn[c] / base_nll[c])
            log(f"  step {st}/{n_steps} frac={frac:.3f} worst={worst}:{gn[worst]/base_nll[worst]:.2f} el={time.time()-t0:.0f}s")
    jl.close()
    ranking = sorted(CAPS, key=lambda c: (fail_frac[c] if fail_frac[c] is not None else 1.0))
    C = {"max_fraction": max_fraction, "failure_def": "nll_ratio>2 (global control)",
         "failure_fraction_pct": {c: fail_frac[c] for c in CAPS},
         "fragility_ranking_most_to_least_fragile": ranking}
    json.dump(C, open(outdir / "C_stress.json", "w"), indent=2)
    log(f"  fragility (most fragile first): {ranking}")
    gitcommit(f"comprehensive C (jsonl) {model_name}")

    # ============ D: 15x15 INTERLINK ============
    log("D: 15x15 interlink")
    def env_mask(c, desc=False):
        r = ranks[c].reshape(-1); elig = np.nonzero(~core_mask_flat)[0]
        order = elig[np.argsort(r[elig], kind="mergesort")]
        if desc: order = order[::-1]
        k = int(ENV_BUDGET * total); return mask_from_flat(order[:k])
    n = len(CAPS)
    dacc_env = np.zeros((n, n)); dloss_env = np.zeros((n, n))
    dacc_hi = np.zeros((n, n)); dloss_hi = np.zeros((n, n))
    t1 = time.time()
    for i, tgt in enumerate(CAPS):
        for desc, dacc, dloss in [(False, dacc_env, dloss_env), (True, dacc_hi, dloss_hi)]:
            m = env_mask(tgt, desc)
            na = nll_all(m); aa = acc_all(m, MAXNEW_SWEEP)
            for j, col in enumerate(CAPS):
                dloss[i, j] = na[col] - base_nll[col]
                if base_acc[col] is not None and aa[col] is not None:
                    dacc[i, j] = aa[col] - base_acc[col]
        json.dump({"budget": ENV_BUDGET, "caps": CAPS,
                   "dacc_env": dacc_env.tolist(), "dloss_env": dloss_env.tolist(),
                   "dacc_high": dacc_hi.tolist(), "dloss_high": dloss_hi.tolist()},
                  open(outdir / "D_interlink.json", "w"), indent=2)
        log(f"  target {tgt} done ({time.time()-t1:.0f}s)")
    gitcommit(f"comprehensive D (json) {model_name}")

    json.dump({"model_name": model_name, "seed": SEED, "threshold_main": THRESH_MAIN,
               "threshold_reason": "per-layer percentile (SiLU scale unbounded/layer-dependent)",
               "total_mlp_neurons": total, "max_fraction": max_fraction,
               "git_hash": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip(),
               "timestamp": datetime.now().isoformat(), "caps": CAPS},
              open(outdir / "manifest.json", "w"), indent=2)
    log(f"DONE C+D -> {outdir}")


if __name__ == "__main__":
    main()
