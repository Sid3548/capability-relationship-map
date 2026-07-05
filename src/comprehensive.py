"""Comprehensive capability<->neuron map + break-point study (0.5B, reusable for 3B).

Deliverables (checkpointed to <outdir> after each):
  A OVERLAP MATRIX   15x15 active-set overlap (overlap-coeff headline + Jaccard),
                     at top-10/20/30% thresholds (sensitivity).
  B ALLOCATION MAP   classify all neurons: DEAD/EXCLUSIVE/SHARED/CORE.
  C GLOBAL STRESS    remove ascending by GLOBAL importance to ~70%, random control
                     (>=5 seeds), NLL all 15 every step (batched), gen at coarse
                     checkpoints; per-task retention/nll_ratio -> fragility ranking.
  D 15x15 INTERLINK  remove each cap's low-envelope (5%) and specific-high (5%),
                     eval all 15 -> delta_acc + delta_loss matrices.

Guardrails: score down_proj-input h (abs); per-layer percentile threshold;
mean-ablate default + zero-ablate available; reversible hooks; <=0.1% steps;
identical eval every step; greedy determinism. Honest framing: "active-set
overlap under this activation basis", not literal localization.

Usage: python -m src.comprehensive <model_name> <outdir> [max_fraction]
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ablation import attach_ablation_hooks_from_mask, remove_ablation_hooks
from src.capture import capture_task_aggregates, compute_calibration_means
from src.eval.base import load_battery, set_all_seeds, batched_teacher_forced_nll
from src.eval.dispatch import gen_accuracy
from src.models import load_config, load_model
from src.plotting import plot_interlink_heatmap
from src.scoring import percentile_rank_per_layer
from src.storage import save_rows_parquet, write_manifest, git_hash

CAPS = ["coding", "math", "formal_logic", "grammar", "translation",
        "reading_comprehension", "history_facts", "philosophy", "science_facts",
        "commonsense", "problem_solving", "creative_writing", "summarization",
        "spatial_pattern", "ethics"]
THRESH_MAIN = 0.80   # active = mean_abs rank >= 0.80 (top 20%) within its layer
THRESH_SENS = [0.90, 0.80, 0.70]  # top-10/20/30% sensitivity
ENV_BUDGET = 0.05
SEED = 1234


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def active_mask(rank, thresh):
    return rank >= thresh


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    outdir = REPO_ROOT / (sys.argv[2] if len(sys.argv) > 2 else "results/comprehensive_qwen0.5b")
    max_fraction = float(sys.argv[3]) if len(sys.argv) > 3 else 0.70
    outdir.mkdir(parents=True, exist_ok=True)

    model_cfg = load_config(REPO_ROOT / "configs/model.yaml")
    model_cfg["model_name"] = model_name
    eval_cfg = load_config(REPO_ROOT / "configs/eval.yaml")
    set_all_seeds(SEED)
    max_new = eval_cfg["decoding"]["max_new_tokens"]
    exec_to = eval_cfg["exec_timeout_sec"]
    tol = eval_cfg["numeric_tolerance"]

    log(f"loading {model_name}")
    bundle = load_model(model_cfg)
    L, I = bundle.num_layers(), bundle.intermediate_size()
    total = L * I
    log(f"layers={L} intermediate={I} total_mlp_neurons={total}")

    bat = {c: load_battery(REPO_ROOT / f"data/batteries/comprehensive/{c}.jsonl") for c in CAPS}
    eval_types = {c: bat[c][0].eval_type for c in CAPS}
    pairs = {c: [(it.prompt, it.gold) for it in bat[c]] for c in CAPS}

    # ---------- BASELINES ----------
    log("baselines: gen accuracy + batched NLL for all 15")
    base = {}
    for c in CAPS:
        acc, pf = gen_accuracy(bundle, bat[c], eval_types[c], max_new, exec_to, tol)
        nll = batched_teacher_forced_nll(bundle, pairs[c])["mean_nll"]
        base[c] = {"acc": acc, "parse_fail": pf, "nll": nll, "eval_type": eval_types[c]}
        log(f"  {c:22s} acc={acc if acc is None else round(acc,3)} nll={nll:.4f}")
    json.dump(base, open(outdir / "baselines.json", "w"), indent=2)

    # ---------- CAPTURE ----------
    log("capturing per-cap aggregates")
    aggs = {}
    for c in CAPS:
        aggs[c] = capture_task_aggregates(bundle, bat[c], task=c, verbose=False)
    np.savez(outdir / "aggregates_mean_abs.npz", **{c: aggs[c]["mean_abs"] for c in CAPS})
    calib = compute_calibration_means(bundle, [it for c in CAPS for it in bat[c]])

    ranks = {c: percentile_rank_per_layer(aggs[c]["mean_abs"]) for c in CAPS}

    # ================= DELIVERABLE A: OVERLAP MATRIX =================
    log("Deliverable A: overlap matrix")
    def overlap_matrices(thresh):
        masks = {c: active_mask(ranks[c], thresh).reshape(-1) for c in CAPS}
        sizes = {c: int(masks[c].sum()) for c in CAPS}
        n = len(CAPS)
        oc = np.zeros((n, n)); jac = np.zeros((n, n))
        for i, a in enumerate(CAPS):
            for j, b in enumerate(CAPS):
                inter = int((masks[a] & masks[b]).sum())
                union = int((masks[a] | masks[b]).sum())
                mn = min(sizes[a], sizes[b])
                oc[i, j] = inter / mn if mn else 0.0
                jac[i, j] = inter / union if union else 0.0
        return oc, jac, sizes
    A = {}
    for th in THRESH_SENS:
        oc, jac, sizes = overlap_matrices(th)
        A[f"top{int((1-th)*100)}"] = {"overlap_coeff": oc.tolist(), "jaccard": jac.tolist(),
                                       "active_sizes": sizes}
    oc_main, jac_main, sizes_main = overlap_matrices(THRESH_MAIN)
    # most/least entangled pairs (headline = overlap-coeff at top-20%), exclude diagonal
    pairs_oc = []
    for i in range(len(CAPS)):
        for j in range(i + 1, len(CAPS)):
            pairs_oc.append((CAPS[i], CAPS[j], oc_main[i, j], jac_main[i, j]))
    pairs_oc.sort(key=lambda x: x[2])
    A["headline_threshold"] = "top20_overlap_coefficient"
    A["most_entangled"] = [[a, b, round(o, 4), round(jc, 4)] for a, b, o, jc in pairs_oc[-5:][::-1]]
    A["least_entangled"] = [[a, b, round(o, 4), round(jc, 4)] for a, b, o, jc in pairs_oc[:5]]
    json.dump(A, open(outdir / "A_overlap.json", "w"), indent=2)
    save_rows_parquet(outdir / "A_overlap_rows.parquet",
                      [{"a": a, "b": b, "overlap_coeff": float(o), "jaccard": float(jc)}
                       for a, b, o, jc in pairs_oc])
    try:
        plot_interlink_heatmap(oc_main, CAPS, "Active-set overlap coefficient (top-20%)",
                               outdir / "A_overlap_coeff.png", cmap="viridis", fmt="{:.2f}", vlim=1.0)
    except Exception as e:
        log(f"  heatmap A warn: {e}")
    log(f"  most entangled: {A['most_entangled'][0]}; least: {A['least_entangled'][0]}")

    # ================= DELIVERABLE B: ALLOCATION MAP =================
    log("Deliverable B: allocation map")
    stack = np.stack([active_mask(ranks[c], THRESH_MAIN).reshape(-1) for c in CAPS], axis=0)  # [15, total]
    active_count = stack.sum(axis=0)  # per neuron, how many caps active
    dead = int((active_count == 0).sum())
    exclusive = int((active_count == 1).sum())
    shared = int(((active_count >= 2) & (active_count <= 12)).sum())
    core = int((active_count >= 13).sum())
    per_cap_excl = {}
    excl_mask = (active_count == 1)
    for i, c in enumerate(CAPS):
        per_cap_excl[c] = int((stack[i] & excl_mask).sum())
    B = {"total_neurons": total, "threshold": THRESH_MAIN,
         "buckets": {"dead": dead, "exclusive": exclusive, "shared": shared, "core": core},
         "buckets_pct": {"dead": dead / total, "exclusive": exclusive / total,
                         "shared": shared / total, "core": core / total},
         "per_cap_exclusive_count": per_cap_excl,
         "per_cap_exclusive_pct": {c: per_cap_excl[c] / total for c in CAPS},
         "active_count_histogram": {int(k): int((active_count == k).sum()) for k in range(16)}}
    json.dump(B, open(outdir / "B_allocation.json", "w"), indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].bar(["dead", "exclusive", "shared", "core"],
                  [dead, exclusive, shared, core],
                  color=["#888", "#5fd0a0", "#7aa2ff", "#f2b45f"])
        ax[0].set_title(f"Neuron allocation (top-20% active, {total} neurons)")
        ax[0].set_ylabel("neurons")
        for k, v in enumerate([dead, exclusive, shared, core]):
            ax[0].text(k, v, f"{100*v/total:.1f}%", ha="center", va="bottom")
        ax[1].bar(range(len(CAPS)), [per_cap_excl[c] for c in CAPS], color="#5fd0a0")
        ax[1].set_xticks(range(len(CAPS))); ax[1].set_xticklabels(CAPS, rotation=60, ha="right", fontsize=8)
        ax[1].set_title("Exclusive neurons per capability")
        fig.tight_layout(); fig.savefig(outdir / "B_allocation.png", dpi=150); plt.close(fig)
    except Exception as e:
        log(f"  plot B warn: {e}")
    log(f"  dead={dead/total:.3f} exclusive={exclusive/total:.3f} shared={shared/total:.3f} core={core/total:.3f}")

    # global importance = mean percentile rank across all 15 caps (higher=more used)
    global_rank = np.mean(np.stack([ranks[c].reshape(-1) for c in CAPS], axis=0), axis=0)  # [total]

    # checkpoint git
    import subprocess
    try:
        subprocess.run(["git", "add", "-A"], cwd=REPO_ROOT, capture_output=True)
        subprocess.run(["git", "-c", "user.name=Claude", "-c", "user.email=noreply@anthropic.com",
                        "commit", "-q", "-m", f"comprehensive A+B checkpoint ({model_name})"],
                       cwd=REPO_ROOT, capture_output=True)
    except Exception:
        pass

    # ================= DELIVERABLE C: GLOBAL STRESS SWEEP =================
    log(f"Deliverable C: global stress sweep to {max_fraction*100:.0f}%")
    step = max(1, round(0.001 * total))
    n_steps = int(max_fraction * total) // step
    seeds = [0, 1, 2, 3, 4]
    order_global = np.argsort(global_rank, kind="mergesort")  # ascending: least-used first
    rng_orders = {s: np.random.default_rng(s).permutation(total) for s in seeds}
    base_nll = {c: base[c]["nll"] for c in CAPS}
    base_acc = {c: base[c]["acc"] for c in CAPS}

    def mask_from_flat(flat_idx):
        m = np.zeros(total, dtype=bool); m[flat_idx] = True
        return m.reshape(L, I)

    def nll_all(mask):
        handles = attach_ablation_hooks_from_mask(bundle, mask, "mean", calib)
        try:
            out = {c: batched_teacher_forced_nll(bundle, pairs[c])["mean_nll"] for c in CAPS}
        finally:
            remove_ablation_hooks(handles)
        return out

    def acc_all(mask):
        handles = attach_ablation_hooks_from_mask(bundle, mask, "mean", calib)
        try:
            out = {}
            for c in CAPS:
                a, _ = gen_accuracy(bundle, bat[c], eval_types[c], max_new, exec_to, tol)
                out[c] = a
        finally:
            remove_ablation_hooks(handles)
        return out

    rows_c = []
    gen_every = max(1, int(round(0.05 / 0.001)))  # ~every 5%
    fail_frac = {c: None for c in CAPS}  # global-control failure fraction (retention<0.5 or nll_ratio>2)
    t0 = time.time()
    for st in range(0, n_steps + 1):
        cum = st * step
        frac = cum / total
        do_gen = (st % gen_every == 0) or (st == n_steps)
        # global control
        gmask = mask_from_flat(order_global[:cum]) if cum > 0 else np.zeros((L, I), bool)
        gn = nll_all(gmask)
        row = {"step": st, "frac": frac, "control": "global", "seed": -1}
        for c in CAPS:
            row[f"nll_{c}"] = gn[c]
            row[f"nllratio_{c}"] = gn[c] / base_nll[c]
        if do_gen:
            ga = acc_all(gmask)
            for c in CAPS:
                row[f"acc_{c}"] = ga[c]
                if base_acc[c]:
                    row[f"ret_{c}"] = (ga[c] / base_acc[c]) if ga[c] is not None else None
        # record first failure fraction per cap (global control) on NLL ratio (available every step)
        for c in CAPS:
            if fail_frac[c] is None and gn[c] / base_nll[c] > 2.0:
                fail_frac[c] = frac
        rows_c.append(row)
        # random control
        for s in seeds:
            rmask = mask_from_flat(rng_orders[s][:cum]) if cum > 0 else np.zeros((L, I), bool)
            rn = nll_all(rmask)
            rr = {"step": st, "frac": frac, "control": "random", "seed": s}
            for c in CAPS:
                rr[f"nll_{c}"] = rn[c]
                rr[f"nllratio_{c}"] = rn[c] / base_nll[c]
            if do_gen:
                ra = acc_all(rmask)
                for c in CAPS:
                    rr[f"acc_{c}"] = ra[c]
            rows_c.append(rr)
        if st % 20 == 0 or do_gen:
            worst = max(CAPS, key=lambda c: gn[c] / base_nll[c])
            log(f"  step {st}/{n_steps} frac={frac:.3f} worst-NLLratio={worst}:{gn[worst]/base_nll[worst]:.2f} "
                f"elapsed={time.time()-t0:.0f}s")
            save_rows_parquet(outdir / "C_stress_rows.parquet", rows_c)  # incremental checkpoint
    save_rows_parquet(outdir / "C_stress_rows.parquet", rows_c)

    # fragility ranking = order caps fail (earliest failure fraction first)
    ranking = sorted(CAPS, key=lambda c: (fail_frac[c] if fail_frac[c] is not None else 1.0))
    C = {"max_fraction": max_fraction, "failure_def": "nll_ratio>2 (global control)",
         "failure_fraction_pct": {c: (fail_frac[c] if fail_frac[c] is not None else None) for c in CAPS},
         "fragility_ranking_most_to_least_fragile": ranking}
    json.dump(C, open(outdir / "C_stress.json", "w"), indent=2)
    log(f"  fragility (most fragile first): {ranking[:5]} ...")

    # retention curves overlay + failure bar
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        import pandas as pd
        df = pd.DataFrame(rows_c)
        g = df[df.control == "global"].sort_values("frac")
        fig, ax = plt.subplots(figsize=(11, 6))
        for c in CAPS:
            ax.plot(g["frac"] * 100, g[f"nllratio_{c}"], label=c, lw=1)
        ax.axhline(2.0, color="k", ls="--", lw=0.8, label="failure (2x NLL)")
        ax.set_xlabel("% neurons removed (global least-used-first)")
        ax.set_ylabel("NLL ratio vs baseline"); ax.set_yscale("log")
        ax.set_title("Global stress: per-capability NLL degradation")
        ax.legend(fontsize=7, ncol=2); fig.tight_layout()
        fig.savefig(outdir / "C_retention_curves.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 5))
        ff = [(fail_frac[c] if fail_frac[c] is not None else max_fraction) * 100 for c in ranking]
        ax.bar(range(len(ranking)), ff, color="#ff7a7a")
        ax.set_xticks(range(len(ranking))); ax.set_xticklabels(ranking, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("% removed at failure (2x NLL)")
        ax.set_title("Fragility ranking (lower = more fragile)")
        fig.tight_layout(); fig.savefig(outdir / "C_fragility_bar.png", dpi=150); plt.close(fig)
    except Exception as e:
        log(f"  plot C warn: {e}")

    try:
        subprocess.run(["git", "add", "-A"], cwd=REPO_ROOT, capture_output=True)
        subprocess.run(["git", "-c", "user.name=Claude", "-c", "user.email=noreply@anthropic.com",
                        "commit", "-q", "-m", f"comprehensive C checkpoint ({model_name})"],
                       cwd=REPO_ROOT, capture_output=True)
    except Exception:
        pass

    # ================= DELIVERABLE D: 15x15 INTERLINK =================
    log("Deliverable D: 15x15 interlink")
    core_mask_flat = (active_count >= 13)  # reuse core from B for exclusion
    def env_mask(c):
        r = ranks[c].reshape(-1).copy()
        elig = np.nonzero(~core_mask_flat)[0]
        order = elig[np.argsort(r[elig], kind="mergesort")]
        k = int(ENV_BUDGET * total)
        return mask_from_flat(order[:k])
    def high_mask(c):
        r = ranks[c].reshape(-1).copy()
        elig = np.nonzero(~core_mask_flat)[0]
        order = elig[np.argsort(r[elig], kind="mergesort")[::-1]]  # descending
        k = int(ENV_BUDGET * total)
        return mask_from_flat(order[:k])

    n = len(CAPS)
    dacc_env = np.zeros((n, n)); dloss_env = np.zeros((n, n))
    dacc_hi = np.zeros((n, n)); dloss_hi = np.zeros((n, n))
    for i, tgt in enumerate(CAPS):
        for cond, mfn, dacc, dloss in [("env", env_mask, dacc_env, dloss_env),
                                       ("high", high_mask, dacc_hi, dloss_hi)]:
            m = mfn(tgt)
            na = nll_all(m); aa = acc_all(m)
            for j, col in enumerate(CAPS):
                dloss[i, j] = na[col] - base_nll[col]
                if base_acc[col] is not None and aa[col] is not None:
                    dacc[i, j] = aa[col] - base_acc[col]
        log(f"  target {tgt} done ({time.time()-t0:.0f}s)")
    D = {"budget": ENV_BUDGET,
         "dacc_env": dacc_env.tolist(), "dloss_env": dloss_env.tolist(),
         "dacc_high": dacc_hi.tolist(), "dloss_high": dloss_hi.tolist(), "caps": CAPS}
    json.dump(D, open(outdir / "D_interlink.json", "w"), indent=2)
    for mat, name, title in [(dacc_env, "D_env_dacc", "Interlink Δacc: remove LOW-ENVELOPE"),
                             (dloss_env, "D_env_dloss", "Interlink Δloss: remove LOW-ENVELOPE"),
                             (dacc_hi, "D_high_dacc", "Interlink Δacc: remove SPECIFIC-HIGH"),
                             (dloss_hi, "D_high_dloss", "Interlink Δloss: remove SPECIFIC-HIGH")]:
        try:
            plot_interlink_heatmap(np.array(mat), CAPS, title, outdir / f"{name}.png")
        except Exception as e:
            log(f"  plot {name} warn: {e}")

    write_manifest(outdir / "manifest.json", extra={
        "model_name": model_name, "seed": SEED, "threshold_main": THRESH_MAIN,
        "threshold_reason": "per-layer percentile (SiLU scale unbounded/layer-dependent; raw cutoff not comparable)",
        "total_mlp_neurons": total, "max_fraction": max_fraction,
        "timestamp": datetime.now().isoformat(), "caps": CAPS,
    })
    log(f"DONE. all deliverables saved to {outdir}")


if __name__ == "__main__":
    main()
