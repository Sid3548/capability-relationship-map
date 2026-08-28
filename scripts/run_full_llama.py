"""Full comprehensive pipeline (A+B+C+D) for Llama-3.1-8B, int8, local dir.
Identical logic to scripts/run_full.py; only model loading swapped for the
int8/bnb/device_map=auto/local-dir/offline path used in scripts/smoke_llama.py.
JSON/npz ONLY -- no pyarrow/matplotlib inside the torch-CUDA process.

Resumable: reuses baselines.json / aggregates_mean_abs.npz if already present.

Usage: python -m scripts.run_full_llama [outdir] [max_fraction]
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import json, sys, time, subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.ablation import attach_ablation_hooks_from_mask, remove_ablation_hooks
from src.capture import capture_task_aggregates, compute_calibration_means
from src.eval.base import load_battery, set_all_seeds, batched_teacher_forced_nll
from src.eval.dispatch import gen_accuracy
from src.models import ModelBundle, ArchInfo, load_config
from src.scoring import percentile_rank_per_layer

MODEL_DIR = r"D:\hf_models\Meta-Llama-3.1-8B"
MODEL_NAME = "Meta-Llama-3.1-8B-int8"

CAPS = ["coding", "math", "formal_logic", "grammar", "translation",
        "reading_comprehension", "history_facts", "philosophy", "science_facts",
        "commonsense", "problem_solving", "creative_writing", "summarization",
        "spatial_pattern", "ethics"]
THRESH_MAIN = 0.80
THRESH_SENS = [0.90, 0.80, 0.70]
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


def load_int8_bundle():
    log(f"loading tokenizer + model (int8) from {MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, quantization_config=bnb, device_map="auto", dtype=torch.float16)
    model.eval()
    cfg = model.config
    arch = ArchInfo(num_layers=int(cfg.num_hidden_layers),
                     intermediate_size=int(cfg.intermediate_size),
                     hidden_size=int(cfg.hidden_size),
                     total_mlp_neurons=int(cfg.num_hidden_layers) * int(cfg.intermediate_size))
    return ModelBundle(model=model, tokenizer=tok, arch=arch, device="cuda", dtype=torch.float16)


def main():
    outdir = REPO_ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results/comprehensive_llama31_8b")
    max_fraction = float(sys.argv[2]) if len(sys.argv) > 2 else 0.70
    outdir.mkdir(parents=True, exist_ok=True)

    ecfg = load_config(REPO_ROOT / "configs/eval.yaml")
    set_all_seeds(SEED)
    max_new = ecfg["decoding"]["max_new_tokens"]; exec_to = ecfg["exec_timeout_sec"]; tol = ecfg["numeric_tolerance"]

    log(f"loading {MODEL_NAME}")
    bundle = load_int8_bundle()
    L, I = bundle.num_layers(), bundle.intermediate_size(); total = L * I
    log(f"layers={L} intermediate={I} total_mlp_neurons={total} 0.1%={max(1, round(0.001*total))}")

    bat = {c: load_battery(REPO_ROOT / f"data/batteries/comprehensive/{c}.jsonl") for c in CAPS}
    etypes = {c: bat[c][0].eval_type for c in CAPS}
    pairs = {c: [(it.prompt, it.gold) for it in bat[c]] for c in CAPS}

    # ---------- BASELINES ----------
    bpath = outdir / "baselines.json"
    if bpath.exists():
        base = json.load(open(bpath)); log("baselines: reused")
    else:
        log("baselines")
        base = {}
        for c in CAPS:
            acc, pf = gen_accuracy(bundle, bat[c], etypes[c], max_new, exec_to, tol)
            nll = batched_teacher_forced_nll(bundle, pairs[c])["mean_nll"]
            base[c] = {"acc": acc, "parse_fail": pf, "nll": nll, "eval_type": etypes[c]}
            log(f"  {c:22s} acc={acc if acc is None else round(acc,3)} nll={nll:.4f}")
        json.dump(base, open(bpath, "w"), indent=2)
    base_nll = {c: base[c]["nll"] for c in CAPS}; base_acc = {c: base[c]["acc"] for c in CAPS}

    # ---------- CAPTURE ----------
    apath = outdir / "aggregates_mean_abs.npz"
    if apath.exists():
        npz = np.load(apath); aggs = {c: npz[c] for c in CAPS}; log("aggregates: reused")
    else:
        log("capture")
        aggs = {c: capture_task_aggregates(bundle, bat[c], task=c, verbose=False)["mean_abs"] for c in CAPS}
        np.savez(apath, **aggs)
    calib = compute_calibration_means(bundle, [it for c in CAPS for it in bat[c]])
    ranks = {c: percentile_rank_per_layer(aggs[c]) for c in CAPS}

    def amask(c, th): return (ranks[c] >= th).reshape(-1)

    # ---------- A: OVERLAP ----------
    log("A: overlap")
    def overlap(th):
        m = {c: amask(c, th) for c in CAPS}; sz = {c: int(m[c].sum()) for c in CAPS}
        n = len(CAPS); oc = np.zeros((n, n)); jac = np.zeros((n, n))
        for i, a in enumerate(CAPS):
            for j, b in enumerate(CAPS):
                inter = int((m[a] & m[b]).sum()); union = int((m[a] | m[b]).sum())
                mn = min(sz[a], sz[b]); oc[i, j] = inter / mn if mn else 0.0; jac[i, j] = inter / union if union else 0.0
        return oc, jac, sz
    A = {}
    for th in THRESH_SENS:
        oc, jac, sz = overlap(th)
        A[f"top{int((1-th)*100)}"] = {"overlap_coeff": oc.tolist(), "jaccard": jac.tolist(), "active_sizes": sz}
    ocm, jacm, _ = overlap(THRESH_MAIN)
    pr = [(CAPS[i], CAPS[j], ocm[i, j], jacm[i, j]) for i in range(len(CAPS)) for j in range(i + 1, len(CAPS))]
    pr.sort(key=lambda x: x[2])
    A["headline_threshold"] = "top20_overlap_coefficient"
    A["most_entangled"] = [[a, b, round(o, 4), round(j, 4)] for a, b, o, j in pr[-5:][::-1]]
    A["least_entangled"] = [[a, b, round(o, 4), round(j, 4)] for a, b, o, j in pr[:5]]
    json.dump(A, open(outdir / "A_overlap.json", "w"), indent=2)
    log(f"  most entangled {A['most_entangled'][0]}; least {A['least_entangled'][0]}")

    # ---------- B: ALLOCATION ----------
    log("B: allocation")
    stack = np.stack([amask(c, THRESH_MAIN) for c in CAPS], axis=0)
    ac = stack.sum(axis=0)
    dead = int((ac == 0).sum()); excl = int((ac == 1).sum())
    shared = int(((ac >= 2) & (ac <= 12)).sum()); core = int((ac >= 13).sum())
    em = (ac == 1)
    B = {"total_neurons": total, "threshold": THRESH_MAIN,
         "buckets": {"dead": dead, "exclusive": excl, "shared": shared, "core": core},
         "buckets_pct": {"dead": dead/total, "exclusive": excl/total, "shared": shared/total, "core": core/total},
         "per_cap_exclusive_count": {c: int((stack[i] & em).sum()) for i, c in enumerate(CAPS)},
         "active_count_histogram": {int(k): int((ac == k).sum()) for k in range(16)}}
    json.dump(B, open(outdir / "B_allocation.json", "w"), indent=2)
    log(f"  dead={dead/total:.3f} excl={excl/total:.3f} shared={shared/total:.3f} core={core/total:.3f}")
    gitcommit(f"comprehensive A+B (json) {MODEL_NAME}")

    core_mask_flat = (ac >= 13)
    global_rank = np.mean(np.stack([ranks[c].reshape(-1) for c in CAPS], axis=0), axis=0)

    def mask_from_flat(idx):
        m = np.zeros(total, dtype=bool); m[idx] = True; return m.reshape(L, I)

    def nll_all(mask):
        h = attach_ablation_hooks_from_mask(bundle, mask, "mean", calib)
        try: return {c: batched_teacher_forced_nll(bundle, pairs[c])["mean_nll"] for c in CAPS}
        finally: remove_ablation_hooks(h)

    def acc_all(mask, mn):
        h = attach_ablation_hooks_from_mask(bundle, mask, "mean", calib)
        try:
            out = {}
            for c in CAPS:
                a, _ = gen_accuracy(bundle, bat[c], etypes[c], mn, exec_to, tol); out[c] = a
            return out
        finally: remove_ablation_hooks(h)

    # ---------- C: GLOBAL STRESS ----------
    log(f"C: stress to {max_fraction*100:.0f}%")
    step = max(1, round(0.001 * total)); n_steps = int(max_fraction * total) // step
    seeds = [0, 1, 2, 3, 4]
    order_global = np.argsort(global_rank, kind="mergesort")
    rng_orders = {s: np.random.default_rng(s).permutation(total) for s in seeds}
    gen_every = 70; rand_every = 10; fail = {c: None for c in CAPS}
    jl = open(outdir / "C_stress_rows.jsonl", "w"); t0 = time.time()
    for st in range(0, n_steps + 1):
        cum = st * step; frac = cum / total; do_gen = (st % gen_every == 0) or (st == n_steps)
        gmask = mask_from_flat(order_global[:cum]) if cum > 0 else np.zeros((L, I), bool)
        gn = nll_all(gmask); row = {"step": st, "frac": frac, "control": "global", "seed": -1}
        for c in CAPS: row[f"nll_{c}"] = gn[c]; row[f"nllratio_{c}"] = gn[c]/base_nll[c]
        if do_gen:
            ga = acc_all(gmask, MAXNEW_SWEEP)
            for c in CAPS:
                row[f"acc_{c}"] = ga[c]
                if base_acc[c]: row[f"ret_{c}"] = (ga[c]/base_acc[c]) if ga[c] is not None else None
        for c in CAPS:
            if fail[c] is None and gn[c]/base_nll[c] > 2.0: fail[c] = frac
        jl.write(json.dumps(row) + "\n")
        if st % rand_every == 0 or st == n_steps:
            for s in seeds:
                rmask = mask_from_flat(rng_orders[s][:cum]) if cum > 0 else np.zeros((L, I), bool)
                rn = nll_all(rmask); rr = {"step": st, "frac": frac, "control": "random", "seed": s}
                for c in CAPS: rr[f"nll_{c}"] = rn[c]; rr[f"nllratio_{c}"] = rn[c]/base_nll[c]
                jl.write(json.dumps(rr) + "\n")
        if st % 20 == 0 or do_gen:
            jl.flush(); worst = max(CAPS, key=lambda c: gn[c]/base_nll[c])
            log(f"  step {st}/{n_steps} frac={frac:.3f} worst={worst}:{gn[worst]/base_nll[worst]:.2f} el={time.time()-t0:.0f}s")
    jl.close()
    ranking = sorted(CAPS, key=lambda c: (fail[c] if fail[c] is not None else 1.0))
    json.dump({"max_fraction": max_fraction, "failure_def": "nll_ratio>2 (global control)",
               "failure_fraction_pct": {c: fail[c] for c in CAPS},
               "fragility_ranking_most_to_least_fragile": ranking},
              open(outdir / "C_stress.json", "w"), indent=2)
    log(f"  fragility: {ranking}")
    gitcommit(f"comprehensive C (json) {MODEL_NAME}")

    # ---------- D: 15x15 INTERLINK ----------
    log("D: interlink")
    def emask(c, desc):
        r = ranks[c].reshape(-1); elig = np.nonzero(~core_mask_flat)[0]
        order = elig[np.argsort(r[elig], kind="mergesort")]
        if desc: order = order[::-1]
        return mask_from_flat(order[:int(ENV_BUDGET*total)])
    n = len(CAPS)
    de, dle, dh, dlh = (np.zeros((n, n)) for _ in range(4)); t1 = time.time()
    for i, tgt in enumerate(CAPS):
        for desc, dacc, dloss in [(False, de, dle), (True, dh, dlh)]:
            m = emask(tgt, desc); na = nll_all(m); aa = acc_all(m, MAXNEW_SWEEP)
            for j, col in enumerate(CAPS):
                dloss[i, j] = na[col] - base_nll[col]
                if base_acc[col] is not None and aa[col] is not None: dacc[i, j] = aa[col] - base_acc[col]
        json.dump({"budget": ENV_BUDGET, "caps": CAPS, "dacc_env": de.tolist(), "dloss_env": dle.tolist(),
                   "dacc_high": dh.tolist(), "dloss_high": dlh.tolist()},
                  open(outdir / "D_interlink.json", "w"), indent=2)
        log(f"  {tgt} done ({time.time()-t1:.0f}s)")
    json.dump({"model_name": MODEL_NAME, "model_dir": MODEL_DIR, "precision": "int8(bnb)", "seed": SEED,
               "threshold_main": THRESH_MAIN, "total_mlp_neurons": total, "max_fraction": max_fraction,
               "git_hash": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip(),
               "timestamp": datetime.now().isoformat(), "caps": CAPS}, open(outdir / "manifest.json", "w"), indent=2)
    gitcommit(f"comprehensive D+manifest (json) {MODEL_NAME}")
    log(f"DONE -> {outdir}")


if __name__ == "__main__":
    main()
