"""Qwen3.5-9B (int8) smoke slice: capture -> score -> 3-control removal sweep.
Mirrors the Llama smoke interface (src/*). JSON/JSONL only (no pyarrow/mpl in torch proc).
Usage: python -m scripts.smoke_qwen
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.models import ModelBundle, ArchInfo
from src.eval.base import load_battery, set_all_seeds, batched_teacher_forced_nll, greedy_generate
from src.eval.dispatch import gen_accuracy
from src.capture import capture_task_aggregates, compute_calibration_means
from src.scoring import percentile_rank_per_layer
from src.ablation import (build_control_orders, channels_for_step,
                          attach_ablation_hooks, remove_ablation_hooks)

MODEL_DIR = r"D:\hf_models\Qwen3.5-9B"
OUT = REPO_ROOT / "results" / "qwen35_9b_smoke"
SEED = 1234
STEP_FRAC = 0.001      # 0.1% ~ neurons/step
MAX_FRAC = 0.05        # smoke: sweep to 5%
GEN_STEPS = {0, 10, 20, 30, 40, 50}   # pass@1 checkpoints
CORE_P = 0.99
EXEC_TO = 5
MAXNEW_CODE = 256


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_int8_bundle():
    log("loading tokenizer + model (int8)")
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
    OUT.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)
    bundle = load_int8_bundle()
    L, I = bundle.num_layers(), bundle.intermediate_size(); total = L * I
    log(f"layers={L} intermediate={I} total_mlp_neurons={total}")

    # sanity generate
    san = greedy_generate(bundle, "def add(a, b):\n    # returns sum\n", max_new_tokens=24)
    log(f"sanity gen: {san!r}")

    coding = load_battery(REPO_ROOT / "data/batteries/coding.jsonl")
    math = load_battery(REPO_ROOT / "data/batteries/math.jsonl")
    allitems = coding + math
    cpairs = [(it.prompt, it.gold) for it in coding]
    mpairs = [(it.prompt, it.gold) for it in math]

    # baselines
    base_code_nll = batched_teacher_forced_nll(bundle, cpairs)["mean_nll"]
    base_math_nll = batched_teacher_forced_nll(bundle, mpairs)["mean_nll"]
    base_code_acc, _ = gen_accuracy(bundle, coding, "exec", MAXNEW_CODE, EXEC_TO, 0.0)
    base_math_acc, _ = gen_accuracy(bundle, math, "numeric", 16, EXEC_TO, 0.0)
    log(f"baseline coding nll={base_code_nll:.4f} pass@1={base_code_acc:.3f} | math nll={base_math_nll:.4f} acc={base_math_acc:.3f}")

    # capture coding aggregates + calibration
    log("capture coding aggregates")
    cap = capture_task_aggregates(bundle, coding, task="coding", verbose=True)
    mean_abs_code = cap["mean_abs"]  # [L,I]
    log("capture math aggregates")
    mean_abs_math = capture_task_aggregates(bundle, math, task="math")["mean_abs"]
    np.savez(OUT / "aggregates_int8.npz", coding=mean_abs_code, math=mean_abs_math)
    log("calibration means")
    calib = compute_calibration_means(bundle, allitems)

    rank_code = percentile_rank_per_layer(mean_abs_code)
    rank_math = percentile_rank_per_layer(mean_abs_math)
    core_mask = (rank_code >= CORE_P) & (rank_math >= CORE_P)
    log(f"core neurons (rank>={CORE_P} both) = {int(core_mask.sum())}")

    seeds = [0, 1, 2, 3, 4]
    orders = build_control_orders(rank_code, core_mask, seeds)

    step_size = max(1, round(STEP_FRAC * total))
    n_steps = int(MAX_FRAC * total) // step_size
    log(f"step_size={step_size} n_steps={n_steps} (to {MAX_FRAC*100:.0f}%)")

    def eval_masked(channels_by_layer, do_gen):
        handles = attach_ablation_hooks(bundle, channels_by_layer, "mean", calib)
        try:
            r = {"code_nll": batched_teacher_forced_nll(bundle, cpairs)["mean_nll"],
                 "math_nll": batched_teacher_forced_nll(bundle, mpairs)["mean_nll"]}
            if do_gen:
                r["code_pass1"], _ = gen_accuracy(bundle, coding, "exec", MAXNEW_CODE, EXEC_TO, 0.0)
        finally:
            remove_ablation_hooks(handles)
        return r

    jl = open(OUT / "sweep_rows.jsonl", "w")
    t0 = time.time()
    for st in range(0, n_steps + 1):
        cum = st * step_size; frac = cum / total; do_gen = st in GEN_STEPS
        for control in ["low", "random", "high"]:
            ctrl_seeds = seeds if control == "random" else [None]
            for sd in ctrl_seeds:
                if cum == 0:
                    chans = [np.array([], dtype=np.int64) for _ in range(L)]
                else:
                    chans = channels_for_step(control, cum, orders, L, seed=sd)
                # gen only for low/high (skip random gen to bound wall clock)
                dg = do_gen and control in ("low", "high")
                r = eval_masked(chans, dg)
                row = {"step": st, "frac": frac, "control": control,
                       "seed": -1 if sd is None else sd,
                       "n_ablated": int(sum(len(c) for c in chans)),
                       "code_nll": r["code_nll"], "math_nll": r["math_nll"],
                       "code_nll_ratio": r["code_nll"] / base_code_nll,
                       "math_nll_ratio": r["math_nll"] / base_math_nll}
                if "code_pass1" in r:
                    row["code_pass1"] = r["code_pass1"]
                    row["code_pass1_ret"] = (r["code_pass1"] / base_code_acc) if base_code_acc else None
                jl.write(json.dumps(row) + "\n")
        jl.flush()
        log(f"  step {st}/{n_steps} frac={frac:.3f} el={time.time()-t0:.0f}s")
    jl.close()

    json.dump({
        "model": "Qwen3.5-9B", "precision": "int8(bnb)", "seed": SEED,
        "layers": L, "intermediate": I, "total_mlp_neurons": total,
        "step_fraction": STEP_FRAC, "max_fraction": MAX_FRAC, "core_percentile": CORE_P,
        "core_neurons": int(core_mask.sum()),
        "baseline": {"code_nll": base_code_nll, "code_pass1": base_code_acc,
                     "math_nll": base_math_nll, "math_acc": base_math_acc},
        "sanity_gen": san,
    }, open(OUT / "smoke_manifest.json", "w"), indent=2)
    log(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
