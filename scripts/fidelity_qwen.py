"""Fidelity check: int8 vs fp16 for Qwen3.5-9B on the coding battery.
fp16 run on CPU (16GB GPU can't hold 9B fp16) -- small subset only.
Compares (i) baseline teacher-forced coding NLL, (ii) top-20%-per-layer active
neuron overlap (Jaccard) on sampled layers. JSON only.
Usage: python -m scripts.fidelity_qwen
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

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models import ModelBundle, ArchInfo
from src.eval.base import load_battery, set_all_seeds, batched_teacher_forced_nll
from src.capture import capture_task_aggregates
from src.scoring import percentile_rank_per_layer

MODEL_DIR = r"D:\hf_models\Qwen3.5-9B"
OUT = REPO_ROOT / "results" / "qwen35_9b_smoke"
SEED = 1234
SAMPLE_LAYERS = [0, 8, 16, 24, 31]
TOPP = 0.80  # top-20% active


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    set_all_seeds(SEED)
    log("loading fp16 on CPU (subset fidelity)")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16)
    model.to("cpu"); model.eval()
    cfg = model.config
    arch = ArchInfo(int(cfg.num_hidden_layers), int(cfg.intermediate_size),
                    int(cfg.hidden_size), int(cfg.num_hidden_layers) * int(cfg.intermediate_size))
    bundle = ModelBundle(model, tok, arch, device="cpu", dtype=torch.float16)

    coding = load_battery(REPO_ROOT / "data/batteries/coding.jsonl")
    cpairs = [(it.prompt, it.gold) for it in coding]

    log("fp16 baseline coding NLL")
    fp16_nll = batched_teacher_forced_nll(bundle, cpairs, max_batch=4)["mean_nll"]
    log(f"fp16 code_nll={fp16_nll:.4f}")

    log("fp16 capture coding aggregates")
    mean_abs_fp16 = capture_task_aggregates(bundle, coding, task="coding")["mean_abs"]

    int8 = np.load(OUT / "aggregates_int8.npz")["coding"]
    r8 = percentile_rank_per_layer(int8)
    r16 = percentile_rank_per_layer(mean_abs_fp16)

    jac = {}
    for l in SAMPLE_LAYERS:
        a = set(np.nonzero(r8[l] >= TOPP)[0].tolist())
        b = set(np.nonzero(r16[l] >= TOPP)[0].tolist())
        inter = len(a & b); union = len(a | b)
        jac[l] = inter / union if union else 0.0
    mean_jac = float(np.mean(list(jac.values())))

    man = json.load(open(OUT / "smoke_manifest.json"))
    int8_nll = man["baseline"]["code_nll"]
    res = {"int8_code_nll": int8_nll, "fp16_code_nll": fp16_nll,
           "nll_delta_abs": abs(fp16_nll - int8_nll),
           "nll_delta_rel": abs(fp16_nll - int8_nll) / int8_nll,
           "sample_layers": SAMPLE_LAYERS, "top20_jaccard_per_layer": jac,
           "top20_jaccard_mean": mean_jac}
    json.dump(res, open(OUT / "fidelity.json", "w"), indent=2)
    log(f"int8_nll={int8_nll:.4f} fp16_nll={fp16_nll:.4f} delta={res['nll_delta_abs']:.4f} "
        f"top20 Jaccard mean={mean_jac:.3f}")
    log(f"DONE -> {OUT / 'fidelity.json'}")


if __name__ == "__main__":
    main()
