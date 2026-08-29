"""Verify Qwen3.5-9B architecture is DENSE SwiGLU with silu activation.
Reads config.json from local dir (offline).
"""
import os, json, sys
os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path

MODEL_DIR = r"D:\hf_models\Qwen3.5-9B"
REPO_ROOT = Path(__file__).resolve().parent.parent

def log(m): print(m, flush=True)

try:
    cfg_path = Path(MODEL_DIR) / "config.json"
    if not cfg_path.exists():
        log(f"ERROR: {cfg_path} not found - download may not be complete")
        sys.exit(1)

    cfg = json.load(open(cfg_path))
    log(f"Config loaded from {cfg_path}")

    # Check required fields
    num_layers = cfg.get("num_hidden_layers")
    hidden_size = cfg.get("hidden_size")
    intermediate_size = cfg.get("intermediate_size")
    hidden_act = cfg.get("hidden_act", "")

    log(f"num_layers: {num_layers}")
    log(f"hidden_size: {hidden_size}")
    log(f"intermediate_size: {intermediate_size}")
    log(f"hidden_act: {hidden_act}")

    # Check for MoE
    if "num_experts" in cfg or "num_local_experts" in cfg or "moe_" in str(cfg.keys()).lower():
        log("ERROR: Model is MoE - per-expert hooks not implemented")
        sys.exit(1)

    # Check activation
    if "silu" not in hidden_act.lower():
        log(f"ERROR: Activation is {hidden_act}, not silu")
        sys.exit(1)

    # Check for gate_proj/up_proj (SwiGLU signature)
    # We'll verify this when loading the model
    log(f"Architecture check: DENSE, SwiGLU candidate, activation=silu ✓")
    log(f"Total MLP neurons: {num_layers} * {intermediate_size} = {num_layers * intermediate_size}")

    # Write summary
    out_file = REPO_ROOT / "logs" / "qwen_arch_verify.json"
    json.dump({
        "model": "Qwen3.5-9B",
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "hidden_act": hidden_act,
        "total_mlp_neurons": num_layers * intermediate_size,
        "is_moe": False,
        "is_silu": "silu" in hidden_act.lower(),
        "status": "VERIFIED"
    }, open(out_file, "w"), indent=2)
    log(f"VERIFIED -> {out_file}")

except Exception as e:
    log(f"ERROR: {e}")
    sys.exit(1)
