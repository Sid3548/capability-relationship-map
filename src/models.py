"""Model loading + architecture adapter.

Exposes MLP hook points generically so capture.py / ablation.py never need
to know about specific model internals. All dims are read from
model.config at runtime -- nothing hardcoded, so this slots in for
Llama/Gemma later (out of scope for this run, but the seam is here).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclasses.dataclass
class ArchInfo:
    num_layers: int
    intermediate_size: int
    hidden_size: int
    total_mlp_neurons: int  # num_layers * intermediate_size


class ModelBundle:
    """Wraps a HF causal LM + tokenizer with MLP hook-point accessors."""

    def __init__(self, model, tokenizer, arch: ArchInfo, device: str, dtype: torch.dtype):
        self.model = model
        self.tokenizer = tokenizer
        self.arch = arch
        self.device = device
        self.dtype = dtype

    def down_proj(self, layer_idx: int) -> torch.nn.Module:
        """Return the down_proj module for layer_idx.

        A forward hook on this module's input == silu(gate_proj(x)) * up_proj(x),
        i.e. exactly the neuron activation vector h we must score/ablate.
        """
        return self.model.model.layers[layer_idx].mlp.down_proj

    def num_layers(self) -> int:
        return self.arch.num_layers

    def intermediate_size(self) -> int:
        return self.arch.intermediate_size


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(cfg: dict) -> ModelBundle:
    model_name = cfg["model_name"]
    dtype = DTYPE_MAP[cfg.get("dtype", "bfloat16")]
    device = cfg.get("device", "cuda")
    cache_dir = cfg.get("cache_dir", None)
    trust_remote_code = cfg.get("trust_remote_code", False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()

    hf_config = model.config
    num_layers = int(hf_config.num_hidden_layers)
    intermediate_size = int(hf_config.intermediate_size)
    hidden_size = int(hf_config.hidden_size)
    arch = ArchInfo(
        num_layers=num_layers,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        total_mlp_neurons=num_layers * intermediate_size,
    )

    return ModelBundle(model=model, tokenizer=tokenizer, arch=arch, device=device, dtype=dtype)
