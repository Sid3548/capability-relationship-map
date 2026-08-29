from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from .common import N_STEPS, exact_mask_count


def global_order(scores: np.ndarray, descending: bool = False) -> np.ndarray:
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("scores must be finite [layers, intermediate] array")
    order = np.argsort(scores.reshape(-1), kind="stable")
    return order[::-1].copy() if descending else order


def mask_for_step(order: np.ndarray, total_channels: int, step: int) -> np.ndarray:
    if order.shape != (total_channels,) or len(np.unique(order)) != total_channels:
        raise ValueError("ranking must be a complete permutation of all channels")
    k = exact_mask_count(total_channels, step)
    return np.asarray(order[:k], dtype=np.int64)


def mask_hash(flat_indices: np.ndarray, total_channels: int, step: int) -> str:
    canonical = np.sort(np.asarray(flat_indices, dtype="<i8"))
    header = f"mask.v1|N={total_channels}|step={step}|k={len(canonical)}|".encode()
    return hashlib.sha256(header + canonical.tobytes(order="C")).hexdigest()


def validate_schedule(order: np.ndarray, total_channels: int) -> list[dict]:
    previous: set[int] = set()
    rows = []
    for step in range(N_STEPS + 1):
        indices = mask_for_step(order, total_channels, step)
        current = set(map(int, indices))
        expected = exact_mask_count(total_channels, step)
        if len(current) != expected:
            raise AssertionError(f"step {step}: unique count {len(current)} != expected {expected}")
        if not previous.issubset(current):
            raise AssertionError(f"step {step}: masks are not nested")
        rows.append({"step": step, "count": expected, "mask_sha256": mask_hash(indices, total_channels, step)})
        previous = current
    return rows


def layer_counts(flat_indices: np.ndarray, intermediate_size: int, n_layers: int) -> np.ndarray:
    layers = np.asarray(flat_indices, dtype=np.int64) // intermediate_size
    return np.bincount(layers, minlength=n_layers).astype(np.int64)


def layer_matched_random_order(primary_order: np.ndarray, n_layers: int, intermediate_size: int, seed: int) -> np.ndarray:
    """Build a nested random order whose cumulative per-layer counts match the primary at every step.

    Each channel in the primary order contributes only its layer identity. A seeded random
    permutation within each layer supplies the channel identity, preserving the layer sequence
    and therefore exact layer counts at every cumulative k.
    """
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(intermediate_size) for _ in range(n_layers)]
    cursors = np.zeros(n_layers, dtype=np.int64)
    result = np.empty_like(primary_order)
    for pos, flat in enumerate(primary_order):
        layer = int(flat) // intermediate_size
        channel = int(pools[layer][cursors[layer]])
        cursors[layer] += 1
        result[pos] = layer * intermediate_size + channel
    if len(np.unique(result)) != n_layers * intermediate_size:
        raise AssertionError("random order is not a complete channel permutation")
    return result


@dataclass
class InterventionController:
    modules: list
    intermediate_size: int
    replacement_means: np.ndarray
    mode: str = "mean"

    def __post_init__(self):
        if self.mode not in {"mean", "zero", "noop"}:
            raise ValueError(f"unsupported intervention mode {self.mode}")
        self.handles = []
        self.indices = [torch.empty(0, dtype=torch.long) for _ in self.modules]
        self._observed_dtypes: set[str] = set()

    def _hook(self, layer: int):
        def apply(module, args):
            h = args[0]
            self._observed_dtypes.add(str(h.dtype))
            idx = self.indices[layer]
            if self.mode == "noop" or idx.numel() == 0:
                return args
            idx = idx.to(h.device)
            patched = h.clone()
            if self.mode == "zero":
                patched[..., idx] = 0
            else:
                means = torch.as_tensor(self.replacement_means[layer], device=h.device, dtype=h.dtype)
                patched[..., idx] = means[idx]
            return (patched,) + tuple(args[1:])
        return apply

    def attach(self) -> None:
        if self.handles:
            raise RuntimeError("controller already attached")
        self.handles = [module.register_forward_pre_hook(self._hook(layer)) for layer, module in enumerate(self.modules)]

    def update_flat_indices(self, flat_indices: np.ndarray) -> None:
        grouped = [[] for _ in self.modules]
        for flat in np.asarray(flat_indices, dtype=np.int64):
            layer, channel = divmod(int(flat), self.intermediate_size)
            grouped[layer].append(channel)
        self.indices = [torch.tensor(values, dtype=torch.long) for values in grouped]

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @property
    def observed_dtypes(self) -> list[str]:
        return sorted(self._observed_dtypes)

