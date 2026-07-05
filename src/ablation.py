"""Control-mask construction (low / random / high) plus the reversible
mean/zero ablation hooks used during the sweep. No weight mutation anywhere
-- only forward pre-hooks patched onto down_proj, attached/removed per step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.hooks import AblationHook


@dataclass
class ControlOrders:
    """Global (across all layers) neuron orderings for the low/high controls,
    and per-seed per-layer random permutations for the random control -- all
    restricted to the non-core candidate pool."""

    low_order: list[tuple[int, int]]     # ascending target score, weakest first (the envelope)
    high_order: list[tuple[int, int]]    # descending target score, strongest first
    non_core_by_layer: list[np.ndarray]  # per layer: array of eligible channel indices
    random_perms: dict[int, list[np.ndarray]]  # seed -> per-layer random permutation of non_core_by_layer[l]


def build_control_orders(
    target_rank: np.ndarray, core_mask: np.ndarray, random_seeds: list[int]
) -> ControlOrders:
    L, I = target_rank.shape
    non_core_mask = ~core_mask

    # Global list of (layer, channel) sorted by target score (rank), non-core only
    flat_rank = target_rank.reshape(-1)
    flat_noncore = non_core_mask.reshape(-1)
    eligible_idx = np.nonzero(flat_noncore)[0]
    order_asc = eligible_idx[np.argsort(flat_rank[eligible_idx], kind="mergesort")]
    order_desc = order_asc[::-1]

    def to_pairs(flat_indices):
        return [(int(idx // I), int(idx % I)) for idx in flat_indices]

    low_order = to_pairs(order_asc)
    high_order = to_pairs(order_desc)

    non_core_by_layer = [np.nonzero(non_core_mask[l])[0] for l in range(L)]

    random_perms = {}
    for seed in random_seeds:
        rng = np.random.default_rng(seed)
        perms = [rng.permutation(non_core_by_layer[l]) for l in range(L)]
        random_perms[seed] = perms

    return ControlOrders(
        low_order=low_order,
        high_order=high_order,
        non_core_by_layer=non_core_by_layer,
        random_perms=random_perms,
    )


def per_layer_counts_from_order(order: list[tuple[int, int]], cumulative_n: int, n_layers: int) -> np.ndarray:
    counts = np.zeros(n_layers, dtype=np.int64)
    for l, _ in order[:cumulative_n]:
        counts[l] += 1
    return counts


def channels_for_step(
    control: str,
    cumulative_n: int,
    orders: ControlOrders,
    n_layers: int,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Return, per layer, the array of channel indices to ablate at this
    cumulative removal count, for the given control type."""
    if control == "low":
        pairs = orders.low_order[:cumulative_n]
        by_layer = [[] for _ in range(n_layers)]
        for l, c in pairs:
            by_layer[l].append(c)
        return [np.array(x, dtype=np.int64) for x in by_layer]

    if control == "high":
        pairs = orders.high_order[:cumulative_n]
        by_layer = [[] for _ in range(n_layers)]
        for l, c in pairs:
            by_layer[l].append(c)
        return [np.array(x, dtype=np.int64) for x in by_layer]

    if control == "random":
        assert seed is not None, "random control requires a seed"
        # match per-layer neuron counts to the LOW control's cumulative counts
        # at this same step (documented design choice: low is the primary
        # envelope comparison; random must remove the same count per layer,
        # just choosing WHICH channels randomly from the non-core pool).
        counts = per_layer_counts_from_order(orders.low_order, cumulative_n, n_layers)
        perms = orders.random_perms[seed]
        return [perms[l][: counts[l]] for l in range(n_layers)]

    raise ValueError(f"unknown control: {control}")


def attach_ablation_hooks(
    bundle,
    channels_by_layer: list[np.ndarray],
    mode: str,
    calibration_means: np.ndarray | None,
) -> list:
    """Attach AblationHook as a forward PRE-hook on each layer's down_proj.
    Returns list of handles to remove() after the eval step."""
    handles = []
    for layer_idx, channels in enumerate(channels_by_layer):
        idx_t = torch.as_tensor(channels, dtype=torch.long)
        mean_vals = None
        if mode == "mean":
            mean_vals = torch.as_tensor(calibration_means[layer_idx], dtype=torch.float32)
        hook = AblationHook(channel_indices=idx_t, mode=mode, mean_values=mean_vals)
        handle = bundle.down_proj(layer_idx).register_forward_pre_hook(hook)
        handles.append(handle)
    return handles


def remove_ablation_hooks(handles: list):
    for h in handles:
        h.remove()
