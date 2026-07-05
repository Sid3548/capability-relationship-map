"""Turn per-(layer,task) activation aggregates into per-layer PERCENTILE
ranks in [0,1] (never fixed raw cutoffs -- activation scale is layer
dependent), and identify the 'core' mask: neurons with high rank on BOTH
tasks, excluded from all removal controls (low/random/high alike).

A-low (removal envelope) and A-specific-high (interlink, out of scope) are
kept as separate concepts throughout -- this module only ever produces a
single generic 'core' exclusion set plus per-task percentile ranks; it does
not conflate "important for task X" with "interlink-specific".
"""
from __future__ import annotations

import numpy as np


def percentile_rank_per_layer(agg: np.ndarray) -> np.ndarray:
    """agg: [L, I] raw aggregate (e.g. mean_abs). Returns [L, I] array of
    percentile ranks in [0, 1], computed independently PER LAYER (row),
    since raw activation scale differs by layer/depth."""
    L, I = agg.shape
    ranks = np.zeros_like(agg, dtype=np.float64)
    for l in range(L):
        row = agg[l]
        order = np.argsort(row, kind="mergesort")  # ascending
        rank_of = np.empty(I, dtype=np.float64)
        rank_of[order] = np.arange(I, dtype=np.float64)
        ranks[l] = rank_of / max(I - 1, 1)
    return ranks


def compute_core_mask(
    target_ranks: np.ndarray,
    other_ranks_list: list[np.ndarray],
    core_percentile: float = 0.99,
) -> np.ndarray:
    """core[l, i] = True iff this neuron's percentile rank is >= core_percentile
    on the target task AND on every task in other_ranks_list (i.e. it's
    high-activation / "similar high rank" across all tasks we scored --
    a shared-importance neuron that should never be a removal candidate,
    for any of the low/random/high controls)."""
    core = target_ranks >= core_percentile
    for other in other_ranks_list:
        core = core & (other >= core_percentile)
    return core


def print_sanity_topbottom(agg: np.ndarray, ranks: np.ndarray, task: str, stat: str, k: int = 5):
    """Print top-k and bottom-k (layer, channel) neurons by rank, as a
    sanity check that percentile normalization behaves sensibly."""
    L, I = agg.shape
    flat_ranks = ranks.reshape(-1)
    flat_vals = agg.reshape(-1)
    order = np.argsort(flat_ranks)
    bottom = order[:k]
    top = order[-k:][::-1]

    def fmt(idx):
        l, i = divmod(int(idx), I)
        return f"(layer={l}, ch={i}, {stat}={flat_vals[idx]:.5f}, rank={flat_ranks[idx]:.4f})"

    print(f"[scoring] task={task} stat={stat} TOP-{k}: " + ", ".join(fmt(x) for x in top))
    print(f"[scoring] task={task} stat={stat} BOTTOM-{k}: " + ", ".join(fmt(x) for x in bottom))


class ScoreBundle:
    """Container for one task's percentile ranks, both stats, plus raw aggs."""

    def __init__(self, task: str, mean_abs: np.ndarray, max_abs: np.ndarray):
        self.task = task
        self.mean_abs = mean_abs
        self.max_abs = max_abs
        self.rank_mean = percentile_rank_per_layer(mean_abs)
        self.rank_max = percentile_rank_per_layer(max_abs)

    def rank(self, stat: str) -> np.ndarray:
        return self.rank_mean if stat == "mean_abs" else self.rank_max

    def agg(self, stat: str) -> np.ndarray:
        return self.mean_abs if stat == "mean_abs" else self.max_abs
