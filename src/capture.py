"""Run the battery through the model with capture hooks attached, aggregate
mean_abs / max_abs of down_proj-input (h = silu(gate)*up) per (layer, task)
over answer-token positions only (never pad tokens). Streams aggregates,
not full per-token tensors.
"""
from __future__ import annotations

import numpy as np
import torch

from src.eval.base import BatteryItem, teacher_forced_nll
from src.hooks import CaptureHook, attach_capture_hooks, remove_handles


@torch.no_grad()
def capture_task_aggregates(bundle, items: list[BatteryItem], task: str, verbose: bool = False) -> dict:
    """Returns dict with mean_abs [L, I] and max_abs [L, I] numpy arrays for
    the given task, aggregated over that task's battery items' answer-token
    positions."""
    hooks, handles = attach_capture_hooks(bundle, dtype=torch.float32)
    n_layers = bundle.num_layers()
    intermediate = bundle.intermediate_size()

    task_items = [it for it in items if it.task == task]
    checked_shape = False
    try:
        for it in task_items:
            nll_info = teacher_forced_nll(bundle, it.prompt, it.gold)
            full_ids = nll_info["full_ids"]
            attn_mask = nll_info["attn_mask"]
            answer_mask = nll_info["answer_mask"]  # [1, T] bool

            for h in hooks:
                h.set_mask(answer_mask)

            out = bundle.model(input_ids=full_ids, attention_mask=attn_mask)

            # Reset mask to None immediately after the real aggregating pass.
            # Otherwise the NEXT item's incidental internal forward (inside
            # teacher_forced_nll, called at the top of the next loop
            # iteration) would fire the hook with THIS item's stale mask,
            # which has the wrong sequence length and silently corrupts (or
            # crashes) the aggregation.
            for h in hooks:
                h.set_mask(None)

            if not checked_shape:
                shape = hooks[0]._last_shape
                assert shape is not None, "capture hook never fired"
                b, s, i = shape
                assert i == intermediate, f"intermediate dim mismatch: got {i}, expected {intermediate}"
                sample = full_ids  # noop just to keep it in scope
                if verbose:
                    print(f"[capture] task={task} activation shape on layer0: {shape} (B,S,intermediate)")
                checked_shape = True
    finally:
        remove_handles(handles)

    mean_abs = np.stack([h.mean_abs().cpu().numpy() for h in hooks], axis=0)  # [L, I]
    max_abs = np.stack([h.max_abs.cpu().numpy() for h in hooks], axis=0)  # [L, I]

    # sanity checks
    assert np.isfinite(mean_abs).all(), "non-finite values in mean_abs aggregate"
    assert np.isfinite(max_abs).all(), "non-finite values in max_abs aggregate"
    assert (mean_abs >= 0).all(), "mean_abs should be non-negative (abs-scored)"
    assert mean_abs.shape == (n_layers, intermediate)

    return {"mean_abs": mean_abs, "max_abs": max_abs, "n_items": len(task_items)}


@torch.no_grad()
def compute_calibration_means(bundle, items: list[BatteryItem]) -> np.ndarray:
    """Held-out calibration pass: plain (signed) per-channel mean activation
    across ALL battery items/tasks, used as the mean-ablation replacement
    value. Returns [L, I] numpy array.
    """
    from src.hooks import attach_calibration_hooks

    hooks, handles = attach_calibration_hooks(bundle, dtype=torch.float32)
    try:
        for it in items:
            nll_info = teacher_forced_nll(bundle, it.prompt, it.gold)
            full_ids = nll_info["full_ids"]
            attn_mask = nll_info["attn_mask"]
            answer_mask = nll_info["answer_mask"]
            for h in hooks:
                h.set_mask(answer_mask)
            bundle.model(input_ids=full_ids, attention_mask=attn_mask)
            for h in hooks:
                h.set_mask(None)
    finally:
        remove_handles(handles)
    means = np.stack([h.mean().cpu().numpy() for h in hooks], axis=0)
    return means
