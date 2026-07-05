"""Forward hooks for (a) activation capture and (b) reversible ablation.

No task logic here -- this module only knows about tensors, masks, and
channel indices. Hooks attach to the `down_proj` module of each MLP block;
the *input* to down_proj is exactly h = silu(gate_proj(x)) * up_proj(x),
i.e. the neuron activation vector we score/ablate (shape [B, S, intermediate_size]).
"""
from __future__ import annotations

from typing import Optional

import torch


class CaptureHook:
    """Forward hook (module, input, output) -> None.

    Accumulates streaming aggregates (sum_abs, max_abs, count) over the
    positions marked True in `current_mask`, instead of keeping full
    per-token tensors around.
    """

    def __init__(self, intermediate_size: int, device: str, dtype=torch.float32):
        self.intermediate_size = intermediate_size
        self.device = device
        self.sum_abs = torch.zeros(intermediate_size, dtype=dtype, device=device)
        self.max_abs = torch.zeros(intermediate_size, dtype=dtype, device=device)
        self.count = 0
        self.current_mask: Optional[torch.Tensor] = None  # [B, S] bool
        self._last_shape = None
        self._sanity_checked = False

    def set_mask(self, mask: torch.Tensor):
        """mask: bool tensor [B, S], True at positions to include in aggregation
        (i.e. non-pad AND within the answer span)."""
        self.current_mask = mask

    def __call__(self, module, inputs, output):
        h = inputs[0]  # [B, S, intermediate_size]
        self._last_shape = tuple(h.shape)
        if self.current_mask is None:
            # Hook fired from an incidental forward pass (e.g. the internal
            # forward inside teacher_forced_nll used only to compute NLL)
            # before capture.py has set the real answer-token mask for this
            # item. Skip aggregation rather than crash; capture.py always
            # follows up with an explicit forward pass with the mask set.
            return
        mask = self.current_mask.to(h.device)
        h_abs = h.detach().float().abs()  # signed -> abs before any scoring, per plan
        # flatten batch/seq, select masked positions
        flat = h_abs.reshape(-1, h_abs.shape[-1])  # [B*S, I]
        flat_mask = mask.reshape(-1)  # [B*S]
        if flat_mask.any():
            sel = flat[flat_mask]  # [n_sel, I]
            self.sum_abs += sel.sum(dim=0)
            step_max = sel.max(dim=0).values
            self.max_abs = torch.maximum(self.max_abs, step_max)
            self.count += int(sel.shape[0])
        self._sanity_checked = True

    def mean_abs(self) -> torch.Tensor:
        if self.count == 0:
            raise RuntimeError("No positions were ever aggregated (count=0)")
        return self.sum_abs / self.count


class MeanCalibrationHook:
    """Forward hook that accumulates the plain (signed) mean activation per
    channel over all (non-pad) positions seen. Used to build the mean-ablation
    replacement vector from a held-out calibration pass."""

    def __init__(self, intermediate_size: int, device: str, dtype=torch.float32):
        self.sum_val = torch.zeros(intermediate_size, dtype=dtype, device=device)
        self.count = 0
        self.current_mask: Optional[torch.Tensor] = None

    def set_mask(self, mask: torch.Tensor):
        self.current_mask = mask

    def __call__(self, module, inputs, output):
        h = inputs[0]
        if self.current_mask is None:
            # Same incidental-forward-pass situation as CaptureHook: skip.
            return
        mask = self.current_mask.to(h.device)
        flat = h.detach().float().reshape(-1, h.shape[-1])
        flat_mask = mask.reshape(-1)
        if flat_mask.any():
            sel = flat[flat_mask]
            self.sum_val += sel.sum(dim=0)
            self.count += int(sel.shape[0])

    def mean(self) -> torch.Tensor:
        if self.count == 0:
            raise RuntimeError("No positions aggregated for calibration mean")
        return self.sum_val / self.count


class AblationHook:
    """Forward PRE-hook: (module, args) -> modified args.

    Replaces the given channel indices of the down_proj input with either
    zero (harsh bound) or a precomputed per-channel calibration mean
    (default, stays on-manifold). Does NOT mutate any weights -- purely a
    reversible activation patch. Remove via the handle returned by
    module.register_forward_pre_hook(hook).
    """

    def __init__(
        self,
        channel_indices: torch.Tensor,
        mode: str = "mean",
        mean_values: Optional[torch.Tensor] = None,
    ):
        assert mode in ("zero", "mean")
        if mode == "mean" and mean_values is None:
            raise ValueError("mean_values required for mode='mean'")
        self.channel_indices = channel_indices
        self.mode = mode
        self.mean_values = mean_values

    def __call__(self, module, args):
        h = args[0]
        if self.channel_indices.numel() == 0:
            return args
        h = h.clone()
        idx = self.channel_indices.to(h.device)
        if self.mode == "zero":
            h[..., idx] = 0.0
        else:
            repl = self.mean_values.to(h.device, h.dtype)[idx]
            h[..., idx] = repl
        return (h,) + tuple(args[1:])


def attach_capture_hooks(bundle, dtype=torch.float32) -> list[CaptureHook]:
    """Attach one CaptureHook per layer's down_proj. Returns list indexed by layer."""
    hooks = []
    handles = []
    for layer_idx in range(bundle.num_layers()):
        hook = CaptureHook(bundle.intermediate_size(), device=bundle.device, dtype=dtype)
        handle = bundle.down_proj(layer_idx).register_forward_hook(hook)
        hooks.append(hook)
        handles.append(handle)
    return hooks, handles


def attach_calibration_hooks(bundle, dtype=torch.float32):
    hooks = []
    handles = []
    for layer_idx in range(bundle.num_layers()):
        hook = MeanCalibrationHook(bundle.intermediate_size(), device=bundle.device, dtype=dtype)
        handle = bundle.down_proj(layer_idx).register_forward_hook(hook)
        hooks.append(hook)
        handles.append(handle)
    return hooks, handles


def remove_handles(handles: list):
    for h in handles:
        h.remove()
