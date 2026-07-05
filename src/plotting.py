"""Retention-curve plotting: performance (accuracy + loss) vs % neurons
removed, with the three controls (low / random / high) overlaid. Interlink
heatmap is a stub -- out of scope for this run."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


COLORS = {"low": "#1f77b4", "random": "#7f7f7f", "high": "#d62728"}


def _agg_random(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Average the random control across its seeds at each frac_removed."""
    sub = df[df["control"] == "random"]
    if sub.empty:
        return sub
    return sub.groupby("frac_removed", as_index=False)[value_col].mean()


def plot_loss_curve(df: pd.DataFrame, target_task: str, out_path: str | Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    col = "nll_target"
    for control in ["low", "high"]:
        sub = df[(df["control"] == control) & df[col].notna()].sort_values("frac_removed")
        ax.plot(sub["frac_removed"] * 100, sub[col], label=control, color=COLORS[control], marker="o", markersize=2)
    rnd = _agg_random(df[df[col].notna()], col).sort_values("frac_removed")
    if not rnd.empty:
        ax.plot(rnd["frac_removed"] * 100, rnd[col], label="random (mean)", color=COLORS["random"], marker="o", markersize=2)
    ax.set_xlabel("% MLP neurons removed")
    ax.set_ylabel(f"teacher-forced NLL on gold ({target_task})")
    ax.set_title(f"Smooth loss retention curve -- target={target_task}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy_curve(df: pd.DataFrame, target_task: str, out_path: str | Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    col = "acc_target"
    for control in ["low", "high"]:
        sub = df[(df["control"] == control) & df[col].notna()].sort_values("frac_removed")
        ax.plot(sub["frac_removed"] * 100, sub[col], label=control, color=COLORS[control], marker="o")
    rnd = _agg_random(df[df[col].notna()], col).sort_values("frac_removed")
    if not rnd.empty:
        ax.plot(rnd["frac_removed"] * 100, rnd[col], label="random (mean)", color=COLORS["random"], marker="o")
    ax.set_xlabel("% MLP neurons removed")
    ax.set_ylabel(f"pass@1 / accuracy ({target_task})")
    ax.set_title(f"Generation-eval retention curve -- target={target_task}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_interlink_heatmap(matrix, tasks, title, out_path, cmap="RdBu_r", fmt="{:+.2f}", vlim=None):
    """matrix: [N, N] array, rows = targeted (ablated) task, cols = collateral
    task measured. Value = delta (post - pre) of acc or loss. Annotated."""
    import numpy as np

    matrix = np.asarray(matrix, dtype=float)
    n = len(tasks)
    if vlim is None:
        vlim = max(1e-9, np.nanmax(np.abs(matrix)))
    fig, ax = plt.subplots(figsize=(1.6 * n + 2, 1.6 * n + 1.5))
    im = ax.imshow(matrix, cmap=cmap, vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(n)); ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(tasks)
    ax.set_xlabel("collateral task (measured)")
    ax.set_ylabel("targeted task (neurons removed)")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color="black" if abs(v) < 0.6 * vlim else "white", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

