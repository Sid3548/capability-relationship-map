"""Persistence: npz for activation aggregates, parquet for sweep result rows,
and a run manifest JSON (git hash, model, seed, dtype, device) for
reproducibility."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def save_aggregates_npz(path: str | Path, mean_abs: np.ndarray, max_abs: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean_abs=mean_abs, max_abs=max_abs)


def load_aggregates_npz(path: str | Path) -> dict:
    d = np.load(path)
    return {"mean_abs": d["mean_abs"], "max_abs": d["max_abs"]}


def save_rows_parquet(path: str | Path, rows: list[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def append_rows_parquet(path: str | Path, rows: list[dict]):
    """Simplest safe approach for a smoke run: read-modify-write."""
    path = Path(path)
    if path.exists():
        existing = pq.read_table(path).to_pylist()
        rows = existing + rows
    save_rows_parquet(path, rows)


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def write_manifest(path: str | Path, extra: dict):
    manifest = {
        "git_hash": git_hash(),
        **extra,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
