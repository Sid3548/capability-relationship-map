from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA_VERSION = "llama_pilot.v1"
CAPABILITIES = (
    "coding", "math", "formal_logic", "grammar", "translation",
    "reading_comprehension", "history_facts", "philosophy", "science_facts",
    "commonsense", "problem_solving", "creative_writing", "summarization",
    "spatial_pattern", "ethics",
)
TARGETS = ("coding", "translation")
MAX_PERCENT = 10.0
N_STEPS = 100
SENTINEL_STEPS = (0, 5, 10, 20, 50, 100)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def append_jsonl_fsync(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f.{path.name}., suffix=.tmp, dir=path.parent)
    try:
        with os.fdopen(fd, w, encoding=utf-8, newline=\n) as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + \n)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def exact_mask_count(total_channels: int, step: int) -> int:
    if not isinstance(step, int) or not 0 <= step <= N_STEPS:
        raise ValueError(f"step must be an integer in [0,{N_STEPS}], got {step!r}")
    return (total_channels * step) // 1000


def validate_scope(model_path: str | Path, targets: list[str] | tuple[str, ...], max_percent: float) -> Path:
    resolved = Path(model_path).resolve()
    if resolved.name != "Meta-Llama-3.1-8B":
        raise RuntimeError(f"SAFETY: expected local Meta-Llama-3.1-8B directory, got {resolved}")
    if "qwen" in str(resolved).lower():
        raise RuntimeError("SAFETY: Qwen is prohibited for this pilot")
    if tuple(targets) != TARGETS:
        raise RuntimeError(f"SAFETY: only targets {TARGETS} are authorized, got {tuple(targets)}")
    if float(max_percent) > MAX_PERCENT:
        raise RuntimeError(f"SAFETY: pilot may not exceed {MAX_PERCENT}%")
    return resolved
