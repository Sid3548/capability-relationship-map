"""Fetch and validate the two pinned experiment checkpoints.

This script never prints an HF token. It fills the existing partial Qwen
snapshot and downloads the matched post-trained Llama checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "model_fetch_v2.json"

MODELS = {
    "qwen": {
        "repo_id": "Qwen/Qwen3.5-9B",
        "local_dir": Path(r"D:\hf_models\Qwen3.5-9B"),
        "required": [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
        ],
        "patterns": [
            "*.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "README.md",
            "LICENSE",
        ],
    },
    "llama": {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir": Path(r"D:\hf_models\Meta-Llama-3.1-8B-Instruct"),
        "required": [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        "patterns": [
            "*.safetensors",
            "*.json",
            "tokenizer*",
            "*.model",
            "*.jinja",
            "README.md",
            "LICENSE*",
            "USE_POLICY.md",
        ],
    },
}


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def fetch_one(name: str, spec: dict, token: str | None) -> dict:
    local_dir: Path = spec["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    info = HfApi().model_info(spec["repo_id"], token=token)
    snapshot_download(
        repo_id=spec["repo_id"],
        revision=info.sha,
        local_dir=local_dir,
        allow_patterns=spec["patterns"],
        token=token,
    )
    missing = [item for item in spec["required"] if not (local_dir / item).is_file()]
    shards = sorted(local_dir.glob("*.safetensors"))
    result = {
        "name": name,
        "repo_id": spec["repo_id"],
        "revision": info.sha,
        "local_dir": str(local_dir),
        "required_missing": missing,
        "safetensor_shards": len(shards),
        "safetensor_bytes": sum(path.stat().st_size for path in shards),
        "config_sha256": sha256(local_dir / "config.json") if (local_dir / "config.json").is_file() else None,
        "elapsed_seconds": time.time() - started,
        "status": "complete" if not missing and shards else "incomplete",
    }
    if result["status"] != "complete":
        raise RuntimeError(f"{name} snapshot incomplete: {result}")
    return result


def main() -> None:
    token = os.environ.get("HF_TOKEN") or None
    state = {"started_at_unix": time.time(), "models": {}, "status": "running"}
    atomic_json(LOG, state)
    try:
        for name, spec in MODELS.items():
            print(f"Fetching {spec['repo_id']} -> {spec['local_dir']}", flush=True)
            state["models"][name] = fetch_one(name, spec, token)
            atomic_json(LOG, state)
        state["status"] = "complete"
    except Exception as exc:
        state["status"] = "error"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        raise
    finally:
        state["finished_at_unix"] = time.time()
        atomic_json(LOG, state)


if __name__ == "__main__":
    main()
