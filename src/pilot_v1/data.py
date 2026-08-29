from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .common import CAPABILITIES, TARGETS, canonical_json, sha256_bytes


@dataclass(frozen=True)
class PilotItem:
    id: str
    capability: str
    split: str
    prompt: str
    gold: str
    eval_type: str
    metadata: dict


def load_jsonl(path: str | Path) -> list[PilotItem]:
    rows: list[PilotItem] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                rows.append(PilotItem(
                    id=str(raw["id"]), capability=str(raw["capability"]), split=str(raw["split"]),
                    prompt=str(raw["prompt"]), gold=str(raw["gold"]), eval_type=str(raw["eval_type"]),
                    metadata=dict(raw.get("metadata") or {}),
                ))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid pilot row: {exc}") from exc
    return rows


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def dataset_hash(items: Iterable[PilotItem]) -> str:
    return sha256_bytes(canonical_json([item.__dict__ for item in items]))


def lint_battery(items: list[PilotItem]) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    ids = Counter(x.id for x in items)
    for item_id, n in ids.items():
        if n != 1:
            errors.append(f"duplicate id {item_id!r}: {n}")
    exact_prompts = Counter(normalized_text(x.prompt) for x in items)
    for prompt, n in exact_prompts.items():
        if n != 1:
            errors.append(f"duplicate normalized prompt ({n} rows): {prompt[:100]!r}")
    split_cap = Counter((x.split, x.capability) for x in items)
    for cap in CAPABILITIES:
        if split_cap[("heldout", cap)] != 40:
            errors.append(f"heldout/{cap}: expected 40, got {split_cap[('heldout', cap)]}")
        if split_cap[("diagnostic", cap)] != 8:
            errors.append(f"diagnostic/{cap}: expected 8, got {split_cap[('diagnostic', cap)]}")
    if len([x for x in items if x.split == "heldout"]) != 600:
        errors.append("heldout total must be 600")
    if len([x for x in items if x.split == "diagnostic"]) != 120:
        errors.append("diagnostic total must be 120")
    permitted = {"heldout", "diagnostic"}
    for x in items:
        if x.capability not in CAPABILITIES:
            errors.append(f"{x.id}: unknown capability {x.capability}")
        if x.split not in permitted:
            errors.append(f"{x.id}: invalid split {x.split}")
        if not x.prompt or not x.gold:
            errors.append(f"{x.id}: empty prompt/gold")
        if not x.gold[0].isspace():
            errors.append(f"{x.id}: gold must begin with whitespace to make the answer boundary explicit")
        if normalized_text(x.gold) in normalized_text(x.prompt):
            warnings.append(f"{x.id}: normalized gold appears in prompt; verify this is not leakage")
    for cap in CAPABILITIES:
        held = [x for x in items if x.split == "heldout" and x.capability == cap]
        labels = Counter(x.gold.strip().upper() for x in held if x.eval_type in {"mcq", "restricted_choice"})
        if labels and (set(labels) != {"A", "B", "C", "D"} or max(labels.values()) - min(labels.values()) > 1):
            errors.append(f"{cap}: heldout answer labels are not balanced: {dict(labels)}")
    return {
        "valid": not errors,
        "n_items": len(items),
        "split_capability_counts": {f"{s}/{c}": n for (s, c), n in sorted(split_cap.items())},
        "dataset_sha256": dataset_hash(items),
        "errors": errors,
        "warnings": warnings,
    }


def lint_localization(items: list[PilotItem], battery_items: list[PilotItem]) -> dict:
    errors: list[str] = []
    ids = Counter(x.id for x in items)
    battery_prompts = {normalized_text(x.prompt) for x in battery_items}
    for item_id, n in ids.items():
        if n != 1:
            errors.append(f"duplicate localization id {item_id}: {n}")
    for target in TARGETS:
        subset = [x for x in items if x.capability == target and x.split == "localization"]
        if len(subset) < 32:
            errors.append(f"{target}: expected at least 32 localization prompts, got {len(subset)}")
        if len({normalized_text(x.prompt) for x in subset}) != len(subset):
            errors.append(f"{target}: duplicate normalized localization prompts")
    for x in items:
        if x.capability not in TARGETS or x.split != "localization":
            errors.append(f"{x.id}: localization rows must be coding/translation with split=localization")
        if normalized_text(x.prompt) in battery_prompts:
            errors.append(f"{x.id}: localization/evaluation prompt overlap")
    return {"valid": not errors, "n_items": len(items), "dataset_sha256": dataset_hash(items), "errors": errors}

