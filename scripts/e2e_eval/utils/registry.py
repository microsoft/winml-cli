# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Model registry loading and filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003


@dataclass
class ModelEntry:
    """A single model entry from models.json."""

    hf_id: str
    task: str
    model_type: str
    group: str
    priority: str
    dataset_config: dict | None = None
    perf_args: list[str] = field(default_factory=list)
    eval_args: list[str] = field(default_factory=list)
    downloads: int = 0
    last_update_time: str | None = None
    optimum_supported: bool = False


_REQUIRED_FIELDS = {"hf_id", "task", "model_type", "group", "priority"}
_VALID_PRIORITIES = {"P0", "P1", "P2"}


def load_registry(path: Path) -> list[ModelEntry]:
    """Load models.json, validate required fields, return entries."""
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Registry must be a JSON array, got {type(raw).__name__}")  # noqa: TRY004

    entries: list[ModelEntry] = []
    for i, item in enumerate(raw):
        missing = _REQUIRED_FIELDS - set(item.keys())
        if missing:
            raise ValueError(f"Entry {i} ({item.get('hf_id', '?')}): missing fields {missing}")

        priority = item["priority"]
        if priority not in _VALID_PRIORITIES:
            raise ValueError(
                f"Entry {i} ({item['hf_id']}): invalid priority '{priority}', "
                f"must be one of {sorted(_VALID_PRIORITIES)}"
            )

        overrides = item.get("config_overrides", {})
        perf_args = overrides.get("perf_args", [])
        eval_args = overrides.get("eval_args", [])
        raw_ds = item.get("dataset_config")
        ds_config = raw_ds if isinstance(raw_ds, dict) else None
        entries.append(
            ModelEntry(
                hf_id=item["hf_id"],
                task=item["task"],
                model_type=item["model_type"],
                group=item["group"],
                priority=priority,
                dataset_config=ds_config,
                perf_args=perf_args,
                eval_args=eval_args,
                downloads=item.get("downloads", 0) or 0,
                last_update_time=item.get("last_update_time"),
                optimum_supported=item.get("optimum_supported", False),
            )
        )

    return entries


def filter_registry(
    entries: list[ModelEntry],
    *,
    task: str | None = None,
    priority: str | None = None,
    model_type: str | None = None,
    group: str | None = None,
) -> list[ModelEntry]:
    """Apply AND-combined filters."""
    result = entries
    if task:
        result = [e for e in result if e.task == task]
    if priority:
        result = [e for e in result if e.priority == priority]
    if model_type:
        result = [e for e in result if e.model_type == model_type]
    if group:
        result = [e for e in result if e.group == group]
    return result


def make_adhoc_entry(hf_id: str, task: str | None = None) -> ModelEntry:
    """Create a synthetic ModelEntry for --hf-model single model mode."""
    return ModelEntry(
        hf_id=hf_id,
        task=task or "",
        model_type="unknown",
        group="adhoc",
        priority="P0",
    )
