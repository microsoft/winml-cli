# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Dataset configuration for accuracy evaluation (Signal 2).

Single source of truth: the model registry (e.g. ``testsets/models_with_acc.json``).
Configs are registered at startup via ``register_from_registry()``.

Resolution:
1. Per-model config registered from the registry's ``dataset_config`` field.
2. None — caller decides whether to skip or let winml eval use its
   built-in task defaults.
"""

from __future__ import annotations


_DATASET_CONFIGS: dict[tuple[str, str], dict] = {}


def register_from_registry(entries: list) -> None:
    """Register dataset configs from registry entries.

    ``entries`` should be a list of ModelEntry (or any object with
    ``hf_id``, ``task``, and ``dataset_config`` attributes).
    """
    for entry in entries:
        ds = getattr(entry, "dataset_config", None)
        if ds is None:
            continue
        key = (entry.hf_id, entry.task)
        cfg = {**ds}
        # Normalise keys: "path" -> "dataset", "name" -> "dataset_config"
        if "path" in cfg:
            cfg["dataset"] = cfg.pop("path")
        if "name" in cfg:
            cfg["dataset_config"] = cfg.pop("name")
        if "samples" in cfg:
            cfg["num_samples"] = cfg.pop("samples")
        _DATASET_CONFIGS[key] = cfg


def get_dataset_config(hf_id: str, task: str) -> dict | None:
    """Return dataset config for a model, or None.

    None means no explicit config was found; the caller can either
    skip or let winml eval / pytorch baseline use built-in task defaults.
    """
    return _DATASET_CONFIGS.get((hf_id, task))
