# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Dataset configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    """Dataset configuration, aligned with HF load_dataset() API.

    Attributes:
        path: HF dataset path (e.g., "imagenet-1k", "glue").
        name: Config name for multi-config datasets (e.g., "mrpc").
        split: Dataset split.
        samples: Number of samples to evaluate.
        shuffle: Whether to shuffle before sampling for label coverage.
        seed: Random seed for reproducible shuffling.
        columns_mapping: Column name overrides as key=value pairs.
            If empty, consumer uses its own defaults.
        streaming: Whether to stream dataset (avoids full download).
    """

    path: str | None = None
    name: str | None = None
    split: str = "validation"
    samples: int = 100
    shuffle: bool = True
    seed: int = 42
    columns_mapping: dict[str, str] = field(default_factory=dict)
    label_mapping: dict[str, int] | None = None
    streaming: bool = False
    # Tracks which fields were explicitly set by the caller (e.g. CLI).
    # Not serialized; used by evaluate.py to merge user overrides onto defaults.
    explicit_fields: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "split": self.split,
            "samples": self.samples,
            "shuffle": self.shuffle,
            "seed": self.seed,
        }
        if self.path is not None:
            result["path"] = self.path
        if self.name is not None:
            result["name"] = self.name
        if self.columns_mapping:
            result["columns_mapping"] = self.columns_mapping
        if self.label_mapping:
            result["label_mapping"] = self.label_mapping
        if self.streaming:
            result["streaming"] = self.streaming
        return result
