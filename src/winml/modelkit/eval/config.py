# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Configuration for evaluation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..datasets.config import DatasetConfig
from ..utils.constants import EPNameOrAlias


@dataclass
class SchemaColumn:
    """Describes one expected dataset column for --schema output.

    Attributes:
        name: Default column name in the dataset.
        type: HF feature type (e.g. "Image", "ClassLabel").
        override: CLI --column key that maps to this column (e.g. "label_column").
        required: Whether the column is mandatory.
        description: Short human-readable description.
        children: Nested columns for dict-type features.
    """

    name: str
    type: str
    override: str | None = None
    required: bool = True
    description: str = ""
    children: list[SchemaColumn] = field(default_factory=list)


@dataclass
class WinMLEvaluationConfig:
    """Configuration for model evaluation.

    Attributes:
        model_id: HuggingFace model ID for config/preprocessor resolution.
        model_path: Path to .onnx model file, or a ``{role: path}`` dict for
            composite models (e.g. ``{"image-encoder": "...", "text-encoder": "..."}``).
            None = build from model_id.
        task: HF pipeline task. Auto-detected from model_id if omitted.
        device: Target device for inference.
        ep: Explicit execution provider (e.g., "qnn", "dml"). Overrides
            device-to-provider mapping when provided.
        dataset: Dataset configuration.
        output_path: Path to write JSON results.

    Usage:
        config = WinMLEvaluationConfig(
            model_id="microsoft/resnet-50",
            dataset=DatasetConfig(path="imagenet-1k", samples=10),
        )
    """

    model_id: str | None = None
    model_path: str | dict[str, str] | None = None
    task: str | None = None
    device: str = "cpu"
    precision: str = "auto"
    ep: EPNameOrAlias | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output_path: Path | None = field(default=None, metadata={"cli_name": "output"})

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result: dict = {}
        if self.model_id is not None:
            result["model_id"] = self.model_id
        if self.model_path is not None:
            result["model_path"] = self.model_path
        if self.task is not None:
            result["task"] = self.task
        result["device"] = self.device
        if self.precision != "auto":
            result["precision"] = self.precision
        if self.ep is not None:
            result["ep"] = self.ep
        result["dataset"] = self.dataset.to_dict()
        if self.output_path is not None:
            result["output_path"] = str(self.output_path)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> WinMLEvaluationConfig:
        """Create from dictionary, ignoring unknown fields."""
        ds_data = data.get("dataset", {})
        dataset = DatasetConfig(
            path=ds_data.get("path"),
            name=ds_data.get("name"),
            split=ds_data.get("split", "validation"),
            samples=ds_data.get("samples", 100),
            shuffle=ds_data.get("shuffle", True),
            seed=ds_data.get("seed", 42),
            columns_mapping=ds_data.get("columns_mapping", {}),
            streaming=ds_data.get("streaming", False),
            build_script=ds_data.get("build_script"),
            label_mapping_file=ds_data.get("label_mapping_file"),
        )
        return cls(
            model_id=data.get("model_id"),
            model_path=data.get("model_path"),
            task=data.get("task"),
            device=data.get("device", "cpu"),
            precision=data.get("precision", "auto"),
            ep=data.get("ep"),
            dataset=dataset,
            output_path=(Path(data["output_path"]) if data.get("output_path") else None),
        )
