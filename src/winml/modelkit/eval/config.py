# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Configuration for evaluation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.constants import EPNameOrAlias
from ..utils.eval_utils import EvalMode


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
        revision: Git revision (branch, tag, or commit) to load. Useful for
            datasets pinned to a specific snapshot (e.g.
            ``refs/convert/parquet``).
        build_script: Path to a Python script that builds the dataset locally.
            When set alongside ``path``, the script is invoked with
            ``--output <path>`` before the dataset is loaded.
        label_mapping_file: Path to a JSON file with label mapping.
            Resolved into ``label_mapping`` at eval time.
    """

    path: str | None = field(default=None, metadata={"cli_name": "dataset_path"})
    name: str | None = field(default=None, metadata={"cli_name": "dataset_name"})
    split: str = "validation"
    samples: int = 100
    shuffle: bool = True
    seed: int = 42
    columns_mapping: dict[str, str] = field(default_factory=dict)
    label_mapping: dict[str, int] | None = None
    streaming: bool = False
    revision: str | None = field(default=None, metadata={"cli_name": "dataset_revision"})
    build_script: str | None = field(default=None, metadata={"cli_name": "dataset_script"})
    label_mapping_file: str | None = None

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
        if self.revision is not None:
            result["revision"] = self.revision
        if self.build_script is not None:
            result["build_script"] = self.build_script
        if self.label_mapping_file is not None:
            result["label_mapping_file"] = self.label_mapping_file
        return result


@dataclass
class WinMLEvaluationConfig:
    """Configuration for model evaluation.

    Attributes:
        model_id: HuggingFace model ID for config/preprocessor resolution.
        model_path: Path to .onnx model file, or a ``{role: path}`` dict for
            composite models (e.g. ``{"image-encoder": "...", "text-encoder": "..."}``).
            None = build from model_id.
        reference_path: Path to a second ``.onnx`` file used as the reference in
            ``--mode compare``. When set, both ``model_path`` and ``reference_path``
            run as raw ORT sessions and their output tensors are compared directly,
            so no ``model_id`` / ``task`` / HF reference is needed.
        input_data: Path to a ``.npz`` archive of real input tensors for
            ``--mode compare``. When set, the candidate and reference are compared
            on these tensors (validated against the candidate's inputs) instead of
            randomly generated ones. The leading axis of each array is the sample
            axis, so one archive can hold ``N`` samples; all inputs must share the
            same leading length.
        task: HF pipeline task. Auto-detected from model_id if omitted.
        device: Target device for inference.
        ep: Explicit execution provider (e.g., "qnn", "dml"). Overrides
            device-to-provider mapping when provided.
        dataset: Dataset configuration.
        output_path: Path to write JSON results.
        mode: Evaluation mode (see :data:`EvalMode`).

            - ``"onnx"`` (default): evaluate the ONNX candidate on the
              labeled dataset.
            - ``"compare"``: compare ONNX vs HF reference output tensors
              on identical random inputs and report tensor-similarity
              metrics per output tensor. When ``reference_path`` is set,
              the reference is a second ONNX file instead of the HF model.
              When ``input_data`` is set, real tensors from a ``.npz`` are
              used instead of random inputs.

    Usage:
        config = WinMLEvaluationConfig(
            model_id="microsoft/resnet-50",
            dataset=DatasetConfig(path="imagenet-1k", samples=10),
        )
    """

    model_id: str | None = None
    model_path: str | dict[str, str] | None = None
    reference_path: str | None = field(default=None, metadata={"cli_name": "reference"})
    input_data: str | None = None
    task: str | None = None
    device: str = "auto"
    precision: str = "auto"
    ep: EPNameOrAlias | None = None
    allow_unsupported_nodes: bool = False
    # Build-pipeline toggles, applied when building from model_id (ignored for
    # pre-built ONNX inputs). Shared semantics with winml build/perf.
    quant: bool = True
    optimize: bool = True
    analyze: bool = True
    max_optim_iterations: int | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output_path: Path | None = field(default=None, metadata={"cli_name": "output"})
    mode: EvalMode = "onnx"
    skip_build: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result: dict = {}
        if self.model_id is not None:
            result["model_id"] = self.model_id
        if self.model_path is not None:
            result["model_path"] = self.model_path
        if self.reference_path is not None:
            result["reference_path"] = self.reference_path
        if self.input_data is not None:
            result["input_data"] = self.input_data
        if self.task is not None:
            result["task"] = self.task
        result["device"] = self.device
        if self.precision != "auto":
            result["precision"] = self.precision
        if self.ep is not None:
            result["ep"] = self.ep
        if self.allow_unsupported_nodes:
            result["allow_unsupported_nodes"] = self.allow_unsupported_nodes
        # Emit build toggles only when they deviate from the default so the
        # serialized config stays minimal.
        if not self.quant:
            result["quant"] = self.quant
        if not self.optimize:
            result["optimize"] = self.optimize
        if not self.analyze:
            result["analyze"] = self.analyze
        if self.max_optim_iterations is not None:
            result["max_optim_iterations"] = self.max_optim_iterations
        result["dataset"] = self.dataset.to_dict()
        if self.output_path is not None:
            result["output_path"] = str(self.output_path)
        if self.mode != "onnx":
            result["mode"] = self.mode
        result["skip_build"] = self.skip_build
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
            revision=ds_data.get("revision"),
            build_script=ds_data.get("build_script"),
            label_mapping_file=ds_data.get("label_mapping_file"),
        )
        return cls(
            model_id=data.get("model_id"),
            model_path=data.get("model_path"),
            reference_path=data.get("reference_path"),
            input_data=data.get("input_data"),
            task=data.get("task"),
            device=data.get("device", "auto"),
            precision=data.get("precision", "auto"),
            ep=data.get("ep"),
            allow_unsupported_nodes=data.get("allow_unsupported_nodes", False),
            quant=data.get("quant", True),
            optimize=data.get("optimize", True),
            analyze=data.get("analyze", True),
            max_optim_iterations=data.get("max_optim_iterations"),
            dataset=dataset,
            output_path=(Path(data["output_path"]) if data.get("output_path") else None),
            mode=data.get("mode", "onnx"),
            skip_build=data.get("skip_build", True),
        )
