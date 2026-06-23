# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Configuration classes for quantizer module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol


if TYPE_CHECKING:
    import numpy as np


class CalibrationDataReader(Protocol):
    """Protocol for calibration data providers."""

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return next calibration sample or None when exhausted."""
        ...

    def rewind(self) -> None:
        """Reset to beginning."""
        ...


@dataclass
class WinMLQuantizationConfig:
    """Configuration for quantize_onnx.

    Defaults:
    - 10 random samples for calibration
    - uint8 for weights and activations
    - MinMax calibration method
    - Asymmetric quantization

    Usage:
        from winml.modelkit.quant import WinMLQuantizationConfig, quantize_onnx

        # Quick quantize with defaults
        config = WinMLQuantizationConfig()
        result = quantize_onnx("model.onnx", config)

        # Custom config
        config = WinMLQuantizationConfig(samples=100, weight_type="int8")
        result = quantize_onnx("model.onnx", config)

        # FP16 post-processing (mixed precision INT8 + FP16)
        config = WinMLQuantizationConfig(fp16=True)
        result = quantize_onnx("model.onnx", config)
    """

    # Quantization algorithm
    algorithm: Literal["static", "dynamic", "rtn"] = "static"
    # "static"  — Calibrated QDQ quantization (requires calibration data)
    # "dynamic" — Dynamic quantization (no calibration) [planned, not yet wired]
    # "rtn"     — Round-To-Nearest weight-only (no calibration, block-wise)

    mode: Literal["qdq", "static", "dynamic"] = "qdq"

    # Calibration settings (static/dynamic)
    samples: int = 10
    calibration_method: Literal["minmax", "entropy", "percentile"] = "minmax"
    calibration_data: CalibrationDataReader | None = None  # None = random data

    # Task-aware calibration (used when calibration_data is None)
    task: str | None = None  # e.g., "image-classification"
    model_name: str | None = None  # e.g., "microsoft/resnet-50"
    dataset_name: str | None = None  # Optional: override default dataset

    # Quantization types (static/dynamic)
    weight_type: Literal["uint8", "int8", "uint16", "int16"] = "uint8"
    activation_type: Literal["uint8", "int8", "uint16", "int16"] = "uint8"

    # Quantization options (static/dynamic)
    per_channel: bool = False
    symmetric: bool = False

    # Output settings
    save_calibration: bool = False

    # Calibration data management (ported from compiler.CalibrationConfig)
    distribution: str = "uniform"
    seed: int | None = None
    calibration_load_path: Path | None = None
    calibration_save_path: Path | None = None

    # Advanced (static/dynamic)
    op_types_to_quantize: list[str] | None = None
    nodes_to_exclude: list[str] | None = None

    # RTN-specific settings (only used when algorithm="rtn")
    rtn_bits: int = 4
    rtn_block_size: int = 128
    rtn_symmetric: bool = True
    rtn_accuracy_level: int = 0

    # FP16 post-processing (can combine with any algorithm)
    # When True, remaining FP32 tensors/ops are converted to FP16 after
    # quantization. This produces mixed-precision models (e.g. INT8 + FP16).
    fp16: bool = False
    fp16_only: bool = False  # When True, skip quantization and only do FP16
    fp16_keep_io_types: bool = True
    fp16_op_block_list: list[str] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Includes all fields that affect quantization behavior so that
        ``generate_cache_key()`` produces distinct hashes for distinct configs.
        Optional fields (task, model_name, dataset_name) are omitted when None
        to keep submodule configs clean.
        """
        result: dict = {
            "algorithm": self.algorithm,
            "mode": self.mode,
            "samples": self.samples,
            "calibration_method": self.calibration_method,
            "weight_type": self.weight_type,
            "activation_type": self.activation_type,
            "per_channel": self.per_channel,
            "symmetric": self.symmetric,
            "save_calibration": self.save_calibration,
            "distribution": self.distribution,
            "seed": self.seed,
            "calibration_load_path": (
                str(self.calibration_load_path) if self.calibration_load_path else None
            ),
            "calibration_save_path": (
                str(self.calibration_save_path) if self.calibration_save_path else None
            ),
            "op_types_to_quantize": self.op_types_to_quantize,
            "nodes_to_exclude": self.nodes_to_exclude,
            "fp16": self.fp16,
            "fp16_only": self.fp16_only,
        }
        if self.task is not None:
            result["task"] = self.task
        if self.model_name is not None:
            result["model_name"] = self.model_name
        if self.dataset_name is not None:
            result["dataset_name"] = self.dataset_name
        if self.algorithm == "rtn":
            result["rtn_bits"] = self.rtn_bits
            result["rtn_block_size"] = self.rtn_block_size
            result["rtn_symmetric"] = self.rtn_symmetric
            result["rtn_accuracy_level"] = self.rtn_accuracy_level
        if self.fp16:
            result["fp16_keep_io_types"] = self.fp16_keep_io_types
            result["fp16_op_block_list"] = self.fp16_op_block_list
        return result

    @classmethod
    def from_dict(cls, data: dict) -> WinMLQuantizationConfig:
        """Create from dictionary, ignoring unknown fields.

        Args:
            data: Configuration dictionary.

        Returns:
            WinMLQuantizationConfig instance.
        """
        return cls(
            algorithm=data.get("algorithm", "static"),
            mode=data.get("mode", "qdq"),
            samples=data.get("samples", data.get("calibration_samples", 10)),
            calibration_method=data.get("calibration_method", "minmax"),
            task=data.get("task"),
            model_name=data.get("model_name"),
            dataset_name=data.get("dataset_name"),
            weight_type=data.get("weight_type", "uint8"),
            activation_type=data.get("activation_type", "uint8"),
            per_channel=data.get("per_channel", False),
            symmetric=data.get("symmetric", False),
            save_calibration=data.get("save_calibration", False),
            distribution=data.get("distribution", "uniform"),
            seed=data.get("seed"),
            calibration_load_path=(
                Path(data["calibration_load_path"]) if data.get("calibration_load_path") else None
            ),
            calibration_save_path=(
                Path(data["calibration_save_path"]) if data.get("calibration_save_path") else None
            ),
            op_types_to_quantize=data.get("op_types_to_quantize"),
            nodes_to_exclude=data.get("nodes_to_exclude"),
            rtn_bits=data.get("rtn_bits", 4),
            rtn_block_size=data.get("rtn_block_size", 128),
            rtn_symmetric=data.get("rtn_symmetric", True),
            rtn_accuracy_level=data.get("rtn_accuracy_level", 0),
            fp16=data.get("fp16", False),
            fp16_only=data.get("fp16_only", False),
            fp16_keep_io_types=data.get("fp16_keep_io_types", True),
            fp16_op_block_list=data.get("fp16_op_block_list"),
        )


@dataclass
class QuantizeResult:
    """Result of quantize_onnx operation."""

    success: bool
    output_path: Path | None
    calibration_path: Path | None = None

    # Timing
    calibration_time_seconds: float = 0.0
    qdq_insertion_time_seconds: float = 0.0
    postproc_time_seconds: float = 0.0
    total_time_seconds: float = 0.0

    # Stats
    nodes_quantized: int = 0
    nodes_skipped: int = 0

    # Errors/warnings
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
