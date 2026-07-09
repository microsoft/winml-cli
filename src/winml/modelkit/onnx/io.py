# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX I/O utilities — tensor specs and ONNX model I/O extraction.

Canonical home for InputTensorSpec / OutputTensorSpec dataclasses and
get_io_config() which combines ONNX graph input specs with
winml.io.inputs metadata (value_range) for a complete I/O configuration.

Example:
    >>> from winml.modelkit.onnx.io import get_io_config, InputTensorSpec
    >>> io_config = get_io_config("model.onnx")
    >>> io_config["input_names"]
    ['input_ids', 'attention_mask', 'token_type_ids']
    >>> io_config["value_ranges"]
    {'input_ids': (0, 30522), 'attention_mask': (0, 2), 'token_type_ids': (0, 2)}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

ShapeDim = int | str


# =============================================================================
# Tensor Specification Dataclasses
# =============================================================================


@dataclass
class InputTensorSpec:
    """Specification for an input tensor.

    All fields are optional - unspecified fields will be inferred during export.

    Attributes:
        name: Input tensor name in the ONNX graph (e.g., "pixel_values", "input_ids")
        dtype: Data type (e.g., "float32", "int64", "float16")
        shape: Tensor shape as tuple (e.g., (1, 3, 224, 224)). String dimensions
            are symbolic ONNX dimensions and use size 1 for dummy tensor generation.
        value_range: Optional (min, max_exclusive) for dummy tensor generation.
            Populated by resolve_export_config() via Optimum's interceptor.
            Integer semantics: torch.randint(min, max) — max is exclusive.
            Float semantics: uniform in [min, max).

    Example:
        # Vision model input
        InputTensorSpec(name="pixel_values", dtype="float32", shape=(1, 3, 224, 224))

        # Language model input with value range
        InputTensorSpec(name="input_ids", dtype="int64", shape=(1, 128), value_range=(0, 30522))

        # Minimal - just specify name
        InputTensorSpec(name="pixel_values")
    """

    name: str | None = None
    dtype: str | None = None  # "float32", "float16", "int64", "int32", etc.
    shape: tuple[ShapeDim, ...] | None = None
    value_range: tuple[float, float] | None = None  # (min, max_exclusive)

    def to_tensor(self) -> Any:
        """Generate a dummy tensor from this spec.

        When value_range is set, generates values in the correct range:
        - Integer: torch.randint(low, high, shape)
        - Float: uniform in [low, high)

        Falls back to ones (int) or rand [0,1) (float) when no range set.

        Returns:
            torch.Tensor with the correct shape, dtype, and value range.

        Raises:
            ValueError: If shape is not set.
        """
        import torch

        if self.shape is None:
            raise ValueError(f"Cannot create tensor: shape is None for '{self.name}'")

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int64": torch.int64,
            "int32": torch.int32,
            "int8": torch.int8,
        }
        torch_dtype = dtype_map.get(self.dtype or "float32", torch.float32)

        concrete_shape = self.concrete_shape()

        if self.value_range is not None:
            lo, hi = self.value_range
            if torch_dtype.is_floating_point:
                return torch.rand(concrete_shape, dtype=torch_dtype) * (hi - lo) + lo
            return torch.randint(int(lo), int(hi), concrete_shape, dtype=torch_dtype)

        # Fallback: no range info (backward compatible)
        if torch_dtype.is_floating_point:
            return torch.rand(concrete_shape, dtype=torch_dtype)
        return torch.ones(concrete_shape, dtype=torch_dtype)

    def concrete_shape(self) -> tuple[int, ...]:
        """Return a torch-compatible dummy shape, replacing symbolic dims with 1."""
        if self.shape is None:
            raise ValueError(f"Cannot create tensor: shape is None for '{self.name}'")

        concrete: list[int] = []
        for dim in self.shape:
            if isinstance(dim, str):
                if not dim:
                    raise ValueError(f"Cannot create tensor: empty symbolic dim for '{self.name}'")
                concrete.append(1)
            elif isinstance(dim, int):
                concrete.append(dim)
            else:
                raise TypeError(
                    f"Cannot create tensor: shape for '{self.name}' contains "
                    f"unsupported dimension {dim!r}"
                )
        return tuple(concrete)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result: dict[str, Any] = {}
        if self.name is not None:
            result["name"] = self.name
        if self.dtype is not None:
            result["dtype"] = self.dtype
        if self.shape is not None:
            result["shape"] = self.shape
        if self.value_range is not None:
            result["value_range"] = list(self.value_range)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputTensorSpec:
        """Create from dictionary."""
        shape = data.get("shape")
        if shape is not None and isinstance(shape, list):
            shape = tuple(shape)
        vr = data.get("value_range")
        value_range = tuple(vr) if vr is not None else None
        return cls(
            name=data.get("name"),
            dtype=data.get("dtype"),
            shape=shape,
            value_range=value_range,
        )


@dataclass
class OutputTensorSpec:
    """Specification for an output tensor.

    All fields are optional - unspecified fields will be inferred during export.

    Attributes:
        name: Output tensor name in the ONNX graph (e.g., "logits", "last_hidden_state")

    Example:
        OutputTensorSpec(name="logits")
    """

    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        if self.name is not None:
            result["name"] = self.name
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputTensorSpec:
        """Create from dictionary."""
        return cls(name=data.get("name"))


# ONNX elem_type to numpy dtype mapping
ONNX_ELEM_TYPE_TO_NUMPY: dict[int, np.dtype] = {
    1: np.dtype("float32"),  # FLOAT
    2: np.dtype("uint8"),  # UINT8
    3: np.dtype("int8"),  # INT8
    5: np.dtype("int16"),  # INT16
    6: np.dtype("int32"),  # INT32
    7: np.dtype("int64"),  # INT64
    10: np.dtype("float16"),  # FLOAT16
    11: np.dtype("float64"),  # DOUBLE
    12: np.dtype("uint32"),  # UINT32
    13: np.dtype("uint64"),  # UINT64
}


def get_io_config(
    model_path: str | Any,
) -> dict[str, Any]:
    """Extract I/O configuration from an ONNX model, including value ranges.

    Combines ONNX graph input/output specs (names, shapes, dtypes) with
    winml.io.inputs metadata (value_range) for a complete I/O picture.

    Args:
        model_path: Path to ONNX model file.

    Returns:
        Dict with:
            - input_names: list of input tensor names
            - input_shapes: list of input shapes (None for dynamic dims)
            - input_types: list of numpy dtypes for inputs
            - value_ranges: dict mapping input names to (min, max) tuples
            - output_names: list of output tensor names
            - output_shapes: list of output shapes
            - output_types: list of numpy dtypes for outputs

    Raises:
        FileNotFoundError: If model_path does not exist.

    Example:
        >>> config = get_io_config("bert_model.onnx")
        >>> config["input_names"]
        ['input_ids', 'attention_mask', 'token_type_ids']
        >>> config["value_ranges"]
        {'input_ids': (0, 30522), 'attention_mask': (0, 2)}
    """
    from .persistence import load_onnx

    model_path = str(model_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    model = load_onnx(model_path, load_weights=False, validate=False)

    io_config: dict[str, Any] = {
        "input_names": [],
        "input_shapes": [],
        "input_types": [],
        "output_names": [],
        "output_shapes": [],
        "output_types": [],
        "value_ranges": {},
    }

    for prefix, ios in [
        ("input", model.graph.input),
        ("output", model.graph.output),
    ]:
        for io in ios:
            tensor_type = io.type.tensor_type

            # Handle sequence types (fallback)
            if tensor_type.elem_type == 0 and io.type.HasField("sequence_type"):
                tensor_type = io.type.sequence_type.elem_type.tensor_type

            # Extract dtype
            dtype = ONNX_ELEM_TYPE_TO_NUMPY.get(tensor_type.elem_type, np.dtype("float32"))

            # Extract shape (None for dynamic dims)
            shape: list[int | None] = []
            if tensor_type.HasField("shape"):
                for dim in tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        shape.append(dim.dim_value)
                    else:
                        shape.append(None)

            io_config[f"{prefix}_names"].append(io.name)
            io_config[f"{prefix}_shapes"].append(shape)
            io_config[f"{prefix}_types"].append(dtype)

    # Enhance with value ranges from winml.io.inputs metadata
    for prop in model.metadata_props:
        if prop.key == "winml.io.inputs":
            try:
                specs = json.loads(prop.value)
                for spec in specs:
                    name = spec.get("name")
                    vr = spec.get("value_range")
                    if name and vr is not None:
                        io_config["value_ranges"][name] = tuple(vr)
            except (json.JSONDecodeError, TypeError):
                logger.debug("Failed to parse winml.io.inputs metadata")
            break

    return io_config


def generate_inputs_from_onnx(model_path: str) -> dict[str, Any]:
    """Generate dummy input tensors from an ONNX model's I/O config and metadata.

    Reads input specs (shapes, dtypes) and value ranges (from winml.io.inputs
    metadata) from the ONNX model, then generates random tensors with correct
    ranges via InputTensorSpec.to_tensor().

    Args:
        model_path: Path to ONNX model file.

    Returns:
        Dict mapping input names to torch tensors with correct shapes,
        dtypes, and value ranges.
    """
    io_config = get_io_config(model_path)
    inputs: dict[str, Any] = {}

    value_ranges = io_config.get("value_ranges", {})

    for name, shape, dtype in zip(
        io_config["input_names"],
        io_config["input_shapes"],
        io_config["input_types"],
        strict=False,
    ):
        # Replace dynamic dims (None) with 1
        resolved_shape = tuple(d if d is not None and d > 0 else 1 for d in shape)

        # Map numpy dtype to string for InputTensorSpec
        dtype_str = str(dtype).replace("numpy.", "")  # e.g. "int32", "float32"

        spec = InputTensorSpec(
            name=name,
            dtype=dtype_str,
            shape=resolved_shape,
            value_range=value_ranges.get(name),
        )
        inputs[name] = spec.to_tensor()

    return inputs
