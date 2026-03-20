# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX model utilities for ModelKit.

Read and write winml.* metadata, extract I/O config from ONNX models.
Canonical home for InputTensorSpec / OutputTensorSpec tensor spec dataclasses.

Inspection utilities (find_undefined_types, get_qdq_param_info, etc.)
are available via ``modelkit.onnx.inspection`` for diagnostic use.
"""

from __future__ import annotations

from .detection import is_compiled_onnx, is_quantized_onnx
from .external_data import copy_onnx_model
from .io import InputTensorSpec, OutputTensorSpec, generate_inputs_from_onnx, get_io_config
from .metadata import capture_metadata, restore_metadata
from .persistence import cleanup_onnx, load_onnx, save_onnx
from .shape import infer_shapes


__all__ = [
    "InputTensorSpec",
    "OutputTensorSpec",
    "capture_metadata",
    "cleanup_onnx",
    "copy_onnx_model",
    "generate_inputs_from_onnx",
    "get_io_config",
    "infer_shapes",
    "is_compiled_onnx",
    "is_quantized_onnx",
    "load_onnx",
    "restore_metadata",
    "save_onnx",
]
