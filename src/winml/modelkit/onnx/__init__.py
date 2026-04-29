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

from .domains import ONNXDomain
from .dtypes import SupportedONNXType, remove_optional_from_type_annotation
from .external_data import copy_onnx_model
from .io import InputTensorSpec, OutputTensorSpec, generate_inputs_from_onnx, get_io_config
from .metadata import capture_metadata, restore_metadata
from .persistence import cleanup_onnx, load_onnx, save_onnx
from .shape import infer_onnx_shapes, infer_shapes
from .utils import EXTERNAL_DATA_THRESHOLD, check_onnx_model, get_model_size


__all__ = [
    "EXTERNAL_DATA_THRESHOLD",
    "InputTensorSpec",
    "ONNXDomain",
    "OutputTensorSpec",
    "SupportedONNXType",
    "capture_metadata",
    "check_onnx_model",
    "cleanup_onnx",
    "copy_onnx_model",
    "generate_inputs_from_onnx",
    "get_io_config",
    "get_model_size",
    "infer_onnx_shapes",
    "infer_shapes",
    "is_compiled_onnx",
    "is_quantized_onnx",
    "load_onnx",
    "remove_optional_from_type_annotation",
    "restore_metadata",
    "save_onnx",
]


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "is_compiled_onnx": (".detection", "is_compiled_onnx"),
    "is_quantized_onnx": (".detection", "is_quantized_onnx"),
}


def __getattr__(name: str):
    """Lazy-load detection module to avoid circular import with compiler."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(set(list(globals()) + __all__))
