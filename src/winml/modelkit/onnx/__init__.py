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

# Lazy imports to avoid circular dependency: onnx → compiler → onnx
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "is_compiled_onnx": (".detection", "is_compiled_onnx"),
    "is_quantized_onnx": (".detection", "is_quantized_onnx"),
    "ONNXDomain": (".domains", "ONNXDomain"),
    "SupportedONNXType": (".dtypes", "SupportedONNXType"),
    "remove_optional_from_type_annotation": (".dtypes", "remove_optional_from_type_annotation"),
    "copy_onnx_model": (".external_data", "copy_onnx_model"),
    "InputTensorSpec": (".io", "InputTensorSpec"),
    "OutputTensorSpec": (".io", "OutputTensorSpec"),
    "generate_inputs_from_onnx": (".io", "generate_inputs_from_onnx"),
    "get_io_config": (".io", "get_io_config"),
    "capture_metadata": (".metadata", "capture_metadata"),
    "restore_metadata": (".metadata", "restore_metadata"),
    "cleanup_onnx": (".persistence", "cleanup_onnx"),
    "load_onnx": (".persistence", "load_onnx"),
    "save_onnx": (".persistence", "save_onnx"),
    "infer_onnx_shapes": (".shape", "infer_onnx_shapes"),
    "infer_shapes": (".shape", "infer_shapes"),
    "EXTERNAL_DATA_THRESHOLD": (".utils", "EXTERNAL_DATA_THRESHOLD"),
    "check_onnx_model": (".utils", "check_onnx_model"),
    "get_model_size": (".utils", "get_model_size"),
}


def __getattr__(name: str):
    """Lazy-load submodule exports on first access (PEP 562)."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy attributes in dir()."""
    return list(set(list(globals()) + __all__))
