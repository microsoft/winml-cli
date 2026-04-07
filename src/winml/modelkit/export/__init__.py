# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WML Export - ONNX Export with Hierarchy Preservation.

This package provides:
- WinMLExportConfig with input/output tensor specifications
- resolve_export_config for unified export + loader config resolution
- resolve_io_specs for resolving I/O tensor specs from OnnxConfig
- export_pytorch / export_onnx for ONNX export
"""

from .config import (
    InputTensorSpec,
    OutputTensorSpec,
    WinMLExportConfig,
    resolve_export_config,
)


def __getattr__(name: str):
    """Lazy-load heavy submodules to avoid importing optimum at startup."""
    _io_names = {
        "MaxLengthTextInputGenerator",
        "ONNXConfigNotFoundError",
        "generate_dummy_inputs",
        "register_onnx_overwrite",
        "resolve_io_specs",
    }
    if name in _io_names:
        from . import io

        return getattr(io, name)

    _pytorch_names = {"export_pytorch", "export_onnx"}
    if name in _pytorch_names:
        from .pytorch import export_pytorch

        if name == "export_onnx":
            return export_pytorch
        return export_pytorch

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "2.1.0"

__all__ = [
    "InputTensorSpec",
    "MaxLengthTextInputGenerator",
    "ONNXConfigNotFoundError",
    "OutputTensorSpec",
    "WinMLExportConfig",
    "export_onnx",
    "export_pytorch",
    "generate_dummy_inputs",
    "register_onnx_overwrite",
    "resolve_export_config",
    "resolve_io_specs",
]
