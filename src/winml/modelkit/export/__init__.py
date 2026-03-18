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
from .io import (
    MaxLengthTextInputGenerator,
    OnnxConfigNotFoundError,
    register_onnx_overwrite,
    resolve_io_specs,
)
from .pytorch import export_pytorch
from .pytorch import export_pytorch as export_onnx


__version__ = "2.1.0"

__all__ = [
    "InputTensorSpec",
    "MaxLengthTextInputGenerator",
    "OnnxConfigNotFoundError",
    "OutputTensorSpec",
    "WinMLExportConfig",
    "export_onnx",
    "export_pytorch",
    "register_onnx_overwrite",
    "resolve_export_config",
    "resolve_io_specs",
]
