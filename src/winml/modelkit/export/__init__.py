# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Export - ONNX Export with Hierarchy Preservation.

This package provides:
- WinMLExportConfig with input/output tensor specifications
- resolve_export_config for unified export + loader config resolution
- resolve_io_specs for resolving I/O tensor specs from OnnxConfig
- export_pytorch / export_onnx for ONNX export
"""

from typing import TYPE_CHECKING, Any

from .config import (
    InputTensorSpec,
    OutputTensorSpec,
    WinMLExportConfig,
    resolve_export_config,
)


# Static type re-exports for the names exposed by ``__getattr__`` below.
# At runtime these are loaded lazily (see _LAZY_IMPORTS); at type-check time
# we want mypy to see real types so callers like ``build.hf.export_onnx(...)``
# get checked instead of resolving to ``Any``.
if TYPE_CHECKING:
    from .io import (
        MaxLengthTextInputGenerator,
        ONNXConfigNotFoundError,
        generate_dummy_inputs,
        register_onnx_overwrite,
        resolve_io_specs,
    )
    from .pytorch import export_pytorch
    from .pytorch import export_pytorch as export_onnx


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


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "MaxLengthTextInputGenerator": (".io", "MaxLengthTextInputGenerator"),
    "ONNXConfigNotFoundError": (".io", "ONNXConfigNotFoundError"),
    "generate_dummy_inputs": (".io", "generate_dummy_inputs"),
    "register_onnx_overwrite": (".io", "register_onnx_overwrite"),
    "resolve_io_specs": (".io", "resolve_io_specs"),
    "export_pytorch": (".pytorch", "export_pytorch"),
    "export_onnx": (".pytorch", "export_pytorch"),  # alias for export_pytorch
}


def __getattr__(name: str) -> Any:
    """Lazy-load heavy exports to avoid importing optimum at package init."""
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
