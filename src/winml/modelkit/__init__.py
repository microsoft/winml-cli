# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML CLI - Accelerate Model Deployment on Windows ML.

WinML CLI provides tools for converting PyTorch models to optimized ONNX format
with support for QNN (Qualcomm Neural Processing SDK) and OpenVINO backends.

Key Features:
- Universal ONNX export with hierarchy preservation
- QNN-optimized configurations
- Static batch enforcement for hardware compatibility
- Model-agnostic design principles
- Automatic task detection and model selection

Usage:
    from winml.modelkit import WinMLAutoModel, WinMLBuildConfig

    # Auto-detect task and load model
    model = WinMLAutoModel.from_pretrained("microsoft/resnet-50")

    # With custom config
    from .optim import WinMLOptimizationConfig
    config = WinMLBuildConfig(
        optim=WinMLOptimizationConfig(gelu_fusion=True),
    )
    model = WinMLAutoModel.from_pretrained("facebook/convnext-tiny-224", config=config)
"""

import logging
import sys
from importlib.metadata import PackageNotFoundError, version


# Force utf-8 stdout/stderr so emoji and Unicode output (rich console, logs,
# CLI banners) does not raise UnicodeEncodeError on Windows shells that start
# Python with a charmap codec (e.g., PYTHONIOENCODING=cp1252).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logging.getLogger(__name__).addHandler(logging.NullHandler())


def _preload_bundled_onnxruntime_dll() -> None:
    # Windows ships C:\Windows\System32\onnxruntime.dll (older API version)
    # as part of the system WindowsML component. When WinML EP plugin DLLs
    # are loaded (via EpCatalog), they import "onnxruntime.dll" by base name
    # and the loader binds them to the system copy, producing
    # "The requested API version [N] is not available" errors.
    # Loading the wheel-bundled DLL by full path first makes later base-name
    # imports resolve to the already-loaded module.
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import os
        from importlib.util import find_spec
        from pathlib import Path

        spec = find_spec("onnxruntime")
        if spec is None or spec.origin is None:
            return
        dll = Path(spec.origin).parent / "capi" / "onnxruntime.dll"
        if not dll.is_file():
            return
        os.add_dll_directory(str(dll.parent))
        ctypes.WinDLL(str(dll))
    except Exception as _e:  # pragma: no cover - best-effort preload
        print(f"Warning: failed to preload bundled onnxruntime.dll: {_e}", file=sys.stderr)


_preload_bundled_onnxruntime_dll()

from . import _warnings  # Configure warning filters before importing subpackages


try:
    __version__ = version("winml-cli")
except PackageNotFoundError:
    __version__ = "0.0.1.dev0"

__all__ = [
    "WinMLAutoModel",
    "WinMLBuildConfig",
    "WinMLModelForImageClassification",
    "WinMLPreTrainedModel",
    "__version__",
]


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "WinMLBuildConfig": (".config", "WinMLBuildConfig"),
    "WinMLAutoModel": (".models", "WinMLAutoModel"),
    "WinMLPreTrainedModel": (".models", "WinMLPreTrainedModel"),
    "WinMLModelForImageClassification": (".models", "WinMLModelForImageClassification"),
}


def __getattr__(name: str):
    """Lazy-load heavy exports on first access (PEP 562).

    This avoids importing torch/transformers/optimum (~30s) when only
    lightweight operations are needed (e.g., ``winml --help``).
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy attributes in dir() for debugger/IPython compatibility."""
    return list(set(list(globals()) + __all__))
