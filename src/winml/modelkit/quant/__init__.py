# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Quantizer module for ONNX models.

Provides QDQ (Quantize-Dequantize) quantization for ONNX models.

Usage:
    from winml.modelkit.quant import quantize_onnx, WinMLQuantizationConfig

    # Quick quantize with defaults (10 samples, uint8)
    result = quantize_onnx("model.onnx")

    # Custom config
    result = quantize_onnx("model.onnx", WinMLQuantizationConfig(samples=100))
"""

from typing import TYPE_CHECKING, Any

from .config import QuantizeResult, WinMLQuantizationConfig


__all__ = [
    "QuantizeResult",
    "WinMLQuantizationConfig",
    "get_quant_finalizer",
    "quantize_onnx",
    "register_quant_finalizer",
]


# Names below are loaded lazily via ``__getattr__`` to avoid pulling in
# onnxruntime.quantization/torch at import time. The TYPE_CHECKING re-imports
# give static analyzers (mypy, CodeQL) visibility into what ``__all__`` exports
# without triggering the heavy imports at runtime.
if TYPE_CHECKING:
    from .calibration import get_quant_finalizer, register_quant_finalizer
    from .quantizer import quantize_onnx


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "quantize_onnx": (".quantizer", "quantize_onnx"),
    "get_quant_finalizer": (".calibration", "get_quant_finalizer"),
    "register_quant_finalizer": (".calibration", "register_quant_finalizer"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load quantizer (imports onnxruntime.quantization)."""
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
