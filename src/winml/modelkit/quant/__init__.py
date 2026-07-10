# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Quantizer module for ONNX models.

Provides QDQ (Quantize-Dequantize) quantization for ONNX models.

Usage:
    from winml.modelkit.quant import (
        quantize_onnx,
        Quantizer,
        expand_precision,
        WinMLQuantizationConfig,
    )

    # Quick quantize with defaults (10 samples, uint8)
    result = quantize_onnx("model.onnx")

    # Custom config
    result = quantize_onnx("model.onnx", WinMLQuantizationConfig(samples=100))

    # Pipeline: RTN int4 weight-only
    config = WinMLQuantizationConfig(mode="rtn", rtn_bits=4)
    result = Quantizer(expand_precision("rtn", config)).run("model.onnx", "out.onnx")
"""

from typing import TYPE_CHECKING, Any

from .calibration import get_quant_finalizer
from .config import QuantizeResult, WinMLQuantizationConfig
from .passes import BaseQuantPass, DynamicPass, FP16Pass, RTNPass, StaticPass


if TYPE_CHECKING:
    from .quantizer import Quantizer, expand_precision, quantize_onnx


__all__ = [
    "BaseQuantPass",
    "DynamicPass",
    "FP16Pass",
    "QuantizeResult",
    "Quantizer",
    "RTNPass",
    "StaticPass",
    "WinMLQuantizationConfig",
    "expand_precision",
    "get_quant_finalizer",
    "quantize_onnx",
]


# ``quantize_onnx`` is loaded lazily via ``__getattr__`` to avoid pulling in
# onnxruntime.quantization at import time. The TYPE_CHECKING re-import gives
# static analyzers (mypy, CodeQL) visibility into what ``__all__`` exports.
# ``get_quant_finalizer`` is imported directly above — its module chain
# (calibration/__init__ -> registry) is lightweight and safe at import time.
if TYPE_CHECKING:
    from .calibration import get_quant_finalizer
    from .quantizer import Quantizer, expand_precision, quantize_onnx


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "quantize_onnx": (".quantizer", "quantize_onnx"),
    "Quantizer": (".quantizer", "Quantizer"),
    "expand_precision": (".quantizer", "expand_precision"),
}


def __getattr__(name: str) -> Any:
    """Lazy-load quantizer module (avoids importing onnxruntime at package import time)."""
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
