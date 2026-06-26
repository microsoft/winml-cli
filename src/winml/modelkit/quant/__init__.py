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

    # Pipeline: RTN int4 followed by FP16 (w4a16)
    config = WinMLQuantizationConfig(mode="w4a16", rtn_bits=4)
    result = Quantizer(expand_precision("w4a16", config)).run("model.onnx", "out.onnx")
"""

from typing import TYPE_CHECKING, Any

from .config import QuantizeResult, WinMLQuantizationConfig
from .passes import BaseQuantPass, FP16Pass, QDQPass, RTNPass


if TYPE_CHECKING:
    from .quantizer import Quantizer, expand_precision, quantize_onnx


__all__ = [
    "BaseQuantPass",
    "FP16Pass",
    "QDQPass",
    "QuantizeResult",
    "Quantizer",
    "RTNPass",
    "WinMLQuantizationConfig",
    "expand_precision",
    "get_quant_finalizer",
    "quantize_onnx",
]


# Names below are loaded lazily via ``__getattr__`` to avoid pulling in
# onnxruntime.quantization/torch at import time. The TYPE_CHECKING re-imports
# give static analyzers (mypy, CodeQL) visibility into what ``__all__`` exports
# without triggering the heavy imports at runtime.
if TYPE_CHECKING:
    from .calibration import get_quant_finalizer
    from .quantizer import Quantizer, expand_precision, quantize_onnx


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "quantize_onnx": (".quantizer", "quantize_onnx"),
    "Quantizer": (".quantizer", "Quantizer"),
    "expand_precision": (".quantizer", "expand_precision"),
    "get_quant_finalizer": (".calibration", "get_quant_finalizer"),
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
