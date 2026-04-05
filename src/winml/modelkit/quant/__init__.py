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

from .config import QuantizeResult, WinMLQuantizationConfig


__all__ = [
    "QuantizeResult",
    "WinMLQuantizationConfig",
    "quantize_onnx",
]


def __getattr__(name: str):
    """Lazy-load quantizer (imports onnxruntime.quantization)."""
    if name == "quantize_onnx":
        from .quantizer import quantize_onnx

        globals()["quantize_onnx"] = quantize_onnx
        return quantize_onnx

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
