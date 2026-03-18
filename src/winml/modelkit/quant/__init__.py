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
from .quantizer import quantize_onnx


__all__ = [
    "QuantizeResult",
    "WinMLQuantizationConfig",
    "quantize_onnx",
]
