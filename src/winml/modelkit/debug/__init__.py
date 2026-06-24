# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Per-op quantization error measurement for ONNX models.

Usage:
    from winml.modelkit.debug import debug_quantization

    errors = debug_quantization("float.onnx", "quantized.onnx")
"""

from .debugger import debug_quantization


__all__ = [
    "debug_quantization",
]
