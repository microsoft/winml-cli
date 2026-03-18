# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX Compiler Module.

This module provides tools for compiling ONNX models to EP-specific formats.

Quantization concerns (QDQ, calibration) are handled separately by
WinMLQuantizationConfig in modelkit.quant.config.

Core Loop:
    [model.onnx] -> [compile] -> [model_ctx.onnx]

Usage:
    from winml.modelkit.compiler import compile_onnx, WinMLCompileConfig

    # Default: QNN compilation
    result = compile_onnx("model.onnx")

    # Custom config
    config = WinMLCompileConfig.for_qnn()
    config.ep_config.provider_options["htp_performance_mode"] = "default"
    result = compile_onnx("model.onnx", config)
"""

from .compiler import Compiler, compile_onnx, list_compilers
from .configs import (
    EPConfig,
    WinMLCompileConfig,
)
from .context import CompileContext
from .result import CompileResult
from .utils import QDQ_OP_TYPES


__all__ = [
    "QDQ_OP_TYPES",
    "CompileContext",
    "CompileResult",
    "Compiler",
    "EPConfig",
    "WinMLCompileConfig",
    "compile_onnx",
    "list_compilers",
]

__version__ = "0.1.0"
