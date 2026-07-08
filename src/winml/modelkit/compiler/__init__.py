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

from typing import TYPE_CHECKING, Any

from .configs import (
    EPConfig,
    WinMLCompileConfig,
)
from .context import CompileContext
from .result import CompileResult
from .transforms import clear_transforms, get_transforms_for_ep, register_transform
from .utils import (
    QDQ_OP_TYPES,
    QOPERATOR_OP_TYPES,
    QUANTIZATION_OP_TYPES,
    needs_format_conversion,
)


# Names below are loaded lazily via ``__getattr__`` to avoid pulling in session/
# torch at import time. The TYPE_CHECKING re-imports give static analyzers
# (mypy, CodeQL) visibility into what ``__all__`` actually exports without
# triggering the heavy imports at runtime.
if TYPE_CHECKING:
    from .compiler import Compiler, compile_multiple_onnx, compile_onnx, list_compilers
    from .stages.compile import CompileStage
    from .stages.optimize import OptimizeStage
    from .stages.qformat import QFormatConvertStage


def __getattr__(name: str) -> Any:
    """Lazy-load heavy symbols that pull in session/torch to speed up import."""
    if name in {"Compiler", "compile_multiple_onnx", "compile_onnx", "list_compilers"}:
        from .compiler import (
            Compiler,
            compile_multiple_onnx,
            compile_onnx,
            list_compilers,
        )

        globals().update(
            Compiler=Compiler,
            compile_multiple_onnx=compile_multiple_onnx,
            compile_onnx=compile_onnx,
            list_compilers=list_compilers,
        )
        return globals()[name]

    _stage_names = {"CompileStage", "OptimizeStage", "QFormatConvertStage"}
    if name in _stage_names:
        from .stages.compile import CompileStage
        from .stages.optimize import OptimizeStage
        from .stages.qformat import QFormatConvertStage

        globals().update(
            CompileStage=CompileStage,
            OptimizeStage=OptimizeStage,
            QFormatConvertStage=QFormatConvertStage,
        )
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "QDQ_OP_TYPES",
    "QOPERATOR_OP_TYPES",
    "QUANTIZATION_OP_TYPES",
    "CompileContext",
    "CompileResult",
    "CompileStage",
    "Compiler",
    "EPConfig",
    "OptimizeStage",
    "QFormatConvertStage",
    "WinMLCompileConfig",
    "clear_transforms",
    "compile_multiple_onnx",
    "compile_onnx",
    "get_transforms_for_ep",
    "list_compilers",
    "needs_format_conversion",
    "register_transform",
]

__version__ = "0.1.0"
