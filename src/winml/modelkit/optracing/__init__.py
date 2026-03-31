# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Operator-level profiling for ModelKit."""

from __future__ import annotations

from .base import OpTracer
from .registry import get_tracer, register_tracer
from .report import display_op_trace_report, write_op_trace_json
from .result import OperatorMetrics, OpTraceResult


def is_qnn_profiling_available() -> bool:
    """Check if QNN EP is available for op-tracing."""
    try:
        import onnxruntime as ort

        return "QNNExecutionProvider" in ort.get_available_providers()
    except (ImportError, AttributeError):
        return False


__all__ = [
    "OpTraceResult",
    "OpTracer",
    "OperatorMetrics",
    "display_op_trace_report",
    "get_tracer",
    "is_qnn_profiling_available",
    "register_tracer",
    "write_op_trace_json",
]
