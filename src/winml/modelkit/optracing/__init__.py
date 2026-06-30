# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Operator-level profiling for WinML CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import OpTracer
from .registry import get_tracer, register_tracer
from .report import display_op_trace_report, write_op_trace_json
from .result import OperatorMetrics, OpTraceResult


if TYPE_CHECKING:
    from ..utils.constants import EPNameOrAlias

# The single EP / device / tracing-level combination op-tracing currently
# supports. Expanded as more tracers land.
_SUPPORTED_EP = "QNNExecutionProvider"
_SUPPORTED_DEVICE = "npu"
_SUPPORTED_LEVEL = "basic"


def is_profiling_available(
    resolved_ep: EPNameOrAlias | None,
    resolved_device: str | None,
    op_tracing: str | None,
) -> bool:
    """Check whether op-tracing is supported for a resolved EP/device/level.

    Op-tracing is currently limited to the QNN EP on NPU at the ``"basic"``
    level; every other combination is unsupported.

    Args:
        resolved_ep: Concrete EP the benchmark resolved to (full name or alias).
        resolved_device: Concrete device the benchmark resolved to (e.g. ``"npu"``).
        op_tracing: Requested tracing level (e.g. ``"basic"``), or ``None``.

    Returns:
        ``True`` only for the QNN + NPU + ``"basic"`` combination.
    """
    from ..utils.constants import normalize_ep_name

    return (
        normalize_ep_name(resolved_ep) == _SUPPORTED_EP
        and (resolved_device or "").lower() == _SUPPORTED_DEVICE
        and (op_tracing or "").lower() == _SUPPORTED_LEVEL
    )


__all__ = [
    "OpTraceResult",
    "OpTracer",
    "OperatorMetrics",
    "display_op_trace_report",
    "get_tracer",
    "is_profiling_available",
    "register_tracer",
    "write_op_trace_json",
]
