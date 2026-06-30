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

# The EP / device / tracing-level combinations op-tracing currently supports,
# as ``(canonical_ep, device, level)`` tuples. Expanded as more tracers land.
_SUPPORTED_COMBINATIONS: set[tuple[str, str, str]] = {
    ("QNNExecutionProvider", "npu", "basic"),
    ("CPUExecutionProvider", "cpu", "basic"),
}


def is_profiling_available(
    resolved_ep: EPNameOrAlias | None,
    resolved_device: str | None,
    op_tracing: str | None,
) -> bool:
    """Check whether op-tracing is supported for a resolved EP/device/level.

    Op-tracing currently supports the QNN EP on NPU and the CPU EP on CPU, both
    at the ``"basic"`` level; every other combination is unsupported.

    Args:
        resolved_ep: Concrete EP the benchmark resolved to (full name or alias).
        resolved_device: Concrete device the benchmark resolved to (e.g. ``"npu"``).
        op_tracing: Requested tracing level (e.g. ``"basic"``), or ``None``.

    Returns:
        ``True`` only for a supported EP/device/level combination.
    """
    from ..utils.constants import normalize_ep_name

    ep = normalize_ep_name(resolved_ep)
    if ep is None:
        return False
    return (
        ep,
        (resolved_device or "").lower(),
        (op_tracing or "").lower(),
    ) in _SUPPORTED_COMBINATIONS


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
