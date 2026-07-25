# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Deprecated compatibility surface for operator tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._compat import warn_deprecated


if TYPE_CHECKING:
    from ..utils.constants import EPNameOrAlias
    from .base import OpTracer
    from .report import display_op_trace_report, write_op_trace_json
    from .result import OperatorMetrics, OpTraceResult


_SUPPORTED_EP = "QNNExecutionProvider"
_SUPPORTED_DEVICE = "npu"
_SUPPORTED_LEVEL = "basic"


def is_profiling_available(
    resolved_ep: EPNameOrAlias | None,
    resolved_device: str | None,
    op_tracing: str | None,
) -> bool:
    """Check whether the legacy optracing entry point would have been available."""
    warn_deprecated(
        "is_profiling_available",
        "winml.modelkit.session.monitor.QNNMonitor",
        stacklevel=2,
    )
    from ..utils.constants import normalize_ep_name

    return (
        normalize_ep_name(resolved_ep) == _SUPPORTED_EP
        and (resolved_device or "").lower() == _SUPPORTED_DEVICE
        and (op_tracing or "").lower() == _SUPPORTED_LEVEL
    )


def __getattr__(name: str) -> Any:
    if name == "OpTraceResult":
        from ..session.monitor import OpTraceResult

        warn_deprecated(name, "winml.modelkit.session.monitor.OpTraceResult", stacklevel=2)
        globals()[name] = OpTraceResult
        return OpTraceResult
    if name == "OperatorMetrics":
        from ..session.monitor import OperatorMetrics

        warn_deprecated(name, "winml.modelkit.session.monitor.OperatorMetrics", stacklevel=2)
        globals()[name] = OperatorMetrics
        return OperatorMetrics
    if name == "display_op_trace_report":
        from .report import _display_op_trace_report_compat

        warn_deprecated(
            name,
            "winml.modelkit.session.monitor.display_op_trace_report",
            stacklevel=2,
        )
        globals()[name] = _display_op_trace_report_compat
        return _display_op_trace_report_compat
    if name == "write_op_trace_json":
        from ..session.monitor import write_op_trace_json

        warn_deprecated(
            name,
            "winml.modelkit.session.monitor.write_op_trace_json",
            stacklevel=2,
        )
        globals()[name] = write_op_trace_json
        return write_op_trace_json
    if name == "OpTracer":
        from .base import _OpTracer

        warn_deprecated(name, "winml.modelkit.session.monitor.QNNMonitor", stacklevel=2)
        globals()[name] = _OpTracer
        return _OpTracer
    if name == "get_tracer":
        from .registry import _get_tracer

        warn_deprecated(name, "winml.modelkit.session.monitor.QNNMonitor", stacklevel=2)
        globals()[name] = _get_tracer
        return _get_tracer
    if name == "register_tracer":
        from .registry import _register_tracer

        warn_deprecated(name, "winml.modelkit.session.monitor.QNNMonitor", stacklevel=2)
        globals()[name] = _register_tracer
        return _register_tracer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


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
