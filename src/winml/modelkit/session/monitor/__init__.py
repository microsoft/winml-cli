# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Lazy public facade for EP monitors and op-tracing reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .ep_monitor import EPMonitor, NullEPMonitor, WinMLEPMonitor
    from .op_metrics import OperatorMetrics, OpTraceResult
    from .openvino_monitor import OpenVinoMonitor
    from .report import display_op_trace_report, write_op_trace_json


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "EPMonitor": (".ep_monitor", "EPMonitor"),
    "NullEPMonitor": (".ep_monitor", "NullEPMonitor"),
    "WinMLEPMonitor": (".ep_monitor", "WinMLEPMonitor"),
    "OperatorMetrics": (".op_metrics", "OperatorMetrics"),
    "OpTraceResult": (".op_metrics", "OpTraceResult"),
    "OpenVinoMonitor": (".openvino_monitor", "OpenVinoMonitor"),
    "display_op_trace_report": (".report", "display_op_trace_report"),
    "write_op_trace_json": (".report", "write_op_trace_json"),
}

__all__ = [
    "EPMonitor",
    "NullEPMonitor",
    "OpTraceResult",
    "OpenVinoMonitor",
    "OperatorMetrics",
    "WinMLEPMonitor",
    "display_op_trace_report",
    "write_op_trace_json",
]


def __getattr__(name: str) -> Any:
    """Load one public monitor symbol without importing other monitors."""
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module_path, attr_name = target
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public symbols without resolving them."""
    return sorted(set(globals()) | set(__all__))
