# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Deprecated compatibility shim for op-tracing report helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..session.monitor.report import display_op_trace_report as _display_op_trace_report
from ..session.monitor.report import write_op_trace_json as _write_op_trace_json
from ._compat import warn_deprecated


if TYPE_CHECKING:
    from rich.console import Console

    from .result import OpTraceResult


def _display_op_trace_report_compat(
    result: OpTraceResult,
    console: Console | None = None,
    top_n: int = 15,
) -> None:
    _display_op_trace_report(result, console=console, top_n=top_n)


_display_op_trace_report_compat.__name__ = "display_op_trace_report"
_display_op_trace_report_compat.__qualname__ = "display_op_trace_report"
_display_op_trace_report_compat.__doc__ = _display_op_trace_report.__doc__


if TYPE_CHECKING:
    display_op_trace_report = _display_op_trace_report_compat
    write_op_trace_json = _write_op_trace_json


def __getattr__(name: str) -> Any:
    if name == "display_op_trace_report":
        warn_deprecated(
            "report.display_op_trace_report",
            "winml.modelkit.session.monitor.display_op_trace_report",
            stacklevel=2,
        )
        return _display_op_trace_report_compat
    if name == "write_op_trace_json":
        warn_deprecated(
            "report.write_op_trace_json",
            "winml.modelkit.session.monitor.write_op_trace_json",
            stacklevel=2,
        )
        return _write_op_trace_json
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = ["display_op_trace_report", "write_op_trace_json"]
