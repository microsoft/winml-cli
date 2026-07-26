# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Deprecated compatibility shim for op-tracing result dataclasses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..session.monitor.op_metrics import OperatorMetrics as _OperatorMetrics
from ..session.monitor.op_metrics import OpTraceResult as _OpTraceResult
from ._compat import warn_deprecated


if TYPE_CHECKING:
    OperatorMetrics = _OperatorMetrics
    OpTraceResult = _OpTraceResult


def __getattr__(name: str) -> Any:
    if name == "OperatorMetrics":
        warn_deprecated(
            "result.OperatorMetrics",
            "winml.modelkit.session.monitor.OperatorMetrics",
            stacklevel=2,
        )
        return _OperatorMetrics
    if name == "OpTraceResult":
        warn_deprecated(
            "result.OpTraceResult",
            "winml.modelkit.session.monitor.OpTraceResult",
            stacklevel=2,
        )
        return _OpTraceResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = ["OpTraceResult", "OperatorMetrics"]
