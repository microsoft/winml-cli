# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""QNN optracing compatibility helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._compat import warn_deprecated


if TYPE_CHECKING:
    from .profiler import QNNProfiler


def __getattr__(name: str) -> Any:
    if name != "QNNProfiler":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .profiler import _QNNProfiler

    warn_deprecated(
        "qnn.QNNProfiler",
        "winml.modelkit.session.monitor.QNNMonitor",
        stacklevel=2,
    )
    globals()[name] = _QNNProfiler
    return _QNNProfiler


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = ["QNNProfiler"]
