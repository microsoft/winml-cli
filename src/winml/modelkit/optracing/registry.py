# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Deprecated compatibility shim for the legacy op-tracer registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._compat import warn_deprecated


if TYPE_CHECKING:
    from ..utils.constants import EPName
    from .base import OpTracer


_TRACERS: dict[str, dict[str, type[OpTracer]]] = {}


def _ensure_defaults() -> None:
    qnn_levels = _TRACERS.setdefault("QNN", {})
    if "basic" in qnn_levels and "detail" in qnn_levels:
        return

    from .qnn.profiler import _QNNProfiler

    qnn_levels.setdefault("basic", _QNNProfiler)
    qnn_levels.setdefault("detail", _QNNProfiler)


def _register_tracer(ep_pattern: str, level: str, tracer_class: type[OpTracer]) -> None:
    _TRACERS.setdefault(ep_pattern, {})[level] = tracer_class


def _get_tracer(ep_name: EPName | str, level: str) -> type[OpTracer] | None:
    _ensure_defaults()
    for pattern, levels in _TRACERS.items():
        if pattern in ep_name and level in levels:
            return levels[level]
    return None


if TYPE_CHECKING:
    get_tracer = _get_tracer
    register_tracer = _register_tracer


def __getattr__(name: str) -> Any:
    if name == "register_tracer":
        warn_deprecated(
            "registry.register_tracer",
            "winml.modelkit.session.monitor.QNNMonitor",
            stacklevel=2,
        )
        return _register_tracer
    if name == "get_tracer":
        warn_deprecated(
            "registry.get_tracer",
            "winml.modelkit.session.monitor.QNNMonitor",
            stacklevel=2,
        )
        return _get_tracer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = ["get_tracer", "register_tracer"]
