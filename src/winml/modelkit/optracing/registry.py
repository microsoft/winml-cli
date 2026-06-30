# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""EP tracer registry with substring-based pattern matching.

Tracers register themselves against an EP *pattern* (e.g. ``"QNN"``) and
a profiling *level* (``"basic"`` or ``"detail"``).  Lookup uses substring
matching so that ``"QNN"`` matches ``"QNNExecutionProvider"`` without
hardcoding full EP names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..utils.constants import EPName
    from .base import OpTracer

# {ep_pattern: {level: tracer_class}}
_TRACERS: dict[str, dict[str, type[OpTracer]]] = {}


def register_tracer(ep_pattern: str, level: str, tracer_class: type[OpTracer]) -> None:
    """Register a tracer class for an EP pattern and profiling level.

    Parameters
    ----------
    ep_pattern:
        Substring that will be matched against EP names (e.g. ``"QNN"``).
    level:
        Profiling level identifier (e.g. ``"basic"``, ``"detail"``).
    tracer_class:
        The ``OpTracer`` subclass to register.
    """
    _TRACERS.setdefault(ep_pattern, {})[level] = tracer_class


def get_tracer(ep_name: EPName, level: str) -> type[OpTracer] | None:
    """Look up a tracer class by EP name and level.

    Uses substring matching: a registered pattern ``"QNN"`` will match
    any *ep_name* that contains ``"QNN"`` (e.g. ``"QNNExecutionProvider"``).

    Returns ``None`` when no matching tracer is found.
    """
    for pattern, levels in _TRACERS.items():
        if pattern in ep_name and level in levels:
            return levels[level]
    return None


def _register_defaults() -> None:
    """Auto-register built-in tracers."""
    from .cpu.profiler import CPUProfiler
    from .qnn.profiler import QNNProfiler

    register_tracer("QNN", "basic", QNNProfiler)
    register_tracer("QNN", "detail", QNNProfiler)
    register_tracer("CPU", "basic", CPUProfiler)


# Eagerly register defaults on import.
_register_defaults()
