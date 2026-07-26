# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Deprecated compatibility shim for the legacy op-tracer base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._compat import warn_deprecated


if TYPE_CHECKING:
    import numpy as np

    from .result import OpTraceResult


class OpTracer(ABC):
    """EP-agnostic operator profiling interface."""

    def __init__(
        self,
        onnx_path: Path,
        *,
        output_dir: Path,
        level: str = "basic",
        input_data: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.output_dir = Path(output_dir)
        self.level = level
        self.input_data = input_data

    @abstractmethod
    def run(self, iterations: int = 5, warmup: int = 2) -> OpTraceResult:
        """Run operator-level tracing and return structured results."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this tracer's runtime dependencies are available."""


def __getattr__(name: str) -> Any:
    if name != "OpTracer":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warn_deprecated("base.OpTracer", "winml.modelkit.session.monitor.QNNMonitor", stacklevel=2)
    return _OpTracer


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = ["OpTracer"]


_OpTracer = OpTracer
del OpTracer

if TYPE_CHECKING:
    OpTracer = _OpTracer  # type: ignore[misc]  # Static-only compatibility export.
