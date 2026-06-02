# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""EP-agnostic operator profiling interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .result import OpTraceResult


class OpTracer(ABC):
    """EP-agnostic operator profiling interface.

    Subclasses implement tracing logic for a specific execution provider
    (e.g. QNN, Dml, CUDA).

    Concrete implementations receive the model path and output directory
    at construction time, then call ``run()`` to execute profiling.

    Subclasses overriding ``__init__`` MUST call ``super().__init__(...)`` so
    that ``onnx_path``, ``output_dir``, and ``level`` are stored on ``self``.
    """

    def __init__(self, onnx_path: Path, *, output_dir: Path, level: str = "basic") -> None:
        """Construct an OpTracer for an ONNX model.

        Args:
            onnx_path: Path to the ONNX model to trace.
            output_dir: Directory for profiling artifacts.
            level: Profiling level ("basic" or "detail").
        """
        self.onnx_path = Path(onnx_path)
        self.output_dir = Path(output_dir)
        self.level = level

    @abstractmethod
    def run(self, iterations: int = 5, warmup: int = 2) -> OpTraceResult:
        """Run operator-level tracing and return structured results."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this tracer's runtime dependencies are available."""
