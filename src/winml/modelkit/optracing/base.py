# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""EP-agnostic operator profiling interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:

    from .result import OpTraceResult


class OpTracer(ABC):
    """EP-agnostic operator profiling interface.

    Subclasses implement tracing logic for a specific execution provider
    (e.g. QNN, DirectML, CUDA).

    Concrete implementations receive the model path and output directory
    at construction time, then call ``run()`` to execute profiling.
    """

    @abstractmethod
    def run(self, iterations: int = 5, warmup: int = 2) -> OpTraceResult:
        """Run operator-level tracing and return structured results."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this tracer's runtime dependencies are available."""
        ...
