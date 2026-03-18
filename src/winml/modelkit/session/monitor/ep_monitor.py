# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""EPMonitor - Abstract base class for EP-specific hardware monitoring.

Defines the common interface that all EP hardware monitors implement.
Each subclass provides data collection for a specific execution provider
(VitisAI, MIGraphX, TensorRT, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from typing_extensions import Self


class EPMonitor(ABC):
    """Base class for EP-specific hardware performance monitoring.

    Used as a context manager alongside ``PerfStats`` to collect
    hardware utilization metrics during inference.

    Example::

        with session.perf(warmup=10) as stats:
            with SomeEPMonitor() as hw:
                for _ in range(110):
                    session.run(inputs)

        print(stats.mean_ms)  # inference timing
        print(hw.to_dict())  # proof-of-execution data
    """

    @abstractmethod
    def __enter__(self) -> Self:
        """Start hardware monitoring."""

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Stop hardware monitoring and finalize metrics."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary of all collected metrics."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Whether this monitor can work on the current system."""


class NullEPMonitor(EPMonitor):
    """No-op EP monitor (Null Object Pattern).

    Used when no vendor-specific EP monitor is available.
    Eliminates null checks in the benchmark loop.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        pass

    @classmethod
    def is_available(cls) -> bool:
        """Always available (it does nothing)."""
        return True

    def to_dict(self) -> dict[str, Any]:
        """No-op: returns empty dict."""
        return {}
