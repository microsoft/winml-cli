# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Performance tracking for WinMLSession."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


@dataclass
class PerfStats:
    """Performance statistics with timing capture and warmup exclusion.

    Collects individual timing samples and provides computed statistics
    including mean, min, max, and percentiles. Supports warmup exclusion
    to ignore initial samples from calculations.

    Example:
        >>> perf = PerfStats(warmup=5)
        >>> for _ in range(15):
        ...     result = perf.record(lambda: expensive_operation())
        >>> print(f"Mean: {perf.mean_ms:.2f} ms")  # Based on last 10 samples
    """

    _samples: list[float] = field(default_factory=list)
    warmup: int = 0  # Exclude first N samples from calculations

    def record(self, func: Callable[[], T]) -> T:
        """Execute func, record timing, return result unchanged.

        Args:
            func: Zero-argument callable to execute and time.
                  Use lambda to wrap functions with arguments.

        Returns:
            The return value of func, unchanged.

        Example:
            >>> result = perf.record(lambda: session.run(names, inputs))
        """
        start = time.perf_counter()
        result = func()
        self._samples.append((time.perf_counter() - start) * 1000)
        return result

    @property
    def _effective_samples(self) -> list[float]:
        """Samples after warmup exclusion."""
        return self._samples[self.warmup :]

    @property
    def samples_ms(self) -> list[float]:
        """Timing samples in milliseconds (after warmup, read-only copy)."""
        return self._effective_samples.copy()

    @property
    def all_samples_ms(self) -> list[float]:
        """All timing samples including warmup (read-only copy)."""
        return self._samples.copy()

    @property
    def count(self) -> int:
        """Number of samples (after warmup exclusion)."""
        return len(self._effective_samples)

    @property
    def total_count(self) -> int:
        """Total number of samples including warmup."""
        return len(self._samples)

    @property
    def total_ms(self) -> float:
        """Total time across all runs (after warmup)."""
        return sum(self._effective_samples)

    @property
    def mean_ms(self) -> float:
        """Mean time per run (after warmup)."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def min_ms(self) -> float:
        """Minimum run time (after warmup)."""
        return min(self._effective_samples) if self._effective_samples else 0.0

    @property
    def max_ms(self) -> float:
        """Maximum run time (after warmup)."""
        return max(self._effective_samples) if self._effective_samples else 0.0

    def percentile(self, p: float) -> float:
        """Get p-th percentile (0-100), after warmup exclusion.

        Args:
            p: Percentile value between 0 and 100.

        Returns:
            The p-th percentile value, or 0.0 if no samples.
        """
        samples = self._effective_samples
        if not samples:
            return 0.0
        sorted_samples = sorted(samples)
        idx = int(len(sorted_samples) * p / 100)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]

    @property
    def p50_ms(self) -> float:
        """Median (50th percentile)."""
        return self.percentile(50)

    @property
    def p90_ms(self) -> float:
        """90th percentile."""
        return self.percentile(90)

    @property
    def p95_ms(self) -> float:
        """95th percentile."""
        return self.percentile(95)

    @property
    def p99_ms(self) -> float:
        """99th percentile."""
        return self.percentile(99)
