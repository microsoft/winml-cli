# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Dependency-light statistics and memory checkpoints for generation perf.

Shared helpers must not import a generation command or session backend: GGUF
uses Task API and does not need the Python ORT/GenAI runtime to report metrics.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass


# Keep the existing default prompt and percentile/JSON semantics for both routes.
_DEFAULT_PROMPT = "Explain the theory of relativity in simple terms."


def _mean(xs: list[float]) -> float:
    """Arithmetic mean, or ``0.0`` for an empty sequence."""
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(sorted_xs: list[float], p: float) -> float:
    """Nearest-rank ``p``-th percentile (0-100) of an already-sorted list.

    Matches :meth:`winml.modelkit.session.stats.PerfStats.percentile` so the
    two perf paths report percentiles the same way.
    """
    if not sorted_xs:
        return 0.0
    idx = int(len(sorted_xs) * p / 100)
    idx = min(idx, len(sorted_xs) - 1)
    return sorted_xs[idx]


def _stats(values: list[float]) -> dict[str, float]:
    """Return the percentile summary shape used by perf JSON blocks."""
    sorted_values = sorted(values)
    mean = _mean(values)
    variance = _mean([(value - mean) ** 2 for value in values]) if values else 0.0
    return {
        "mean": mean,
        "std": variance**0.5,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "p50": _percentile(sorted_values, 50),
        "p90": _percentile(sorted_values, 90),
        "p95": _percentile(sorted_values, 95),
        "p99": _percentile(sorted_values, 99),
    }


def _round_stats(values: dict[str, float]) -> dict[str, float]:
    """Round a stats block for JSON output."""
    return {key: round(value, 3) for key, value in values.items()}


def _get_rss_mb() -> float:
    """Return current process RSS in MB."""
    from ..session.monitor import get_rss_mb

    return get_rss_mb()


def _get_vram_mb(adapter_luid: str | None) -> tuple[float | None, float | None]:
    """Return current process device-memory usage as local/shared MB."""
    from ..session.monitor import get_vram_mb

    return get_vram_mb(adapter_luid)


@dataclass
class _MemorySnapshot:
    """Point-in-time process RAM and device-memory usage."""

    rss_mb: float = 0.0
    vram_local_mb: float | None = None
    vram_shared_mb: float | None = None


class _GenerationMemoryTracker:
    """Best-effort process memory checkpoints shared by generation benchmarks."""

    def __init__(self, *, adapter_luid: str | None) -> None:
        self._adapter_luid = adapter_luid
        self._baseline = _MemorySnapshot()
        self._after_load = _MemorySnapshot()
        self._after_benchmark = _MemorySnapshot()

    def _snapshot(self) -> _MemorySnapshot:
        gc.collect()
        rss = _get_rss_mb()
        local, shared = _get_vram_mb(self._adapter_luid) if self._adapter_luid else (None, None)
        return _MemorySnapshot(rss_mb=rss, vram_local_mb=local, vram_shared_mb=shared)

    def record_baseline(self) -> None:
        self._baseline = self._snapshot()

    def record_after_load(self) -> None:
        self._after_load = self._snapshot()

    def record_after_benchmark(self) -> None:
        self._after_benchmark = self._snapshot()

    @staticmethod
    def _delta(after: float | None, before: float | None) -> float | None:
        return round(after - before, 2) if after is not None and before is not None else None

    def to_dict(self) -> dict[str, float | None]:
        baseline = self._baseline
        after_load = self._after_load
        after_benchmark = self._after_benchmark
        result = {
            "rss_baseline_mb": round(baseline.rss_mb, 2),
            "rss_after_compile_mb": round(after_load.rss_mb, 2),
            "rss_after_inference_mb": round(after_benchmark.rss_mb, 2),
            "rss_checkpoint_peak_mb": round(
                max(baseline.rss_mb, after_load.rss_mb, after_benchmark.rss_mb), 2
            ),
            "rss_model_load_delta_mb": self._delta(after_load.rss_mb, baseline.rss_mb),
            "rss_inference_delta_mb": self._delta(after_benchmark.rss_mb, after_load.rss_mb),
            "rss_total_delta_mb": self._delta(after_benchmark.rss_mb, baseline.rss_mb),
        }
        if self._adapter_luid is None:
            return result
        for key in ("vram_local", "vram_shared"):
            points = [
                getattr(snapshot, key + "_mb")
                for snapshot in (baseline, after_load, after_benchmark)
            ]
            for phase, value in zip(
                ("baseline", "after_compile", "after_inference"), points, strict=True
            ):
                result[f"{key}_{phase}_mb"] = round(value, 2) if value is not None else None
            result[f"{key}_checkpoint_peak_mb"] = (
                round(max(points), 2) if all(value is not None for value in points) else None
            )
            for phase, start, end in (("model_load", 0, 1), ("inference", 1, 2), ("total", 0, 2)):
                result[f"{key}_{phase}_delta_mb"] = self._delta(points[end], points[start])
        return result
