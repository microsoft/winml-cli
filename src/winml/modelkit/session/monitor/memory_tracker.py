# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Process memory tracking for perf benchmarking.

Measures RSS at benchmark phase boundaries to compute memory deltas for
model loading and inference.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

import psutil


logger = logging.getLogger(__name__)

_MB = 1024 * 1024


def _get_rss_mb() -> float:
    """Return current RSS in MB for this process."""
    return psutil.Process(os.getpid()).memory_info().rss / _MB


def _get_device_memory_mb(luid: str | None) -> float:
    """Single-shot PDH query for device local memory in MB."""
    if luid is None or sys.platform != "win32":
        return 0.0

    try:
        from ._pdh import PdhQuery

        pid = os.getpid()
        query = PdhQuery()
        query.open()

        ok = query.add_counter(
            "local",
            rf"\GPU Process Memory(pid_{pid}_luid_{luid}_phys_0)\Local Usage",
            fmt="large",
        )
        if not ok:
            query.close()
            return 0.0

        query.prime()
        values = query.collect()
        query.close()
        return (values.get("local") or 0) / _MB
    except Exception:
        logger.debug("Device memory query failed", exc_info=True)
        return 0.0


@dataclass
class MemoryProfile:
    """Memory measurements across benchmark phases."""

    rss_baseline_mb: float
    rss_after_compile_mb: float
    rss_after_inference_mb: float
    device_local_mb: float = 0.0

    @property
    def model_load_delta_mb(self) -> float:
        """RSS increase from model loading + compilation."""
        return self.rss_after_compile_mb - self.rss_baseline_mb

    @property
    def inference_alloc_delta_mb(self) -> float:
        """RSS increase from inference (warmup + benchmark)."""
        return self.rss_after_inference_mb - self.rss_after_compile_mb

    @property
    def total_delta_mb(self) -> float:
        """Total RSS increase from baseline."""
        return self.rss_after_inference_mb - self.rss_baseline_mb

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dictionary."""
        return {
            "rss_baseline_mb": round(self.rss_baseline_mb, 2),
            "rss_after_compile_mb": round(self.rss_after_compile_mb, 2),
            "rss_after_inference_mb": round(self.rss_after_inference_mb, 2),
            "model_load_delta_mb": round(self.model_load_delta_mb, 2),
            "inference_alloc_delta_mb": round(self.inference_alloc_delta_mb, 2),
            "total_delta_mb": round(self.total_delta_mb, 2),
            "device_local_mb": round(self.device_local_mb, 2),
        }


class MemoryTracker:
    """Lightweight memory tracker that takes RSS snapshots at phase boundaries.

    Usage::

        tracker = MemoryTracker()
        tracker.snapshot_baseline()
        # ... load model + compile ...
        tracker.snapshot_post_compile(adapter_luid="0x...")
        # ... run benchmark ...
        tracker.snapshot_post_inference(adapter_luid="0x...")
        profile = tracker.profile()
    """

    def __init__(self) -> None:
        self._baseline: float | None = None
        self._post_compile: float | None = None
        self._post_inference: float | None = None
        self._device_local_mb: float = 0.0

    def snapshot_baseline(self) -> None:
        """Capture baseline RSS (call after EP warmup)."""
        self._baseline = _get_rss_mb()

    def snapshot_post_compile(self, adapter_luid: str | None = None) -> None:
        """Capture RSS after model load + compile."""
        self._post_compile = _get_rss_mb()
        self._device_local_mb = max(self._device_local_mb, _get_device_memory_mb(adapter_luid))

    def snapshot_post_inference(self, adapter_luid: str | None = None) -> None:
        """Capture RSS after inference."""
        self._post_inference = _get_rss_mb()
        self._device_local_mb = max(self._device_local_mb, _get_device_memory_mb(adapter_luid))

    def profile(self) -> MemoryProfile | None:
        """Build MemoryProfile. Returns None if any snapshot is missing."""
        if any(s is None for s in (self._baseline, self._post_compile, self._post_inference)):
            return None

        assert self._baseline is not None
        assert self._post_compile is not None
        assert self._post_inference is not None

        return MemoryProfile(
            rss_baseline_mb=self._baseline,
            rss_after_compile_mb=self._post_compile,
            rss_after_inference_mb=self._post_inference,
            device_local_mb=self._device_local_mb,
        )
