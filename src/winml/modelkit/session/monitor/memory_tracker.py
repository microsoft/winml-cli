# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Process memory helper for perf benchmarking."""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import psutil


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
_DEFAULT_POLL_INTERVAL_SECONDS = 0.05

DEPRECATED_MEMORY_FIELDS = (
    "rss_baseline_mb",
    "rss_after_compile_mb",
    "rss_checkpoint_peak_mb",
    "rss_model_load_delta_mb",
    "rss_inference_delta_mb",
    "rss_total_delta_mb",
)

RSS_MEMORY_PROFILE_FIELDS = frozenset(
    (
    "setup_duration_ms",
    "rss_process_baseline_mb",
    "rss_before_session_mb",
    "rss_after_session_mb",
    "rss_peak_during_inference_mb",
    "rss_after_inference_mb",
    "rss_session_delta_mb",
    "rss_peak_delta_mb",
    "rss_end_delta_mb",
    *DEPRECATED_MEMORY_FIELDS,
    "deprecated_fields",
    )
)


def get_rss_mb() -> float:
    """Return current RSS in MB for this process."""
    return psutil.Process(os.getpid()).memory_info().rss / _MB


def get_vram_mb(adapter_luid: str | None) -> tuple[float, float]:
    """Return current VRAM usage as (local_mb, shared_mb) via PDH.

    Returns (0.0, 0.0) on non-Windows, if no adapter_luid, or on failure.
    """
    if sys.platform != "win32" or not adapter_luid:
        return 0.0, 0.0

    try:
        from ._pdh import PdhQuery

        pid = os.getpid()
        q = PdhQuery()
        q.open()
        q.add_counter(
            "local",
            rf"\GPU Process Memory(pid_{pid}_luid_{adapter_luid}_phys_0)\Local Usage",
        )
        q.add_counter(
            "shared",
            rf"\GPU Process Memory(pid_{pid}_luid_{adapter_luid}_phys_0)\Shared Usage",
        )
        # Memory counters are absolute (not rate-based), single collect suffices.
        values = q.collect()
        q.close()
        local = (values.get("local") or 0) / _MB
        shared = (values.get("shared") or 0) / _MB
        return local, shared
    except Exception:
        logger.debug("VRAM query failed", exc_info=True)
        return 0.0, 0.0


@dataclass(frozen=True)
class _MemorySnapshot:
    """One process-memory sample."""

    rss_mb: float


class ProcessMemoryTracker:
    """Track consistent process-memory lifecycle stages for perf runtimes."""

    def __init__(
        self,
        *,
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        rss_reader: Callable[[], float] | None = None,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._poll_interval_seconds = poll_interval_seconds
        self._rss_reader = rss_reader or get_rss_mb
        self._process_baseline: _MemorySnapshot | None = None
        self._before_session: _MemorySnapshot | None = None
        self._after_session: _MemorySnapshot | None = None
        self._after_inference: _MemorySnapshot | None = None
        self._rss_inference_peak: float | None = None
        self._setup_duration_ms = 0.0
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def record_process_baseline(self) -> None:
        """Record RSS before runtime setup or model/session construction."""
        self._process_baseline = self._snapshot(collect=True)

    def record_before_session(self, *, setup_duration_ms: float) -> None:
        """Record the post-setup baseline immediately before session creation."""
        self._setup_duration_ms = setup_duration_ms
        self._before_session = self._snapshot(collect=True)

    def record_after_session(self) -> None:
        """Record retained memory after eager or lazy session creation."""
        self._after_session = self._snapshot(collect=True)

    def start_inference(self) -> None:
        """Start continuous memory sampling for the inference phase."""
        if self._poll_thread is not None:
            raise RuntimeError("Inference sampling is already active")
        self._after_inference = None
        with self._lock:
            self._rss_inference_peak = None
        self._stop_event.clear()
        self._record_inference_sample(self._snapshot())
        self._poll_thread = threading.Thread(
            target=self._poll_inference,
            name="winml-memory-tracker",
            daemon=True,
        )
        self._poll_thread.start()

    def stop_inference(self) -> None:
        """Stop sampling and record the final post-inference snapshot."""
        poll_thread = self._poll_thread
        if poll_thread is None:
            raise RuntimeError("Inference sampling is not active")
        self._stop_event.set()
        poll_thread.join()
        self._poll_thread = None
        final_snapshot = self._snapshot()
        self._record_inference_sample(final_snapshot)
        self._after_inference = final_snapshot

    @contextmanager
    def track_inference(self) -> Iterator[None]:
        """Continuously sample memory while the wrapped inference runs."""
        self.start_inference()
        try:
            yield
        finally:
            self.stop_inference()

    def _snapshot(self, *, collect: bool = False) -> _MemorySnapshot:
        if collect:
            gc.collect()
        return _MemorySnapshot(rss_mb=self._rss_reader())

    def _poll_inference(self) -> None:
        while not self._stop_event.wait(self._poll_interval_seconds):
            self._sample_inference_best_effort()

    def _sample_inference_best_effort(self) -> None:
        try:
            self._record_inference_sample(self._snapshot())
        except Exception:
            logger.warning("Continuous process-memory sampling failed", exc_info=True)

    def _record_inference_sample(self, snapshot: _MemorySnapshot) -> None:
        with self._lock:
            self._rss_inference_peak = self._optional_max(
                self._rss_inference_peak, snapshot.rss_mb
            )

    @staticmethod
    def _optional_max(current: float | None, sample: float | None) -> float | None:
        if sample is None:
            return current
        return sample if current is None else max(current, sample)

    @staticmethod
    def _round(value: float | None) -> float | None:
        return round(value, 2) if value is not None else None

    @classmethod
    def _delta(cls, after: float | None, before: float | None) -> float | None:
        if after is None or before is None:
            return None
        return cls._round(after - before)

    @classmethod
    def _peak_delta(cls, peak: float | None, baseline: float | None) -> float | None:
        delta = cls._delta(peak, baseline)
        return max(0.0, delta) if delta is not None else None

    @classmethod
    def _checkpoint_peak(cls, *values: float | None) -> float | None:
        available = [value for value in values if value is not None]
        return cls._round(max(available)) if available else None

    def to_dict(self) -> dict[str, object]:
        """Return canonical lifecycle metrics plus explicitly deprecated aliases."""
        if (
            self._process_baseline is None
            or self._before_session is None
            or self._after_session is None
            or self._after_inference is None
            or self._rss_inference_peak is None
        ):
            raise RuntimeError("Memory lifecycle tracking is incomplete")

        process = self._process_baseline
        before = self._before_session
        after_session = self._after_session
        after_inference = self._after_inference
        rss_session_delta = self._delta(after_session.rss_mb, before.rss_mb)
        rss_inference_delta = self._delta(after_inference.rss_mb, after_session.rss_mb)
        rss_end_delta = self._delta(after_inference.rss_mb, process.rss_mb)

        result: dict[str, object] = {
            "setup_duration_ms": self._round(self._setup_duration_ms),
            "rss_process_baseline_mb": self._round(process.rss_mb),
            "rss_before_session_mb": self._round(before.rss_mb),
            "rss_after_session_mb": self._round(after_session.rss_mb),
            "rss_peak_during_inference_mb": self._round(self._rss_inference_peak),
            "rss_after_inference_mb": self._round(after_inference.rss_mb),
            "rss_session_delta_mb": rss_session_delta,
            "rss_peak_delta_mb": self._peak_delta(
                self._rss_inference_peak, process.rss_mb
            ),
            "rss_end_delta_mb": rss_end_delta,
        }

        result.update(
            {
                "rss_baseline_mb": result["rss_process_baseline_mb"],
                "rss_after_compile_mb": result["rss_after_session_mb"],
                "rss_checkpoint_peak_mb": self._checkpoint_peak(
                    process.rss_mb,
                    after_session.rss_mb,
                    after_inference.rss_mb,
                ),
                "rss_model_load_delta_mb": rss_session_delta,
                "rss_inference_delta_mb": rss_inference_delta,
                "rss_total_delta_mb": rss_end_delta,
            }
        )
        result["deprecated_fields"] = list(DEPRECATED_MEMORY_FIELDS)
        return result
