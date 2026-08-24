# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the memory_tracker module."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from winml.modelkit.session.monitor.memory_tracker import (
    DEPRECATED_MEMORY_FIELDS,
    RSS_MEMORY_PROFILE_FIELDS,
    ProcessMemoryTracker,
    get_rss_mb,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


class TestGetRssMb:
    """Test process RSS retrieval."""

    def test_returns_positive_float(self) -> None:
        rss = get_rss_mb()
        assert isinstance(rss, float)
        assert rss > 0

    def test_increases_after_allocation(self) -> None:
        before = get_rss_mb()
        _data = [bytearray(1024 * 1024) for _ in range(10)]  # ~10 MB
        after = get_rss_mb()
        assert after >= before
        assert _data is not None


def _rss_reader(values: list[float]) -> Iterator[float]:
    yield from values


class TestProcessMemoryTracker:
    """Test lifecycle semantics and continuously sampled peaks."""

    def test_negative_end_delta_keeps_nonnegative_peak_delta(self) -> None:
        rss_values = _rss_reader([100.0, 110.0, 150.0, 140.0, 90.0])
        tracker = ProcessMemoryTracker(
            poll_interval_seconds=60.0,
            rss_reader=lambda: next(rss_values),
        )

        tracker.record_process_baseline()
        tracker.record_before_session(setup_duration_ms=12.5)
        tracker.record_after_session()
        with tracker.track_inference():
            pass
        result = tracker.to_dict()

        assert result["rss_end_delta_mb"] == -10.0
        assert result["rss_peak_delta_mb"] == 40.0
        assert result["rss_total_delta_mb"] == -10.0
        assert result["setup_duration_ms"] == 12.5
        assert result["deprecated_fields"] == list(DEPRECATED_MEMORY_FIELDS)
        assert set(result) == RSS_MEMORY_PROFILE_FIELDS

    def test_peak_comes_from_continuous_sampling_not_checkpoints(self) -> None:
        peak_sampled = threading.Event()
        values = iter([100.0, 110.0, 150.0, 160.0, 300.0])

        def read_rss() -> float:
            try:
                value = next(values)
            except StopIteration:
                return 90.0
            if value == 300.0:
                peak_sampled.set()
            return value

        tracker = ProcessMemoryTracker(
            poll_interval_seconds=0.001,
            rss_reader=read_rss,
        )
        tracker.record_process_baseline()
        tracker.record_before_session(setup_duration_ms=0.0)
        tracker.record_after_session()
        tracker.start_inference()
        assert peak_sampled.wait(timeout=1.0)
        tracker.stop_inference()

        result = tracker.to_dict()
        assert result["rss_peak_during_inference_mb"] == 300.0
        assert result["rss_checkpoint_peak_mb"] == 150.0
        assert result["rss_peak_delta_mb"] == 200.0
