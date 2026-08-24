# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the memory_tracker module."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from winml.modelkit.session.monitor import memory_tracker
from winml.modelkit.session.monitor.memory_tracker import (
    DEPRECATED_MEMORY_FIELDS,
    MEMORY_PROFILE_FIELDS,
    ProcessMemoryTracker,
    get_rss_mb,
    get_vram_mb,
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


class _FakePdhQuery:
    def __init__(
        self,
        values: dict[str, float] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._values = values or {}
        self._error = error
        self.closed = False

    def open(self) -> None:
        if self._error is not None:
            raise self._error

    def add_counter(self, _name: str, _path: str) -> None:
        pass

    def collect(self) -> dict[str, float]:
        return self._values

    def close(self) -> None:
        self.closed = True


class TestGetVramMb:
    """Test nullable process VRAM retrieval."""

    def test_missing_adapter_is_unknown(self) -> None:
        assert get_vram_mb(None) == (None, None)

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ({}, (None, None)),
            ({"local": 0.0, "shared": 0.0}, (0.0, 0.0)),
            ({"local": 2 * 1024 * 1024}, (2.0, None)),
        ],
    )
    def test_missing_counters_are_not_reported_as_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        values: dict[str, float],
        expected: tuple[float | None, float | None],
    ) -> None:
        query = _FakePdhQuery(values)
        monkeypatch.setattr(memory_tracker.sys, "platform", "win32")
        monkeypatch.setattr(
            "winml.modelkit.session.monitor._pdh.PdhQuery",
            lambda: query,
        )

        assert get_vram_mb("0x00000000_0x00000001") == expected
        assert query.closed

    def test_multi_gpu_query_failure_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        query = _FakePdhQuery(error=OSError("adapter counter disappeared"))
        monkeypatch.setattr(memory_tracker.sys, "platform", "win32")
        monkeypatch.setattr(
            "winml.modelkit.session.monitor._pdh.PdhQuery",
            lambda: query,
        )

        assert get_vram_mb("0x00000000_0x00000002") == (None, None)
        assert query.closed


def _rss_reader(values: list[float]) -> Iterator[float]:
    yield from values


class TestProcessMemoryTracker:
    """Test lifecycle semantics and continuously sampled peaks."""

    def test_negative_end_delta_keeps_nonnegative_peak_delta(self) -> None:
        rss_values = _rss_reader([100.0, 110.0, 150.0, 140.0, 90.0])
        tracker = ProcessMemoryTracker(
            poll_interval_seconds=60.0,
            rss_reader=lambda: next(rss_values),
            vram_reader=lambda _luid: (None, None),
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
        assert set(result) == MEMORY_PROFILE_FIELDS

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
            vram_reader=lambda _luid: (None, None),
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

    def test_unavailable_adapter_keeps_all_vram_fields_nullable(self) -> None:
        rss_values = _rss_reader([100.0, 100.0, 100.0, 100.0, 100.0])
        tracker = ProcessMemoryTracker(
            poll_interval_seconds=60.0,
            rss_reader=lambda: next(rss_values),
            vram_reader=lambda _luid: (None, None),
        )
        tracker.record_process_baseline()
        tracker.record_before_session(setup_duration_ms=0.0)
        tracker.record_after_session()
        with tracker.track_inference():
            pass

        result = tracker.to_dict()
        vram_values = {
            key: value
            for key, value in result.items()
            if key.startswith("vram_")
        }
        assert vram_values
        assert set(vram_values.values()) == {None}
