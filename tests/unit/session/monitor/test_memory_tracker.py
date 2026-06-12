# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the memory_tracker module."""

from __future__ import annotations

import pytest

from winml.modelkit.session.monitor.memory_tracker import (
    MemoryProfile,
    MemorySnapshot,
    MemoryTracker,
    _get_memory_mb,
)


class TestGetMemoryMb:
    """Test the process memory retrieval function."""

    def test_returns_dict_with_expected_keys(self) -> None:
        result = _get_memory_mb()
        assert "rss_mb" in result
        assert "peak_wset_mb" in result

    def test_rss_positive(self) -> None:
        result = _get_memory_mb()
        assert result["rss_mb"] > 0
        assert result["peak_wset_mb"] >= result["rss_mb"]


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass."""

    def test_to_dict(self) -> None:
        snap = MemorySnapshot(
            rss_mb=100.127,
            peak_wset_mb=120.456,
            device_local_mb=50.347,
            device_shared_mb=10.678,
        )
        d = snap.to_dict()
        assert d["rss_mb"] == 100.13
        assert d["peak_wset_mb"] == 120.46
        assert d["device_local_mb"] == 50.35
        assert d["device_shared_mb"] == 10.68

    def test_defaults_are_zero(self) -> None:
        snap = MemorySnapshot()
        assert snap.rss_mb == 0.0
        assert snap.device_local_mb == 0.0


class TestMemoryProfile:
    """Test MemoryProfile computed properties."""

    @pytest.fixture
    def profile(self) -> MemoryProfile:
        return MemoryProfile(
            baseline=MemorySnapshot(rss_mb=100.0, peak_wset_mb=100.0),
            post_compile=MemorySnapshot(
                rss_mb=320.0,
                peak_wset_mb=350.0,
                device_local_mb=50.0,
            ),
            post_inference=MemorySnapshot(
                rss_mb=330.0,
                peak_wset_mb=360.0,
                device_local_mb=52.0,
                device_shared_mb=8.0,
            ),
        )

    def test_model_load_delta(self, profile: MemoryProfile) -> None:
        assert profile.model_load_delta_mb == pytest.approx(220.0)

    def test_inference_alloc_delta(self, profile: MemoryProfile) -> None:
        assert profile.inference_alloc_delta_mb == pytest.approx(10.0)

    def test_total_delta(self, profile: MemoryProfile) -> None:
        assert profile.total_delta_mb == pytest.approx(230.0)

    def test_peak_wset(self, profile: MemoryProfile) -> None:
        assert profile.peak_wset_mb == pytest.approx(360.0)

    def test_peak_delta(self, profile: MemoryProfile) -> None:
        assert profile.peak_delta_mb == pytest.approx(260.0)

    def test_peak_device_local(self, profile: MemoryProfile) -> None:
        assert profile.peak_device_local_mb == pytest.approx(52.0)

    def test_to_dict(self, profile: MemoryProfile) -> None:
        d = profile.to_dict()
        assert d["rss_baseline_mb"] == 100.0
        assert d["rss_after_compile_mb"] == 320.0
        assert d["rss_after_inference_mb"] == 330.0
        assert d["model_load_delta_mb"] == 220.0
        assert d["inference_alloc_delta_mb"] == 10.0
        assert d["total_delta_mb"] == 230.0
        assert d["peak_working_set_mb"] == 360.0
        assert d["peak_delta_mb"] == 260.0
        assert d["device_local_mb"] == 52.0


class TestMemoryTracker:
    """Test MemoryTracker snapshot collection."""

    def test_full_workflow(self) -> None:
        tracker = MemoryTracker()
        tracker.snapshot_baseline()
        tracker.snapshot_post_compile()
        tracker.snapshot_post_inference()
        profile = tracker.profile()

        assert profile is not None
        assert profile.baseline.rss_mb > 0
        assert profile.post_inference.rss_mb > 0

    def test_incomplete_returns_none(self) -> None:
        tracker = MemoryTracker()
        tracker.snapshot_baseline()
        # Missing other phases
        profile = tracker.profile()
        assert profile is None

    def test_snapshots_are_nondecreasing(self) -> None:
        """RSS should generally not decrease between adjacent snapshots."""
        tracker = MemoryTracker()
        tracker.snapshot_baseline()

        # Allocate something to ensure memory grows
        _data = [bytearray(1024 * 1024) for _ in range(5)]  # ~5 MB

        tracker.snapshot_post_compile()
        tracker.snapshot_post_inference()
        profile = tracker.profile()

        assert profile is not None
        # post_compile should be >= baseline (we allocated memory)
        assert profile.post_compile.rss_mb >= profile.baseline.rss_mb
        # Keep _data alive until assertions complete
        assert _data is not None
