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
    _get_process_memory,
)


class TestGetProcessMemory:
    """Test the process memory retrieval function."""

    def test_returns_four_floats(self) -> None:
        result = _get_process_memory()
        assert len(result) == 4
        for val in result:
            assert isinstance(val, float)

    def test_working_set_positive(self) -> None:
        ws, peak_ws, priv, peak_priv = _get_process_memory()
        # Our process should be using *some* memory
        assert ws > 0
        assert peak_ws >= ws
        assert priv > 0
        assert peak_priv >= priv


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass."""

    def test_to_dict(self) -> None:
        snap = MemorySnapshot(
            working_set_mb=100.123,
            peak_working_set_mb=120.456,
            private_bytes_mb=80.789,
            peak_private_bytes_mb=90.012,
            device_local_mb=50.347,
            device_shared_mb=10.678,
        )
        d = snap.to_dict()
        assert d["working_set_mb"] == 100.12
        assert d["peak_working_set_mb"] == 120.46
        assert d["private_bytes_mb"] == 80.79
        assert d["device_local_mb"] == 50.35
        assert d["device_shared_mb"] == 10.68

    def test_defaults_are_zero(self) -> None:
        snap = MemorySnapshot()
        assert snap.working_set_mb == 0.0
        assert snap.device_local_mb == 0.0


class TestMemoryProfile:
    """Test MemoryProfile computed properties."""

    @pytest.fixture
    def profile(self) -> MemoryProfile:
        return MemoryProfile(
            baseline=MemorySnapshot(
                working_set_mb=100.0,
                peak_working_set_mb=100.0,
                private_bytes_mb=120.0,
                peak_private_bytes_mb=120.0,
            ),
            post_load=MemorySnapshot(
                working_set_mb=300.0,
                peak_working_set_mb=310.0,
                private_bytes_mb=350.0,
                peak_private_bytes_mb=350.0,
            ),
            post_compile=MemorySnapshot(
                working_set_mb=320.0,
                peak_working_set_mb=325.0,
                private_bytes_mb=370.0,
                peak_private_bytes_mb=375.0,
                device_local_mb=50.0,
            ),
            post_inference=MemorySnapshot(
                working_set_mb=330.0,
                peak_working_set_mb=340.0,
                private_bytes_mb=380.0,
                peak_private_bytes_mb=385.0,
                device_local_mb=52.0,
                device_shared_mb=8.0,
            ),
        )

    def test_load_delta(self, profile: MemoryProfile) -> None:
        assert profile.load_delta_mb == pytest.approx(200.0)

    def test_compile_delta(self, profile: MemoryProfile) -> None:
        assert profile.compile_delta_mb == pytest.approx(20.0)

    def test_inference_delta(self, profile: MemoryProfile) -> None:
        assert profile.inference_delta_mb == pytest.approx(10.0)

    def test_total_delta(self, profile: MemoryProfile) -> None:
        assert profile.total_delta_mb == pytest.approx(230.0)

    def test_peak_working_set(self, profile: MemoryProfile) -> None:
        assert profile.peak_working_set_mb == pytest.approx(340.0)

    def test_peak_device_local(self, profile: MemoryProfile) -> None:
        assert profile.peak_device_local_mb == pytest.approx(52.0)

    def test_peak_device_shared(self, profile: MemoryProfile) -> None:
        assert profile.peak_device_shared_mb == pytest.approx(8.0)

    def test_to_dict(self, profile: MemoryProfile) -> None:
        d = profile.to_dict()
        assert "baseline" in d
        assert "post_load" in d
        assert "post_compile" in d
        assert "post_inference" in d
        assert d["peak_working_set_mb"] == 340.0
        assert d["total_delta_working_set_mb"] == 230.0


class TestMemoryTracker:
    """Test MemoryTracker snapshot collection."""

    def test_full_workflow(self) -> None:
        tracker = MemoryTracker()
        tracker.snapshot_baseline()
        tracker.snapshot_post_load()
        tracker.snapshot_post_compile()
        tracker.snapshot_post_inference()
        profile = tracker.profile()

        assert profile is not None
        assert profile.baseline.working_set_mb > 0
        assert profile.post_inference.working_set_mb > 0

    def test_incomplete_returns_none(self) -> None:
        tracker = MemoryTracker()
        tracker.snapshot_baseline()
        # Missing other phases
        profile = tracker.profile()
        assert profile is None

    def test_snapshots_are_nondecreasing(self) -> None:
        """Working set should generally not decrease between adjacent snapshots."""
        tracker = MemoryTracker()
        tracker.snapshot_baseline()

        # Allocate something to ensure memory grows
        _data = [bytearray(1024 * 1024) for _ in range(5)]  # ~5 MB

        tracker.snapshot_post_load()
        tracker.snapshot_post_compile()
        tracker.snapshot_post_inference()
        profile = tracker.profile()

        assert profile is not None
        # post_load should be >= baseline (we allocated memory)
        assert profile.post_load.working_set_mb >= profile.baseline.working_set_mb
        # Keep _data alive until assertions complete so memory isn't reclaimed early
        assert _data is not None
