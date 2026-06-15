# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the memory_tracker module."""

from __future__ import annotations

import pytest

from winml.modelkit.session.monitor.memory_tracker import (
    MemoryProfile,
    MemoryTracker,
    _get_rss_mb,
)


class TestGetRssMb:
    """Test the process RSS retrieval function."""

    def test_returns_positive_float(self) -> None:
        rss = _get_rss_mb()
        assert isinstance(rss, float)
        assert rss > 0


class TestMemoryProfile:
    """Test MemoryProfile computed properties."""

    @pytest.fixture
    def profile(self) -> MemoryProfile:
        return MemoryProfile(
            rss_baseline_mb=100.0,
            rss_after_compile_mb=320.0,
            rss_after_inference_mb=330.0,
            device_local_mb=52.0,
        )

    def test_model_load_delta(self, profile: MemoryProfile) -> None:
        assert profile.model_load_delta_mb == pytest.approx(220.0)

    def test_inference_alloc_delta(self, profile: MemoryProfile) -> None:
        assert profile.inference_alloc_delta_mb == pytest.approx(10.0)

    def test_total_delta(self, profile: MemoryProfile) -> None:
        assert profile.total_delta_mb == pytest.approx(230.0)

    def test_to_dict(self, profile: MemoryProfile) -> None:
        d = profile.to_dict()
        assert d["rss_baseline_mb"] == 100.0
        assert d["rss_after_compile_mb"] == 320.0
        assert d["rss_after_inference_mb"] == 330.0
        assert d["model_load_delta_mb"] == 220.0
        assert d["inference_alloc_delta_mb"] == 10.0
        assert d["total_delta_mb"] == 230.0
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
        assert profile.rss_baseline_mb > 0
        assert profile.rss_after_inference_mb > 0

    def test_incomplete_returns_none(self) -> None:
        tracker = MemoryTracker()
        tracker.snapshot_baseline()
        assert tracker.profile() is None

    def test_deltas_nonnegative_with_allocation(self) -> None:
        tracker = MemoryTracker()
        tracker.snapshot_baseline()

        # Allocate ~5 MB to ensure RSS grows
        _data = [bytearray(1024 * 1024) for _ in range(5)]

        tracker.snapshot_post_compile()
        tracker.snapshot_post_inference()
        profile = tracker.profile()

        assert profile is not None
        assert profile.model_load_delta_mb >= 0
        assert _data is not None
