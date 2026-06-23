# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the memory_tracker module."""

from __future__ import annotations

from winml.modelkit.session.monitor.memory_tracker import get_rss_mb


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
