# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for shared CLI helpers in winml.modelkit.utils.cli."""

from __future__ import annotations

import click
import pytest

from winml.modelkit.utils.cli import parse_ep_options


class TestParseEpOptions:
    """Tests for parse_ep_options()."""

    def test_empty_returns_none(self) -> None:
        """No values -> None so callers leave the session default untouched."""
        assert parse_ep_options(()) is None

    def test_single_pair(self) -> None:
        assert parse_ep_options(("htp_performance_mode=burst",)) == {
            "htp_performance_mode": "burst"
        }

    def test_multiple_pairs(self) -> None:
        result = parse_ep_options(
            ("htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3")
        )
        assert result == {
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        }

    def test_value_may_contain_equals(self) -> None:
        """Only the first '=' splits key from value."""
        assert parse_ep_options(("key=a=b=c",)) == {"key": "a=b=c"}

    def test_last_wins_on_duplicate_key(self) -> None:
        assert parse_ep_options(("k=1", "k=2")) == {"k": "2"}

    def test_whitespace_stripped_from_key_and_value(self) -> None:
        """Surrounding whitespace (e.g. from shell quoting) is stripped."""
        assert parse_ep_options(("htp_performance_mode= burst ",)) == {
            "htp_performance_mode": "burst"
        }
        assert parse_ep_options(("  k  =  v  ",)) == {"k": "v"}

    def test_missing_equals_raises(self) -> None:
        with pytest.raises(click.BadParameter):
            parse_ep_options(("no_equals_sign",))

    def test_empty_key_raises(self) -> None:
        with pytest.raises(click.BadParameter):
            parse_ep_options(("=value",))
