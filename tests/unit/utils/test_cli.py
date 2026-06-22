# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for shared CLI helpers in winml.modelkit.utils.cli."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from winml.modelkit.utils.cli import parse_ep_options, precision_option


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


class TestPrecisionOption:
    """Tests for the shared precision_option() decorator."""

    @staticmethod
    def _make_cmd(**kwargs: object) -> click.Command:
        @click.command()
        @precision_option(**kwargs)  # type: ignore[arg-type]
        def cmd(precision: str) -> None:
            click.echo(precision)

        return cmd

    def test_default_is_auto(self) -> None:
        """No --precision yields the default 'auto'."""
        result = CliRunner().invoke(self._make_cmd(), [])
        assert result.exit_code == 0
        assert result.output.strip() == "auto"

    def test_short_flag_p(self) -> None:
        """-p is registered by default and maps to the precision param."""
        result = CliRunner().invoke(self._make_cmd(), ["-p", "fp16"])
        assert result.exit_code == 0
        assert result.output.strip() == "fp16"

    def test_long_flag(self) -> None:
        """--precision sets the value."""
        result = CliRunner().invoke(self._make_cmd(), ["--precision", "w8a16"])
        assert result.exit_code == 0
        assert result.output.strip() == "w8a16"

    def test_include_short_false_drops_p(self) -> None:
        """include_short=False removes the -p alias but keeps --precision."""
        cmd = self._make_cmd(include_short=False)
        assert CliRunner().invoke(cmd, ["-p", "fp16"]).exit_code != 0
        assert CliRunner().invoke(cmd, ["--precision", "fp16"]).exit_code == 0

    def test_custom_default(self) -> None:
        """default overrides the resolved value when flag is omitted."""
        result = CliRunner().invoke(self._make_cmd(default="int8"), [])
        assert result.output.strip() == "int8"

    def test_optional_message_appended_to_help(self) -> None:
        """optional_message is appended after the base help text."""
        result = CliRunner().invoke(self._make_cmd(optional_message="Extra note."), ["--help"])
        assert "w8a16" in result.output  # base help present
        assert "Extra note." in result.output

    def test_accepts_arbitrary_string(self) -> None:
        """type=str: validation is deferred downstream, so any string parses."""
        result = CliRunner().invoke(self._make_cmd(), ["--precision", "bf16"])
        assert result.exit_code == 0
        assert result.output.strip() == "bf16"

    def test_default_none(self) -> None:
        """default=None (e.g. quantize) yields no value when the flag is omitted."""
        result = CliRunner().invoke(self._make_cmd(default=None), [])
        assert result.exit_code == 0
        assert result.output.strip() == ""

    def test_help_text_override_replaces_base(self) -> None:
        """help_text replaces the default float+int help; optional_message still appends."""
        cmd = self._make_cmd(
            help_text="Quantization precision: int8, int16",
            optional_message="Overridden by --weight-type",
        )
        result = CliRunner().invoke(cmd, ["--help"])
        # Collapse whitespace so wrapped help lines compare as one string.
        flat = " ".join(result.output.split())
        assert "Quantization precision: int8, int16" in flat
        assert "Overridden by --weight-type" in flat
        assert "fp16" not in flat  # base float help is gone
