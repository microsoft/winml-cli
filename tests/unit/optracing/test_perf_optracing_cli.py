# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the --op-tracing CLI option on winml perf."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from winml.modelkit.commands.perf import perf


def _invoke_perf(args: list[str]):
    """Invoke perf CLI with PerfBenchmark.run mocked to prevent model loading."""
    runner = CliRunner()
    with patch(
        "winml.modelkit.commands.perf.PerfBenchmark.run",
        side_effect=RuntimeError("mocked — not running benchmark"),
    ):
        return runner.invoke(perf, args, obj={})


class TestOpTracingOptionParsing:
    """Verify --op-tracing is recognized and validates choices."""

    def test_option_is_recognized(self):
        """--op-tracing is accepted as a valid CLI option."""
        result = _invoke_perf(["--op-tracing", "basic", "-m", "nonexistent"])
        assert "no such option" not in (result.output or "").lower()

    def test_basic_choice_accepted(self):
        """--op-tracing basic is a valid choice."""
        result = _invoke_perf(["--op-tracing", "basic", "-m", "nonexistent"])
        assert "no such option" not in (result.output or "").lower()
        assert "invalid choice" not in (result.output or "").lower()

    def test_detail_choice_accepted(self):
        """--op-tracing detail is a valid choice."""
        result = _invoke_perf(["--op-tracing", "detail", "-m", "nonexistent"])
        assert "no such option" not in (result.output or "").lower()
        assert "invalid choice" not in (result.output or "").lower()

    def test_invalid_choice_rejected(self):
        """--op-tracing with an invalid value is rejected by Click."""
        runner = CliRunner()
        result = runner.invoke(perf, ["--op-tracing", "invalid", "-m", "test"])
        assert result.exit_code != 0
        output_lower = (result.output or "").lower()
        assert "invalid" in output_lower or "choice" in output_lower

    def test_case_insensitive(self):
        """--op-tracing accepts mixed-case values (e.g. Basic, DETAIL)."""
        result = _invoke_perf(["--op-tracing", "BASIC", "-m", "nonexistent"])
        assert "invalid choice" not in (result.output or "").lower()

    def test_without_op_tracing_flag(self):
        """Command works without --op-tracing (default is None)."""
        result = _invoke_perf(["-m", "nonexistent"])
        assert "no such option" not in (result.output or "").lower()

    def test_model_required_with_op_tracing(self):
        """--op-tracing alone without -m still requires a model."""
        runner = CliRunner()
        result = runner.invoke(perf, ["--op-tracing", "basic"])
        assert result.exit_code != 0
