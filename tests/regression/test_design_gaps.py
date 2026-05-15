# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests documenting current CLI behavior and known design gaps.

These tests document the CURRENT behavior of the CLI. They should PASS
as-is. When a gap is addressed, the corresponding test should be updated
to reflect the new behavior.

Design Gap IDs reference the CLI verification plan.

Markers:
    regression: Regression test suite
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.build import build
from winml.modelkit.commands.inspect import inspect
from winml.modelkit.commands.optimize import optimize
from winml.modelkit.commands.perf import perf


pytestmark = [pytest.mark.regression]


# ===========================================================================
# A-1: All --help text must be ASCII-safe (no non-ASCII chars in descriptions)
# ===========================================================================


class TestA1HelpTextAsciiSafe:
    """Verify that --help output contains no non-ASCII characters (regression: #228).

    Non-ASCII chars in capability/rewrite descriptions (e.g. U+2192 →) crash
    winml on Windows terminals using cp1252 encoding.
    """

    def _assert_ascii_help(self, cmd, args: list) -> None:
        runner = CliRunner()
        result = runner.invoke(cmd, args, obj={})
        assert result.exit_code == 0, f"--help exited {result.exit_code}: {result.exception}"
        non_ascii = [(i, c) for i, c in enumerate(result.output) if ord(c) > 127]
        assert not non_ascii, "Non-ASCII characters found in help output: " + ", ".join(
            f"pos {i} U+{ord(c):04X}" for i, c in non_ascii[:5]
        )

    def test_optimize_help_is_ascii_safe(self):
        """winml optimize --help must not contain non-ASCII characters."""
        self._assert_ascii_help(optimize, ["--help"])

    def test_optimize_list_capabilities_is_ascii_safe(self):
        """winml optimize --list-capabilities must not contain non-ASCII characters."""
        self._assert_ascii_help(optimize, ["--list-capabilities"])

    def test_optimize_list_rewrites_is_ascii_safe(self):
        """winml optimize --list-rewrites must not contain non-ASCII characters."""
        self._assert_ascii_help(optimize, ["--list-rewrites"])


# ===========================================================================
# M-1: --list-tasks IS in inspect --help (implemented in MVP v2 port)
# ===========================================================================


class TestM1ListTasksPresent:
    """Verify that --list-tasks is implemented in inspect."""

    def test_list_tasks_in_help(self):
        """inspect --help should contain --list-tasks option."""
        runner = CliRunner()
        result = runner.invoke(inspect, ["--help"], obj={})
        assert result.exit_code == 0
        assert "--list-tasks" in result.output


# ===========================================================================
# M-5: --no-analyze NOT in build --help
# ===========================================================================


class TestM5NoAnalyzePresent:
    """Verify that --no-analyze IS now implemented in build (M-5 fixed)."""

    def test_no_analyze_in_help(self):
        """build --help should now contain --no-analyze option."""
        runner = CliRunner()
        result = runner.invoke(build, ["--help"], obj={})
        assert result.exit_code == 0
        assert "--no-analyze" in result.output


# ===========================================================================
# B-2: precision values accepted by perf CLI
# ===========================================================================


class TestB2PerfPrecisionValues:
    """Document current precision values accepted by perf --precision."""

    def test_accepted_precision_values(self):
        """perf --help should list auto, fp32, fp16, int8 as precision choices."""
        runner = CliRunner()
        result = runner.invoke(perf, ["--help"], obj={})
        assert result.exit_code == 0
        # The help text shows the choices
        assert "auto" in result.output
        assert "fp32" in result.output
        assert "fp16" in result.output
        assert "int8" in result.output

    def test_int16_not_accepted(self):
        """perf does NOT accept int16 precision (unlike config command)."""
        runner = CliRunner()
        result = runner.invoke(perf, ["--help"], obj={})
        assert result.exit_code == 0
        # int16 is in config's choices but NOT in perf's
        # Verify by checking the precision option line does not include int16
        help_text = result.output
        # Find the precision line
        for line in help_text.split("\n"):
            if "--precision" in line and "[" in line:
                assert "int16" not in line
                break


# ===========================================================================
# D-6: DEFAULT_VOCAB_SIZE == 30522
# ===========================================================================


class TestD6NoHardcodedVocab:
    """Verify that DEFAULT_VOCAB_SIZE is removed from perf.py (D-6 fixed)."""

    def test_no_default_vocab_size_constant(self):
        """perf.py should NOT have DEFAULT_VOCAB_SIZE constant after D-6 fix."""
        import winml.modelkit.commands.perf as perf_mod

        assert not hasattr(perf_mod, "DEFAULT_VOCAB_SIZE"), (
            "DEFAULT_VOCAB_SIZE still exists — D-6 not fixed"
        )


# ===========================================================================
# B-1: inspect help says "HuggingFace" (no ONNX mention)
# ===========================================================================


class TestB1InspectHelpScope:
    """Document that inspect only targets HuggingFace models."""

    def test_help_mentions_huggingface(self):
        """inspect --help should mention HuggingFace."""
        runner = CliRunner()
        result = runner.invoke(inspect, ["--help"], obj={})
        assert result.exit_code == 0
        assert "HuggingFace" in result.output

    def test_help_does_not_mention_onnx_input(self):
        """inspect --help should NOT mention ONNX file input.

        Unlike config/build/perf which accept .onnx files, inspect
        only works with HuggingFace model IDs.
        """
        runner = CliRunner()
        result = runner.invoke(inspect, ["--help"], obj={})
        assert result.exit_code == 0
        # The help text should not suggest ONNX file as an input option
        assert ".onnx" not in result.output


# ===========================================================================
# B-4: config --model-type bert without --task (e2e-level)
# ===========================================================================


class TestB4ConfigModelTypeNoTask:
    """Document that config --model-type without --task does not crash."""

    @pytest.fixture(autouse=True)
    def _mock_device(self):
        """Mock hardware detection."""
        with (
            patch(
                "winml.modelkit.session.auto_detect_device",
                return_value="cpu",
            ),
            patch(
                "winml.modelkit.sysinfo.hardware.get_available_devices",
                return_value=["cpu"],
            ),
        ):
            yield

    @pytest.mark.e2e
    @pytest.mark.network
    def test_model_type_bert_no_task_succeeds(self):
        """config --model-type bert (no --task) should auto-select a task.

        This exercises Scenario C: model-type-only config generation.
        The command should not crash and should produce valid JSON.
        """
        from winml.modelkit.commands.config import config

        runner = CliRunner()
        result = runner.invoke(config, ["--model-type", "bert"])
        assert result.exit_code == 0, (
            f"config --model-type bert crashed (exit {result.exit_code}):\n{result.output}"
        )
