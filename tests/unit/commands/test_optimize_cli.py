# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for optimize CLI command.

Covers: flag surface (no --preset), help text, model-required guard,
--list-capabilities, --list-rewrites, and basic optimization invocation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch


if TYPE_CHECKING:
    from pathlib import Path

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.optimize import optimize


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# =============================================================================
# PRESET REMOVAL TESTS
# =============================================================================


class TestPresetRemoved:
    """--preset flag and PRESETS dict must not exist."""

    def test_preset_flag_absent_from_help(self, runner: CliRunner) -> None:
        result = runner.invoke(optimize, ["--help"])
        assert result.exit_code == 0
        assert "--preset" not in result.output

    def test_preset_choice_values_absent_from_help(self, runner: CliRunner) -> None:
        result = runner.invoke(optimize, ["--help"])
        assert result.exit_code == 0
        for name in ("qnn-compatible", "transformer-optimized", "full", "minimal"):
            assert name not in result.output

    def test_preset_flag_rejected_as_unknown_option(self, runner: CliRunner) -> None:
        result = runner.invoke(optimize, ["--preset", "full"])
        assert result.exit_code != 0

    def test_presets_dict_not_in_module(self) -> None:
        import winml.modelkit.commands.optimize as mod

        assert not hasattr(mod, "PRESETS"), "PRESETS dict must be removed from optimize.py"


# =============================================================================
# CLI INTERFACE TESTS
# =============================================================================


class TestOptimizeCliInterface:
    """Basic flag parsing and help text."""

    def test_help_exits_cleanly(self, runner: CliRunner) -> None:
        result = runner.invoke(optimize, ["--help"])
        assert result.exit_code == 0

    def test_help_shows_required_flags(self, runner: CliRunner) -> None:
        result = runner.invoke(optimize, ["--help"])
        assert result.exit_code == 0
        for flag in ("--model", "-m", "--output", "-o", "--config", "-c", "--verbose", "-v"):
            assert flag in result.output, f"Missing flag {flag} in help"

    def test_model_required_without_list_flags(self, runner: CliRunner) -> None:
        result = runner.invoke(optimize, [], obj={})
        assert result.exit_code != 0

    def test_model_required_error_message(self, runner: CliRunner) -> None:
        result = runner.invoke(optimize, [], obj={})
        assert "--model" in result.output or "model" in result.output.lower()


# =============================================================================
# --list-capabilities / --list-rewrites TESTS
# =============================================================================


class TestListFlags:
    """--list-capabilities and --list-rewrites exit without requiring --model."""

    def test_list_capabilities_exits_cleanly(self, runner: CliRunner) -> None:
        result = runner.invoke(optimize, ["--list-capabilities"])
        assert result.exit_code == 0

    def test_list_capabilities_short_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(optimize, ["-l"])
        assert result.exit_code == 0

    def test_list_capabilities_verbose(self, runner: CliRunner) -> None:
        result = runner.invoke(optimize, ["--list-capabilities", "--verbose"])
        assert result.exit_code == 0

    def test_list_rewrites_exits_cleanly(self, runner: CliRunner) -> None:
        result = runner.invoke(optimize, ["--list-rewrites"])
        assert result.exit_code == 0


# =============================================================================
# BASIC OPTIMIZATION INVOCATION TESTS
# =============================================================================

_LOAD_ONNX = "winml.modelkit.commands.optimize.load_onnx"
_SAVE_ONNX = "winml.modelkit.commands.optimize.save_onnx"
_OPTIMIZER = "winml.modelkit.optim.Optimizer"


def _make_mock_model(num_nodes: int = 10) -> MagicMock:
    model = MagicMock()
    model.graph.node = [MagicMock()] * num_nodes
    return model


class TestOptimizeInvocation:
    """Happy-path and error-path for model optimization."""

    def test_basic_optimization_succeeds(self, runner: CliRunner, tmp_path: Path) -> None:
        model_file = tmp_path / "model.onnx"
        model_file.touch()
        out_file = tmp_path / "out.onnx"

        mock_model = _make_mock_model()
        with (
            patch(_LOAD_ONNX, return_value=mock_model),
            patch(_SAVE_ONNX),
            patch(_OPTIMIZER) as mock_opt_cls,
        ):
            mock_opt_cls.return_value.optimize.return_value = mock_model
            result = runner.invoke(optimize, ["-m", str(model_file), "-o", str(out_file)])

        assert result.exit_code == 0, result.output
        assert "Success" in result.output

    def test_default_output_path_derived_from_input(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        model_file = tmp_path / "mymodel.onnx"
        model_file.touch()

        mock_model = _make_mock_model()
        with (
            patch(_LOAD_ONNX, return_value=mock_model),
            patch(_SAVE_ONNX) as mock_save,
            patch(_OPTIMIZER) as mock_opt_cls,
        ):
            mock_opt_cls.return_value.optimize.return_value = mock_model
            result = runner.invoke(optimize, ["-m", str(model_file)])

        assert result.exit_code == 0, result.output
        saved_path = mock_save.call_args[0][1]
        assert saved_path.name == "mymodel_opt.onnx"

    def test_optimization_failure_exits_nonzero(self, runner: CliRunner, tmp_path: Path) -> None:
        model_file = tmp_path / "model.onnx"
        model_file.touch()

        with (
            patch(_LOAD_ONNX, side_effect=RuntimeError("corrupt model")),
        ):
            result = runner.invoke(optimize, ["-m", str(model_file)])

        assert result.exit_code != 0

    def test_node_reduction_reported(self, runner: CliRunner, tmp_path: Path) -> None:
        model_file = tmp_path / "model.onnx"
        model_file.touch()

        original = _make_mock_model(num_nodes=10)
        optimized = _make_mock_model(num_nodes=8)
        with (
            patch(_LOAD_ONNX, return_value=original),
            patch(_SAVE_ONNX),
            patch(_OPTIMIZER) as mock_opt_cls,
        ):
            mock_opt_cls.return_value.optimize.return_value = optimized
            result = runner.invoke(optimize, ["-m", str(model_file)])

        assert result.exit_code == 0, result.output
        assert "10" in result.output
        assert "8" in result.output


# =============================================================================
# CONFIG FILE TESTS
# =============================================================================


class TestConfigFile:
    """Config file loading and precedence."""

    def test_json_config_accepted(self, runner: CliRunner, tmp_path: Path) -> None:
        model_file = tmp_path / "model.onnx"
        model_file.touch()
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"gelu-fusion": True}))

        mock_model = _make_mock_model()
        with (
            patch(_LOAD_ONNX, return_value=mock_model),
            patch(_SAVE_ONNX),
            patch(_OPTIMIZER) as mock_opt_cls,
        ):
            mock_opt_cls.return_value.optimize.return_value = mock_model
            result = runner.invoke(optimize, ["-m", str(model_file), "-c", str(config_file)])

        assert result.exit_code == 0, result.output

    def test_missing_config_exits_nonzero(self, runner: CliRunner, tmp_path: Path) -> None:
        model_file = tmp_path / "model.onnx"
        model_file.touch()

        result = runner.invoke(optimize, ["-m", str(model_file), "-c", str(tmp_path / "no.json")])
        assert result.exit_code != 0
