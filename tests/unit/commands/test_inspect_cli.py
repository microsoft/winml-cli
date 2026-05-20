# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for inspect CLI command -- mock-based, no network calls.

Tests the CLI wrapper around inspect_model() API.
NO actual HuggingFace downloads or model loading.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_inspect_result() -> MagicMock:
    """Create a minimal mock InspectResult for happy-path tests."""
    result = MagicMock()
    result.model_id = "test"
    result.model_type = "bert"
    result.architectures = ["BertForMaskedLM"]
    result.task = "fill-mask"
    result.task_source = "auto"
    result.overall_support = MagicMock(value="supported")
    result.support_notes = []
    result.build_config = {}
    result.hierarchy = None
    result.cache = None
    result.processor = None
    result.io_config = None
    result.loader = MagicMock(
        hf_model_class="BertForMaskedLM",
        hf_model_class_source="task_defaults",
        support_level=MagicMock(value="supported"),
    )
    result.exporter = MagicMock(
        onnx_config_class="BertOnnxConfig",
        onnx_config_source="optimum",
        support_level=MagicMock(value="supported"),
        opset_version=14,
        input_tensors=[],
        output_tensors=[],
    )
    result.winml = MagicMock(
        winml_class="WinMLBert",
        winml_class_source="registry",
        support_level=MagicMock(value="supported"),
    )
    return result


# The inspect command calls _inspect_model_v2 (a module-level function in
# commands/inspect.py) then dispatches to output_json / output_table from
# the formatter module.  We patch at their actual locations.
_INSPECT_MODEL = "winml.modelkit.commands.inspect._inspect_model_v2"
_OUTPUT_JSON = "winml.modelkit.inspect.formatter.output_json"
_OUTPUT_TABLE = "winml.modelkit.inspect.formatter.output_table"


# =============================================================================
# CLI INTERFACE TESTS
# =============================================================================


class TestInspectCliInterface:
    """Test CLI flag parsing and help text."""

    def test_help_shows_all_options(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.inspect import inspect

        result = runner.invoke(inspect, ["--help"])
        assert result.exit_code == 0
        for flag in [
            "--model",
            "-m",
            "--format",
            "-f",
            "--verbose",
            "-v",
            "--task",
            "-t",
            "--hierarchy",
            "-H",
        ]:
            assert flag in result.output, f"Missing flag {flag} in help output"

    def test_model_required(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.inspect import inspect

        result = runner.invoke(inspect, [], obj={})
        assert result.exit_code != 0

    def test_invalid_format_rejected(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.inspect import inspect

        result = runner.invoke(inspect, ["-m", "test", "-f", "xml"], obj={})
        assert result.exit_code != 0


# =============================================================================
# OUTPUT FORMAT TESTS
# =============================================================================


class TestInspectOutputFormat:
    """Test output format dispatching (json vs table)."""

    def test_json_format_accepted(
        self,
        runner: CliRunner,
        mock_inspect_result: MagicMock,
    ) -> None:
        from winml.modelkit.commands.inspect import inspect

        with (
            patch(_INSPECT_MODEL, return_value=mock_inspect_result),
            patch(_OUTPUT_JSON, return_value="{}") as mock_json,
            patch(_OUTPUT_TABLE) as mock_table,
        ):
            result = runner.invoke(inspect, ["-m", "test", "-f", "json"], obj={})
            assert result.exit_code == 0, f"Failed: {result.output}"
            mock_json.assert_called_once()
            mock_table.assert_not_called()

    def test_table_format_default(
        self,
        runner: CliRunner,
        mock_inspect_result: MagicMock,
    ) -> None:
        from winml.modelkit.commands.inspect import inspect

        with (
            patch(_INSPECT_MODEL, return_value=mock_inspect_result),
            patch(_OUTPUT_JSON) as mock_json,
            patch(_OUTPUT_TABLE) as mock_table,
        ):
            result = runner.invoke(inspect, ["-m", "test"], obj={})
            assert result.exit_code == 0, f"Failed: {result.output}"
            mock_table.assert_called_once()
            mock_json.assert_not_called()


# =============================================================================
# FLAG COMBINATION TESTS
# =============================================================================


class TestInspectFlagCombinations:
    """Test flag combinations and kwarg passing."""

    def test_all_flags_combine(
        self,
        runner: CliRunner,
        mock_inspect_result: MagicMock,
    ) -> None:
        from winml.modelkit.commands.inspect import inspect

        with (
            patch(_INSPECT_MODEL, return_value=mock_inspect_result),
            patch(_OUTPUT_JSON, return_value="{}"),
            patch(_OUTPUT_TABLE),
        ):
            result = runner.invoke(
                inspect,
                ["-m", "test", "-v", "-H", "-t", "fill-mask", "-f", "json"],
                obj={},
            )
            assert result.exit_code == 0, f"Failed: {result.output}"

    def test_task_override_passed_to_api(
        self,
        runner: CliRunner,
        mock_inspect_result: MagicMock,
    ) -> None:
        from winml.modelkit.commands.inspect import inspect

        with (
            patch(_INSPECT_MODEL, return_value=mock_inspect_result) as mock_api,
            patch(_OUTPUT_TABLE),
        ):
            runner.invoke(inspect, ["-m", "test", "-t", "fill-mask"], obj={})
            mock_api.assert_called_once()
            # inspect_model(model, include_hierarchy=..., task_override=...)
            _, call_kwargs = mock_api.call_args
            assert call_kwargs["task_override"] == "fill-mask"

    def test_hierarchy_flag_passed_to_api(
        self,
        runner: CliRunner,
        mock_inspect_result: MagicMock,
    ) -> None:
        from winml.modelkit.commands.inspect import inspect

        with (
            patch(_INSPECT_MODEL, return_value=mock_inspect_result) as mock_api,
            patch(_OUTPUT_TABLE),
        ):
            runner.invoke(inspect, ["-m", "test", "-H"], obj={})
            mock_api.assert_called_once()
            # inspect_model(model, include_hierarchy=..., task_override=...)
            _, call_kwargs = mock_api.call_args
            assert call_kwargs["include_hierarchy"] is True

    def test_verbose_flag_default_false(
        self,
        runner: CliRunner,
        mock_inspect_result: MagicMock,
    ) -> None:
        from winml.modelkit.commands.inspect import inspect

        with (
            patch(_INSPECT_MODEL, return_value=mock_inspect_result),
            patch(_OUTPUT_TABLE) as mock_table,
        ):
            runner.invoke(inspect, ["-m", "test"], obj={})
            mock_table.assert_called_once()
            # output_table(console, result, verbose=verbose)
            _, call_kwargs = mock_table.call_args
            assert call_kwargs["verbose"] is False


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestInspectErrors:
    """Test error handling for various exception types."""

    def test_model_not_found_error(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.inspect import inspect
        from winml.modelkit.inspect import ModelNotFoundError

        with patch(_INSPECT_MODEL, side_effect=ModelNotFoundError("no-such-model")):
            result = runner.invoke(inspect, ["-m", "no-such-model"], obj={})
            assert result.exit_code != 0
            assert "Model not found" in result.output

    def test_network_error(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.inspect import inspect
        from winml.modelkit.inspect import NetworkError

        with patch(_INSPECT_MODEL, side_effect=NetworkError("connection timed out")):
            result = runner.invoke(inspect, ["-m", "test"], obj={})
            assert result.exit_code != 0
            assert "Network error" in result.output

    def test_inspect_error(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.inspect import inspect
        from winml.modelkit.inspect import InspectError

        with patch(_INSPECT_MODEL, side_effect=InspectError("unexpected failure")):
            result = runner.invoke(inspect, ["-m", "test"], obj={})
            assert result.exit_code != 0
            assert "Inspection error" in result.output

    def test_missing_local_file_shows_path_error(
        self, runner: CliRunner, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Absolute path to a missing file shows a clear local error, not 'Network error'."""
        from winml.modelkit.commands.inspect import inspect

        missing = str(tmp_path / "missing.onnx")
        result = runner.invoke(inspect, ["-m", missing], obj={})
        assert result.exit_code != 0
        assert "does not exist" in result.output
        assert "Network error" not in result.output

    def test_missing_local_relative_path_shows_path_error(self, runner: CliRunner) -> None:
        """Relative path starting with '.' shows a clear local error, not 'Network error'."""
        from winml.modelkit.commands.inspect import inspect

        result = runner.invoke(inspect, ["-m", "./does-not-exist.onnx"], obj={})
        assert result.exit_code != 0
        assert "does not exist" in result.output
        assert "Network error" not in result.output

    def test_bogus_hf_id_shows_model_not_found(self, runner: CliRunner) -> None:
        """RepositoryNotFoundError from HF Hub maps to 'Model not found', not 'Network error'."""
        from huggingface_hub.utils import RepositoryNotFoundError

        from winml.modelkit.commands.inspect import inspect

        with patch(
            "transformers.AutoConfig.from_pretrained",
            side_effect=RepositoryNotFoundError("totally-bogus/does-not-exist"),
        ):
            result = runner.invoke(inspect, ["-m", "totally-bogus/does-not-exist"], obj={})
            assert result.exit_code != 0
            assert "Model not found" in result.output
            assert "Network error" not in result.output
