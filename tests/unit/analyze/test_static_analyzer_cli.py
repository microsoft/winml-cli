# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for analyze CLI command.

Tests verify:
- Command registration and discovery
- Argument validation
- Option parsing
- Exit codes
- Output formats (stdout/file)
- Error handling
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.analyze import analyze


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_analyzer_result() -> Mock:
    """Create a mock AnalysisOutput result."""
    mock_result = Mock()
    mock_result.is_fully_supported.return_value = True
    mock_result.get_unsupported_operators.return_value = []
    mock_result.to_json.return_value = json.dumps(
        {
            "analyzer_version": "0.1.0",
            "analysis_timestamp": "2025-12-05T12:00:00",
            "metadata": {
                "model_path": "test.onnx",
                "opset_version": 13,
                "total_operators": 10,
                "operator_counts": {"Conv": 5, "Add": 3, "ReLU": 2},
                "unique_operator_types": 3,
            },
            "results": [],
        }
    )
    return mock_result


@pytest.fixture
def mock_analyzer_partial_support() -> Mock:
    """Create a mock result with partial support."""
    mock_result = Mock()
    mock_result.is_fully_supported.return_value = False
    mock_result.get_unsupported_operators.return_value = ["Conv", "Gemm", "Add"]
    mock_result.to_json.return_value = json.dumps(
        {
            "analyzer_version": "0.1.0",
            "analysis_timestamp": "2025-12-05T12:00:00",
            "metadata": {
                "model_path": "test.onnx",
                "opset_version": 13,
                "total_operators": 6,
                "operator_counts": {"Conv": 2, "Gemm": 2, "Add": 2},
                "unique_operator_types": 3,
            },
            "results": [],
        }
    )
    return mock_result


class TestAnalyzeCommand:
    """Test analyze command."""

    def test_command_exists(self, runner: CliRunner) -> None:
        """Test that analyze command is registered."""
        result = runner.invoke(analyze, ["--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output.lower()


class TestAnalyzeCommandArguments:
    """Test analyze command argument validation."""

    def test_requires_model_argument(self, runner: CliRunner) -> None:
        """Test that --model argument is required."""
        result = runner.invoke(analyze, [])
        assert result.exit_code != 0
        assert "model" in result.output.lower() or "missing" in result.output.lower()

    def test_ep_argument_optional(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that --ep argument is optional (will analyze all EPs if not provided)."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        # Command without --ep should not fail due to missing argument
        # It may fail for other reasons (invalid model), but not missing --ep
        result = runner.invoke(analyze, ["--model", str(model_file)])
        # Should not complain about missing --ep argument
        assert "ep" not in result.output.lower() or "missing" not in result.output.lower()

    def test_device_argument_optional(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that --device argument is optional (will use default NPU if not provided)."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        # Command without --device should not fail due to missing argument
        result = runner.invoke(
            analyze, ["--model", str(model_file), "--ep", "QNNExecutionProvider"]
        )
        # Should not complain about missing --device argument
        assert "device" not in result.output.lower() or "missing" not in result.output.lower()

    def test_validates_ep_choice(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that --ep only accepts valid execution providers."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "InvalidEP",
                "--device",
                "NPU",
            ],
        )
        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "choice" in result.output.lower()

    def test_validates_device_choice(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that --device only accepts valid device types."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "INVALID",
            ],
        )
        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "choice" in result.output.lower()

    def test_model_file_must_exist(self, runner: CliRunner) -> None:
        """Test that model file path must exist."""
        result = runner.invoke(
            analyze,
            [
                "--model",
                "nonexistent.onnx",
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
            ],
        )
        # Click should catch this with path validation
        assert result.exit_code != 0


class TestAnalyzeCommandExecution:
    """Test analyze command execution with mocked analyzer."""

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_successful_analysis_exits_zero(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test that successful analysis exits with code 0."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        # Setup mock
        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
            ],
        )

        assert result.exit_code == 0
        mock_instance.analyze.assert_called_once()

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_partial_support_exits_one(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_partial_support: Mock,
    ) -> None:
        """Test that partial support exits with code 1."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        # Setup mock
        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_partial_support
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
            ],
        )

        assert result.exit_code == 1

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_analysis_failure_exits_two(
        self, mock_analyzer_class: MagicMock, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that analysis failure exits with code 2."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        # Setup mock to raise exception
        mock_instance = Mock()
        mock_instance.analyze.side_effect = RuntimeError("Analysis failed")
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
            ],
        )

        assert result.exit_code == 2


class TestAnalyzeCommandOptions:
    """Test analyze command options."""

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_information_flag_enables_recommendations(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test that --information flag is passed to analyzer."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
                "--information",
            ],
        )

        # Verify analyze was called with enable_information=True
        call_kwargs = mock_instance.analyze.call_args[1]
        assert call_kwargs["enable_information"] is True

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_no_information_flag_disables_recommendations(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test that --no-information flag is passed to analyzer."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
                "--no-information",
            ],
        )

        # Verify analyze was called with enable_information=False
        call_kwargs = mock_instance.analyze.call_args[1]
        assert call_kwargs["enable_information"] is False

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_verbose_flag_enables_debug_logging(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test that --verbose flag enables debug output."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
                "--verbose",
            ],
        )

        # Should complete successfully
        assert result.exit_code == 0

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_quiet_flag_suppresses_warnings(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test that --quiet flag suppresses non-error output."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
                "--quiet",
            ],
        )

        assert result.exit_code == 0


class TestAnalyzeCommandOutput:
    """Test analyze command output formats."""

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_output_to_stdout_by_default(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test that results are written to stdout by default."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
            ],
        )

        # Output should contain formatted report (not JSON by default)
        assert result.exit_code == 0
        # Check for report title or analysis summary in output
        assert "analysis" in result.output.lower() or "model" in result.output.lower()

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_output_to_file_with_option(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test that --output saves results to file."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")
        output_file = tmp_path / "results.json"

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify file contains valid JSON
        content = json.loads(output_file.read_text())
        assert "analyzer_version" in content

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_output_file_not_written_on_error(
        self, mock_analyzer_class: MagicMock, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that output file is not created when analysis fails."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")
        output_file = tmp_path / "results.json"

        mock_instance = Mock()
        mock_instance.analyze.side_effect = RuntimeError("Analysis failed")
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 2
        assert not output_file.exists()


class TestAnalyzeCommandIntegration:
    """Integration tests for analyze command."""

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_all_supported_eps(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test all supported execution providers."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        eps = ["QNNExecutionProvider", "OpenVINOExecutionProvider", "VitisAIExecutionProvider"]

        for ep in eps:
            result = runner.invoke(
                analyze,
                [
                    "--model",
                    str(model_file),
                    "--ep",
                    ep,
                    "--device",
                    "NPU",
                ],
            )
            assert result.exit_code == 0, f"Failed for EP: {ep}"

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_all_supported_devices(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test all supported device types."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        devices = ["CPU", "GPU", "NPU"]

        for device in devices:
            result = runner.invoke(
                analyze,
                [
                    "--model",
                    str(model_file),
                    "--ep",
                    "QNNExecutionProvider",
                    "--device",
                    device,
                ],
            )
            assert result.exit_code == 0, f"Failed for device: {device}"

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_analyze_called_with_correct_parameters(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test that analyze() is called with correct parameters."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "OpenVINOExecutionProvider",
                "--device",
                "GPU",
                "--information",
            ],
        )

        # Verify analyze was called with correct parameters
        mock_instance.analyze.assert_called_once()
        call_kwargs = mock_instance.analyze.call_args[1]
        assert call_kwargs["model_path"] == model_file
        assert call_kwargs["ep"] == "OpenVINOExecutionProvider"
        assert call_kwargs["device"] == "GPU"
        assert call_kwargs["enable_information"] is True
