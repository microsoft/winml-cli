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
from rich.console import Console


if TYPE_CHECKING:
    from pathlib import Path

from winml.modelkit.commands.analyze import analyze


# Fixed simulated local availability derived from `ort.get_ep_devices()` after
# WinML registration and `.AUTO` filtering.
SIMULATED_LOCAL_EP_DEVICE_PAIRS = [
    ("CPUExecutionProvider", "CPU"),
    ("DmlExecutionProvider", "GPU"),
    ("OpenVINOExecutionProvider", "NPU"),
    ("OpenVINOExecutionProvider", "CPU"),
    ("NvTensorRTRTXExecutionProvider", "GPU"),
]


@pytest.fixture(autouse=True)
def _mock_local_ep_device_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix local EP/device availability for deterministic CLI behavior tests."""
    monkeypatch.setattr(
        "winml.modelkit.commands.analyze._get_local_ep_device_pairs",
        lambda: list(SIMULATED_LOCAL_EP_DEVICE_PAIRS),
    )
    # Defensive mocks: the analyze command derives devices/eps from local_pairs
    # in auto mode, but other code paths (and any future code) may still call
    # these helpers — keep them consistent with the simulated local matrix so
    # tests stay environment-independent.
    simulated_devices = tuple(sorted({d for _, d in SIMULATED_LOCAL_EP_DEVICE_PAIRS}))
    # Sort eps so iteration order is deterministic across runs (the real helper
    # returns a frozenset whose iteration depends on PYTHONHASHSEED).
    simulated_eps = tuple(sorted({e for e, _ in SIMULATED_LOCAL_EP_DEVICE_PAIRS}))
    monkeypatch.setattr(
        "winml.modelkit.sysinfo.device._get_available_devices",
        lambda: simulated_devices,
    )
    monkeypatch.setattr(
        "winml.modelkit.sysinfo.device._get_available_eps",
        lambda: simulated_eps,
    )


@pytest.fixture(autouse=True)
def _mock_any_runtime_rule_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide deterministic non-empty runtime-rule availability for CLI tests.

    Most tests validate EP/device selection logic and should not depend on
    machine/environment-specific parquet assets being present on disk.
    """
    monkeypatch.setattr(
        "winml.modelkit.analyze.utils.ep_utils.has_any_rule_data",
        lambda: True,
    )


@pytest.fixture(autouse=True)
def _mock_has_rule_data_for_ep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix rule-data availability so tests do not depend on CI assets.

    The analyze command gates execution on has_rule_data_for_ep before it
    invokes ONNXStaticAnalyzer. Keep this matrix deterministic in unit tests.
    """
    simulated_rule_pairs = {
        ("OpenVINOExecutionProvider", "NPU"),
        ("OpenVINOExecutionProvider", "GPU"),
        ("OpenVINOExecutionProvider", "CPU"),
        ("QNNExecutionProvider", "NPU"),
        ("NvTensorRTRTXExecutionProvider", "GPU"),
    }

    monkeypatch.setattr(
        "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
        lambda ep_name, device_name: (ep_name, str(device_name).upper()) in simulated_rule_pairs,
    )


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_analyzer_result() -> Mock:
    """Create a mock AnalysisResult (returned by ONNXStaticAnalyzer.analyze).

    The command accesses ``result.output.results`` (list of EPSupport) for
    Rich live display, ``result.is_fully_supported()`` for exit code, and
    ``result.to_json()`` for JSON output.
    """
    mock_result = Mock()
    mock_result.is_fully_supported.return_value = True
    mock_result.get_unsupported_operators.return_value = []
    mock_result.output.results = []  # empty EP results list (iterable)
    mock_result.to_json.return_value = json.dumps(
        {
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
    mock_result.output.results = []  # empty EP results list (iterable)
    mock_result.to_json.return_value = json.dumps(
        {
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

    def test_unknown_ep_with_device_exits_two(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test that unknown EP + explicit device exits with code 2."""
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
        assert result.exit_code == 2

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

    def test_missing_runtime_rule_parquet_exits_two(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no parquet is found in search dirs, analyze should fail fast."""
        monkeypatch.setattr(
            "winml.modelkit.analyze.utils.ep_utils.has_any_rule_data",
            lambda: False,
        )

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
                "NPU",
            ],
        )

        assert result.exit_code == 2
        assert "no runtime rule parquet files were found" in result.output.lower()
        assert "reinstall" in result.output.lower()


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
    def test_debug_flag_enables_runtime_debug(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test that --debug writes runtime debug summary JSON near the model."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")
        output_file = tmp_path / "results.json"
        debug_rules_dir = tmp_path / "rules_debug"
        debug_rules_subdir = debug_rules_dir / "QNNExecutionProvider_NPU"
        debug_rules_subdir.mkdir(parents=True, exist_ok=True)
        (debug_rules_subdir / "placeholder.parquet").write_bytes(b"dummy")
        monkeypatch.setenv("WINMLCLI_RULES_DIR_FOR_DEBUG", str(debug_rules_dir))

        mock_analyzer_result.to_json.return_value = json.dumps(
            {
                "analysis_timestamp": "2025-12-05T12:00:00",
                "metadata": {
                    "model_path": "test.onnx",
                    "opset_version": 13,
                    "total_operators": 1,
                    "operator_counts": {"Conv": 1},
                    "unique_operator_types": 1,
                },
                "results": [
                    {
                        "ep_type": "QNNExecutionProvider",
                        "device_type": "NPU",
                        "runtime_debug_details_summary": {
                            "supported": {
                                "node_conv": {
                                    "case_indices": ["case_7"],
                                    "table_path": "rules/conv.parquet",
                                    "table_file": "conv.parquet",
                                }
                            },
                            "partial": {},
                            "unsupported": {},
                        },
                    }
                ],
            }
        )

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
                "--debug",
                "--output",
                str(output_file),
            ],
        )

        # Should complete successfully
        assert result.exit_code == 0
        call_kwargs = mock_instance.analyze.call_args.kwargs
        assert call_kwargs["for_debug"] is True

        debug_file = tmp_path / "test.analyze.QNNExecutionProvider.NPU.debug.json"
        assert debug_file.exists()

        debug_content = json.loads(debug_file.read_text())
        assert set(debug_content.keys()) == {"supported", "partial", "unsupported"}
        assert debug_content["supported"]["node_conv"] == {
            "case_indices": ["case_7"],
            "table_path": "rules/conv.parquet",
            "table_file": "conv.parquet",
        }

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_verbose_flag_no_longer_enables_runtime_debug(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_analyzer_result: Mock,
    ) -> None:
        """--verbose should not enable runtime debug without --debug."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")
        output_file = tmp_path / "results.json"
        debug_rules_dir = tmp_path / "rules_debug"
        debug_rules_subdir = debug_rules_dir / "QNNExecutionProvider_NPU"
        debug_rules_subdir.mkdir(parents=True, exist_ok=True)
        (debug_rules_subdir / "placeholder.parquet").write_bytes(b"dummy")
        monkeypatch.setenv("WINMLCLI_RULES_DIR_FOR_DEBUG", str(debug_rules_dir))

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
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_instance.analyze.call_args.kwargs
        assert call_kwargs["for_debug"] is False

        debug_file = tmp_path / "test.analyze.QNNExecutionProvider.NPU.debug.json"
        assert not debug_file.exists()

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_debug_flag_requires_debug_env_with_parquet(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--debug should fail fast when debug env var is missing."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")
        monkeypatch.delenv("WINMLCLI_RULES_DIR_FOR_DEBUG", raising=False)

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
                "--debug",
            ],
        )

        assert result.exit_code == 2
        assert "--debug requires" in result.output.lower()
        assert "winmlcli_rules_dir_for_debug" in result.output.lower()
        assert not mock_analyzer_class.called

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_debug_flag_requires_second_level_parquet(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--debug should fail when debug dir has no */*.parquet files."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        debug_rules_dir = tmp_path / "rules_debug"
        debug_rules_dir.mkdir(parents=True, exist_ok=True)
        # Root-level parquet should not satisfy */*.parquet requirement.
        (debug_rules_dir / "root.parquet").write_bytes(b"dummy")
        monkeypatch.setenv("WINMLCLI_RULES_DIR_FOR_DEBUG", str(debug_rules_dir))

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
                "--debug",
            ],
        )

        assert result.exit_code == 2
        assert "--debug requires" in result.output.lower()
        assert "winmlcli_rules_dir_for_debug" in result.output.lower()
        assert not mock_analyzer_class.called

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
        assert "metadata" in content

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

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_output_creates_parent_dirs(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test that --output creates missing parent directories."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")
        output_file = tmp_path / "nested" / "dir" / "results.json"

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        assert not output_file.parent.exists()

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

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_optim_config_to_file(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test that --optim-config saves optimization config to file."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")
        config_file = tmp_path / "optim.json"

        mock_analyzer_result.get_optimization_config.return_value.to_dict.return_value = {
            "gelu_fusion": True,
        }
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
                "--optim-config",
                str(config_file),
            ],
        )

        assert result.exit_code == 0
        assert config_file.exists()
        content = json.loads(config_file.read_text())
        assert content == {"gelu_fusion": True}

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_optim_config_creates_parent_dirs(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test that --optim-config creates missing parent directories."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")
        config_file = tmp_path / "nested" / "dir" / "optim.json"

        mock_analyzer_result.get_optimization_config.return_value.to_dict.return_value = {
            "gelu_fusion": True,
        }
        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        assert not config_file.parent.exists()

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "QNNExecutionProvider",
                "--device",
                "NPU",
                "--optim-config",
                str(config_file),
            ],
        )

        assert result.exit_code == 0
        assert config_file.exists()


class TestAnalyzeCommandIntegration:
    """Integration tests for analyze command."""

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    @patch(
        "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
        return_value=True,
    )
    def test_all_supported_eps(
        self,
        _mock_has_rule: Mock,
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

        eps = ["QNNExecutionProvider", "OpenVINOExecutionProvider"]

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
    @patch(
        "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
        return_value=True,
    )
    def test_all_supported_devices(
        self,
        _mock_has_rule: Mock,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Test all supported device types (with rule data validation bypassed)."""
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
                    "OpenVINOExecutionProvider",
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
        assert call_kwargs["model_path"] == str(model_file)
        assert call_kwargs["ep"] == "OpenVINOExecutionProvider"
        assert call_kwargs["device"] == "GPU"
        assert call_kwargs["enable_information"] is True
        assert call_kwargs["for_debug"] is False


class TestAnalyzeEPDeviceValidation:
    """Test EP + device validation in analyze command."""

    def test_dml_cpu_rejected_with_only_supports(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DML + CPU should be rejected: DML does not support CPU per EP_SUPPORTED_DEVICES."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "dml", "--device", "CPU"],
        )
        assert result.exit_code == 2
        assert "no ep/device combination matched" in result.output.lower()

    def test_cpu_ep_npu_rejected(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CPU EP + NPU should be rejected: CPU EP does not support NPU."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "cpu", "--device", "NPU"],
        )
        assert result.exit_code == 2
        assert "no ep/device combination matched" in result.output.lower()

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    @patch(
        "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
        return_value=True,
    )
    def test_valid_combo_passes_validation(
        self,
        _mock_has_rule: Mock,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Valid EP+device combo should proceed to analysis."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "qnn", "--device", "NPU"],
        )
        assert result.exit_code == 0
        mock_instance.analyze.assert_called_once()

    def test_ep_alias_cpu_resolves(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'cpu' alias resolves to CPUExecutionProvider, which doesn't support GPU."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "cpu", "--device", "GPU"],
        )
        assert result.exit_code == 2
        assert "no ep/device combination matched" in result.output.lower()

    def test_ep_alias_dml_resolves(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'dml' alias resolves to DmlExecutionProvider, which doesn't support NPU."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "dml", "--device", "NPU"],
        )
        assert result.exit_code == 2
        assert "no ep/device combination matched" in result.output.lower()

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_ep_without_device_auto_resolves_local_device(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """With --device auto and a specific EP, analyze runs on the matching local device.

        Rule-data availability no longer gates execution — the per-pair OP CHECK
        section just renders an "Op check skipped — no rule data" row inline.
        """
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        # dml is locally available on GPU per the fixture; auto picks GPU.
        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "dml"],
        )
        assert result.exit_code == 0
        mock_instance.analyze.assert_called_once()
        call_kwargs = mock_instance.analyze.call_args.kwargs
        assert call_kwargs["ep"] == "DmlExecutionProvider"
        assert call_kwargs["device"] == "GPU"

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_ep_without_device_auto_run_unknown_op_executes_no_rule_data_pair(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """--run-unknown-op should execute local parsed pair even without rule data."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "dml", "--run-unknown-op"],
        )
        assert result.exit_code == 0
        mock_instance.analyze.assert_called_once()

        call_kwargs = mock_instance.analyze.call_args.kwargs
        assert call_kwargs["ep"] == "DmlExecutionProvider"
        assert call_kwargs["device"] == "GPU"


class TestAnalyzeEPDeviceSelectionMatrix:
    """Matrix tests for EP/device resolution with fixed local availability."""

    @pytest.mark.parametrize(
        ("ep_arg", "device_arg", "expect_exit", "expect_calls", "expect_error"),
        [
            # Both auto: filter to local_pairs. Output sorted by EP_SUPPORTED_DEVICES.
            (
                None,
                None,
                0,
                [
                    ("NvTensorRTRTXExecutionProvider", "GPU"),
                    ("OpenVINOExecutionProvider", "NPU"),
                    ("OpenVINOExecutionProvider", "CPU"),
                    ("DmlExecutionProvider", "GPU"),
                    ("CPUExecutionProvider", "CPU"),
                ],
                None,
            ),
            # ep=auto, device=gpu: warn about non-local but run all eps that support GPU.
            (
                None,
                "gpu",
                0,
                [
                    ("NvTensorRTRTXExecutionProvider", "GPU"),
                    ("OpenVINOExecutionProvider", "GPU"),
                    ("DmlExecutionProvider", "GPU"),
                ],
                None,
            ),
            # ep=openvino, device=auto: warn about non-local pairs, run all 3.
            (
                "openvino",
                None,
                0,
                [
                    ("OpenVINOExecutionProvider", "NPU"),
                    ("OpenVINOExecutionProvider", "GPU"),
                    ("OpenVINOExecutionProvider", "CPU"),
                ],
                None,
            ),
            # ep=qnn, device=auto: QNN is not local, but we warn (not filter) and run.
            (
                "qnn",
                None,
                0,
                [
                    ("QNNExecutionProvider", "NPU"),
                    ("QNNExecutionProvider", "GPU"),
                ],
                None,
            ),
            (
                "qnn",
                "all",
                0,
                [
                    ("QNNExecutionProvider", "NPU"),
                    ("QNNExecutionProvider", "GPU"),
                ],
                None,
            ),
            ("openvino", "gpu", 0, [("OpenVINOExecutionProvider", "GPU")], None),
            # ep=all, device=all: every (ep, device) combo allowed by EP_SUPPORTED_DEVICES.
            (
                "all",
                "all",
                0,
                [
                    ("NvTensorRTRTXExecutionProvider", "GPU"),
                    ("CUDAExecutionProvider", "GPU"),
                    ("MIGraphXExecutionProvider", "GPU"),
                    ("QNNExecutionProvider", "NPU"),
                    ("QNNExecutionProvider", "GPU"),
                    ("OpenVINOExecutionProvider", "NPU"),
                    ("OpenVINOExecutionProvider", "GPU"),
                    ("OpenVINOExecutionProvider", "CPU"),
                    ("DmlExecutionProvider", "GPU"),
                    ("CPUExecutionProvider", "CPU"),
                    ("VitisAIExecutionProvider", "NPU"),
                ],
                None,
            ),
        ],
        ids=[
            "empty-empty",
            "empty-gpu",
            "openvino-empty",
            "qnn-empty",
            "qnn-all",
            "openvino-gpu",
            "all-all",
        ],
    )
    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_selection_matrix(
        self,
        mock_analyzer_class: MagicMock,
        ep_arg: str | None,
        device_arg: str | None,
        expect_exit: int,
        expect_calls: list[tuple[str, str]],
        expect_error: str | None,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Assert execute targets selected from requested EP/device pair."""
        matrix_rule_pairs = {
            ("OpenVINOExecutionProvider", "NPU"),
            ("OpenVINOExecutionProvider", "CPU"),
            ("OpenVINOExecutionProvider", "GPU"),
            ("NvTensorRTRTXExecutionProvider", "GPU"),
            ("QNNExecutionProvider", "NPU"),
            ("QNNExecutionProvider", "GPU"),
        }
        monkeypatch.setattr(
            "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
            lambda ep_name, device_name: (ep_name, device_name) in matrix_rule_pairs,
        )

        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        args = ["--model", str(model_file)]
        if ep_arg is not None:
            args.extend(["--ep", ep_arg])
        if device_arg is not None:
            args.extend(["--device", device_arg])

        result = runner.invoke(analyze, args)
        assert result.exit_code == expect_exit

        if expect_exit == 0:
            assert mock_instance.analyze.call_count == len(expect_calls)
            actual_calls = [
                (call.kwargs["ep"], call.kwargs["device"])
                for call in mock_instance.analyze.call_args_list
            ]
            assert actual_calls == expect_calls
        else:
            assert not mock_instance.analyze.called
            assert expect_error is not None
            assert expect_error in result.output.lower()

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_no_rule_data_pair_runs_with_inline_skip_marker(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """A pair without rule data still runs — OP CHECK renders 'skipped' inline."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "dml", "--device", "gpu"],
        )
        assert result.exit_code == 0
        mock_instance.analyze.assert_called_once()
        call_kwargs = mock_instance.analyze.call_args.kwargs
        assert call_kwargs["ep"] == "DmlExecutionProvider"
        assert call_kwargs["device"] == "GPU"

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_qnn_auto_warns_about_non_local_pairs(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """qnn + auto device: QNN isn't locally supported but we warn (not error) and run."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(analyze, ["--model", str(model_file), "--ep", "qnn"])
        assert result.exit_code == 0
        assert "not available on this machine" in result.output.lower()
        actual_calls = [
            (call.kwargs["ep"], call.kwargs["device"])
            for call in mock_instance.analyze.call_args_list
        ]
        assert actual_calls == [
            ("QNNExecutionProvider", "NPU"),
            ("QNNExecutionProvider", "GPU"),
        ]

    @patch(
        "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
        return_value=False,
    )
    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_auto_specific_device_run_unknown_op_executes_local_pairs_without_rule_data(
        self,
        mock_analyzer_class: MagicMock,
        _mock_has_rule: Mock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """ep=auto + specific device should run all locally-eligible (ep, device) pairs.

        With ep=auto and device specified, no local filter is applied — pairs the
        local machine doesn't support are kept (a warning is emitted) and analysis
        runs for each. has_rule_data_for_ep returning False here only affects
        per-pair OP CHECK rendering (op-check-skipped), not which pairs run.
        """
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--device", "gpu", "--run-unknown-op"],
        )
        assert result.exit_code == 0

        actual_calls = [
            (call.kwargs["ep"], call.kwargs["device"])
            for call in mock_instance.analyze.call_args_list
        ]
        assert actual_calls == [
            ("NvTensorRTRTXExecutionProvider", "GPU"),
            ("OpenVINOExecutionProvider", "GPU"),
            ("DmlExecutionProvider", "GPU"),
        ]


class TestQDQNodeDisplayMapping:
    """Tests for QDQ node result mapping in the op progress table.

    QDQ-wrapped ops (e.g. Conv surrounded by DQ/Q nodes) produce pattern IDs
    like 'OP/ai.onnx/Conv (QDQ)'.  The live table keys come from
    metadata.operator_counts which uses bare op types ('Conv').  The
    on_node_result callback must strip the ' (QDQ)' suffix so results are
    attributed to the right row instead of being silently dropped.
    """

    def test_qdq_pattern_id_maps_to_base_op_for_table_key(self) -> None:
        """_display_name maps QDQ-suffixed and EP-suffixed pattern IDs to base
        op types so instance_counts keys match all_op_counts keys."""
        from winml.modelkit.commands.analyze import _display_name

        # QDQ suffix
        assert _display_name("OP/ai.onnx/Conv (QDQ)") == "Conv"
        assert _display_name("OP/ai.onnx/Add (QDQ)") == "Add"
        assert _display_name("OP/ai.onnx/Pad (QDQ)") == "Pad"
        # No suffix
        assert _display_name("OP/ai.onnx/DequantizeLinear") == "DequantizeLinear"
        assert _display_name("OP/ai.onnx/Reshape") == "Reshape"
        # EP-prefix suffix from EPContextNodeChecker
        assert _display_name("OP/com.microsoft/EPContext (QNN)") == "EPContext"
        assert _display_name("OP/com.microsoft/EPContext (Dml)") == "EPContext"

    @patch("winml.modelkit.commands.analyze.Live")
    @patch("winml.modelkit.commands.analyze.Console")
    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_qdq_wrapped_ops_tracked_under_base_type(
        self,
        mock_analyzer_class: MagicMock,
        mock_console_class: MagicMock,
        mock_live_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """on_node_result must map 'Conv (QDQ)' → 'Conv' so the table row
        shows support counts instead of '...'."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        # Accumulate per-EP instance counts written by on_node_result so we
        # can assert that QDQ-wrapped ops land under the base op type key.
        captured_ep_counts: dict = {}

        mock_console = MagicMock()
        mock_console_class.return_value = mock_console

        ep_support_mock = Mock()
        ep_support_mock.ep_type = "QNNExecutionProvider"
        ep_support_mock.classification = {}
        ep_support_mock.information = []
        mock_analyzer_result.output.results = [ep_support_mock]

        def invoke_callbacks(**kwargs):
            on_ep_start = kwargs.get("on_ep_start")
            on_node_result = kwargs.get("on_node_result")
            if on_ep_start:
                on_ep_start("QNNExecutionProvider", {"Conv": 2, "DequantizeLinear": 4})
            if on_node_result:
                for _ in range(2):
                    pr = Mock()
                    pr.pattern_id = "OP/ai.onnx/Conv (QDQ)"
                    pr.result.compile = True
                    pr.result.run = True
                    pr.result.no_data = False
                    pr.result.classification.value = "supported"
                    on_node_result(pr)
                for _ in range(4):
                    pr = Mock()
                    pr.pattern_id = "OP/ai.onnx/DequantizeLinear"
                    pr.result.compile = True
                    pr.result.run = True
                    pr.result.no_data = False
                    pr.result.classification.value = "supported"
                    on_node_result(pr)
            # Capture the instance_counts via _render_analysis_summary call args
            return mock_analyzer_result

        mock_instance = Mock()
        mock_instance.analyze.side_effect = invoke_callbacks
        mock_analyzer_class.return_value = mock_instance

        # Intercept _render_analysis_summary to capture ep_instance_counts
        with patch("winml.modelkit.commands.analyze._render_analysis_summary") as mock_summary:
            result = runner.invoke(
                analyze,
                ["--model", str(model_file), "--ep", "QNNExecutionProvider", "--device", "NPU"],
            )
            if mock_summary.called:
                captured_ep_counts = mock_summary.call_args[0][2]  # 3rd positional arg

        assert result.exit_code == 0
        # After the fix, 'Conv (QDQ)' is keyed as 'Conv' in instance_counts.
        # ep_instance_counts[("QNNExecutionProvider", "NPU")]['Conv'] must be populated
        # (not 'Conv (QDQ)') so the Conv row shows counts instead of '...'.
        assert mock_summary.called
        qnn_counts = captured_ep_counts.get(("QNNExecutionProvider", "NPU"), {})
        assert "Conv" in qnn_counts, "Conv (QDQ) results must be stored under 'Conv'"
        assert "Conv (QDQ)" not in qnn_counts, "QDQ suffix must be stripped"
        assert qnn_counts["Conv"] == {"supported": 2}
        assert qnn_counts["DequantizeLinear"] == {"supported": 4}


class TestAnalyzeSummaryRendering:
    """Summary rendering behavior for no-rule-data fallback cases."""

    def test_no_rule_data_with_instance_counts_renders_op_summary(self) -> None:
        """When unknown-op probing produced counts, summary should not show skip message."""
        from winml.modelkit.commands.analyze import _render_analysis_summary

        console = Console(record=True, force_terminal=False, width=120)

        ep_support = Mock()
        ep_support.ep_type = "DmlExecutionProvider"
        ep_support.device_type = "GPU"
        ep_support.classification = {}
        ep_support.information = []

        _render_analysis_summary(
            console,
            [ep_support],
            ep_instance_counts={("DmlExecutionProvider", "GPU"): {"Conv": {"supported": 2}}},
            ep_patterns={},
            ep="DmlExecutionProvider",
            device="GPU",
            no_data_eps={("DmlExecutionProvider", "GPU")},
        )

        output = console.export_text()
        assert "DmlExecutionProvider (GPU)" in output
        assert "2/0/0" in output
        assert "Op check skipped" not in output


# ---------------------------------------------------------------------------
# --format json
# ---------------------------------------------------------------------------


class TestAnalyzeFormatJson:
    """Test --format json produces structured JSON to stdout."""

    def test_help_shows_format_option(self, runner: CliRunner) -> None:
        """--format flag must appear in --help output."""
        result = runner.invoke(analyze, ["--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "json" in result.output

    def test_invalid_format_rejected(self, runner: CliRunner, tmp_path: Path) -> None:
        """An invalid --format value must be rejected by Click."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--format", "xml"],
        )
        assert result.exit_code != 0

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_format_json_emits_valid_json(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """--format json output must contain parseable JSON."""
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
                "--format",
                "json",
                "--quiet",
            ],
        )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "metadata" in parsed
        assert parsed["metadata"]["model_path"] == "test.onnx"

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_format_json_emits_on_partial_support(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_partial_support: Mock,
    ) -> None:
        """--format json must still emit JSON when exit code is 1 (partial support)."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

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
                "--format",
                "json",
                "--quiet",
            ],
        )

        assert result.exit_code == 1
        parsed = json.loads(result.output)
        assert "metadata" in parsed

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_format_json_with_output_file(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """--format json + --output should emit JSON to stdout AND save file."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")
        output_file = tmp_path / "result.json"

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
                "--format",
                "json",
                "--output",
                str(output_file),
                "--quiet",
            ],
        )

        assert result.exit_code == 0
        # stdout has JSON
        parsed = json.loads(result.output)
        assert "metadata" in parsed
        # File also has JSON
        assert output_file.exists()
        file_data = json.loads(output_file.read_text())
        assert "metadata" in file_data
