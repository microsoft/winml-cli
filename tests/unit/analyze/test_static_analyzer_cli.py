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
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.analyze import analyze


@pytest.fixture(autouse=True)
def _mock_rule_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass rule-data validation so CLI tests don't depend on rule artifacts."""
    monkeypatch.setattr(
        "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "winml.modelkit.commands.analyze._discover_runtime_rule_parquet_files",
        lambda: ([Path("runtime_check_rules")], [Path("runtime_check_rules/mock.parquet")]),
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
            "winml.modelkit.commands.analyze._discover_runtime_rule_parquet_files",
            lambda: ([Path("runtime_check_rules")], []),
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
        assert call_kwargs["model_path"] == str(model_file)
        assert call_kwargs["ep"] == "OpenVINOExecutionProvider"
        assert call_kwargs["device"] == "GPU"
        assert call_kwargs["enable_information"] is True


class TestAnalyzeEPDeviceValidation:
    """Test EP + device validation in analyze command."""

    def test_dml_cpu_rejected_with_only_supports(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DML only supports GPU — passing CPU should exit 2 with helpful message."""
        # Override the autouse mock to restore real validation
        monkeypatch.setattr(
            "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
            lambda ep, dev: False,
        )
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "dml", "--device", "CPU"],
        )
        assert result.exit_code == 2
        assert "only supports" in result.output.lower()
        assert "gpu" in result.output.lower()

    def test_cpu_ep_npu_rejected(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CPUExecutionProvider only supports CPU — passing NPU should exit 2."""
        monkeypatch.setattr(
            "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
            lambda ep, dev: False,
        )
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "cpu", "--device", "NPU"],
        )
        assert result.exit_code == 2
        assert "only supports" in result.output.lower()
        assert "cpu" in result.output.lower()

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
        """'cpu' alias should resolve to CPUExecutionProvider."""
        monkeypatch.setattr(
            "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
            lambda ep, dev: False,
        )
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        # CPU EP + GPU → should fail validation with "only supports CPU"
        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "cpu", "--device", "GPU"],
        )
        assert result.exit_code == 2
        assert "cpuexecutionprovider" in result.output.lower()

    def test_ep_alias_dml_resolves(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'dml' alias should resolve to DmlExecutionProvider."""
        monkeypatch.setattr(
            "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
            lambda ep, dev: False,
        )
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        # DML EP + NPU → should fail validation with "only supports GPU"
        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "dml", "--device", "NPU"],
        )
        assert result.exit_code == 2
        assert "dmlexecutionprovider" in result.output.lower()

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_ep_without_device_skips_validation(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """When --device is omitted, EP+device validation should be skipped."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        # dml without --device should not exit 2 on validation
        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "dml"],
        )
        # Should proceed to analysis (not fail on validation)
        assert result.exit_code == 0
        assert mock_instance.analyze.called


class TestQDQNodeDisplayMapping:
    """Tests for QDQ node result mapping in the op progress table.

    QDQ-wrapped ops (e.g. Conv surrounded by DQ/Q nodes) produce pattern IDs
    like 'OP/ai.onnx/Conv (QDQ)'.  The live table keys come from
    metadata.operator_counts which uses bare op types ('Conv').  The
    on_node_result callback must strip the ' (QDQ)' suffix so results are
    attributed to the right row instead of being silently dropped.
    """

    def test_qdq_pattern_id_maps_to_base_op_for_table_key(self) -> None:
        """_display_name + removesuffix(QDQ_SUFFIX) maps QDQ pattern IDs to base
        op types so instance_counts keys match all_op_counts keys."""
        from winml.modelkit.analyze import QDQ_SUFFIX
        from winml.modelkit.commands.analyze import _display_name

        assert _display_name("OP/ai.onnx/Conv (QDQ)").removesuffix(QDQ_SUFFIX) == "Conv"
        assert _display_name("OP/ai.onnx/Add (QDQ)").removesuffix(QDQ_SUFFIX) == "Add"
        assert _display_name("OP/ai.onnx/Pad (QDQ)").removesuffix(QDQ_SUFFIX) == "Pad"
        assert (
            _display_name("OP/ai.onnx/DequantizeLinear").removesuffix(QDQ_SUFFIX)
            == "DequantizeLinear"
        )
        assert _display_name("OP/ai.onnx/Reshape").removesuffix(QDQ_SUFFIX) == "Reshape"

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
                    pr.result.classification.value = "supported"
                    on_node_result(pr)
                for _ in range(4):
                    pr = Mock()
                    pr.pattern_id = "OP/ai.onnx/DequantizeLinear"
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
        # ep_instance_counts['QNNExecutionProvider']['Conv'] must be populated
        # (not 'Conv (QDQ)') so the Conv row shows counts instead of '...'.
        assert mock_summary.called
        qnn_counts = captured_ep_counts.get("QNNExecutionProvider", {})
        assert "Conv" in qnn_counts, "Conv (QDQ) results must be stored under 'Conv'"
        assert "Conv (QDQ)" not in qnn_counts, "QDQ suffix must be stripped"
        assert qnn_counts["Conv"] == {"supported": 2}
        assert qnn_counts["DequantizeLinear"] == {"supported": 4}
