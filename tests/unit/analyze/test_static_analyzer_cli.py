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
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner
from rich.console import Console
from rich.progress import Progress


if TYPE_CHECKING:
    from pathlib import Path

from winml.modelkit.commands.analyze import _get_local_ep_device_pairs, analyze
from winml.modelkit.utils.constants import EP_SUPPORTED_DEVICES, normalize_ep_name


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
    # Keep the legacy sysinfo shims aligned with the simulated matrix for any
    # indirect callers that still read them.
    device_ep_map: dict[str, list[str]] = {}
    for _ep, _device in SIMULATED_LOCAL_EP_DEVICE_PAIRS:
        device_ep_map.setdefault(_device.lower(), []).append(_ep)
    simulated_device_ep_map = {d: tuple(eps) for d, eps in device_ep_map.items()}
    monkeypatch.setattr(
        "winml.modelkit.sysinfo.device._get_device_ep_map_from_ort",
        lambda: simulated_device_ep_map,
    )

    local_pairs_by_device = {
        device_name.lower(): [
            ep_name
            for ep_name in EP_SUPPORTED_DEVICES
            if (ep_name, device_name.upper()) in SIMULATED_LOCAL_EP_DEVICE_PAIRS
        ]
        for device_name in {"cpu", "gpu", "npu"}
    }

    def _resolve_target(target):
        requested_ep = target.ep
        if target.device != "auto":
            return target

        if requested_ep == "auto":
            for device_name in ("npu", "gpu", "cpu"):
                eps = local_pairs_by_device.get(device_name, [])
                if eps:
                    return type(target)(ep=eps[0], device=device_name, source=target.source)
            raise ValueError("No execution provider is available on this system.")

        canonical_ep = normalize_ep_name(requested_ep)
        for device_name in ("npu", "gpu", "cpu"):
            if canonical_ep in local_pairs_by_device.get(device_name, []):
                return type(target)(ep=canonical_ep, device=device_name, source=target.source)
        raise ValueError(f"{requested_ep} is not available on this system.")

    monkeypatch.setattr("winml.modelkit.session.resolve_device", _resolve_target)
    monkeypatch.setattr(
        "winml.modelkit.session.available_eps_for_device",
        lambda device_name: list(local_pairs_by_device.get(str(device_name).lower(), [])),
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
        ("TensorrtExecutionProvider", "GPU"),
    }

    monkeypatch.setattr(
        "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
        lambda ep_name, device_name: (ep_name, str(device_name).upper()) in simulated_rule_pairs,
    )


def test_local_ep_device_pairs_use_cli_device_casing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local ORT devices use the same casing as the analyze execution matrix."""
    import onnxruntime as ort

    registered = SimpleNamespace(
        ep_name="DmlExecutionProvider",
        device=SimpleNamespace(type=ort.OrtHardwareDeviceType.GPU),
    )
    monkeypatch.setattr(
        "winml.modelkit.winml.get_registered_ep_devices",
        lambda: (registered,),
    )

    assert _get_local_ep_device_pairs() == [("DmlExecutionProvider", "GPU")]


def test_local_ep_device_pair_query_propagates_registration_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration failures must not be reported as an empty local inventory."""
    monkeypatch.setattr(
        "winml.modelkit.winml.get_registered_ep_devices",
        Mock(side_effect=RuntimeError("registration failed")),
    )

    with pytest.raises(RuntimeError, match="registration failed"):
        _get_local_ep_device_pairs()


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
        result = runner.invoke(analyze, ["--model", str(model_file), "--ep", "qnn"])
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
                "qnn",
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
                "qnn",
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
                "qnn",
                "--device",
                "NPU",
            ],
        )

        assert result.exit_code == 0
        mock_instance.analyze.assert_called_once()

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_pattern_progress_prints_only_final_table(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """Output must not append one pattern table per result callback."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        def analyze_with_pattern_progress(**kwargs: object) -> Mock:
            on_start = kwargs["on_pattern_query_start"]
            on_result = kwargs["on_pattern_query_result"]
            on_summary_ready = kwargs["on_pattern_summary_ready"]
            assert callable(on_start)
            assert callable(on_result)
            assert callable(on_summary_ready)
            on_start("DmlExecutionProvider", {"SUBGRAPH/Test": 2}, True)
            on_result("DmlExecutionProvider", "SUBGRAPH/Test", "supported")
            on_result("DmlExecutionProvider", "SUBGRAPH/Test", "unsupported")
            on_summary_ready("DmlExecutionProvider", {"patterns": []})
            return mock_analyzer_result

        mock_instance = Mock()
        mock_instance.analyze.side_effect = analyze_with_pattern_progress
        mock_analyzer_class.return_value = mock_instance

        with patch(
            "winml.modelkit.commands.analyze.Progress",
            wraps=Progress,
        ) as mock_progress:
            result = runner.invoke(
                analyze,
                [
                    "--model",
                    str(model_file),
                    "--ep",
                    "DmlExecutionProvider",
                    "--device",
                    "GPU",
                    "--run-unknown-op",
                ],
            )

        assert result.exit_code == 0
        assert mock_progress.call_args.kwargs["redirect_stdout"] is False
        assert mock_progress.call_args.kwargs["redirect_stderr"] is False
        assert result.output.count("PATTERN CHECK") == 1
        assert "Pattern progress" in result.output
        assert "2/2" in result.output
        assert "1/0/1/0" in result.output

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
                "qnn",
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
                "qnn",
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
                "qnn",
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
                "qnn",
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
                            "unknown": ["node_customop"],
                            "supported": {
                                "node_conv": {
                                    "case_indices": ["case_7"],
                                    "table_path": "rules/conv.parquet",
                                    "table_file": "conv.parquet",
                                    "match_status": "op_match",
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
        assert set(debug_content.keys()) == {"unknown", "supported", "partial", "unsupported"}
        # "unknown" must be the first key in the written debug.json.
        assert next(iter(debug_content)) == "unknown"
        assert debug_content["unknown"] == ["node_customop"]
        assert debug_content["supported"]["node_conv"] == {
            "case_indices": ["case_7"],
            "table_path": "rules/conv.parquet",
            "table_file": "conv.parquet",
            "match_status": "op_match",
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
                "qnn",
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
                "qnn",
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
                "qnn",
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
                "qnn",
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
                "qnn",
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

        eps = ["qnn", "openvino", "vitisai"]

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
        """QNN only succeeds on its catalogued device pairs."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        expected_exit_by_device = {
            "CPU": 2,
            "GPU": 0,
            "NPU": 0,
        }

        for device, expected_exit in expected_exit_by_device.items():
            result = runner.invoke(
                analyze,
                [
                    "--model",
                    str(model_file),
                    "--ep",
                    "qnn",
                    "--device",
                    device,
                ],
            )
            assert result.exit_code == expected_exit, f"Unexpected result for device: {device}"

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
                "openvino",
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

        def analyze_with_unknown_op_progress(**kwargs: object) -> Mock:
            on_ep_start = kwargs["on_ep_start"]
            assert callable(on_ep_start)
            on_ep_start("DmlExecutionProvider", {"Add": 2}, False)
            return mock_analyzer_result

        mock_instance = Mock()
        mock_instance.analyze.side_effect = analyze_with_unknown_op_progress
        mock_analyzer_class.return_value = mock_instance

        with (
            patch(
                "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
                return_value=False,
            ),
            patch(
                "winml.modelkit.commands.analyze.Progress",
                wraps=Progress,
            ) as mock_progress,
        ):
            result = runner.invoke(
                analyze,
                ["--model", str(model_file), "--ep", "dml", "--run-unknown-op"],
            )
        assert result.exit_code == 0
        mock_instance.analyze.assert_called_once()
        assert mock_progress.call_args.kwargs["redirect_stdout"] is False
        assert mock_progress.call_args.kwargs["redirect_stderr"] is False

        call_kwargs = mock_instance.analyze.call_args.kwargs
        assert call_kwargs["ep"] == "DmlExecutionProvider"
        assert call_kwargs["device"] == "GPU"


class TestAnalyzeEPDeviceSelectionMatrix:
    """Matrix tests for EP/device resolution with fixed local availability."""

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_explicit_pair_does_not_require_local_inventory(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A concrete static target remains usable when local registration fails."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        inventory = Mock(side_effect=RuntimeError("registration failed"))
        monkeypatch.setattr(
            "winml.modelkit.commands.analyze._get_local_ep_device_pairs",
            inventory,
        )

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "openvino", "--device", "gpu"],
        )

        assert result.exit_code == 0
        inventory.assert_not_called()
        mock_instance.analyze.assert_called_once()
        assert mock_instance.analyze.call_args.kwargs["ep"] == "OpenVINOExecutionProvider"
        assert mock_instance.analyze.call_args.kwargs["device"] == "GPU"

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_all_device_static_targets_do_not_require_local_inventory(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A static ``all`` fan-out must not depend on installed local EPs."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        inventory = Mock(side_effect=RuntimeError("registration failed"))
        monkeypatch.setattr(
            "winml.modelkit.commands.analyze._get_local_ep_device_pairs",
            inventory,
        )

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(
            analyze,
            ["--model", str(model_file), "--ep", "qnn", "--device", "all"],
        )

        assert result.exit_code == 0
        inventory.assert_not_called()
        assert {
            (call.kwargs["ep"], call.kwargs["device"])
            for call in mock_instance.analyze.call_args_list
        } == {
            ("QNNExecutionProvider", "GPU"),
            ("QNNExecutionProvider", "NPU"),
        }

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_concrete_ep_auto_device_uses_exact_local_binding(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Concrete EP + auto device must use an exact local binding for that EP."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        monkeypatch.setattr(
            "winml.modelkit.commands.analyze._get_local_ep_device_pairs",
            lambda: [("OpenVINOExecutionProvider", "CPU")],
        )

        def _wrong_resolve_device(target):
            raise RuntimeError(f"wrong auto-resolution target: {target.ep}/{target.device}")

        monkeypatch.setattr("winml.modelkit.session.resolve_device", _wrong_resolve_device)

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(analyze, ["--model", str(model_file), "--ep", "openvino"])
        assert result.exit_code == 0

        actual_calls = [
            (call.kwargs["ep"], call.kwargs["device"])
            for call in mock_instance.analyze.call_args_list
        ]
        assert actual_calls == [("OpenVINOExecutionProvider", "CPU")]

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_auto_ep_all_device_uses_best_exact_local_binding_per_device(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """auto + all must derive one best locally bound pair per represented device."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        monkeypatch.setattr(
            "winml.modelkit.commands.analyze._get_local_ep_device_pairs",
            lambda: [
                ("OpenVINOExecutionProvider", "CPU"),
                ("DmlExecutionProvider", "GPU"),
                ("NvTensorRTRTXExecutionProvider", "GPU"),
                ("OpenVINOExecutionProvider", "NPU"),
            ],
        )

        available_eps_calls: list[str] = []
        fabricated_eps_by_device = {
            "cpu": ["CPUExecutionProvider"],
            "gpu": ["TensorrtExecutionProvider"],
            "npu": ["QNNExecutionProvider"],
        }

        def _fabricated_available_eps_for_device(device_name: str) -> list[str]:
            available_eps_calls.append(str(device_name).lower())
            return list(fabricated_eps_by_device.get(str(device_name).lower(), []))

        monkeypatch.setattr(
            "winml.modelkit.session.available_eps_for_device",
            _fabricated_available_eps_for_device,
        )

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(analyze, ["--model", str(model_file), "--device", "all"])
        assert result.exit_code == 0

        actual_calls = [
            (call.kwargs["ep"], call.kwargs["device"])
            for call in mock_instance.analyze.call_args_list
        ]
        assert actual_calls == [
            ("NvTensorRTRTXExecutionProvider", "GPU"),
            ("OpenVINOExecutionProvider", "NPU"),
            ("OpenVINOExecutionProvider", "CPU"),
        ]
        assert available_eps_calls == ["cpu", "gpu", "npu"]

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_auto_ep_all_device_ignores_unsupported_local_bindings(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """auto + all must skip unsupported local bindings and keep supported ones."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        monkeypatch.setattr(
            "winml.modelkit.commands.analyze._get_local_ep_device_pairs",
            lambda: [
                ("QNNExecutionProvider", "CPU"),
                ("OpenVINOExecutionProvider", "CPU"),
            ],
        )

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(analyze, ["--model", str(model_file), "--device", "all"])
        assert result.exit_code == 0

        actual_calls = [
            (call.kwargs["ep"], call.kwargs["device"])
            for call in mock_instance.analyze.call_args_list
        ]
        assert actual_calls == [("OpenVINOExecutionProvider", "CPU")]

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_concrete_ep_auto_device_requires_supported_local_binding(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Concrete EP + auto device should fail when only unsupported local bindings exist."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        monkeypatch.setattr(
            "winml.modelkit.commands.analyze._get_local_ep_device_pairs",
            lambda: [("QNNExecutionProvider", "CPU")],
        )

        result = runner.invoke(analyze, ["--model", str(model_file), "--ep", "qnn"])
        assert result.exit_code == 2
        assert "no supported local binding" in result.output.lower()
        assert "available on this system" in result.output.lower()
        assert not mock_analyzer_class.called

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_auto_ep_auto_device_uses_device_priority_before_resolved_device_fallback(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default analyze target should use the strongest exact local binding."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        multi_device_ep = next(
            ep_name
            for ep_name, devices in EP_SUPPORTED_DEVICES.items()
            if {"npu", "gpu"}.issubset(devices)
        )

        monkeypatch.setattr(
            "winml.modelkit.commands.analyze._get_local_ep_device_pairs",
            lambda: [
                (multi_device_ep, "GPU"),
                (multi_device_ep, "NPU"),
                ("CPUExecutionProvider", "CPU"),
            ],
        )
        monkeypatch.setattr(
            "winml.modelkit.session.resolve_device",
            lambda target: type(target)(
                ep=multi_device_ep,
                device="gpu",
                source=target.source,
            ),
        )
        monkeypatch.setattr(
            "winml.modelkit.session.available_eps_for_device",
            lambda device_name: (
                [multi_device_ep] if str(device_name).lower() in {"gpu", "npu"} else []
            ),
        )

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(analyze, ["--model", str(model_file)])
        assert result.exit_code == 0

        actual_calls = [
            (call.kwargs["ep"], call.kwargs["device"])
            for call in mock_instance.analyze.call_args_list
        ]
        assert actual_calls == [(multi_device_ep, "NPU")]

    @pytest.mark.parametrize(
        "ranked_gpu_eps",
        [
            pytest.param(
                ["OpenVINOExecutionProvider", "DmlExecutionProvider"],
                id="fabricated-leading-ep",
            )
        ],
    )
    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_auto_ep_concrete_device_uses_only_exact_local_supported_pairs(
        self,
        mock_analyzer_class: MagicMock,
        ranked_gpu_eps: list[str],
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """auto + concrete device must ignore ranked EPs without an exact local binding."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        monkeypatch.setattr(
            "winml.modelkit.commands.analyze._get_local_ep_device_pairs",
            lambda: [
                ("OpenVINOExecutionProvider", "NPU"),
                ("DmlExecutionProvider", "GPU"),
            ],
        )
        monkeypatch.setattr(
            "winml.modelkit.session.available_eps_for_device",
            lambda device_name: list(ranked_gpu_eps) if str(device_name).lower() == "gpu" else [],
        )

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(analyze, ["--model", str(model_file), "--device", "gpu"])
        assert result.exit_code == 0

        actual_calls = [
            (call.kwargs["ep"], call.kwargs["device"])
            for call in mock_instance.analyze.call_args_list
        ]
        assert actual_calls == [("DmlExecutionProvider", "GPU")]

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_auto_ep_concrete_and_all_device_share_same_exact_local_ranking(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """auto + gpu and auto + all should choose the same exact local GPU pair."""
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        monkeypatch.setattr(
            "winml.modelkit.commands.analyze._get_local_ep_device_pairs",
            lambda: [
                ("TensorrtExecutionProvider", "GPU"),
                ("DmlExecutionProvider", "GPU"),
            ],
        )
        monkeypatch.setattr(
            "winml.modelkit.session.available_eps_for_device",
            lambda device_name: (
                ["DmlExecutionProvider", "TensorrtExecutionProvider"]
                if str(device_name).lower() == "gpu"
                else []
            ),
        )

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        concrete_result = runner.invoke(analyze, ["--model", str(model_file), "--device", "gpu"])
        assert concrete_result.exit_code == 0
        concrete_calls = [
            (call.kwargs["ep"], call.kwargs["device"])
            for call in mock_instance.analyze.call_args_list
        ]
        assert concrete_calls == [("DmlExecutionProvider", "GPU")]

        mock_instance.analyze.reset_mock()

        all_result = runner.invoke(analyze, ["--model", str(model_file), "--device", "all"])
        assert all_result.exit_code == 0
        all_calls = [
            (call.kwargs["ep"], call.kwargs["device"])
            for call in mock_instance.analyze.call_args_list
        ]
        assert all_calls == [("DmlExecutionProvider", "GPU")]

    @pytest.mark.parametrize(
        ("ep_arg", "device_arg", "expect_exit", "expect_calls", "expect_error"),
        [
            # Both auto: resolve a single best target via shared sysinfo helpers.
            # Best device is NPU (priority npu>gpu>cpu); its best local EP is
            # OpenVINO (only npu EP in the simulated matrix).
            (
                None,
                None,
                0,
                [("OpenVINOExecutionProvider", "NPU")],
                None,
            ),
            # ep=auto, device=gpu: single best local EP for GPU. _DEVICE_EP_MAP
            # ranks NvTensorRTRTX above Dml, both locally available on GPU.
            (
                None,
                "gpu",
                0,
                [("NvTensorRTRTXExecutionProvider", "GPU")],
                None,
            ),
            # ep=openvino, device=auto: single best local device for OpenVINO.
            # OpenVINO is local on NPU and CPU; NPU wins on priority.
            (
                "openvino",
                None,
                0,
                [("OpenVINOExecutionProvider", "NPU")],
                None,
            ),
            # ep=qnn, device=auto: QNN is not local, so resolving a device fails
            # the same way build/run fail — exit 2 with a clear message.
            (
                "qnn",
                None,
                2,
                [],
                "no supported local binding",
            ),
            # ep=qnn, device=all: `all` keeps the full static fan-out, so
            # both catalog-supported QNN devices run without a local check.
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
            # ep=auto, device=all: best available EP *per device* rather than one
            # ref-device EP fanned across all devices. GPU->NvTensorRTRTX,
            # NPU->OpenVINO, CPU->OpenVINO from the simulated local matrix.
            (
                None,
                "all",
                0,
                [
                    ("NvTensorRTRTXExecutionProvider", "GPU"),
                    ("OpenVINOExecutionProvider", "NPU"),
                    ("OpenVINOExecutionProvider", "CPU"),
                ],
                None,
            ),
            # ep=all, device=all: every static catalog-supported pair.
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
                    ("TensorrtExecutionProvider", "GPU"),
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
            "auto-all",
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
            ("TensorrtExecutionProvider", "GPU"),
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
    def test_qnn_device_auto_errors_when_not_local(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """qnn + auto device: QNN isn't local, so device resolution fails (exit 2).

        ``auto`` resolves from local availability via the shared sysinfo helpers,
        exactly like build/run. To statically analyze a non-local EP the user must
        pin the device (``--device npu``) or use ``--device all``.
        """
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"dummy")

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        result = runner.invoke(analyze, ["--model", str(model_file), "--ep", "qnn"])
        assert result.exit_code == 2
        assert "no supported local binding" in result.output.lower()
        assert not mock_instance.analyze.called

    @patch(
        "winml.modelkit.analyze.utils.ep_utils.has_rule_data_for_ep",
        return_value=False,
    )
    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_auto_ep_specific_device_run_unknown_op_executes_single_local_pair(
        self,
        mock_analyzer_class: MagicMock,
        _mock_has_rule: Mock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
    ) -> None:
        """ep=auto + specific device resolves a single best local (ep, device) pair.

        With ep=auto the shared resolver picks the highest-priority EP locally
        available on the requested device (NvTensorRTRTX on GPU). The pair is
        local, so --run-unknown-op stays enabled. has_rule_data_for_ep returning
        False only affects per-pair OP CHECK rendering, not which pair runs.
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
        assert actual_calls == [("NvTensorRTRTXExecutionProvider", "GPU")]


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

    def test_summary_heading_includes_per_ep_analyze_elapsed(self) -> None:
        """Heading should show elapsed analyze time annotation for EP/device."""
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
            ep_instance_counts={("DmlExecutionProvider", "GPU"): {"Conv": {"supported": 1}}},
            ep_patterns={},
            ep="DmlExecutionProvider",
            device="GPU",
            analyze_elapsed_ms=1234,
        )

        output = console.export_text()
        assert "ANALYSIS SUMMARY" in output
        assert "Analyze total: DmlExecutionProvider (GPU), 1.23s" in output

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


class TestAnalyzeCheckOptim:
    """Test the --check-optim flag wiring and rendering."""

    @staticmethod
    def _write_model(path: Path) -> None:
        """Write a small valid MatMul+Add model (a Gemm-fusion candidate)."""
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper, save

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 16])
        w = numpy_helper.from_array(np.random.randn(8, 16).astype(np.float32), "W")
        b = numpy_helper.from_array(np.random.randn(16).astype(np.float32), "B")
        nodes = [
            helper.make_node("MatMul", ["x", "W"], ["mm"], name="mm"),
            helper.make_node("Add", ["mm", "B"], ["y"], name="addbias"),
        ]
        graph = helper.make_graph(nodes, "matmul_add", [x], [y], initializer=[w, b])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 10
        save(model, str(path))

    def test_flag_appears_in_help(self, runner: CliRunner) -> None:
        result = runner.invoke(analyze, ["--help"])
        assert result.exit_code == 0
        assert "--check-optim" in result.output

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_opt_in_renders_section(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--check-optim renders a produced-operator support section."""
        from winml.modelkit.analyze.models.support_level import SupportLevel
        from winml.modelkit.analyze.optim_output import (
            OptimizationOutputSupport,
            ProducedOperatorSupport,
        )

        model_file = tmp_path / "model.onnx"
        self._write_model(model_file)

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        crafted = [
            OptimizationOutputSupport(
                name="matmul-add-fusion",
                enable_flag="--enable-matmul-add-fusion",
                category="matmul",
                description="Fuse MatMul followed by Add into a single Gemm.",
                pipe_name="fusion",
                operators=[
                    ProducedOperatorSupport(
                        "Gemm", "Gemm 'gemm'", "modified", SupportLevel.SUPPORTED
                    )
                ],
            )
        ]
        monkeypatch.setattr(
            "winml.modelkit.analyze.optim_output.check_optimization_output_support",
            lambda *a, **k: crafted,
        )

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--check-optim",
            ],
        )

        assert result.exit_code == 0
        assert "OPTIMIZATION OUTPUT SUPPORT" in result.output
        assert "--enable-matmul-add-fusion" in result.output
        assert "supported" in result.output.lower()

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_disabled_by_default(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without the flag the section is not rendered and the bridge is not called."""
        model_file = tmp_path / "model.onnx"
        self._write_model(model_file)

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        def _should_not_run(*_a: object, **_k: object) -> list:
            raise AssertionError("check_optimization_output_support should not be called")

        monkeypatch.setattr(
            "winml.modelkit.analyze.optim_output.check_optimization_output_support",
            _should_not_run,
        )

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "qnn",
                "--device",
                "NPU",
            ],
        )

        assert result.exit_code == 0
        assert "OPTIMIZATION OUTPUT SUPPORT" not in result.output

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_quiet_json_includes_structured_optimization_support(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--check-optim produces JSON even when Rich rendering is quiet."""
        from winml.modelkit.analyze.models.support_level import SupportLevel
        from winml.modelkit.analyze.optim_output import (
            OptimizationOutputSupport,
            ProducedOperatorSupport,
        )

        model_file = tmp_path / "model.onnx"
        output_file = tmp_path / "analysis.json"
        self._write_model(model_file)

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        monkeypatch.setattr(
            "winml.modelkit.optim.iter_optimization_outputs",
            lambda *_args, **_kwargs: [(object(), object())],
        )
        monkeypatch.setattr("winml.modelkit.optim.get_all_capabilities", dict)
        monkeypatch.setattr(
            "winml.modelkit.analyze.optim_output.check_optimization_output_support",
            lambda *_args, **_kwargs: [
                OptimizationOutputSupport(
                    name="static-split-to-slice",
                    enable_flag="--enable-static-split-to-slice",
                    category="rewrite",
                    description="Replace static Split with Slice.",
                    pipe_name="algebraic",
                    operators=[
                        ProducedOperatorSupport(
                            "Slice",
                            "Slice 'slice_0'",
                            "added",
                            SupportLevel.SUPPORTED,
                        )
                    ],
                )
            ],
        )

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--check-optim",
                "--format",
                "json",
                "--output",
                str(output_file),
                "--quiet",
            ],
        )

        assert result.exit_code == 0, result.output
        stdout_data = json.loads(result.output)
        file_data = json.loads(output_file.read_text(encoding="utf-8"))
        assert file_data == stdout_data
        support = stdout_data["optimization_output_support"]
        assert support["ep_type"] == "QNNExecutionProvider"
        assert support["device_type"] == "NPU"
        assert support["probe_error"] is None
        assert support["support_error"] is None
        assert support["optimizations"][0]["enable_flag"] == ("--enable-static-split-to-slice")
        assert support["optimizations"][0]["worst_support"] == "supported"

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_quiet_json_preserves_optimization_probe_error(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Optimization probe failures remain visible to JSON consumers."""
        model_file = tmp_path / "model.onnx"
        self._write_model(model_file)

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        def _fail_probe(*_args: object, **_kwargs: object) -> list:
            raise RuntimeError("probe failed")

        monkeypatch.setattr("winml.modelkit.optim.iter_optimization_outputs", _fail_probe)

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--check-optim",
                "--format",
                "json",
                "--quiet",
            ],
        )

        assert result.exit_code == 0, result.output
        support = json.loads(result.output)["optimization_output_support"]
        assert support["probe_error"] == "probe failed"
        assert support["support_error"] is None
        assert support["optimizations"] == []

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_quiet_json_preserves_target_support_error(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Target support failures are distinct from graph probe failures."""
        model_file = tmp_path / "model.onnx"
        self._write_model(model_file)

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        monkeypatch.setattr(
            "winml.modelkit.optim.iter_optimization_outputs",
            lambda *_args, **_kwargs: [(object(), object())],
        )
        monkeypatch.setattr("winml.modelkit.optim.get_all_capabilities", dict)

        def _fail_support(*_args: object, **_kwargs: object) -> list:
            raise RuntimeError("support check failed")

        monkeypatch.setattr(
            "winml.modelkit.analyze.optim_output.check_optimization_output_support",
            _fail_support,
        )

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "qnn",
                "--device",
                "NPU",
                "--check-optim",
                "--format",
                "json",
                "--quiet",
            ],
        )

        assert result.exit_code == 0, result.output
        support = json.loads(result.output)["optimization_output_support"]
        assert support["probe_error"] is None
        assert support["support_error"] == "support check failed"
        assert support["optimizations"] == []

    @patch("winml.modelkit.analyze.ONNXStaticAnalyzer")
    def test_multi_device_json_aligns_optimization_support_with_each_target(
        self,
        mock_analyzer_class: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        mock_analyzer_result: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each result in a fan-out carries support for its own EP/device."""
        from winml.modelkit.analyze.optim_output import OptimizationOutputSupport

        model_file = tmp_path / "model.onnx"
        self._write_model(model_file)

        mock_instance = Mock()
        mock_instance.analyze.return_value = mock_analyzer_result
        mock_analyzer_class.return_value = mock_instance

        monkeypatch.setattr(
            "winml.modelkit.optim.iter_optimization_outputs",
            lambda *_args, **_kwargs: [(object(), object())],
        )
        monkeypatch.setattr("winml.modelkit.optim.get_all_capabilities", dict)

        def _support_for_target(*_args: object, **kwargs: object) -> list:
            device = str(kwargs["device"])
            return [
                OptimizationOutputSupport(
                    name=f"optimization-for-{device.lower()}",
                    enable_flag=f"--enable-for-{device.lower()}",
                    category="rewrite",
                    description="Target-specific test result.",
                    pipe_name="algebraic",
                )
            ]

        monkeypatch.setattr(
            "winml.modelkit.analyze.optim_output.check_optimization_output_support",
            _support_for_target,
        )

        result = runner.invoke(
            analyze,
            [
                "--model",
                str(model_file),
                "--ep",
                "qnn",
                "--device",
                "all",
                "--check-optim",
                "--format",
                "json",
                "--quiet",
            ],
        )

        assert result.exit_code == 0, result.output
        payloads = json.loads(result.output)
        assert len(payloads) == 2
        for payload in payloads:
            support = payload["optimization_output_support"]
            device = support["device_type"]
            assert support["ep_type"] == "QNNExecutionProvider"
            assert support["optimizations"][0]["name"] == (f"optimization-for-{device.lower()}")
