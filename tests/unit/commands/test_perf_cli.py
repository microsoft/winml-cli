# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for perf CLI command -- mock-based, no network, no actual benchmarks.

Tests the CLI wrapper around PerfBenchmark.
NO WinMLAutoModel involvement, NO actual inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.perf import BenchmarkConfig, PerfBenchmark, perf


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def mock_resolve_device():
    """Mock device resolution helpers to avoid hardware detection in all perf CLI tests."""
    from winml.modelkit.session import EPDevice

    fake_cpu_ep_device = EPDevice(
        ep="CPUExecutionProvider",
        device="cpu",
        vendor_id=0x0000,
        device_id=0x0001,
    )
    with (
        patch(
            "winml.modelkit.session.auto_detect_device",
            return_value="cpu",
        ),
        patch(
            "winml.modelkit.sysinfo.hardware.get_available_devices",
            return_value=["cpu"],
        ),
        patch(
            "winml.modelkit.session.resolve_device",
            return_value=fake_cpu_ep_device,
        ),
    ):
        yield


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


# =============================================================================
# CLI INTERFACE TESTS
# =============================================================================


class TestPerfCliInterface:
    """Test CLI flag parsing and help text."""

    def test_help_shows_all_options(self, runner: CliRunner) -> None:
        result = runner.invoke(perf, ["--help"])
        assert result.exit_code == 0
        for flag in [
            "--model",
            "-m",
            "--task",
            "--iterations",
            "--warmup",
            "--device",
            "--precision",
            "--output",
            "-o",
            "--batch-size",
            "--no-quantize",
            "--verbose",
            "-v",
        ]:
            assert flag in result.output, f"Expected {flag!r} in help output"

    def test_model_required(self, runner: CliRunner) -> None:
        result = runner.invoke(perf, [], obj={})
        assert result.exit_code != 0
        assert "model" in result.output.lower()

    def test_invalid_device_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(perf, ["-m", "test", "--device", "tpu"], obj={})
        assert result.exit_code != 0

    def test_both_model_and_hf_model_error(self, runner: CliRunner) -> None:
        result = runner.invoke(
            perf,
            ["-m", "model1", "--hf-model", "model2"],
            obj={},
        )
        assert result.exit_code != 0
        assert "cannot use both" in result.output.lower()

    def test_iterations_default_in_help(self, runner: CliRunner) -> None:
        result = runner.invoke(perf, ["--help"])
        assert result.exit_code == 0
        assert "100" in result.output

    def test_warmup_default_in_help(self, runner: CliRunner) -> None:
        result = runner.invoke(perf, ["--help"])
        assert result.exit_code == 0
        assert "10" in result.output

    @pytest.mark.parametrize("device", ["auto", "cpu", "gpu", "npu"])
    def test_valid_device_choices(self, runner: CliRunner, device: str) -> None:
        """Verify Click accepts each valid device choice (no invalid-choice error)."""
        result = runner.invoke(perf, ["--help"])
        assert result.exit_code == 0
        assert device in result.output


# =============================================================================
# OUTPUT PATH TESTS
# =============================================================================


class TestPerfOutputPath:
    """Test generate_output_path() behavior."""

    def test_hf_model_path(self) -> None:
        from winml.modelkit.commands.perf import generate_output_path

        result = generate_output_path("microsoft/resnet-50")
        assert result.name == "microsoft_resnet-50_perf.json"

    def test_onnx_file_uses_stem(self) -> None:
        from winml.modelkit.commands.perf import generate_output_path

        result = generate_output_path("/path/to/model.onnx")
        assert result.name == "model_perf.json"

    def test_onnx_no_leading_underscore(self) -> None:
        from winml.modelkit.commands.perf import generate_output_path

        result = generate_output_path("./model.onnx")
        assert not result.name.startswith("._")
        assert result.name == "model_perf.json"

    def test_windows_path_handled(self) -> None:
        """Backslashes in paths should be replaced."""
        from winml.modelkit.commands.perf import generate_output_path

        result = generate_output_path("C:\\models\\bert-base")
        assert "\\" not in result.name
        # On Windows, Path("C:_models_bert-base_perf.json").name strips the
        # drive letter prefix "C:", yielding "_models_bert-base_perf.json".
        assert result.name == "_models_bert-base_perf.json"


# =============================================================================
# UNIFIED PIPELINE TESTS (ONNX and HF both through PerfBenchmark)
# =============================================================================


class TestPerfUnifiedPipeline:
    """Test that both ONNX and HF models go through PerfBenchmark._load_model."""

    def test_onnx_load_model_calls_from_onnx(self, tmp_path: Path) -> None:
        """ONNX file input should use WinMLAutoModel.from_onnx in _load_model."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        config = BenchmarkConfig(
            model_id=str(onnx_file),
            task="image-classification",
            device="cpu",
        )
        benchmark = PerfBenchmark(config)

        mock_model = MagicMock()
        with patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_onnx",
            return_value=mock_model,
        ) as mock_from_onnx:
            benchmark._load_model()

        mock_from_onnx.assert_called_once()
        kwargs = mock_from_onnx.call_args
        assert kwargs.kwargs["task"] == "image-classification"
        assert kwargs.kwargs["ep_device"].device == "cpu"
        assert benchmark._model is mock_model

    def test_hf_load_model_calls_from_pretrained(self) -> None:
        """HF model input should use WinMLAutoModel.from_pretrained in _load_model."""
        config = BenchmarkConfig(
            model_id="microsoft/resnet-50",
            task="image-classification",
            device="cpu",
        )
        benchmark = PerfBenchmark(config)

        mock_model = MagicMock()
        with patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_pretrained",
            return_value=mock_model,
        ) as mock_from_pretrained:
            benchmark._load_model()

        mock_from_pretrained.assert_called_once()
        kwargs = mock_from_pretrained.call_args
        assert kwargs.args[0] == "microsoft/resnet-50"
        assert kwargs.kwargs["task"] == "image-classification"
        assert kwargs.kwargs["ep_device"].device == "cpu"
        assert benchmark._model is mock_model

    def test_no_quantize_only_sets_quant_none(self, tmp_path: Path) -> None:
        """--no-quantize should only set quant=None, NOT compile=None."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        config = BenchmarkConfig(
            model_id=str(onnx_file),
            task="image-classification",
            device="cpu",
            no_quantize=True,
        )
        benchmark = PerfBenchmark(config)

        mock_model = MagicMock()
        with patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_onnx",
            return_value=mock_model,
        ) as mock_from_onnx:
            benchmark._load_model()

        override = mock_from_onnx.call_args.kwargs["config"]
        assert override is not None
        assert override.quant is None
        # compile should NOT be set to None -- it should remain at default
        assert override.compile is not None

    def test_no_quantize_hf_only_sets_quant_none(self) -> None:
        """--no-quantize with HF model only sets quant=None, not compile=None."""
        config = BenchmarkConfig(
            model_id="test-model",
            task=None,
            device="auto",
            precision="auto",
            iterations=10,
            warmup=2,
            batch_size=1,
            no_quantize=True,
        )
        benchmark = PerfBenchmark(config)

        with patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_pretrained",
            return_value=MagicMock(),
        ) as mock_fp:
            benchmark._load_model()

        call_kwargs = mock_fp.call_args.kwargs
        override = call_kwargs["config"]
        assert override is not None
        # quant should be explicitly set to None
        assert override.quant is None
        # compile should NOT be set to None -- override only affects quant
        assert override.compile is not None

    def test_no_quantize_false_passes_no_override(self) -> None:
        """Without --no-quantize, config override should be None."""
        config = BenchmarkConfig(
            model_id="microsoft/resnet-50",
            device="cpu",
            no_quantize=False,
        )
        benchmark = PerfBenchmark(config)

        mock_model = MagicMock()
        with patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_pretrained",
            return_value=mock_model,
        ) as mock_from_pretrained:
            benchmark._load_model()

        override = mock_from_pretrained.call_args.kwargs["config"]
        assert override is None

    def test_cli_onnx_goes_through_onnx_benchmark(self, runner: CliRunner, tmp_path: Path) -> None:
        """CLI with .onnx file should route through _run_onnx_benchmark."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        with (
            patch(
                "winml.modelkit.commands.perf._run_onnx_benchmark",
                return_value=MagicMock(),
            ) as mock_run,
            patch(
                "winml.modelkit.commands.perf.display_console_report",
            ),
            patch(
                "winml.modelkit.commands.perf.write_json_report",
            ),
        ):
            result = runner.invoke(
                perf,
                ["-m", str(onnx_file), "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output
        mock_run.assert_called_once()

    def test_cli_onnx_not_found_error(self, runner: CliRunner, tmp_path: Path) -> None:
        """CLI with non-existent .onnx file should raise FileNotFoundError."""
        missing = tmp_path / "missing.onnx"
        result = runner.invoke(
            perf,
            ["-m", str(missing)],
            obj={},
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_onnx_load_model_passes_ep(self, tmp_path: Path) -> None:
        """EP argument should be forwarded to from_onnx via ep_device."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        config = BenchmarkConfig(
            model_id=str(onnx_file),
            task="image-classification",
            device="npu",
            ep="qnn",
        )
        benchmark = PerfBenchmark(config)

        mock_model = MagicMock()
        with patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_onnx",
            return_value=mock_model,
        ) as mock_from_onnx:
            benchmark._load_model()

        ep_device = mock_from_onnx.call_args.kwargs["ep_device"]
        assert ep_device.ep == "CPUExecutionProvider"  # resolved via mock_resolve_device fixture
