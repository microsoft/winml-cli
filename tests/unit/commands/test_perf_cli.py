# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for perf CLI command -- mock-based, no network, no actual benchmarks.

Tests the CLI wrapper around PerfBenchmark.
NO WinMLAutoModel involvement, NO actual inference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.perf import (
    BenchmarkConfig,
    PerfBenchmark,
    generate_output_path,
    perf,
)


@pytest.fixture(autouse=True)
def mock_resolve_device():
    """Mock resolve_device to avoid hardware detection in all perf CLI tests."""
    with patch(
        "winml.modelkit.sysinfo.resolve_device",
        return_value=("cpu", ["cpu"]),
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
    """Test generate_output_path() behavior.

    The default output lives under ~/.cache/winml/perf/<slug>/<timestamp>.json
    so repeated `winml perf` runs don't pollute CWD (#551).
    """

    _TIMESTAMP_RE = r"^\d{8}-\d{6}\.json$"

    @property
    def _cache_root(self) -> Path:
        return Path.home() / ".cache" / "winml" / "perf"

    def test_hf_model_path(self) -> None:
        result = generate_output_path("microsoft/resnet-50")
        assert result.parent == self._cache_root / "microsoft_resnet-50"
        assert re.match(self._TIMESTAMP_RE, result.name)

    def test_onnx_file_uses_stem(self) -> None:
        result = generate_output_path("/path/to/model.onnx")
        assert result.parent == self._cache_root / "model"
        assert re.match(self._TIMESTAMP_RE, result.name)

    def test_onnx_no_leading_underscore(self) -> None:
        result = generate_output_path("./model.onnx")
        assert result.parent == self._cache_root / "model"
        assert re.match(self._TIMESTAMP_RE, result.name)

    def test_windows_path_handled(self) -> None:
        """Backslashes in paths are replaced in the slug directory name."""
        result = generate_output_path("C:\\models\\bert-base")
        # On Windows the "C:" drive letter is stripped by Path().name, yielding
        # "_models_bert-base" — match the legacy slug semantics.
        assert result.parent == self._cache_root / "_models_bert-base"
        assert "\\" not in result.parent.name
        assert re.match(self._TIMESTAMP_RE, result.name)

    def test_module_class_adds_subdir(self) -> None:
        """--module CLASSNAME nests results under <slug>/<module_class>/."""
        result = generate_output_path("bert-base-uncased", module_class="BertAttention")
        assert result.parent == self._cache_root / "bert-base-uncased" / "BertAttention"
        assert re.match(self._TIMESTAMP_RE, result.name)

    def test_path_is_under_user_home(self) -> None:
        """Sanity: regardless of input, the file lands under ~/.cache/winml/perf."""
        result = generate_output_path("microsoft/resnet-50")
        assert self._cache_root in result.parents


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
        assert kwargs.kwargs["device"] == "cpu"
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
        assert kwargs.kwargs["device"] == "cpu"
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

    def test_cli_onnx_routes_through_perf_benchmark(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """CLI with .onnx file should route through the same PerfBenchmark as HF.

        Both paths must share the build+benchmark pipeline so latency numbers
        from `winml perf -m hf/id` and `winml perf -m <built.onnx>` are
        comparable (issue #596).
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        with (
            patch.object(
                PerfBenchmark,
                "run",
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

    def test_cli_onnx_clears_shape_config_with_warning(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """ONNX input with --shape-config: warn + clear shape_config before PerfBenchmark.

        Shapes are baked into a pre-exported ONNX, so --shape-config is silently
        ignored; we want to be sure the CLI both surfaces the warning to the
        user and actually drops the override before constructing PerfBenchmark.
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        shape_cfg_file = tmp_path / "shapes.json"
        shape_cfg_file.write_text(json.dumps({"input_ids": [1, 128]}))

        captured: dict[str, BenchmarkConfig] = {}

        def capture_config(config: BenchmarkConfig) -> MagicMock:
            captured["config"] = config
            mock = MagicMock()
            mock.run.return_value = MagicMock()
            return mock

        with (
            patch(
                "winml.modelkit.commands.perf.PerfBenchmark",
                side_effect=capture_config,
            ),
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            result = runner.invoke(
                perf,
                [
                    "-m",
                    str(onnx_file),
                    "--shape-config",
                    str(shape_cfg_file),
                    "-o",
                    str(tmp_path / "out.json"),
                ],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert "shape-config is ignored" in result.output
        assert "Benchmarking ONNX" in result.output
        assert captured["config"].shape_config is None

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
        """EP argument should be forwarded to from_onnx."""
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

        assert mock_from_onnx.call_args.kwargs["ep"] == "qnn"


# =============================================================================
# --FORMAT JSON TESTS
# =============================================================================


class TestPerfFormatJson:
    """Test --format json produces structured JSON to stdout."""

    def test_help_shows_format_option(self, runner: CliRunner) -> None:
        """--format flag must appear in --help output."""
        result = runner.invoke(perf, ["--help"])
        assert result.exit_code == 0
        assert "--format" in result.output
        assert "json" in result.output

    def test_invalid_format_rejected(self, runner: CliRunner) -> None:
        """An invalid --format value must be rejected by Click."""
        result = runner.invoke(perf, ["-m", "test", "--format", "xml"], obj={})
        assert result.exit_code != 0

    @patch("winml.modelkit.commands.perf.PerfBenchmark")
    def test_format_json_emits_valid_json(
        self, mock_benchmark_class: MagicMock, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--format json must produce parseable JSON on stdout.

        Note: CliRunner mixes stderr into result.output; in production the
        Console(stderr=True) keeps stdout clean. Extract JSON from mixed output.
        """
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "benchmark_info": {
                "model_id": "microsoft/resnet-50",
                "task": "image-classification",
                "device": "cpu",
                "ep": None,
            },
            "latency_ms": {"mean": 18.3, "p50": 17.5, "p90": 21.7},
            "throughput": {"samples_per_sec": 54.6},
        }
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        mock_benchmark_class.return_value = mock_instance

        output_file = tmp_path / "result.json"

        result = runner.invoke(
            perf,
            [
                "-m",
                "microsoft/resnet-50",
                "--format",
                "json",
                "--output",
                str(output_file),
            ],
            obj={},
        )

        assert result.exit_code == 0
        # Extract JSON object from mixed output (CliRunner mixes stderr)
        output = result.output
        json_start = output.index("{")
        json_end = output.rindex("}") + 1
        parsed = json.loads(output[json_start:json_end])
        assert parsed["benchmark_info"]["model_id"] == "microsoft/resnet-50"
        assert "latency_ms" in parsed

    @patch("winml.modelkit.commands.perf.PerfBenchmark")
    def test_format_text_shows_console_report(
        self, mock_benchmark_class: MagicMock, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Default --format text must not emit raw JSON."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"benchmark_info": {"model_id": "test"}}
        mock_result.config = MagicMock()
        mock_result.config.model_id = "test"
        mock_result.actual_device = "cpu"
        mock_result.actual_task = "cls"
        mock_result.actual_ep = None
        mock_result.mean_ms = 10.0
        mock_result.min_ms = 9.0
        mock_result.max_ms = 11.0
        mock_result.p50_ms = 10.0
        mock_result.p90_ms = 10.5
        mock_result.p95_ms = 10.8
        mock_result.p99_ms = 11.0
        mock_result.std_ms = 0.5
        mock_result.warmup_mean_ms = 12.0
        mock_result.samples_per_sec = 100.0
        mock_result.batches_per_sec = 100.0
        mock_result.hw_monitor = None
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        mock_benchmark_class.return_value = mock_instance

        output_file = tmp_path / "result.json"

        result = runner.invoke(
            perf,
            [
                "-m",
                "test",
                "--output",
                str(output_file),
            ],
            obj={},
        )

        assert result.exit_code == 0
        # Should NOT be parseable as JSON (it's console text)
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.output)
