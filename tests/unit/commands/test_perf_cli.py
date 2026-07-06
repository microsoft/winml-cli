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
    BenchmarkResult,
    PerfBenchmark,
    generate_output_path,
    perf,
)


@pytest.fixture(autouse=True)
def mock_resolve_device():
    """Mock device/EP resolution to avoid hardware detection in all perf CLI tests.

    perf() resolves the device (and, when --ep is omitted, derives a concrete EP
    via resolve_eps) up front, so both are stubbed to a deterministic CPU result.
    """
    with (
        patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("cpu", ["cpu"]),
        ),
        patch(
            "winml.modelkit.sysinfo.resolve_eps",
            return_value=["CPUExecutionProvider"],
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

    def test_cli_onnx_preserves_shape_config(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """ONNX input with --shape-config keeps the override for dummy inputs.

        Regression: perf previously warned that shape config was ignored for
        ONNX inputs and force-cleared the override. The ONNX path now honors
        user-provided shapes during random input generation.
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
        assert "shape-config is ignored" not in result.output
        assert "Benchmarking ONNX" in result.output
        assert captured["config"].shape_config == {"input_ids": [1, 128]}

    def test_cli_onnx_warns_ignored_build_flags(self, runner: CliRunner, tmp_path: Path) -> None:
        """Build-pipeline flags are no-ops for a pre-built ONNX with skip_build,
        so the CLI surfaces a warning naming the flags the user set."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        def capture_config(_config: BenchmarkConfig) -> MagicMock:
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
                    "--no-quant",
                    "--no-optimize",
                    "-o",
                    str(tmp_path / "out.json"),
                ],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert "--no-quant" in result.output
        assert "--no-optimize" in result.output
        assert "pre-built ONNX" in result.output

    def test_cli_onnx_no_build_flag_warning_at_defaults(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """No ignored-build-flags warning when the flags are left at defaults."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        def capture_config(_config: BenchmarkConfig) -> MagicMock:
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
                ["-m", str(onnx_file), "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert "ignored for pre-built ONNX inputs (no build runs" not in result.output

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

    def test_cli_hub_onnx_ref_is_resolved(self, runner: CliRunner, tmp_path: Path) -> None:
        """CLI with a Hub-style ONNX ref must download once before the
        ``Path(...).suffix == '.onnx' and exists()`` check, otherwise the
        ref string is mistaken for a missing local file and rejected with
        ``FileNotFoundError`` before any HF Hub call happens.

        Regression test for ``winml perf -m
        onnx-community/sam3-tracker-ONNX/onnx/...``.
        """
        local = tmp_path / "vision_encoder_int8.onnx"
        local.write_bytes(b"fake onnx")
        hub_ref = "onnx-community/sam3-tracker-ONNX/onnx/vision_encoder_int8.onnx"

        mock_result = MagicMock()
        mock_result.to_dict = MagicMock(return_value={})

        # Stub PerfBenchmark so the test stays fast and EP-independent;
        # capture the BenchmarkConfig it was constructed with so we can
        # assert ``model_id`` is the resolved local path, not the Hub ref.
        captured_configs: list = []
        original_init = PerfBenchmark.__init__

        def _capturing_init(self_, config, *args, **kwargs):
            captured_configs.append(config)
            original_init(self_, config, *args, **kwargs)

        with (
            patch(
                "winml.modelkit.loader.onnx_hub.resolve_hf_onnx_path",
                return_value=local,
            ) as mock_resolve,
            patch.object(PerfBenchmark, "__init__", _capturing_init),
            patch.object(PerfBenchmark, "run", return_value=mock_result) as mock_run,
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            result = runner.invoke(
                perf,
                ["-m", hub_ref, "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output
        # ``resolve_model_input`` forwards revision/cache_dir/token kwargs
        # to the downloader; only the positional Hub ref is meaningful here.
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.args == (hub_ref,)
        # After resolution, the PerfBenchmark sees the LOCAL path on its
        # config.model_id -- not the original Hub ref string.
        mock_run.assert_called_once()
        assert len(captured_configs) == 1
        assert Path(captured_configs[0].model_id) == local

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

    def test_onnx_load_model_passes_ep_options(self, tmp_path: Path) -> None:
        """--ep-options should reach from_onnx as provider_options (ONNX path)."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        config = BenchmarkConfig(
            model_id=str(onnx_file),
            task="image-classification",
            device="npu",
            ep="qnn",
            ep_options={"htp_performance_mode": "burst"},
        )
        benchmark = PerfBenchmark(config)

        with patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_onnx",
            return_value=MagicMock(),
        ) as mock_from_onnx:
            benchmark._load_model()

        assert mock_from_onnx.call_args.kwargs["provider_options"] == {
            "htp_performance_mode": "burst"
        }

    def test_hf_load_model_passes_ep_options(self) -> None:
        """--ep-options should reach from_pretrained as provider_options (HF path)."""
        config = BenchmarkConfig(
            model_id="microsoft/resnet-50",
            task="image-classification",
            device="npu",
            ep="qnn",
            ep_options={"htp_performance_mode": "burst"},
        )
        benchmark = PerfBenchmark(config)

        with patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_pretrained",
            return_value=MagicMock(),
        ) as mock_from_pretrained:
            benchmark._load_model()

        assert mock_from_pretrained.call_args.kwargs["provider_options"] == {
            "htp_performance_mode": "burst"
        }

    def test_cli_ep_options_parsed_into_config(self, runner: CliRunner, tmp_path: Path) -> None:
        """Repeated --ep-options KEY=VALUE are parsed into BenchmarkConfig.ep_options."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        captured: dict[str, BenchmarkConfig] = {}

        def capture_config(config: BenchmarkConfig) -> MagicMock:
            captured["config"] = config
            return MagicMock()

        with (
            patch("winml.modelkit.commands.perf.PerfBenchmark", side_effect=capture_config),
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            result = runner.invoke(
                perf,
                [
                    "-m",
                    str(onnx_file),
                    "--ep-options",
                    "htp_performance_mode=burst",
                    "--ep-options",
                    "htp_graph_finalization_optimization_mode=3",
                    "-o",
                    str(tmp_path / "out.json"),
                ],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert captured["config"].ep_options == {
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        }

    def test_cli_ep_options_invalid_format_rejected(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """An --ep-options value without '=' is rejected with a clear error."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        result = runner.invoke(
            perf,
            ["-m", str(onnx_file), "--ep-options", "no_equals_sign"],
            obj={},
        )

        assert result.exit_code != 0
        assert "KEY=VALUE" in result.output

    def test_load_model_no_ep_derives_concrete_ep(self, tmp_path: Path) -> None:
        """Without an EP, PerfBenchmark resolves a concrete one before building.

        Regression guard: previously ep stayed None down to the build, so the
        static analyzer ran with ep=None and aggregated across all EPs (and
        logged a warning). PerfBenchmark now resolves the EP from the device
        (autouse fixture stubs resolve_eps -> ["CPUExecutionProvider"]) and
        passes it to from_onnx. The config keeps the raw request (ep=None);
        the resolved value lives on the instance.
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        config = BenchmarkConfig(model_id=str(onnx_file), task="image-classification")
        benchmark = PerfBenchmark(config)

        with patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_onnx",
            return_value=MagicMock(),
        ) as mock_from_onnx:
            benchmark._load_model()

        assert mock_from_onnx.call_args.kwargs["ep"] == "CPUExecutionProvider"
        assert benchmark._resolved_ep == "CPUExecutionProvider"
        assert config.ep is None

    def test_load_model_explicit_ep_passed_through_verbatim(self, tmp_path: Path) -> None:
        """An explicit EP reaches from_onnx unchanged (no normalization).

        Downstream build/session stages normalize aliases themselves, so
        PerfBenchmark must not rewrite the user's value (e.g. 'qnn' stays 'qnn').
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        config = BenchmarkConfig(
            model_id=str(onnx_file), task="image-classification", device="npu", ep="qnn"
        )
        benchmark = PerfBenchmark(config)

        with patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_onnx",
            return_value=MagicMock(),
        ) as mock_from_onnx:
            benchmark._load_model()

        assert mock_from_onnx.call_args.kwargs["ep"] == "qnn"
        assert benchmark._resolved_ep == "qnn"

    def test_load_model_unavailable_device_ep_fails_before_build(self, tmp_path: Path) -> None:
        """An unavailable device/EP combo fails before the build pipeline runs.

        PerfBenchmark resolves device+EP at the start of _load_model, so an
        unavailable combo (resolve_device raises ValueError) surfaces before
        from_onnx kicks off the build — the user does not wait for the whole
        build only to fail at session.compile().
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        config = BenchmarkConfig(model_id=str(onnx_file), task="image-classification", device="npu")
        benchmark = PerfBenchmark(config)

        with (
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                side_effect=ValueError("no compatible EP is available"),
            ),
            patch("winml.modelkit.models.auto.WinMLAutoModel.from_onnx") as mock_from_onnx,
            pytest.raises(ValueError, match="no compatible EP is available"),
        ):
            benchmark._load_model()

        mock_from_onnx.assert_not_called()

    def test_cli_unavailable_device_ep_surfaces_error(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """The CLI surfaces the fail-fast resolution error with a non-zero exit."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        with (
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                side_effect=ValueError("no compatible EP is available"),
            ),
            patch("winml.modelkit.models.auto.WinMLAutoModel.from_onnx") as mock_from_onnx,
        ):
            result = runner.invoke(
                perf,
                ["-m", str(onnx_file), "--device", "npu", "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code != 0
        assert "no compatible EP is available" in result.output
        mock_from_onnx.assert_not_called()

    def test_help_shows_ep_options(self, runner: CliRunner) -> None:
        result = runner.invoke(perf, ["--help"])
        assert result.exit_code == 0
        assert "--ep-options" in result.output

    def test_ep_options_captured_in_to_dict(self) -> None:
        """ep_options must be written into benchmark_info so saved JSON is reproducible."""
        ep_options = {"htp_performance_mode": "burst"}
        config = BenchmarkConfig(model_id="m", ep_options=ep_options)
        result = BenchmarkResult(config=config)

        assert result.to_dict()["benchmark_info"]["ep_options"] == ep_options

    def test_ep_options_none_when_not_set_in_to_dict(self) -> None:
        """When no EP options are given, benchmark_info records None."""
        config = BenchmarkConfig(model_id="m")
        result = BenchmarkResult(config=config)

        assert result.to_dict()["benchmark_info"]["ep_options"] is None


class TestEffectiveBatchSize:
    """Throughput must scale by the batch the session actually ran.

    ``--batch-size`` only lands on inputs whose leading dim is dynamic, so a
    static-batch model silently runs a different batch than requested. The
    reported ``samples_per_sec`` must reflect the actual batch, not the request.
    """

    def test_helper_reads_dynamic_batch_from_inputs(self) -> None:
        import numpy as np

        from winml.modelkit.commands.perf import effective_batch_size

        inputs = {"pixel_values": np.zeros((8, 3, 224, 224), dtype=np.float32)}
        assert effective_batch_size(inputs, ["pixel_values"], requested=8) == 8

    def test_helper_reads_static_batch_not_requested(self) -> None:
        import numpy as np

        from winml.modelkit.commands.perf import effective_batch_size

        # Model has a static batch of 1; the requested 8 never reached the input.
        inputs = {"pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)}
        assert effective_batch_size(inputs, ["pixel_values"], requested=8) == 1

    def test_helper_skips_scalar_inputs(self) -> None:
        import numpy as np

        from winml.modelkit.commands.perf import effective_batch_size

        # First input is a rank-0 scalar (no batch dim); fall through to the
        # first batched input for the batch reading.
        inputs = {
            "scalar": np.array(3, dtype=np.int64),
            "tokens": np.zeros((4, 128), dtype=np.int64),
        }
        assert effective_batch_size(inputs, ["scalar", "tokens"], requested=4) == 4

    def test_helper_falls_back_when_all_scalar(self) -> None:
        import numpy as np

        from winml.modelkit.commands.perf import effective_batch_size

        inputs = {"scalar": np.array(3, dtype=np.int64)}
        assert effective_batch_size(inputs, ["scalar"], requested=8) == 8

    def _fake_stats(self) -> MagicMock:
        stats = MagicMock()
        stats.mean_ms = 10.0  # 0.01 s -> 100 batches/sec
        stats.min_ms = 9.0
        stats.max_ms = 11.0
        stats.p50_ms = 10.0
        stats.p90_ms = 10.5
        stats.p95_ms = 10.8
        stats.p99_ms = 11.0
        stats.samples_ms = [10.0, 10.0]
        stats.all_samples_ms = [10.0, 10.0]
        return stats

    def _benchmark_with_single(self, *, batch_size: int, effective_batch: int) -> PerfBenchmark:
        config = BenchmarkConfig(model_id="m", batch_size=batch_size, warmup=0)
        benchmark = PerfBenchmark(config)
        single = MagicMock()
        single.io_config = {
            "input_names": ["pixel_values"],
            "input_shapes": [[effective_batch, 3, 224, 224]],
            "input_types": ["float32"],
            "output_names": ["logits"],
            "output_shapes": [[effective_batch, 1000]],
        }
        single.device = "cpu"
        single.ep_name = None
        single.task = "image-classification"
        single.running_model_path = "model.onnx"
        benchmark._model = single
        benchmark._effective_batch = effective_batch
        return benchmark

    def test_throughput_scales_by_effective_not_requested(self) -> None:
        # Requested batch 8, but model ran batch 1: 100 batches/sec -> 100 sps,
        # NOT 800. This is the bug guard.
        benchmark = self._benchmark_with_single(batch_size=8, effective_batch=1)
        result = benchmark._collect_results(self._fake_stats())

        assert result.effective_batch_size == 1
        assert result.batches_per_sec == pytest.approx(100.0)
        assert result.samples_per_sec == pytest.approx(100.0)

    def test_throughput_scales_when_batch_applied(self) -> None:
        # Dynamic batch honored: 100 batches/sec * 8 = 800 samples/sec.
        benchmark = self._benchmark_with_single(batch_size=8, effective_batch=8)
        result = benchmark._collect_results(self._fake_stats())

        assert result.effective_batch_size == 8
        assert result.batches_per_sec == pytest.approx(100.0)
        assert result.samples_per_sec == pytest.approx(800.0)

    def test_generate_inputs_warns_on_static_batch(self) -> None:
        import numpy as np

        config = BenchmarkConfig(model_id="m", batch_size=8)
        benchmark = PerfBenchmark(config)
        single = MagicMock()
        single.io_config = {
            "input_names": ["pixel_values"],
            "input_shapes": [[1, 3, 224, 224]],
            "input_types": ["float32"],
        }
        benchmark._model = single

        # Static batch of 1: generate_random_inputs ignores the requested 8.
        static_inputs = {"pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)}
        with (
            patch(
                "winml.modelkit.commands.perf.generate_random_inputs",
                return_value=static_inputs,
            ),
            patch("winml.modelkit.commands.perf.logger") as mock_logger,
        ):
            benchmark._generate_inputs()

        assert benchmark._effective_batch == 1
        mock_logger.warning.assert_called_once()

    def test_to_dict_emits_effective_batch_size(self) -> None:
        config = BenchmarkConfig(model_id="m", batch_size=8)
        result = BenchmarkResult(config=config, effective_batch_size=1)

        info = result.to_dict()["benchmark_info"]
        assert info["batch_size"] == 8
        assert info["effective_batch_size"] == 1


# =============================================================================
# --FORMAT JSON TESTS
# =============================================================================


class TestFormatInputShape:
    """Dynamic dims render as ``dynamic(<actual>)`` with real generated sizes."""

    def test_dynamic_dim_shows_actual_value(self) -> None:
        from winml.modelkit.commands.perf import _format_input_shape

        assert _format_input_shape([None, 3, 64, 64], (1, 3, 64, 64)) == "[dynamic(1), 3, 64, 64]"

    def test_multiple_dynamic_dims(self) -> None:
        from winml.modelkit.commands.perf import _format_input_shape

        assert _format_input_shape([None, None], (2, 128)) == "[dynamic(2), dynamic(128)]"

    def test_all_static_dims_unchanged(self) -> None:
        from winml.modelkit.commands.perf import _format_input_shape

        assert _format_input_shape([1, 3, 224, 224], (1, 3, 224, 224)) == "[1, 3, 224, 224]"

    def test_dynamic_without_actual_falls_back_to_bare_dynamic(self) -> None:
        from winml.modelkit.commands.perf import _format_input_shape

        assert _format_input_shape([None, 3], None) == "[dynamic, 3]"

    def test_dynamic_shape_survives_rich_rendering(self) -> None:
        # Regression: a lowercase ``[dynamic(...)]`` is valid Rich markup and
        # gets swallowed unless escaped, leaving the shape column blank.
        import contextlib
        import io as _io

        from winml.modelkit.commands.perf import _print_model_info

        io_config = {
            "input_names": ["pixel_values"],
            "input_shapes": [[None, 3, 64, 64]],
            "input_types": ["float32"],
            "output_names": ["logits"],
            "output_shapes": [[None, 1000]],
        }
        buf = _io.StringIO()
        with contextlib.redirect_stderr(buf):
            _print_model_info(
                io_config,
                actual_shapes={"pixel_values": (10, 3, 64, 64)},
            )
        out = buf.getvalue()
        assert "[dynamic(10), 3, 64, 64]" in out
        # Outputs have no generated data, so dynamic dims render bare.
        assert "[dynamic, 1000]" in out


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
        mock_result.memory_profile = None
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
