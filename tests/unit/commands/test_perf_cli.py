# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for perf CLI command -- mock-based, no network, no actual benchmarks.

Tests the CLI wrapper around PerfBenchmark.
NO WinMLAutoModel involvement, NO actual inference.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import re
import sys
import warnings
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner, Result
from rich.console import Console

import winml.modelkit.commands.perf as perf_module
from winml.modelkit.commands.perf import (
    BenchmarkConfig,
    BenchmarkResult,
    PerfBenchmark,
    display_console_report,
    generate_output_path,
    perf,
)
from winml.modelkit.utils.console import SafeConsole


class TestPerfCacheOptions:
    @staticmethod
    def _capture_config(
        runner: CliRunner,
        tmp_path: Path,
        extra_args: list[str],
    ) -> tuple[Result, BenchmarkConfig | None]:
        captured: dict[str, BenchmarkConfig] = {}

        def _benchmark(config: BenchmarkConfig) -> MagicMock:
            captured["config"] = config
            instance = MagicMock()
            instance.run.return_value = MagicMock()
            return instance

        with (
            patch("winml.modelkit.commands.perf.PerfBenchmark", side_effect=_benchmark),
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            result = runner.invoke(
                perf,
                [
                    "-m",
                    "test/model",
                    "-o",
                    str(tmp_path / "result.json"),
                    *extra_args,
                ],
                obj={},
            )

        return result, captured.get("config")

    def test_help_shows_canonical_flags(
        self,
        runner: CliRunner,
    ) -> None:
        result = runner.invoke(perf, ["--help"])

        assert result.exit_code == 0, result.output
        assert "--use-cache / --no-use-cache" in result.output
        assert "--rebuild / --no-rebuild" in result.output

    @pytest.mark.parametrize(
        ("extra_args", "use_cache", "rebuild"),
        [
            ([], True, False),
            (["--no-use-cache"], False, False),
            (["--rebuild"], True, True),
        ],
    )
    def test_canonical_flags_reach_benchmark_config(
        self,
        runner: CliRunner,
        tmp_path: Path,
        extra_args: list[str],
        use_cache: bool,
        rebuild: bool,
    ) -> None:
        result, config = self._capture_config(runner, tmp_path, extra_args)

        assert result.exit_code == 0, result.output
        assert config is not None
        assert config.use_cache is use_cache
        assert config.rebuild is rebuild


@pytest.fixture(autouse=True)
def mock_resolve_device():
    """Mock device resolution helpers to avoid hardware detection in all perf CLI tests."""
    from winml.modelkit.session import EPDeviceTarget

    fake_cpu_ep_device = EPDeviceTarget(ep="CPUExecutionProvider", device="cpu")
    fake_winml_ep_device = MagicMock()
    fake_winml_ep_device.device.ep_name = "CPUExecutionProvider"
    fake_winml_ep_device.device.device_type = "CPU"
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
        patch("winml.modelkit.session.WinMLEPRegistry") as mock_reg,
    ):
        mock_reg.instance.return_value.auto_device.return_value = fake_winml_ep_device
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
            "--input-specs",
            "--export-config",
            "--dynamic-axes",
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

    def test_load_model_does_not_forward_export_policy_details(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-resolved runtime EPs should not force target-specific export policy."""
        from winml.modelkit.models import WinMLAutoModel

        benchmark = PerfBenchmark(
            BenchmarkConfig(
                model_id="microsoft/resnet-50",
                task="image-classification",
            )
        )
        fake_ep_device = MagicMock()
        benchmark._ep_device = fake_ep_device
        benchmark._resolved_device = "gpu"
        benchmark._resolved_ep = "DmlExecutionProvider"
        monkeypatch.setattr(benchmark, "_resolve_device_ep", lambda: None)

        received: dict[str, object] = {}

        def _from_pretrained(*args: object, **kwargs: object) -> MagicMock:
            received["args"] = args
            received.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(WinMLAutoModel, "from_pretrained", _from_pretrained)

        benchmark._load_model()

        assert received["ep_device"] is fake_ep_device

    def test_close_releases_single_model_session(self) -> None:
        """Closing a benchmark resets the loaded model's native session."""
        benchmark = PerfBenchmark(BenchmarkConfig(model_id="m"))
        model = MagicMock()
        model._session = MagicMock()
        benchmark._model = model
        benchmark._inputs = {"x": object()}

        benchmark.close()

        model._session.reset.assert_called_once()
        assert benchmark._model is None
        assert benchmark._inputs is None

    def test_close_suppresses_native_warning_from_session_reset(self, capfd) -> None:
        """Closing a benchmark hides warning-level native stderr from session teardown."""
        benchmark = PerfBenchmark(BenchmarkConfig(model_id="m"))
        model = MagicMock()
        model._session = MagicMock()

        def write_reset_warning() -> int:
            return os.write(
                2,
                b"2026 [W:custom-native:, file.cc:1 ResetWarn] reset warning\n",
            )

        model._session.reset.side_effect = write_reset_warning
        benchmark._model = model

        benchmark.close()

        assert "reset warning" not in capfd.readouterr().err

    def test_close_releases_composite_sub_model_sessions(self) -> None:
        """Composite benchmarks reset every sub-model session before process teardown."""
        benchmark = PerfBenchmark(BenchmarkConfig(model_id="m"))
        first = MagicMock()
        first._session = MagicMock()
        second = MagicMock()
        second._session = MagicMock()
        composite = MagicMock()
        composite._session = None
        composite.sub_models = {"first": first, "second": second}
        benchmark._model = composite

        benchmark.close()

        first._session.reset.assert_called_once()
        second._session.reset.assert_called_once()
        assert benchmark._model is None

    def test_run_does_not_redirect_native_stderr_around_model_load_or_ui(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Perf must not route model-load UI or pre-bench rendering through fd2 filtering."""
        inside_native_suppression = False
        observed: dict[str, bool] = {}

        @contextmanager
        def mark_native_suppression(*_args: object, **_kwargs: object):
            nonlocal inside_native_suppression
            previous = inside_native_suppression
            inside_native_suppression = True
            try:
                yield
            finally:
                inside_native_suppression = previous

        def fake_load(self: PerfBenchmark) -> None:
            observed["load"] = inside_native_suppression
            self._model = MagicMock()

        def fake_run_single(self: PerfBenchmark) -> MagicMock:
            observed["run_single"] = inside_native_suppression
            return MagicMock()

        def is_not_composite(self: PerfBenchmark) -> bool:
            return False

        monkeypatch.setattr(
            "winml.modelkit.commands.perf.suppress_native_warnings",
            mark_native_suppression,
        )
        monkeypatch.setattr(PerfBenchmark, "_load_model", fake_load)
        monkeypatch.setattr(PerfBenchmark, "_run_single", fake_run_single)
        monkeypatch.setattr(
            PerfBenchmark,
            "_is_composite",
            property(is_not_composite),
        )

        benchmark = PerfBenchmark(BenchmarkConfig(model_id="m"))
        benchmark.run()

        assert observed == {"load": False, "run_single": False}

    def test_native_perf_context_filters_enter_exit_not_benchmark_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """session.perf native setup/teardown is filtered without wrapping the loop."""
        inside_native_suppression = False
        observed: list[tuple[str, bool]] = []

        @contextmanager
        def mark_native_suppression(*_args: object, **_kwargs: object):
            nonlocal inside_native_suppression
            previous = inside_native_suppression
            inside_native_suppression = True
            try:
                yield
            finally:
                inside_native_suppression = previous

        class FakePerfContext:
            def __enter__(self) -> SimpleNamespace:
                observed.append(("enter", inside_native_suppression))
                return SimpleNamespace(stats=MagicMock())

            def __exit__(self, *exc: object) -> bool:
                observed.append(("exit", inside_native_suppression))
                return False

        class FakeSession:
            def perf(self, **_kwargs: object) -> FakePerfContext:
                return FakePerfContext()

        monkeypatch.setattr(
            "winml.modelkit.commands.perf.suppress_native_warnings",
            mark_native_suppression,
        )

        with perf_module._native_warning_filtered_perf(FakeSession(), warmup=1) as ctx:
            assert ctx.stats is not None
            observed.append(("body", inside_native_suppression))

        assert observed == [("enter", True), ("body", False), ("exit", True)]

    def test_native_perf_context_suppresses_native_warning_from_exit(
        self, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """Native warning lines from session.perf teardown stay hidden."""

        class FakePerfContext:
            def __enter__(self) -> SimpleNamespace:
                return SimpleNamespace(stats=MagicMock())

            def __exit__(self, *exc: object) -> bool:
                os.write(2, b"2026 [W:custom-native:, file.cc:1 PerfExit] hidden warning\n")
                return False

        class FakeSession:
            def perf(self, **_kwargs: object) -> FakePerfContext:
                return FakePerfContext()

        with perf_module._native_warning_filtered_perf(FakeSession(), warmup=1):
            pass

        assert "hidden warning" not in capfd.readouterr().err

    def test_run_single_filters_native_warnings_only_around_session_compile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Session compile is native-heavy, but pre-bench UI must stay outside fd2 filtering."""
        inside_native_suppression = False
        observed: dict[str, bool] = {}

        @contextmanager
        def mark_native_suppression(*_args: object, **_kwargs: object):
            nonlocal inside_native_suppression
            previous = inside_native_suppression
            inside_native_suppression = True
            try:
                yield
            finally:
                inside_native_suppression = previous

        class FakeSession:
            def compile(self) -> None:
                observed["compile"] = inside_native_suppression

        fake_model = MagicMock()
        fake_model._session = FakeSession()
        fake_model.io_config = {
            "input_names": ["input"],
            "input_shapes": [(1,)],
            "input_types": ["float32"],
            "output_names": ["output"],
            "output_shapes": [(1,)],
            "output_types": ["float32"],
        }
        fake_model.task = "image-classification"
        fake_model.device = "npu"
        fake_model.ep_name = "QNNExecutionProvider"
        fake_model._onnx_path = None

        benchmark = PerfBenchmark(BenchmarkConfig(model_id="m", warmup=0, iterations=1))
        benchmark._model = fake_model
        benchmark._ep_device = MagicMock()

        def generate_inputs() -> None:
            return None

        def pre_bench_kwargs_from_ep_device(*_args: object, **_kwargs: object) -> dict[str, object]:
            return {}

        def print_pre_bench(*_args: object, **_kwargs: object) -> bool:
            return observed.setdefault("pre_bench", inside_native_suppression)

        def run_benchmark() -> MagicMock:
            return MagicMock()

        def collect_results(_stats: object) -> MagicMock:
            return MagicMock()

        monkeypatch.setattr(
            "winml.modelkit.commands.perf.suppress_native_warnings",
            mark_native_suppression,
        )
        monkeypatch.setattr(benchmark, "_generate_inputs", generate_inputs)
        monkeypatch.setattr(
            perf_module,
            "_pre_bench_kwargs_from_ep_device",
            pre_bench_kwargs_from_ep_device,
        )
        monkeypatch.setattr(perf_module, "print_pre_bench_block", print_pre_bench)
        monkeypatch.setattr(benchmark, "_run_benchmark", run_benchmark)
        monkeypatch.setattr(benchmark, "_collect_results", collect_results)

        benchmark._run_single()

        assert observed["compile"] is True
        assert observed["pre_bench"] is False

    def test_load_model_filters_native_warnings_around_auto_model_factory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Import-time and model-factory native warnings are filtered."""
        inside_native_suppression = False
        import_observed: list[bool] = []
        load_observed: list[bool] = []

        @contextmanager
        def mark_native_suppression(*_args: object, **_kwargs: object):
            nonlocal inside_native_suppression
            previous = inside_native_suppression
            inside_native_suppression = True
            try:
                yield
            finally:
                inside_native_suppression = previous

        class FakeWinMLAutoModel:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> MagicMock:
                load_observed.append(inside_native_suppression)
                return MagicMock()

        def fake_model_getattr(name: str) -> object:
            if name == "WinMLAutoModel":
                import_observed.append(inside_native_suppression)
                return FakeWinMLAutoModel
            raise AttributeError(name)

        def resolve_ep_device(self: PerfBenchmark) -> None:
            self._ep_device = MagicMock()
            self._resolved_device = "cpu"
            self._resolved_ep = "cpu"

        fake_models_pkg = ModuleType("winml.modelkit.models")
        fake_models_pkg.__getattr__ = fake_model_getattr
        monkeypatch.setitem(sys.modules, "winml.modelkit.models", fake_models_pkg)
        monkeypatch.setattr(
            "winml.modelkit.commands.perf.suppress_native_warnings",
            mark_native_suppression,
        )
        monkeypatch.setattr(PerfBenchmark, "_resolve_device_ep", resolve_ep_device)

        benchmark = PerfBenchmark(
            BenchmarkConfig(model_id="microsoft/resnet-50", task="image-classification")
        )
        benchmark._load_model()

        assert import_observed
        assert all(import_observed)
        assert load_observed == [True]

    def test_resolve_device_ep_filters_native_warnings_and_preserves_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Device/EP probes can hit ORT and should hide only warning-level noise."""
        from winml.modelkit import session as session_module

        monkeypatch.delenv("WINMLCLI_SHOW_ALL_WARNINGS", raising=False)
        logging.getLogger().setLevel(logging.WARNING)
        fake_ep_device = MagicMock()
        fake_ep_device.device.ep_name = "QNNExecutionProvider"
        fake_ep_device.device.device_type = "NPU"

        def fake_resolve_device(target: object) -> object:
            os.write(2, b"2026 [W:custom-native:, file.cc:1 Probe] hidden warning\n")
            return target

        class FakeRegistry:
            def auto_device(self, target: object) -> object:
                os.write(2, b"2026 [E:custom-native:, file.cc:2 Probe] useful error\n")
                return fake_ep_device

        def registry_instance() -> FakeRegistry:
            return FakeRegistry()

        with monkeypatch.context() as local_patch:
            local_patch.setattr(session_module, "resolve_device", fake_resolve_device)
            local_patch.setattr(
                session_module.WinMLEPRegistry,
                "instance",
                staticmethod(registry_instance),
            )

            benchmark = PerfBenchmark(BenchmarkConfig(model_id="m", ep="qnn", device="npu"))
            benchmark._resolve_device_ep()

        stderr = capfd.readouterr().err
        assert "hidden warning" not in stderr
        assert "useful error" in stderr

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
        # ep_device is now a WinMLEPDevice — its .device is a WinMLDevice whose
        # .device_type holds the upper-cased class string.
        assert kwargs.kwargs["ep_device"].device.device_type.lower() == "cpu"
        assert kwargs.kwargs["use_cache"] is True
        assert kwargs.kwargs["force_rebuild"] is False
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
        assert kwargs.kwargs["ep_device"].device.device_type.lower() == "cpu"
        assert kwargs.kwargs["use_cache"] is True
        assert kwargs.kwargs["force_rebuild"] is False
        assert benchmark._model is mock_model

    @pytest.mark.parametrize(
        ("use_cache", "rebuild", "force_rebuild"),
        [
            (False, False, True),
            (True, True, True),
            (False, True, True),
        ],
    )
    def test_hf_load_model_maps_cache_policy(
        self,
        use_cache: bool,
        rebuild: bool,
        force_rebuild: bool,
    ) -> None:
        benchmark = PerfBenchmark(
            BenchmarkConfig(
                model_id="test/model",
                device="cpu",
                use_cache=use_cache,
                rebuild=rebuild,
            )
        )

        with patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_pretrained",
            return_value=MagicMock(),
        ) as mock_from_pretrained:
            benchmark._load_model()

        kwargs = mock_from_pretrained.call_args.kwargs
        assert kwargs["use_cache"] is use_cache
        assert kwargs["force_rebuild"] is force_rebuild

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

    def test_export_overrides_hf_passed_as_sparse_build_override(self) -> None:
        """Export overrides should not construct a full default build config."""
        config = BenchmarkConfig(
            model_id="microsoft/resnet-50",
            task="image-classification",
            device="cpu",
            export_overrides={"dynamic_axes": {"pixel_values": {"0": "batch"}}},
        )
        benchmark = PerfBenchmark(config)

        mock_model = MagicMock()
        with patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_pretrained",
            return_value=mock_model,
        ) as mock_from_pretrained:
            benchmark._load_model()

        override = mock_from_pretrained.call_args.kwargs["config"]
        assert override == {"export": {"dynamic_axes": {"pixel_values": {"0": "batch"}}}}

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
            patch(
                "winml.modelkit.commands.perf.PerfBenchmark",
            ) as mock_perf_cls,
            patch(
                "winml.modelkit.commands.perf.display_console_report",
            ),
            patch(
                "winml.modelkit.commands.perf.write_json_report",
            ),
        ):
            mock_perf_cls.return_value.run.return_value = MagicMock()
            result = runner.invoke(
                perf,
                ["-m", str(onnx_file), "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output
        mock_perf_cls.assert_called_once()

    def test_cli_onnx_preserves_shape_config(self, runner: CliRunner, tmp_path: Path) -> None:
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

    def test_cli_onnx_hub_resolution_suppresses_huggingface_warnings_by_default(
        self, runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Default perf output hides Hugging Face warnings from Hub ONNX resolution."""
        resolved = tmp_path / "model.onnx"
        resolved.write_bytes(b"fake onnx")

        def resolve_with_warning(model: str) -> str:
            logging.getLogger("huggingface_hub.utils._http").warning(
                "Warning: You are sending unauthenticated requests to the HF Hub."
            )
            return str(resolved)

        with (
            patch(
                "winml.modelkit.commands.perf.cli_utils.normalize_model_arg",
                side_effect=resolve_with_warning,
            ),
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_perf,
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            mock_perf.return_value.run.return_value = MagicMock()
            caplog.clear()
            result = runner.invoke(
                perf,
                ["-m", "org/repo/path/model.onnx", "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert "Benchmarking ONNX" in result.output
        assert "unauthenticated requests" not in result.output
        assert not any("unauthenticated requests" in record.message for record in caplog.records)

    def test_cli_onnx_hub_resolution_reveals_huggingface_warnings_when_verbose(
        self, runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verbose perf output keeps Hub ONNX resolution warnings visible."""
        resolved = tmp_path / "model.onnx"
        resolved.write_bytes(b"fake onnx")

        def resolve_with_warning(model: str) -> str:
            logging.getLogger("huggingface_hub.utils._http").warning(
                "Warning: You are sending unauthenticated requests to the HF Hub."
            )
            return str(resolved)

        with (
            patch(
                "winml.modelkit.commands.perf.cli_utils.normalize_model_arg",
                side_effect=resolve_with_warning,
            ),
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_perf,
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            mock_perf.return_value.run.return_value = MagicMock()
            caplog.clear()
            result = runner.invoke(
                perf,
                ["-m", "org/repo/path/model.onnx", "-v", "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert any("unauthenticated requests" in record.message for record in caplog.records)

    def test_cli_hf_forwards_export_overrides(self, runner: CliRunner, tmp_path: Path) -> None:
        """HF perf builds should pass export-related CLI overrides into BenchmarkConfig."""
        input_specs = tmp_path / "inputs.json"
        input_specs.write_text(
            json.dumps({"pixel_values": {"dtype": "float32", "shape": ["batch", 3, 224, 224]}})
        )
        export_config = tmp_path / "export.json"
        export_config.write_text(json.dumps({"opset_version": 18}))
        dynamic_axes = tmp_path / "dynamic_axes.json"
        dynamic_axes.write_text(json.dumps({"pixel_values": {"0": "batch"}}))

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
                    "microsoft/resnet-50",
                    "--input-specs",
                    str(input_specs),
                    "--export-config",
                    str(export_config),
                    "--dynamic-axes",
                    str(dynamic_axes),
                    "-o",
                    str(tmp_path / "out.json"),
                ],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert captured["config"].export_overrides is not None
        export_override = captured["config"].export_overrides
        assert export_override["opset_version"] == 18
        assert export_override["dynamic_axes"] == {"pixel_values": {"0": "batch"}}
        assert export_override["input_tensors"][0].name == "pixel_values"
        assert export_override["input_tensors"][0].shape == ("batch", 3, 224, 224)

    def test_cli_hf_suppresses_huggingface_warning_logs_by_default(
        self, runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Default perf output hides Hugging Face warning chatter during model load."""

        def emit_huggingface_warnings() -> MagicMock:
            logging.getLogger("huggingface_hub.utils._http").warning(
                "Warning: You are sending unauthenticated requests to the HF Hub."
            )
            logging.getLogger("transformers").warning("`Siglip2ImageProcessorFast` is deprecated.")
            return MagicMock()

        with (
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_perf,
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            mock_perf.return_value.run.side_effect = emit_huggingface_warnings
            caplog.clear()
            result = runner.invoke(
                perf,
                ["-m", "microsoft/resnet-50", "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert "Loading model" in result.output
        assert "unauthenticated requests" not in result.output
        assert "Siglip2ImageProcessorFast" not in result.output
        assert not any("unauthenticated requests" in record.message for record in caplog.records)
        assert not any("Siglip2ImageProcessorFast" in record.message for record in caplog.records)

    def test_cli_hf_suppresses_huggingface_python_warnings_by_default(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Default perf output hides Hugging Face warnings.warn chatter during model load."""

        def emit_huggingface_warning() -> MagicMock:
            warnings.warn_explicit(
                "Warning: You are sending unauthenticated requests to the HF Hub.",
                UserWarning,
                filename="huggingface_hub/utils/_http.py",
                lineno=1,
                module="huggingface_hub.utils._http",
            )
            return MagicMock()

        with (
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_perf,
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            mock_perf.return_value.run.side_effect = emit_huggingface_warning
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                result = runner.invoke(
                    perf,
                    ["-m", "microsoft/resnet-50", "-o", str(tmp_path / "out.json")],
                    obj={},
                )

        assert result.exit_code == 0, result.output
        assert not any("unauthenticated requests" in str(record.message) for record in records)

    def test_cli_does_not_wrap_entire_benchmark_in_native_stderr_redirect(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Perf CLI must not route Rich/report output through native stderr filtering."""
        inside_native_suppression = False

        @contextmanager
        def mark_native_suppression(*_args: object, **_kwargs: object):
            nonlocal inside_native_suppression
            previous = inside_native_suppression
            inside_native_suppression = True
            try:
                yield
            finally:
                inside_native_suppression = previous

        def assert_run_not_suppressed() -> MagicMock:
            assert not inside_native_suppression
            return MagicMock()

        def assert_report_not_suppressed(*_args: object, **_kwargs: object) -> MagicMock:
            assert not inside_native_suppression
            return MagicMock()

        monkeypatch.setattr(
            "winml.modelkit.commands.perf.suppress_native_warnings",
            mark_native_suppression,
        )

        with (
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_perf,
            patch(
                "winml.modelkit.commands.perf.display_console_report",
                side_effect=assert_report_not_suppressed,
            ),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            mock_perf.return_value.run.side_effect = assert_run_not_suppressed
            result = runner.invoke(
                perf,
                ["-m", "microsoft/resnet-50", "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output

    def test_op_tracing_monitor_probe_suppresses_native_warning_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ORT imports/probes used to choose op-tracing monitor should be filtered."""
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        inside_native_suppression = False
        import_observed: list[bool] = []
        probe_observed: list[bool] = []

        @contextmanager
        def mark_native_suppression(*_args: object, **_kwargs: object):
            nonlocal inside_native_suppression
            previous = inside_native_suppression
            inside_native_suppression = True
            try:
                yield
            finally:
                inside_native_suppression = previous

        real_import = builtins.__import__

        def observe_session_import(
            name: str,
            globals: dict[str, object] | None = None,
            locals: dict[str, object] | None = None,
            fromlist: tuple[str, ...] | None = (),
            level: int = 0,
        ) -> object:
            requested_names = tuple(fromlist or ())
            if "short_ep_name" in requested_names and (level or name.endswith("session")):
                import_observed.append(inside_native_suppression)
            return real_import(name, globals, locals, fromlist, level)

        def qnn_is_available() -> bool:
            probe_observed.append(inside_native_suppression)
            return True

        monkeypatch.setattr(
            "winml.modelkit.commands.perf.suppress_native_warnings",
            mark_native_suppression,
        )
        monkeypatch.setattr(builtins, "__import__", observe_session_import)
        monkeypatch.setattr(QNNMonitor, "is_available", qnn_is_available)

        monitor = perf_module._resolve_ep_monitor("qnn", "basic", tmp_path, device="npu")

        assert monitor is not None
        assert import_observed == [True]
        assert probe_observed == [True]

    def test_cli_hf_disables_third_party_progress_by_default(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default perf hides third-party Hub/datasets progress during benchmark."""
        monkeypatch.delenv("HF_DATASETS_DISABLE_PROGRESS_BARS", raising=False)
        monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)

        observed: dict[str, str | None] = {}

        def capture_progress_env() -> MagicMock:
            observed["datasets_disable"] = os.environ.get("HF_DATASETS_DISABLE_PROGRESS_BARS")
            observed["hub_disable"] = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            return MagicMock()

        with (
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_perf,
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            mock_perf.return_value.run.side_effect = capture_progress_env
            result = runner.invoke(
                perf,
                ["-m", "microsoft/resnet-50", "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert observed["datasets_disable"] == "1"
        assert observed["hub_disable"] == "1"
        assert "HF_DATASETS_DISABLE_PROGRESS_BARS" not in os.environ
        assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ

    def test_cli_hf_keeps_third_party_progress_when_verbose(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verbose perf preserves third-party progress diagnostics."""
        monkeypatch.delenv("HF_DATASETS_DISABLE_PROGRESS_BARS", raising=False)
        monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)

        observed: dict[str, str | None] = {}

        def capture_progress_env() -> MagicMock:
            observed["datasets_disable"] = os.environ.get("HF_DATASETS_DISABLE_PROGRESS_BARS")
            observed["hub_disable"] = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            return MagicMock()

        with (
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_perf,
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            mock_perf.return_value.run.side_effect = capture_progress_env
            result = runner.invoke(
                perf,
                ["-m", "microsoft/resnet-50", "-v", "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert observed["datasets_disable"] is None
        assert observed["hub_disable"] is None

    def test_cli_hf_reveals_huggingface_warning_logs_when_verbose(
        self, runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verbose perf output keeps Hugging Face warning chatter visible."""

        def emit_huggingface_warnings() -> MagicMock:
            logging.getLogger("huggingface_hub.utils._http").warning(
                "Warning: You are sending unauthenticated requests to the HF Hub."
            )
            logging.getLogger("transformers").warning("`Siglip2ImageProcessorFast` is deprecated.")
            return MagicMock()

        with (
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_perf,
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            mock_perf.return_value.run.side_effect = emit_huggingface_warnings
            caplog.clear()
            result = runner.invoke(
                perf,
                ["-m", "microsoft/resnet-50", "-v", "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert any("unauthenticated requests" in record.message for record in caplog.records)
        assert any("Siglip2ImageProcessorFast" in record.message for record in caplog.records)

    def test_cli_hf_reveals_huggingface_python_warnings_when_verbose(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Verbose perf output keeps Hugging Face warnings.warn chatter visible."""

        def emit_huggingface_warning() -> MagicMock:
            warnings.warn_explicit(
                "Warning: You are sending unauthenticated requests to the HF Hub.",
                UserWarning,
                filename="huggingface_hub/utils/_http.py",
                lineno=1,
                module="huggingface_hub.utils._http",
            )
            return MagicMock()

        with (
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_perf,
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            mock_perf.return_value.run.side_effect = emit_huggingface_warning
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                result = runner.invoke(
                    perf,
                    ["-m", "microsoft/resnet-50", "-v", "-o", str(tmp_path / "out.json")],
                    obj={},
                )

        assert result.exit_code == 0, result.output
        assert any("unauthenticated requests" in str(record.message) for record in records)

    def test_cli_hf_reveals_native_warning_logs_when_verbose(
        self, runner: CliRunner, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """Verbose perf output keeps native ORT warning-level stderr visible."""

        def emit_native_warning() -> MagicMock:
            os.write(
                2,
                b"2026 [W:onnxruntime:Default, onnxruntime_pybind_module.cc:44 "
                b"onnxruntime::python::CreateOrtEnv] Init provider bridge failed.\n",
            )
            return MagicMock()

        with (
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_perf,
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            mock_perf.return_value.run.side_effect = emit_native_warning
            result = runner.invoke(
                perf,
                ["-m", "microsoft/resnet-50", "-v", "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert "Init provider bridge failed" in capfd.readouterr().err

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
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
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
                    "--no-use-cache",
                    "-o",
                    str(tmp_path / "out.json"),
                ],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert "--no-quant" in result.output
        assert "--no-optimize" in result.output
        assert "--no-use-cache" in result.output
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

    def test_compiled_onnx_warns_for_cache_control_with_build_enabled(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        onnx_file = tmp_path / "compiled.onnx"
        onnx_file.write_bytes(b"fake compiled onnx")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=True),
            patch("winml.modelkit.commands.perf.PerfBenchmark") as benchmark_type,
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            benchmark_type.return_value.run.return_value = MagicMock()
            result = runner.invoke(
                perf,
                [
                    "-m",
                    str(onnx_file),
                    "--no-skip-build",
                    "--no-use-cache",
                    "--no-optimize",
                    "-o",
                    str(tmp_path / "out.json"),
                ],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert "--no-use-cache ignored for pre-built ONNX inputs" in result.output
        assert "--no-optimize ignored for pre-built ONNX inputs" in result.output
        assert "pass --no-skip-build to rebuild" not in result.output

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
        # ep_device is a WinMLEPDevice; .device.ep_name holds the canonical EP name.
        assert ep_device.device.ep_name == "CPUExecutionProvider"

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

        Regression guard: previously ep stayed None down to the build. Now
        PerfBenchmark resolves via WinMLEPRegistry.auto_device and hands
        WinMLAutoModel.from_onnx an ``ep_device`` whose ``.device.ep_name``
        carries the concrete EP. The config keeps the raw request (ep=None).
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

        # New API: ep_device carries the resolved EP.
        kwargs = mock_from_onnx.call_args.kwargs
        assert kwargs.get("ep_device") is not None, "expected ep_device kwarg"
        # The autouse fixture returns a fake WinMLEPDevice — its device.ep_name is a canonical EP.
        assert kwargs["ep_device"].device.ep_name.endswith("ExecutionProvider")
        assert config.ep is None

    def test_load_model_explicit_ep_passed_through_verbatim(self, tmp_path: Path) -> None:
        """An explicit EP reaches from_onnx via the resolved ep_device.

        Downstream build/session stages normalize aliases themselves; PerfBenchmark
        threads the resolved (EP, device) target into ``ep_device.device.ep_name``.
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        config = BenchmarkConfig(
            model_id=str(onnx_file), task="image-classification", device="npu", ep="qnn"
        )
        benchmark = PerfBenchmark(config)

        # The autouse fixture pins a static CPU device/registry so hardware
        # detection never runs. Override both locally so an explicit --ep qnn
        # resolves to a QNN ep_device (the fixture's CPU stub would otherwise
        # mask the EP threading this test guards).
        from winml.modelkit.session import EPDeviceTarget

        fake_qnn_ep_device = MagicMock()
        fake_qnn_ep_device.device.ep_name = "QNNExecutionProvider"
        fake_qnn_ep_device.device.device_type = "NPU"

        with (
            patch(
                "winml.modelkit.session.resolve_device",
                return_value=EPDeviceTarget(ep="QNNExecutionProvider", device="npu"),
            ),
            patch("winml.modelkit.session.WinMLEPRegistry") as mock_reg,
            patch(
                "winml.modelkit.models.auto.WinMLAutoModel.from_onnx",
                return_value=MagicMock(),
            ) as mock_from_onnx,
        ):
            mock_reg.instance.return_value.auto_device.return_value = fake_qnn_ep_device
            benchmark._load_model()

        kwargs = mock_from_onnx.call_args.kwargs
        assert kwargs.get("ep_device") is not None
        # Fake resolver expands 'qnn' to canonical 'QNNExecutionProvider'.
        assert kwargs["ep_device"].device.ep_name == "QNNExecutionProvider"

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
                "winml.modelkit.session.resolve_device",
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
                "winml.modelkit.session.resolve_device",
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

    def test_to_dict_includes_schema_version_and_runtime(self) -> None:
        """Classic perf reports the same schema markers as GenAI perf."""
        config = BenchmarkConfig(model_id="m")
        result = BenchmarkResult(config=config)
        d = result.to_dict()

        assert d["schema_version"] == 2
        assert d["benchmark_info"]["runtime"] == "winml-ort"

    def test_iterations_reports_configured_count_without_duration(self) -> None:
        """Without --duration, benchmark_info.iterations is the configured value."""
        config = BenchmarkConfig(model_id="m", iterations=100)
        result = BenchmarkResult(config=config, raw_samples_ms=[1.0, 2.0, 3.0])

        info = result.to_dict()["benchmark_info"]
        assert info["iterations"] == 100
        assert info["duration_sec"] is None

    def test_iterations_reports_actual_sample_count_with_duration(self) -> None:
        """In duration mode, benchmark_info.iterations is the actual sample count."""
        config = BenchmarkConfig(model_id="m", iterations=100, duration=5.0)
        result = BenchmarkResult(config=config, raw_samples_ms=[1.0, 2.0, 3.0, 4.0])

        info = result.to_dict()["benchmark_info"]
        # 4 timed samples were collected, not the unused --iterations=100.
        assert info["iterations"] == 4
        assert info["duration_sec"] == 5.0


class TestResolveShape:
    def test_symbolic_override_rejects_list_value_with_clean_error(self) -> None:
        import click

        from winml.modelkit.commands.perf import _resolve_shape

        with pytest.raises(click.ClickException, match="symbolic dimension 'seq'"):
            _resolve_shape(
                [1, "seq"],
                input_name="tokens",
                batch_size=1,
                symbolic_shape=[None, "seq"],
                shape_config={"seq": [1, 128]},
            )

    def test_symbolic_override_rejects_non_integer_float_value(self) -> None:
        import click

        from winml.modelkit.commands.perf import _resolve_shape

        with pytest.raises(click.ClickException, match="scalar integer"):
            _resolve_shape(
                [1, "seq"],
                input_name="tokens",
                batch_size=1,
                symbolic_shape=[None, "seq"],
                shape_config={"seq": 128.5},
            )


class TestGenerateRandomInputs:
    """Generated build-config ranges must reach default perf inputs."""

    @pytest.mark.parametrize(
        "value_range",
        [
            [0, 1],
            [3, 7],
        ],
    )
    def test_integer_ranges_remain_high_exclusive(self, value_range: list[int]) -> None:
        import numpy as np

        from winml.modelkit.commands.perf import generate_random_inputs

        io_config = {
            "input_names": ["tokens"],
            "input_shapes": [[1, 32]],
            "input_types": ["int32"],
            "input_value_ranges": {"tokens": value_range},
        }

        with patch("numpy.random.randint", wraps=np.random.randint) as randint:
            values = generate_random_inputs(io_config)["tokens"]

        assert randint.call_args.args[:2] == tuple(value_range)
        assert np.all(values >= value_range[0])
        assert np.all(values < value_range[1])

    def test_float_range_and_unspecified_defaults_are_preserved(self) -> None:
        import numpy as np

        from winml.modelkit.commands.perf import generate_random_inputs

        io_config = {
            "input_names": ["ranged", "default_int", "default_float"],
            "input_shapes": [[1, 1024], [1, 1024], [1, 1024]],
            "input_types": ["float32", "int64", "float32"],
            "input_value_ranges": {"ranged": [-2.0, -1.0]},
        }

        inputs = generate_random_inputs(io_config)

        assert np.all(inputs["ranged"] >= -2.0)
        assert np.all(inputs["ranged"] < -1.0)
        assert set(np.unique(inputs["default_int"])) == {0, 1}
        assert np.all(inputs["default_float"] >= 0.0)
        assert np.all(inputs["default_float"] < 1.0)

    def test_bbox_range_generates_ordered_positive_area_boxes(self) -> None:
        """Default perf inputs must preserve the canonical bbox corner contract."""
        import numpy as np

        from winml.modelkit.commands.perf import generate_random_inputs

        io_config = {
            "input_names": ["bbox"],
            "input_shapes": [[1, 512, 4]],
            "input_types": ["int32"],
            "input_value_ranges": {"bbox": [0, 1001]},
        }

        bbox = generate_random_inputs(io_config)["bbox"]

        assert bbox.shape == (1, 512, 4)
        assert np.all(bbox >= 0)
        assert np.all(bbox < 1001)
        assert np.all(bbox[..., 0] < bbox[..., 2])
        assert np.all(bbox[..., 1] < bbox[..., 3])

    def test_bbox_range_requires_two_coordinates(self) -> None:
        """A degenerate bbox range must fail instead of producing invalid corners."""
        from winml.modelkit.commands.perf import generate_random_inputs

        io_config = {
            "input_names": ["bbox"],
            "input_shapes": [[1, 8, 4]],
            "input_types": ["int32"],
            "input_value_ranges": {"bbox": [0, 1]},
        }

        with pytest.raises(ValueError, match="at least two coordinates"):
            generate_random_inputs(io_config)


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


class TestClassicMemoryProfile:
    def test_memory_profile_includes_additive_peak_and_compile_fields(self, monkeypatch) -> None:
        """Classic perf emits the same additive memory fields as GenAI."""
        config = BenchmarkConfig(model_id="m", memory=True, warmup=0)
        benchmark = PerfBenchmark(config)
        single = MagicMock()
        single.io_config = {
            "input_names": ["pixel_values"],
            "input_shapes": [[1, 3, 224, 224]],
            "input_types": ["float32"],
            "output_names": ["logits"],
            "output_shapes": [[1, 1000]],
        }
        single.device = "npu"
        single.ep_name = "QNNExecutionProvider"
        single.task = "image-classification"
        single.running_model_path = "model.onnx"
        single._session.compile.return_value = None
        benchmark._model = single
        benchmark._ep_device = MagicMock()

        stats = MagicMock()
        stats.mean_ms = 10.0
        stats.min_ms = 9.0
        stats.max_ms = 11.0
        stats.p50_ms = 10.0
        stats.p90_ms = 10.5
        stats.p95_ms = 10.8
        stats.p99_ms = 11.0
        stats.samples_ms = [10.0]
        stats.all_samples_ms = [10.0]

        rss_values = iter([100.0, 150.0, 180.0])
        vram_values = iter([(10.0, 20.0), (30.0, 50.0), (40.0, 70.0)])

        monkeypatch.setattr(benchmark, "_resolve_adapter_luid", lambda: "luid")
        monkeypatch.setattr(benchmark, "_run_benchmark", lambda: stats)
        monkeypatch.setattr(
            benchmark,
            "_generate_inputs",
            lambda: setattr(
                benchmark,
                "_inputs",
                {"pixel_values": MagicMock(shape=(1, 3, 224, 224))},
            ),
        )
        monkeypatch.setattr(
            "winml.modelkit.session.monitor.memory_tracker.get_rss_mb",
            lambda: next(rss_values),
        )
        monkeypatch.setattr(
            "winml.modelkit.session.monitor.memory_tracker.get_vram_mb",
            lambda _adapter_luid: next(vram_values),
        )
        monkeypatch.setattr("winml.modelkit.commands.perf._print_model_info", lambda *_, **__: None)

        result = benchmark._run_single()

        assert result.memory_profile == {
            "rss_baseline_mb": 100.0,
            "rss_after_compile_mb": 150.0,
            "rss_after_inference_mb": 180.0,
            "rss_checkpoint_peak_mb": 180.0,
            "rss_model_load_delta_mb": 50.0,
            "rss_inference_delta_mb": 30.0,
            "rss_total_delta_mb": 80.0,
            "vram_local_baseline_mb": 10.0,
            "vram_shared_baseline_mb": 20.0,
            "vram_local_after_compile_mb": 30.0,
            "vram_shared_after_compile_mb": 50.0,
            "vram_local_after_inference_mb": 40.0,
            "vram_shared_after_inference_mb": 70.0,
            "vram_local_checkpoint_peak_mb": 40.0,
            "vram_shared_checkpoint_peak_mb": 70.0,
            "vram_local_model_load_delta_mb": 20.0,
            "vram_shared_model_load_delta_mb": 30.0,
            "vram_local_inference_delta_mb": 10.0,
            "vram_shared_inference_delta_mb": 20.0,
            "vram_local_total_delta_mb": 30.0,
            "vram_shared_total_delta_mb": 50.0,
        }


# =============================================================================
# --INPUT-DATA TESTS
# =============================================================================


class TestLoadInputData:
    """Loading real benchmark inputs from a .npz file (issue #1065)."""

    _IO: ClassVar[dict] = {
        "input_names": ["pixel_values"],
        "input_shapes": [[None, 3, 64, 64]],
        "input_types": ["float32"],
    }

    def _write_npz(self, tmp_path, **arrays):
        import numpy as np

        path = tmp_path / "inputs.npz"
        np.savez(path, **arrays)
        return path

    def test_loads_matching_npz(self, tmp_path) -> None:
        import numpy as np

        from winml.modelkit.commands.perf import load_input_data

        path = self._write_npz(tmp_path, pixel_values=np.zeros((4, 3, 64, 64), dtype=np.float32))
        inputs = load_input_data(path, self._IO)

        assert list(inputs) == ["pixel_values"]
        assert inputs["pixel_values"].shape == (4, 3, 64, 64)

    def test_missing_input_name_errors(self, tmp_path) -> None:
        import numpy as np

        from winml.modelkit.commands.perf import load_input_data

        path = self._write_npz(tmp_path, wrong_name=np.zeros((1, 3, 64, 64), dtype=np.float32))
        with pytest.raises(click.UsageError, match="do not match"):
            load_input_data(path, self._IO)

    def test_unexpected_key_errors(self, tmp_path) -> None:
        import numpy as np

        from winml.modelkit.commands.perf import load_input_data

        path = self._write_npz(
            tmp_path,
            pixel_values=np.zeros((1, 3, 64, 64), dtype=np.float32),
            extra=np.zeros((1,), dtype=np.float32),
        )
        with pytest.raises(click.UsageError, match="unexpected"):
            load_input_data(path, self._IO)

    def test_dtype_cast_with_warning(self, tmp_path, caplog) -> None:
        import logging

        import numpy as np

        from winml.modelkit.commands.perf import load_input_data

        # int64 literals against an int32 input: a normal run casts silently,
        # so load_input_data casts (with a warning) rather than hard-failing.
        io = {
            "input_names": ["input_ids"],
            "input_shapes": [[None, 8]],
            "input_types": ["int32"],
        }
        path = self._write_npz(tmp_path, input_ids=np.zeros((1, 8), dtype=np.int64))

        with caplog.at_level(logging.WARNING, logger="winml.modelkit.datasets.input_data"):
            inputs = load_input_data(path, io)

        assert inputs["input_ids"].dtype == np.int32
        assert "casting" in caplog.text.lower()

    def test_corrupt_npz_errors(self, tmp_path) -> None:
        from winml.modelkit.commands.perf import load_input_data

        path = tmp_path / "corrupt.npz"
        path.write_bytes(b"not an archive")
        with pytest.raises(click.UsageError, match="Could not read"):
            load_input_data(path, self._IO)

    def test_npy_rejected(self, tmp_path) -> None:
        import numpy as np

        from winml.modelkit.commands.perf import load_input_data

        path = tmp_path / "inputs.npy"
        np.save(path, np.zeros((1, 3, 64, 64), dtype=np.float32))
        with pytest.raises(click.UsageError, match=r"does not support \.npy"):
            load_input_data(path, self._IO)

    def test_non_npz_rejected(self, tmp_path) -> None:
        from winml.modelkit.commands.perf import load_input_data

        path = tmp_path / "inputs.bin"
        path.write_bytes(b"not an archive")
        with pytest.raises(click.UsageError, match=r"must be a \.npz"):
            load_input_data(path, self._IO)

    def test_generate_inputs_uses_provided_data(self, tmp_path) -> None:
        import numpy as np

        path = self._write_npz(tmp_path, pixel_values=np.zeros((7, 3, 64, 64), dtype=np.float32))
        config = BenchmarkConfig(model_id="m", batch_size=1, input_data=path)
        benchmark = PerfBenchmark(config)
        single = MagicMock()
        single.io_config = self._IO
        benchmark._model = single

        # No random generation when real inputs are supplied.
        with patch("winml.modelkit.commands.perf.generate_random_inputs") as mock_gen:
            benchmark._generate_inputs()

        mock_gen.assert_not_called()
        assert benchmark._inputs["pixel_values"].shape == (7, 3, 64, 64)
        # Effective batch is read from the provided data, not config.batch_size.
        assert benchmark._effective_batch == 7


class TestPerfInputDataCli:
    """CLI-level guards for --input-data."""

    def test_input_data_rejected_with_module(self, tmp_path, runner) -> None:
        import numpy as np

        path = tmp_path / "inputs.npz"
        np.savez(path, pixel_values=np.zeros((1, 3, 64, 64), dtype=np.float32))

        result = runner.invoke(
            perf,
            ["-m", "bert-base-uncased", "--module", "BertAttention", "--input-data", str(path)],
        )

        assert result.exit_code != 0
        assert "--input-data is not supported in --module mode" in result.output

    def test_input_data_composite_model_rejected(self, tmp_path) -> None:
        """Composite (dual-encoder) models reject --input-data up front.

        Composite-ness is only known after the model loads, so the guard lives
        in run() rather than with the --module / --runtime checks. Without it,
        each sub-model's child benchmark would hit a re-wrapped RuntimeError.
        """
        from unittest.mock import PropertyMock

        import numpy as np

        path = tmp_path / "inputs.npz"
        np.savez(path, pixel_values=np.zeros((1, 3, 64, 64), dtype=np.float32))

        config = BenchmarkConfig(model_id="openai/clip-vit-base-patch32", input_data=path)
        benchmark = PerfBenchmark(config)

        def fake_load(self) -> None:
            self._model = MagicMock()

        with (
            patch.object(PerfBenchmark, "_load_model", fake_load),
            patch.object(
                PerfBenchmark, "_is_composite", new_callable=PropertyMock, return_value=True
            ),
            pytest.raises(click.UsageError, match="composite"),
        ):
            benchmark.run()

    def test_shape_config_ignored_suppresses_overrides_print(self, tmp_path, runner) -> None:
        """--shape-config + --input-data: warn once, don't first announce the overrides.

        Printing "Shape overrides: {…}" and then immediately warning they're
        ignored is confusing, so the parse/print is skipped entirely when
        --input-data is set.
        """
        import numpy as np

        npz = tmp_path / "inputs.npz"
        np.savez(npz, pixel_values=np.zeros((1, 3, 64, 64), dtype=np.float32))
        shape_cfg = tmp_path / "shapes.json"
        shape_cfg.write_text(json.dumps({"height": 480, "width": 480}))

        def capture_config(config: BenchmarkConfig) -> MagicMock:
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
                    "bert-base-uncased",
                    "--input-data",
                    str(npz),
                    "--shape-config",
                    str(shape_cfg),
                    "-o",
                    str(tmp_path / "out.json"),
                ],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert "shape-config is ignored" in result.output
        assert "Shape overrides:" not in result.output


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

    @patch("winml.modelkit.commands.perf.PerfBenchmark")
    def test_cli_closes_benchmark_after_success(
        self, mock_benchmark_class: MagicMock, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Successful CLI runs release native sessions before process teardown."""
        mock_result = BenchmarkResult(
            config=BenchmarkConfig(model_id="test", output_path=tmp_path / "result.json"),
            actual_device="cpu",
            actual_task="cls",
            mean_ms=1.0,
            min_ms=1.0,
            max_ms=1.0,
            p50_ms=1.0,
            p90_ms=1.0,
            p95_ms=1.0,
            p99_ms=1.0,
            samples_per_sec=1.0,
            batches_per_sec=1.0,
        )
        mock_instance = MagicMock()
        mock_instance.run.return_value = mock_result
        mock_benchmark_class.return_value = mock_instance

        output_file = tmp_path / "result.json"

        result = runner.invoke(
            perf,
            ["-m", "test", "--output", str(output_file)],
            obj={},
        )

        assert result.exit_code == 0, result.output
        mock_instance.close.assert_called_once()


class TestDisplayConsoleReport:
    class _FailingConsoleFile:
        encoding = "utf-8"

        def write(self, _text: str) -> int:
            raise OSError(1, "Incorrect function")

        def flush(self) -> None:
            pass

        def isatty(self) -> bool:
            return True

    def test_prefers_adapter_block_over_gpu_aggregate(self) -> None:
        result = BenchmarkResult(
            config=BenchmarkConfig(model_id="microsoft/resnet-50", warmup=1),
            mean_ms=10.0,
            min_ms=9.0,
            max_ms=11.0,
            p50_ms=10.0,
            p90_ms=10.5,
            p95_ms=10.8,
            p99_ms=11.0,
            std_ms=0.5,
            warmup_mean_ms=12.0,
            samples_per_sec=100.0,
            effective_batch_size=1,
            actual_device="gpu",
            actual_task="image-classification",
            hw_monitor={
                "device_kind": "gpu",
                "adapter": {
                    "mean_pct": 91.2,
                    "peak_pct": 98.8,
                    "sample_count": 5,
                },
                "gpu": {
                    "mean_pct": 1.1,
                    "peak_pct": 2.2,
                    "sample_count": 11,
                    "luids": ["0x0_0xBEEF"],
                },
                "cpu": {"mean_pct": 12.3, "peak_pct": 34.5, "sample_count": 5},
                "ram": {"used_mb": 1024.0, "peak_mb": 2048.0},
                "device_memory": {"local_peak_mb": 0.0, "shared_peak_mb": 0.0},
                "running_time_ns": 0,
            },
        )
        console = Console(file=StringIO(), width=200, force_terminal=False, record=True)

        display_console_report(result, console)

        out = console.export_text()
        assert "GPU: 91.2% avg, 98.8% peak" in out
        assert "GPU: 1.1% avg, 2.2% peak" not in out

    def test_ignores_windows_console_write_oserror(self) -> None:
        result = BenchmarkResult(
            config=BenchmarkConfig(model_id="microsoft/resnet-50", warmup=1),
            mean_ms=10.0,
            min_ms=9.0,
            max_ms=11.0,
            p50_ms=10.0,
            p90_ms=10.5,
            p95_ms=10.8,
            p99_ms=11.0,
            std_ms=0.5,
            warmup_mean_ms=12.0,
            samples_per_sec=100.0,
            effective_batch_size=1,
            actual_device="gpu",
            actual_task="image-classification",
        )
        console = SafeConsole(file=self._FailingConsoleFile(), width=120, force_terminal=False)

        display_console_report(result, console)


class TestPerfSubmodel:
    """--submodel narrows a composite model to a single sub-component."""

    _COMPONENTS: ClassVar[dict[str, str]] = {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }

    def test_submodel_loads_component_as_single_with_its_task(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--submodel rewrites the load task to the component's own task."""
        captured: dict[str, BenchmarkConfig] = {}

        def capture_config(config: BenchmarkConfig) -> MagicMock:
            captured["config"] = config
            mock = MagicMock()
            mock.run.return_value = MagicMock()
            return mock

        with (
            patch(
                "winml.modelkit.commands.perf._resolve_composite_components_for_perf",
                return_value=dict(self._COMPONENTS),
            ),
            patch("winml.modelkit.commands.perf.PerfBenchmark", side_effect=capture_config),
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            result = runner.invoke(
                perf,
                [
                    "-m",
                    "google-t5/t5-small",
                    "--submodel",
                    "encoder",
                    "-o",
                    str(tmp_path / "out.json"),
                ],
                obj={},
            )

        assert result.exit_code == 0, result.output
        # The selected component is loaded as a single model using its own task.
        assert captured["config"].task == "feature-extraction"
        assert captured["config"].submodel == "encoder"

    def test_submodel_rejects_unknown_name(self, runner: CliRunner, tmp_path: Path) -> None:
        """--submodel with an invalid name is a clean error listing the available ones."""
        with (
            patch(
                "winml.modelkit.commands.perf._resolve_composite_components_for_perf",
                return_value=dict(self._COMPONENTS),
            ),
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_bench,
        ):
            result = runner.invoke(
                perf,
                ["-m", "google-t5/t5-small", "--submodel", "bogus"],
                obj={},
            )

        assert result.exit_code != 0
        assert "Unknown sub-model 'bogus'" in result.output
        assert "encoder" in result.output
        mock_bench.assert_not_called()

    def test_submodel_rejects_non_composite(self, runner: CliRunner, tmp_path: Path) -> None:
        """--submodel on a non-composite model is a clean error."""
        with (
            patch(
                "winml.modelkit.commands.perf._resolve_composite_components_for_perf",
                return_value=None,
            ),
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_bench,
        ):
            result = runner.invoke(
                perf,
                ["-m", "prajjwal1/bert-tiny", "--submodel", "encoder"],
                obj={},
            )

        assert result.exit_code != 0
        assert "not a composite model" in result.output
        mock_bench.assert_not_called()

    def test_submodel_rejects_onnx_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """--submodel on an ONNX file is rejected (already a single model)."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        result = runner.invoke(
            perf,
            ["-m", str(onnx_file), "--submodel", "encoder"],
            obj={},
        )

        assert result.exit_code != 0
        assert "not supported for ONNX files" in result.output

    def test_submodel_rejects_with_module(self, runner: CliRunner, tmp_path: Path) -> None:
        """--submodel cannot be combined with --module."""
        with patch(
            "winml.modelkit.commands.perf._resolve_composite_components_for_perf",
            return_value=dict(self._COMPONENTS),
        ):
            result = runner.invoke(
                perf,
                ["-m", "google-t5/t5-small", "--submodel", "encoder", "--module", "T5Block"],
                obj={},
            )

        assert result.exit_code != 0
        assert "cannot be combined with --module" in result.output


# =============================================================================
# --DURATION (TIME-BUDGETED BENCHMARKING)
# =============================================================================


class TestPerfDuration:
    """--duration runs the benchmark for a wall-clock budget instead of a fixed
    iteration count (ideal with --monitor; rejected with --op-tracing)."""

    def test_duration_shown_in_help(self, runner: CliRunner) -> None:
        result = runner.invoke(perf, ["--help"])
        assert result.exit_code == 0
        assert "--duration" in result.output

    def test_duration_forwarded_into_config(self, runner: CliRunner, tmp_path: Path) -> None:
        """--duration lands in BenchmarkConfig.duration for the benchmark run."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

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
                ["-m", str(onnx_file), "--duration", "5", "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert captured["config"].duration == 5.0

    def test_duration_defaults_to_none(self, runner: CliRunner, tmp_path: Path) -> None:
        """Without --duration the config keeps the iteration-count behavior."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

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
                ["-m", str(onnx_file), "-o", str(tmp_path / "out.json")],
                obj={},
            )

        assert result.exit_code == 0, result.output
        assert captured["config"].duration is None

    def test_duration_rejected_with_op_tracing(self, runner: CliRunner, tmp_path: Path) -> None:
        """--duration cannot be combined with --op-tracing."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        with patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_bench:
            result = runner.invoke(
                perf,
                ["-m", str(onnx_file), "--duration", "5", "--op-tracing", "basic"],
                obj={},
            )

        assert result.exit_code != 0
        assert "not valid with --op-tracing" in result.output
        mock_bench.assert_not_called()

    def test_duration_rejects_non_positive(self, runner: CliRunner, tmp_path: Path) -> None:
        """--duration must be strictly positive (a 0s budget benchmarks nothing)."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        result = runner.invoke(
            perf,
            ["-m", str(onnx_file), "--duration", "0"],
            obj={},
        )

        assert result.exit_code != 0

    @pytest.mark.parametrize("bad", ["nan", "inf"])
    def test_duration_rejects_non_finite(self, runner: CliRunner, tmp_path: Path, bad: str) -> None:
        """Non-finite --duration slips past FloatRange (nan/inf <= 0 is false) but
        would never terminate the timed loop, so it must be rejected up front."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake onnx")

        with patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_bench:
            result = runner.invoke(
                perf,
                ["-m", str(onnx_file), "--duration", bad],
                obj={},
            )

        assert result.exit_code != 0
        assert "finite" in result.output
        mock_bench.assert_not_called()


class TestBenchmarkIndices:
    """_benchmark_indices drives either a fixed iteration count or a timed loop."""

    def test_iteration_mode_yields_total(self) -> None:
        from winml.modelkit.commands.perf import _benchmark_indices

        indices = list(_benchmark_indices(total_iterations=5, warmup=2, duration_sec=None))
        assert indices == [0, 1, 2, 3, 4]

    def test_duration_mode_runs_warmup_then_timed_budget(self, monkeypatch) -> None:
        """Warmup indices come first, then the loop runs until the budget elapses."""
        from winml.modelkit.commands import perf as perf_mod

        clock = {"t": 0.0}

        def perf_counter() -> float:
            return clock["t"]

        monkeypatch.setattr(perf_mod.time, "perf_counter", perf_counter)

        indices = []
        # total_iterations is huge so only the time budget can end the loop.
        for idx in perf_mod._benchmark_indices(total_iterations=10_000, warmup=2, duration_sec=1.0):
            indices.append(idx)
            clock["t"] += 0.3  # advance 0.3s per iteration
            assert len(indices) < 100, "duration loop failed to terminate"

        # First two indices are warmup; the timed phase (budget captured at t=0.6)
        # runs until elapsed >= 1.0s.
        assert indices[:2] == [0, 1]
        assert indices == [0, 1, 2, 3, 4, 5]

    def test_duration_mode_runs_at_least_one_benchmark_iter(self, monkeypatch) -> None:
        """Even if the budget is already exceeded, one benchmark run still happens."""
        from winml.modelkit.commands import perf as perf_mod

        clock = {"t": 0.0}

        def perf_counter() -> float:
            return clock["t"]

        monkeypatch.setattr(perf_mod.time, "perf_counter", perf_counter)

        indices = []
        for idx in perf_mod._benchmark_indices(
            total_iterations=10_000, warmup=0, duration_sec=0.001
        ):
            indices.append(idx)
            clock["t"] += 10.0  # blow past the budget immediately
            assert len(indices) < 10, "duration loop failed to terminate"

        assert indices == [0]
