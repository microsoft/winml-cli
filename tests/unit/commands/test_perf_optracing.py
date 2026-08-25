# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the --op-tracing CLI option on winml perf and _resolve_ep_monitor."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


if TYPE_CHECKING:
    from pathlib import Path

from winml.modelkit.commands.perf import _resolve_ep_monitor, perf
from winml.modelkit.session.monitor.op_metrics import TraceFallbackReason


def _invoke_perf(args: list[str]):
    """Invoke perf CLI with PerfBenchmark.run mocked to prevent model loading."""
    runner = CliRunner()
    with patch(
        "winml.modelkit.commands.perf.PerfBenchmark.run",
        side_effect=RuntimeError("mocked — not running benchmark"),
    ):
        return runner.invoke(perf, args, obj={})


class TestOpTracingOptionParsing:
    """Verify --op-tracing is recognized and validates choices."""

    def test_option_is_recognized(self):
        """--op-tracing is accepted as a valid CLI option."""
        result = _invoke_perf(["--op-tracing", "basic", "-m", "nonexistent"])
        assert "no such option" not in (result.output or "").lower()

    def test_basic_choice_accepted(self):
        """--op-tracing basic is a valid choice."""
        result = _invoke_perf(["--op-tracing", "basic", "-m", "nonexistent"])
        assert "no such option" not in (result.output or "").lower()
        assert "invalid choice" not in (result.output or "").lower()

    def test_detail_choice_accepted(self):
        """--op-tracing detail is a valid choice."""
        result = _invoke_perf(["--op-tracing", "detail", "-m", "nonexistent"])
        assert "no such option" not in (result.output or "").lower()
        assert "invalid choice" not in (result.output or "").lower()

    def test_invalid_choice_rejected(self):
        """--op-tracing with an invalid value is rejected by Click."""
        runner = CliRunner()
        result = runner.invoke(perf, ["--op-tracing", "invalid", "-m", "test"])
        assert result.exit_code != 0
        output_lower = (result.output or "").lower()
        assert "invalid" in output_lower or "choice" in output_lower

    def test_case_insensitive(self):
        """--op-tracing accepts mixed-case values (e.g. Basic, DETAIL)."""
        result = _invoke_perf(["--op-tracing", "BASIC", "-m", "nonexistent"])
        assert "invalid choice" not in (result.output or "").lower()

    def test_without_op_tracing_flag(self):
        """Command works without --op-tracing (default is None)."""
        result = _invoke_perf(["-m", "nonexistent"])
        assert "no such option" not in (result.output or "").lower()

    def test_model_required_with_op_tracing(self):
        """--op-tracing alone without -m still requires a model."""
        runner = CliRunner()
        result = runner.invoke(perf, ["--op-tracing", "basic"])
        assert result.exit_code != 0


class TestResolveEpMonitor:
    """Unit tests for the _resolve_ep_monitor dispatch helper."""

    def test_no_op_tracing_no_ep_returns_null(self, tmp_path: Path):
        """With no op_tracing and no matching EP, returns NullEPMonitor."""
        from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor

        monitor = _resolve_ep_monitor(ep=None, op_tracing=None, output_dir=tmp_path)
        assert isinstance(monitor, NullEPMonitor)

    def test_no_op_tracing_cpu_ep_returns_null(self, tmp_path: Path):
        """CPU EP with no op_tracing yields NullEPMonitor."""
        from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor

        monitor = _resolve_ep_monitor(ep="cpu", op_tracing=None, output_dir=tmp_path)
        assert isinstance(monitor, NullEPMonitor)

    def test_vitisai_ep_no_op_tracing_returns_vitisai_when_available(self, tmp_path: Path):
        """vitisai EP with no op_tracing returns VitisAIMonitor when available."""
        from winml.modelkit.session.monitor.vitisai_monitor import VitisAIMonitor

        with patch.object(VitisAIMonitor, "is_available", return_value=True):
            monitor = _resolve_ep_monitor(ep="vitisai", op_tracing=None, output_dir=tmp_path)
        assert isinstance(monitor, VitisAIMonitor)

    def test_vitisai_ep_unavailable_returns_null(self, tmp_path: Path):
        """vitisai EP with no op_tracing returns NullEPMonitor when VitisAI is unavailable."""
        from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor
        from winml.modelkit.session.monitor.vitisai_monitor import VitisAIMonitor

        with patch.object(VitisAIMonitor, "is_available", return_value=False):
            monitor = _resolve_ep_monitor(ep="vitisai", op_tracing=None, output_dir=tmp_path)
        assert isinstance(monitor, NullEPMonitor)

    def test_op_tracing_qnn_available_returns_qnn_monitor(self, tmp_path: Path):
        """qnn EP with op_tracing returns QNNMonitor when QNN is available."""
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with patch.object(QNNMonitor, "is_available", return_value=True):
            monitor = _resolve_ep_monitor(ep="qnn", op_tracing="basic", output_dir=tmp_path)
        assert isinstance(monitor, QNNMonitor)

    def test_op_tracing_qnn_unavailable_raises(self, tmp_path: Path):
        """qnn EP with op_tracing raises RuntimeError when QNN is not available."""
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with (
            patch.object(QNNMonitor, "is_available", return_value=False),
            pytest.raises(RuntimeError, match="QNN is not available"),
        ):
            _resolve_ep_monitor(ep="qnn", op_tracing="basic", output_dir=tmp_path)

    def test_op_tracing_unsupported_ep_raises(self, tmp_path: Path):
        """Unsupported EP with op_tracing raises RuntimeError (NFR-2 hard-fail)."""
        with pytest.raises(RuntimeError, match="Op-tracing not available for EP 'dml'"):
            _resolve_ep_monitor(ep="dml", op_tracing="basic", output_dir=tmp_path)

    def test_op_tracing_passes_level_to_qnn_monitor(self, tmp_path: Path):
        """QNNMonitor receives the correct level from _resolve_ep_monitor."""
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with patch.object(QNNMonitor, "is_available", return_value=True):
            monitor = _resolve_ep_monitor(ep="qnn", op_tracing="detail", output_dir=tmp_path)
        assert isinstance(monitor, QNNMonitor)
        assert monitor._level == "detail"

    def test_auto_infers_qnn_from_npu_device(self, tmp_path: Path):
        """--device npu --op-tracing basic must engage QNNMonitor without --ep qnn (SC-1)."""
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with patch.object(QNNMonitor, "is_available", return_value=True):
            monitor = _resolve_ep_monitor(
                ep=None,
                op_tracing="basic",
                output_dir=tmp_path,
                device="npu",
            )
        assert isinstance(monitor, QNNMonitor)

    def test_auto_infers_qnn_from_npu_device_case_insensitive(self, tmp_path: Path):
        """--device NPU (uppercase) also auto-infers QNN."""
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with patch.object(QNNMonitor, "is_available", return_value=True):
            monitor = _resolve_ep_monitor(
                ep=None,
                op_tracing="basic",
                output_dir=tmp_path,
                device="NPU",
            )
        assert isinstance(monitor, QNNMonitor)

    @pytest.mark.parametrize("device_input", ["auto", "AUTO", "", None])
    def test_auto_infers_qnn_from_default_device_when_op_tracing(
        self, tmp_path: Path, device_input
    ):
        """--device auto (default) and empty/None must also auto-infer QNN.

        --op-tracing is itself a strong intent signal; users invoking the
        common pattern ``wmk perf -m <model> --op-tracing basic`` should not
        need to also pass --device npu.
        """
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with patch.object(QNNMonitor, "is_available", return_value=True):
            monitor = _resolve_ep_monitor(
                ep=None,
                op_tracing="basic",
                output_dir=tmp_path,
                device=device_input,
            )
        assert isinstance(monitor, QNNMonitor)

    @pytest.mark.parametrize("device_input", ["cpu", "gpu"])
    def test_explicit_non_npu_device_still_hard_fails(self, tmp_path: Path, device_input):
        """--device cpu/gpu --op-tracing basic must still hard-fail.

        Auto-infer only fires when device is unset (auto/empty) or npu;
        explicit user choice of cpu/gpu must be honored as "no, I do not
        want NPU" and produce a clear error rather than silently switching.
        """
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with (
            patch.object(QNNMonitor, "is_available", return_value=True),
            pytest.raises(RuntimeError, match="Op-tracing not available"),
        ):
            _resolve_ep_monitor(
                ep=None,
                op_tracing="basic",
                output_dir=tmp_path,
                device=device_input,
            )

    @pytest.mark.parametrize("ep_input", ["qnn", "QNN", "Qnn", "qNN"])
    def test_ep_matching_case_insensitive(self, tmp_path: Path, ep_input: str):
        """--ep QNN, --ep Qnn, --ep qnn all behave identically."""
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with patch.object(QNNMonitor, "is_available", return_value=True):
            monitor = _resolve_ep_monitor(
                ep=ep_input,
                op_tracing="basic",
                output_dir=tmp_path,
                device="npu",
            )
        assert isinstance(monitor, QNNMonitor)

    def test_npu_device_qnn_unavailable_raises_descriptive(self, tmp_path: Path):
        """--device npu --op-tracing when QNN unavailable raises with diagnostic message."""
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with (
            patch.object(QNNMonitor, "is_available", return_value=False),
            pytest.raises(RuntimeError, match="not available for EP"),
        ):
            _resolve_ep_monitor(
                ep=None,
                op_tracing="basic",
                output_dir=tmp_path,
                device="npu",
            )

    def test_explicit_qnn_ep_unavailable_message_mentions_install(self, tmp_path: Path):
        """When --ep qnn is explicit and unavailable, message hints at install paths."""
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with (
            patch.object(QNNMonitor, "is_available", return_value=False),
            pytest.raises(RuntimeError) as excinfo,
        ):
            _resolve_ep_monitor(
                ep="qnn",
                op_tracing="basic",
                output_dir=tmp_path,
                device="npu",
            )
        msg = str(excinfo.value)
        assert "QNN is not available" in msg
        assert "onnxruntime" in msg

    def test_op_tracing_openvino_uses_generic_unsupported_error(self, tmp_path: Path):
        """OpenVINO is not advertised as an op-tracing implementation."""
        with pytest.raises(RuntimeError) as excinfo:
            _resolve_ep_monitor(ep="openvino", op_tracing="basic", output_dir=tmp_path)

        message = str(excinfo.value)
        assert "not available for EP 'openvino'" in message
        assert "Supported EPs: qnn." in message

    def test_npu_op_tracing_without_qnn_raises_no_openvino_fallback(self, tmp_path: Path):
        """Unavailable QNN does not silently select another NPU monitor."""
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with (
            patch.object(QNNMonitor, "is_available", return_value=False),
            pytest.raises(RuntimeError, match="Op-tracing not available for EP"),
        ):
            _resolve_ep_monitor(ep=None, op_tracing="basic", output_dir=tmp_path, device="npu")

    def test_gpu_op_tracing_raises_no_openvino_fallback(self, tmp_path: Path):
        """No supported op-tracing backend is selected for a GPU request."""
        with pytest.raises(RuntimeError, match="Op-tracing not available for EP"):
            _resolve_ep_monitor(ep=None, op_tracing="basic", output_dir=tmp_path, device="gpu")

    def test_unsupported_ep_error_mentions_only_supported_ep(self, tmp_path: Path):
        """The diagnostic advertises only implemented tracing backends."""
        with pytest.raises(RuntimeError) as excinfo:
            _resolve_ep_monitor(
                ep="dml",
                op_tracing="basic",
                output_dir=tmp_path,
            )
        msg = str(excinfo.value)
        assert "qnn" in msg.lower()
        assert "openvino" not in msg.lower()


class TestOpTracingIterationsSmartDefault:
    """--op-tracing collapses default iterations to 1 unless user overrides."""

    @staticmethod
    def _capture_config(args: list[str]) -> dict:
        """Run perf CLI with BenchmarkConfig captured before benchmark runs."""
        captured: dict = {}
        runner = CliRunner()

        with (
            patch(
                "winml.modelkit.commands.perf.BenchmarkConfig",
                side_effect=lambda **kw: (captured.update(kw), _ConfigStub(**kw))[1],
            ),
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_bench,
        ):
            # Fail fast in benchmark to avoid model loading
            mock_bench.return_value.run.side_effect = RuntimeError("stop")
            runner.invoke(perf, args, obj={})
        return captured

    def test_op_tracing_without_iterations_collapses_to_1(self):
        """--op-tracing basic without --iterations -> iterations=1."""
        captured = self._capture_config(["--op-tracing", "basic", "-m", "fake/model"])
        assert captured.get("iterations") == 1

    def test_op_tracing_with_explicit_iterations_honored(self):
        """--op-tracing basic --iterations 50 -> iterations=50 (user override wins)."""
        captured = self._capture_config(
            ["--op-tracing", "basic", "--iterations", "50", "-m", "fake/model"]
        )
        assert captured.get("iterations") == 50

    def test_op_tracing_with_explicit_default_value_honored(self):
        """--op-tracing basic --iterations 100 -> iterations=100.

        Even when the user explicitly passes the value that matches the
        normal default, the smart override does not fire — the parameter
        source is COMMANDLINE, not DEFAULT.
        """
        captured = self._capture_config(
            ["--op-tracing", "basic", "--iterations", "100", "-m", "fake/model"]
        )
        assert captured.get("iterations") == 100

    def test_no_op_tracing_uses_default_100(self):
        """Without --op-tracing the normal default of 100 stands."""
        captured = self._capture_config(["-m", "fake/model"])
        assert captured.get("iterations") == 100


class TestDetailOpTracingAutoCompile:
    """Raw ONNX detail tracing automatically enables EPContext compilation."""

    @staticmethod
    def _capture_config(tmp_path: Path, *extra_args: str):
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"raw onnx")
        captured: dict = {}
        runner = CliRunner()

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch(
                "winml.modelkit.commands.perf.BenchmarkConfig",
                side_effect=lambda **kw: (captured.update(kw), _ConfigStub(**kw))[1],
            ),
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_bench,
        ):
            mock_bench.return_value.run.side_effect = RuntimeError("stop")
            result = runner.invoke(
                perf,
                [
                    "-m",
                    str(model_path),
                    "--ep",
                    "qnn",
                    "--device",
                    "npu",
                    "--op-tracing",
                    "detail",
                    "-o",
                    str(tmp_path / "result.json"),
                    *extra_args,
                ],
                obj={},
            )
        return result, captured

    def test_raw_onnx_enables_compile_pipeline(self, tmp_path: Path) -> None:
        result, captured = self._capture_config(tmp_path)

        assert result.exit_code != 0
        assert captured, (result.output, result.exception)
        assert captured["no_compile"] is False
        assert captured["skip_build"] is False
        assert captured["compile_ep_options"]["profiling_level"] == "optrace"
        assert captured["compile_ep_options"]["profiling_file_path"].endswith(
            "result_compile_optrace.csv"
        )
        assert "Raw ONNX detected" in result.output

    @pytest.mark.parametrize(
        ("flag", "message"),
        [
            ("--no-compile", "--no-compile prevents automatic compilation"),
            ("--skip-build", "--skip-build prevents automatic compilation"),
        ],
    )
    def test_explicit_compile_opt_out_is_rejected(
        self, tmp_path: Path, flag: str, message: str
    ) -> None:
        result, captured = self._capture_config(tmp_path, flag)

        assert result.exit_code == 2
        assert message in result.output
        assert captured == {}

    def test_basic_tracing_does_not_auto_compile(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"raw onnx")
        captured: dict = {}
        runner = CliRunner()

        with (
            patch(
                "winml.modelkit.commands.perf.BenchmarkConfig",
                side_effect=lambda **kw: (captured.update(kw), _ConfigStub(**kw))[1],
            ),
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_bench,
        ):
            mock_bench.return_value.run.side_effect = RuntimeError("stop")
            runner.invoke(
                perf,
                [
                    "-m",
                    str(model_path),
                    "--op-tracing",
                    "basic",
                    "-o",
                    str(tmp_path / "result.json"),
                ],
                obj={},
            )

        assert captured["no_compile"] is True
        assert captured["skip_build"] is True
        assert captured["compile_ep_options"] is None

    def test_unsupported_ep_does_not_auto_compile(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"raw onnx")
        captured: dict = {}
        runner = CliRunner()

        with (
            patch(
                "winml.modelkit.commands.perf.BenchmarkConfig",
                side_effect=lambda **kw: (captured.update(kw), _ConfigStub(**kw))[1],
            ),
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_bench,
        ):
            mock_bench.return_value.run.side_effect = RuntimeError("stop")
            result = runner.invoke(
                perf,
                [
                    "-m",
                    str(model_path),
                    "--ep",
                    "openvino",
                    "--device",
                    "npu",
                    "--op-tracing",
                    "detail",
                    "-o",
                    str(tmp_path / "result.json"),
                ],
                obj={},
            )

        assert result.exit_code != 0
        assert captured["no_compile"] is True
        assert captured["skip_build"] is True
        assert captured["compile_ep_options"] is None
        assert "Raw ONNX detected" not in result.output

    def test_auto_device_without_qnn_does_not_auto_compile(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"raw onnx")
        captured: dict = {}
        runner = CliRunner()

        with (
            patch(
                "winml.modelkit.session.monitor.qnn_monitor.QNNMonitor.is_available",
                return_value=False,
            ),
            patch(
                "winml.modelkit.commands.perf.BenchmarkConfig",
                side_effect=lambda **kw: (captured.update(kw), _ConfigStub(**kw))[1],
            ),
            patch("winml.modelkit.commands.perf.PerfBenchmark") as mock_bench,
        ):
            mock_bench.return_value.run.side_effect = RuntimeError("stop")
            result = runner.invoke(
                perf,
                [
                    "-m",
                    str(model_path),
                    "--op-tracing",
                    "detail",
                    "-o",
                    str(tmp_path / "result.json"),
                ],
                obj={},
            )

        assert result.exit_code != 0
        assert captured["no_compile"] is True
        assert captured["skip_build"] is True
        assert captured["compile_ep_options"] is None
        assert "Raw ONNX detected" not in result.output


class TestOpTracingHardwareMonitor:
    """--op-tracing must not implicitly enable system hardware monitoring."""

    def test_op_tracing_without_monitor_uses_only_ep_monitor(self):
        from winml.modelkit.commands.perf import BenchmarkConfig, PerfBenchmark

        config = BenchmarkConfig(
            model_id="fake/model",
            device="npu",
            ep="qnn",
            iterations=1,
            warmup=0,
            monitor=False,
            op_tracing="detail",
        )
        benchmark = PerfBenchmark(config)
        benchmark._inputs = {}

        ep_monitor = MagicMock()
        ep_monitor.result = None
        ep_monitor.to_dict.return_value = {"ep": "qnn"}
        ctx = MagicMock()
        ctx.monitor = ep_monitor
        ctx.stats = MagicMock()
        session = MagicMock()
        session.perf.return_value.__enter__.return_value = ctx
        benchmark._model = SimpleNamespace(
            _session=session,
            ep_name="QNNExecutionProvider",
            device="npu",
        )

        hw_monitor = MagicMock()
        with (
            patch(
                "winml.modelkit.commands.perf._open_ep_monitor_or_exit",
                return_value=ep_monitor,
            ),
            patch(
                "winml.modelkit.session.monitor.hw_monitor.HWMonitor",
                hw_monitor,
            ),
            patch("winml.modelkit.commands.perf._run_simple_loop") as simple_loop,
            patch("winml.modelkit.commands.perf._run_monitored_loop") as monitored_loop,
        ):
            result = benchmark._run_benchmark_monitored()

        assert result is ctx.stats
        hw_monitor.is_available.assert_not_called()
        hw_monitor.assert_not_called()
        simple_loop.assert_called_once()
        monitored_loop.assert_not_called()
        session.perf.assert_called_once_with(warmup=0, monitor=ep_monitor)

    def test_op_tracing_with_monitor_enables_hardware_monitor(self):
        from winml.modelkit.commands.perf import BenchmarkConfig, PerfBenchmark

        config = BenchmarkConfig(
            model_id="fake/model",
            device="npu",
            ep="qnn",
            iterations=1,
            warmup=0,
            monitor=True,
            op_tracing="detail",
        )
        benchmark = PerfBenchmark(config)
        benchmark._inputs = {}

        ep_monitor = MagicMock()
        ep_monitor.result = None
        ep_monitor.to_dict.return_value = {"ep": "qnn"}
        ctx = MagicMock()
        ctx.monitor = ep_monitor
        ctx.stats = MagicMock()
        session = MagicMock()
        session.perf.return_value.__enter__.return_value = ctx
        benchmark._model = SimpleNamespace(
            _session=session,
            ep_name="QNNExecutionProvider",
            device="npu",
        )
        benchmark._ep_device = SimpleNamespace(
            device=SimpleNamespace(
                device_type="NPU",
                ort_handle=SimpleNamespace(
                    device=SimpleNamespace(metadata={"LUID": "99219"})
                )
            )
        )

        hw = MagicMock()
        hw.__enter__.return_value = hw
        hw.to_dict.return_value = {"monitor": "HWMonitor"}
        hw_monitor = MagicMock(return_value=hw)
        hw_monitor.is_available.return_value = True
        with (
            patch(
                "winml.modelkit.commands.perf._open_ep_monitor_or_exit",
                return_value=ep_monitor,
            ),
            patch(
                "winml.modelkit.session.monitor.hw_monitor.HWMonitor",
                hw_monitor,
            ),
            patch("winml.modelkit.commands.perf._run_simple_loop") as simple_loop,
            patch("winml.modelkit.commands.perf._run_monitored_loop") as monitored_loop,
        ):
            result = benchmark._run_benchmark_monitored()

        assert result is ctx.stats
        hw_monitor.is_available.assert_called_once_with()
        hw_monitor.assert_called_once_with(
            poll_interval_ms=200,
            device="npu",
            ep_name="QNNExecutionProvider",
            adapter_luid="0x00000000_0x00018393",
            adapter_device="npu",
        )
        simple_loop.assert_not_called()
        monitored_loop.assert_called_once()


class TestEPDeviceMonitorBinding:
    def test_provider_options_select_matching_ep_device(self) -> None:
        from winml.modelkit.commands.perf import _get_ep_device_binding

        def make_device(device_id: str, luid: str) -> SimpleNamespace:
            return SimpleNamespace(
                device_type="GPU",
                ep_name="DmlExecutionProvider",
                ort_handle=SimpleNamespace(
                    ep_options={"device_id": device_id},
                    device=SimpleNamespace(metadata={"LUID": luid}),
                ),
            )

        gpu0 = make_device("0", "99219")
        gpu1 = make_device("1", "99220")
        ep_device = SimpleNamespace(
            device=gpu0,
            ep=SimpleNamespace(devices=(gpu0, gpu1)),
        )

        assert _get_ep_device_binding(ep_device, {"device_id": "1"}) == (
            "0x00000000_0x00018394",
            "gpu",
        )

    def test_unmatched_device_override_disables_bound_luid(self) -> None:
        from winml.modelkit.commands.perf import _get_ep_device_binding

        gpu = SimpleNamespace(
            device_type="GPU",
            ep_name="DmlExecutionProvider",
            ort_handle=SimpleNamespace(
                ep_options={"device_id": "0"},
                device=SimpleNamespace(metadata={"LUID": "99219"}),
            ),
        )
        ep_device = SimpleNamespace(
            device=gpu,
            ep=SimpleNamespace(devices=(gpu,)),
        )

        assert _get_ep_device_binding(ep_device, {"device_id": "1"}) == (
            None,
            None,
        )

    def test_unadvertised_qnn_backend_override_selects_matching_device(self) -> None:
        from winml.modelkit.commands.perf import _get_ep_device_binding

        def make_device(device_type: str, luid: str) -> SimpleNamespace:
            return SimpleNamespace(
                device_type=device_type,
                ep_name="QNNExecutionProvider",
                ort_handle=SimpleNamespace(
                    ep_options={},
                    device=SimpleNamespace(metadata={"LUID": luid}),
                ),
            )

        npu = make_device("NPU", "99219")
        gpu = make_device("GPU", "99220")
        ep_device = SimpleNamespace(
            device=npu,
            ep=SimpleNamespace(devices=(npu, gpu)),
        )

        assert _get_ep_device_binding(ep_device, {"backend_type": "gpu"}) == (
            "0x00000000_0x00018394",
            "gpu",
        )

    def test_qnn_backend_without_candidate_disables_adapter_monitoring(self) -> None:
        from winml.modelkit.commands.perf import (
            _get_ep_device_binding,
            _get_monitor_binding,
        )

        npu = SimpleNamespace(
            device_type="NPU",
            ep_name="QNNExecutionProvider",
            ort_handle=SimpleNamespace(
                ep_options={},
                device=SimpleNamespace(metadata={"LUID": "99219"}),
            ),
        )
        ep_device = SimpleNamespace(
            device=npu,
            ep=SimpleNamespace(devices=(npu,)),
        )

        assert _get_ep_device_binding(ep_device, {"backend_type": "gpu"}) == (
            None,
            "gpu",
        )
        assert _get_monitor_binding(
            ep_device,
            "npu",
            {"backend_type": "gpu"},
        ) == ("cpu", None, None)

    def test_qnn_backend_without_candidate_disables_vram_discovery(self) -> None:
        from winml.modelkit.commands.perf import BenchmarkConfig, PerfBenchmark

        npu = SimpleNamespace(
            device_type="NPU",
            ep_name="QNNExecutionProvider",
            ort_handle=SimpleNamespace(
                ep_options={},
                device=SimpleNamespace(metadata={"LUID": "99219"}),
            ),
        )
        ep_device = SimpleNamespace(
            device=npu,
            ep=SimpleNamespace(devices=(npu,)),
        )
        benchmark = PerfBenchmark(
            BenchmarkConfig(
                model_id="fake/model",
                device="npu",
                ep_options={"backend_type": "gpu"},
            )
        )
        benchmark._model = SimpleNamespace(
            device="npu",
            ep_name="QNNExecutionProvider",
        )
        benchmark._ep_device = ep_device

        with patch(
            "winml.modelkit.sysinfo.pdh_adapters.resolve_adapter_luid"
        ) as resolve_adapter_luid:
            assert benchmark._resolve_adapter_luid() is None

        resolve_adapter_luid.assert_not_called()

    def test_unknown_qnn_backend_disables_bound_adapter(self) -> None:
        from winml.modelkit.commands.perf import (
            _get_ep_device_binding,
            _get_monitor_binding,
        )

        npu = SimpleNamespace(
            device_type="NPU",
            ep_name="QNNExecutionProvider",
            ort_handle=SimpleNamespace(
                ep_options={},
                device=SimpleNamespace(metadata={"LUID": "99219"}),
            ),
        )
        ep_device = SimpleNamespace(
            device=npu,
            ep=SimpleNamespace(devices=(npu,)),
        )

        assert _get_ep_device_binding(ep_device, {"backend_type": "unknown"}) == (
            None,
            None,
        )
        assert _get_monitor_binding(
            ep_device,
            "npu",
            {"backend_type": "unknown"},
        ) == ("cpu", None, None)

    def test_qnn_cpu_backend_disables_accelerator_monitoring(self) -> None:
        from winml.modelkit.commands.perf import _get_monitor_binding

        npu = SimpleNamespace(
            device_type="NPU",
            ep_name="QNNExecutionProvider",
            ort_handle=SimpleNamespace(
                ep_options={},
                device=SimpleNamespace(metadata={"LUID": "99219"}),
            ),
        )
        ep_device = SimpleNamespace(
            device=npu,
            ep=SimpleNamespace(devices=(npu,)),
        )

        assert _get_monitor_binding(
            ep_device,
            "npu",
            {"backend_type": "cpu"},
        ) == ("cpu", None, None)


class _ConfigStub:
    """Lightweight stand-in for BenchmarkConfig used by capture tests."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _benchmark_result_for_json():
    from winml.modelkit.commands.perf import BenchmarkResult

    config = SimpleNamespace(
        model_id="fake/model",
        task=None,
        device="npu",
        precision="auto",
        iterations=1,
        warmup=0,
        batch_size=1,
        ep_source=None,
        ep_options=None,
    )
    return BenchmarkResult(
        config=config,
        input_names=["input"],
        input_shapes=[[1]],
        input_types=["float32"],
        output_names=["output"],
        output_shapes=[[1]],
        mean_ms=1.0,
        min_ms=1.0,
        max_ms=1.0,
        raw_samples_ms=[1.0],
        samples_per_sec=1.0,
        batches_per_sec=1.0,
        actual_device="npu",
        actual_task="fake-task",
        running_model_path="model.onnx",
    )


def _mock_benchmark_with_trace(trace_result):
    mock_ctx = MagicMock()
    mock_ctx.monitor.result = trace_result
    mock_benchmark = MagicMock()
    mock_benchmark.run.return_value = _benchmark_result_for_json()
    mock_benchmark._perf_ctx = mock_ctx
    return mock_benchmark


def _invoke_json_op_trace_failure(tmp_path: Path, trace_result):
    mock_benchmark = _mock_benchmark_with_trace(trace_result)

    runner = CliRunner()
    with (
        patch(
            "winml.modelkit.commands.perf.PerfBenchmark",
            return_value=mock_benchmark,
        ),
        patch("winml.modelkit.commands.perf.write_json_report"),
    ):
        return runner.invoke(
            perf,
            [
                "-m",
                "fake/model",
                "--device",
                "npu",
                "--op-tracing",
                "basic",
                "--format",
                "json",
                "-o",
                str(tmp_path / "perf_result.json"),
            ],
            obj={},
        )


def _invoke_text_op_trace_failure(tmp_path: Path, trace_result):
    mock_benchmark = _mock_benchmark_with_trace(trace_result)

    runner = CliRunner()
    with (
        patch(
            "winml.modelkit.commands.perf.PerfBenchmark",
            return_value=mock_benchmark,
        ),
        patch("winml.modelkit.commands.perf.display_console_report") as display_report,
        patch("winml.modelkit.commands.perf.write_json_report") as write_json,
        patch("winml.modelkit.session.monitor.report.display_op_trace_report") as display_trace,
        patch("winml.modelkit.session.monitor.report.write_op_trace_json") as write_trace,
    ):
        result = runner.invoke(
            perf,
            [
                "-m",
                "fake/model",
                "--device",
                "npu",
                "--op-tracing",
                "basic",
                "-o",
                str(tmp_path / "perf_result.json"),
            ],
            obj={},
        )

    return result, display_report, write_json, display_trace, write_trace


class TestCliOpTracingDispatch:
    """CLI-level integration tests for --op-tracing dispatch (mocked benchmark)."""

    @pytest.mark.parametrize(
        ("status", "error"),
        [
            ("no_data", "profiler CSV missing"),
            ("parse_failed", "invalid CSV header"),
            ("not_run", None),
        ],
    )
    def test_json_mode_failure_status_does_not_emit_benchmark_json(
        self, tmp_path: Path, status: str, error: str | None
    ) -> None:
        from winml.modelkit.session.monitor.op_metrics import OpTraceResult

        trace = OpTraceResult(
            model="fake/model",
            device="npu",
            tracing_level="basic",
            status=status,
            error=error,
        )

        result = _invoke_json_op_trace_failure(tmp_path, trace)

        assert result.exit_code == 4
        assert "benchmark_info" not in result.output
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.output)

    def test_text_mode_not_run_status_exits_before_success_reports(self, tmp_path: Path) -> None:
        from winml.modelkit.session.monitor.op_metrics import OpTraceResult

        trace = OpTraceResult(
            model="fake/model",
            device="npu",
            tracing_level="basic",
            status="not_run",
        )

        result, display_report, write_json, display_trace, write_trace = (
            _invoke_text_op_trace_failure(tmp_path, trace)
        )

        assert result.exit_code == 4
        assert "not_run" in result.output
        display_report.assert_not_called()
        write_json.assert_not_called()
        display_trace.assert_not_called()
        write_trace.assert_not_called()

    def test_json_mode_missing_trace_result_does_not_emit_benchmark_json(
        self, tmp_path: Path
    ) -> None:
        result = _invoke_json_op_trace_failure(tmp_path, None)

        assert result.exit_code == 4
        assert "benchmark_info" not in result.output
        with pytest.raises(json.JSONDecodeError):
            json.loads(result.output)

    def test_no_data_status_exits_4(self, tmp_path: Path):
        """When op-tracing returns status='no_data', CLI exits 4 — not exit 0 with warning."""
        from unittest.mock import MagicMock

        from winml.modelkit.commands.perf import BenchmarkResult
        from winml.modelkit.session.monitor.op_metrics import OpTraceResult

        # Fabricate a BenchmarkResult and a no_data OpTraceResult.
        config = MagicMock()
        config.model_id = "fake/model"
        config.task = None
        config.device = "npu"
        config.precision = "auto"
        config.iterations = 1
        config.warmup = 0
        config.batch_size = 1
        bench_result = BenchmarkResult(config=config)

        trace = OpTraceResult(
            model="fake/model",
            device="npu",
            tracing_level="basic",
            status="no_data",
            error="profiler CSV missing",
        )

        # Mock benchmark to return the fabricated result and expose _perf_ctx.
        mock_ctx = MagicMock()
        mock_ctx.monitor.result = trace
        mock_benchmark = MagicMock()
        mock_benchmark.run.return_value = bench_result
        mock_benchmark._perf_ctx = mock_ctx

        runner = CliRunner()
        with (
            patch(
                "winml.modelkit.commands.perf.PerfBenchmark",
                return_value=mock_benchmark,
            ),
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            result = runner.invoke(
                perf,
                ["-m", "fake/model", "--device", "npu", "--op-tracing", "basic"],
                obj={},
            )

        assert result.exit_code == 4, f"Expected exit 4, got {result.exit_code}: {result.output}"
        assert "no profiling data" in result.output.lower()

    def test_parse_failed_status_exits_4(self, tmp_path: Path):
        """parse_failed status exits 4 with the parser error message."""
        from unittest.mock import MagicMock

        from winml.modelkit.commands.perf import BenchmarkResult
        from winml.modelkit.session.monitor.op_metrics import OpTraceResult

        config = MagicMock()
        config.model_id = "fake/model"
        config.device = "npu"
        config.precision = "auto"
        config.iterations = 1
        config.warmup = 0
        config.batch_size = 1
        config.task = None
        bench_result = BenchmarkResult(config=config)

        trace = OpTraceResult(
            model="fake/model",
            device="npu",
            tracing_level="detail",
            status="parse_failed",
            error="invalid CSV header",
        )
        mock_ctx = MagicMock()
        mock_ctx.monitor.result = trace
        mock_benchmark = MagicMock()
        mock_benchmark.run.return_value = bench_result
        mock_benchmark._perf_ctx = mock_ctx

        runner = CliRunner()
        with (
            patch(
                "winml.modelkit.commands.perf.PerfBenchmark",
                return_value=mock_benchmark,
            ),
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
        ):
            result = runner.invoke(
                perf,
                ["-m", "fake/model", "--device", "npu", "--op-tracing", "detail"],
                obj={},
            )

        assert result.exit_code == 4
        assert "parse failed" in result.output.lower()
        assert "invalid CSV header" in result.output

    def test_basic_fallback_status_exits_0_with_notice(self, tmp_path: Path):
        """basic_fallback status is degraded-success (exit 0 with yellow notice)."""
        from unittest.mock import MagicMock

        from winml.modelkit.commands.perf import BenchmarkResult
        from winml.modelkit.session.monitor.op_metrics import OpTraceResult

        config = MagicMock()
        config.model_id = "fake/model"
        config.device = "npu"
        config.precision = "auto"
        config.iterations = 1
        config.warmup = 0
        config.batch_size = 1
        config.task = None
        bench_result = BenchmarkResult(config=config)
        bench_result.running_model_path = str(tmp_path / "model_ctx.onnx")

        trace = OpTraceResult(
            model="fake/model",
            device="npu",
            tracing_level="detail",
            status="basic_fallback",
            fallback_reason=TraceFallbackReason.SCHEMATIC_MISSING,
        )
        mock_ctx = MagicMock()
        mock_ctx.monitor.result = trace
        mock_benchmark = MagicMock()
        mock_benchmark.run.return_value = bench_result
        mock_benchmark._perf_ctx = mock_ctx

        runner = CliRunner()
        with (
            patch(
                "winml.modelkit.commands.perf.PerfBenchmark",
                return_value=mock_benchmark,
            ),
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
            patch("winml.modelkit.session.monitor.report.display_op_trace_report"),
            patch("winml.modelkit.session.monitor.report.write_op_trace_json"),
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=True),
        ):
            result = runner.invoke(
                perf,
                ["-m", "fake/model", "--device", "npu", "--op-tracing", "detail"],
                obj={},
            )

        assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
        assert "schematic" in result.output.lower()
        assert "raw onnx" in result.output.lower()
        assert "qnn_sdk_root" not in result.output.lower()

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            (TraceFallbackReason.QNN_LOG_MISSING, "optrace log"),
            (TraceFallbackReason.SCHEMATIC_MISSING, "raw ONNX"),
            (TraceFallbackReason.SDK_MISSING, "QNN_SDK_ROOT"),
            (TraceFallbackReason.VIEWER_FAILED, "viewer"),
            (TraceFallbackReason.QHAS_OUTPUT_MISSING, "output was not found"),
            (TraceFallbackReason.QHAS_PARSE_FAILED, "could not be parsed"),
            (None, "post-processing was unavailable"),
        ],
    )
    def test_detail_fallback_guidance_is_reason_specific(
        self, reason: TraceFallbackReason | None, expected: str
    ) -> None:
        from winml.modelkit.commands.perf import _detail_fallback_guidance

        assert expected in _detail_fallback_guidance(reason)

    def test_detail_fallback_guidance_distinguishes_schematic_publish_failure(self) -> None:
        from winml.modelkit.commands.perf import _detail_fallback_guidance

        guidance = _detail_fallback_guidance(TraceFallbackReason.SCHEMATIC_PUBLISH_FAILED)

        assert "copy" in guidance.lower() or "publish" in guidance.lower()
        assert "raw ONNX" not in guidance

    def test_basic_fallback_status_rejects_raw_running_model(self, tmp_path: Path):
        """Detail tracing cannot degrade successfully when ORT ran the raw model."""
        from unittest.mock import MagicMock

        from winml.modelkit.commands.perf import BenchmarkResult
        from winml.modelkit.session.monitor.op_metrics import OpTraceResult

        config = MagicMock()
        config.model_id = "fake/model"
        config.device = "npu"
        config.precision = "auto"
        config.iterations = 1
        config.warmup = 0
        config.batch_size = 1
        config.task = None
        bench_result = BenchmarkResult(config=config)
        bench_result.running_model_path = str(tmp_path / "raw.onnx")

        trace = OpTraceResult(
            model="fake/model",
            device="npu",
            tracing_level="detail",
            status="basic_fallback",
        )
        mock_ctx = MagicMock()
        mock_ctx.monitor.result = trace
        mock_benchmark = MagicMock()
        mock_benchmark.run.return_value = bench_result
        mock_benchmark._perf_ctx = mock_ctx

        runner = CliRunner()
        with (
            patch(
                "winml.modelkit.commands.perf.PerfBenchmark",
                return_value=mock_benchmark,
            ),
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report"),
            patch("winml.modelkit.session.monitor.report.display_op_trace_report"),
            patch("winml.modelkit.session.monitor.report.write_op_trace_json"),
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
        ):
            result = runner.invoke(
                perf,
                ["-m", "fake/model", "--device", "npu", "--op-tracing", "detail"],
                obj={},
            )

        assert result.exit_code == 4
        assert "epcontext" in result.output.lower()

    def test_no_data_status_does_not_write_json(self, tmp_path: Path):
        """A3: when op-tracing returns status='no_data', the benchmark JSON
        must NOT be written to disk before the CLI exits 4.

        Pre-fix: write_json_report() ran before the op-trace status check,
        so the JSON artifact was always written — misleading CI consumers
        that parsed the file despite the exit-4 signal.

        Post-fix: the JSON write is inside the valid-trace branch, after all
        sys.exit(4) guards.
        """
        from unittest.mock import MagicMock, call, patch

        from winml.modelkit.commands.perf import BenchmarkResult
        from winml.modelkit.session.monitor.op_metrics import OpTraceResult

        config = MagicMock()
        config.model_id = "fake/model"
        config.task = None
        config.device = "npu"
        config.precision = "auto"
        config.iterations = 1
        config.warmup = 0
        config.batch_size = 1
        bench_result = BenchmarkResult(config=config)

        trace = OpTraceResult(
            model="fake/model",
            device="npu",
            tracing_level="basic",
            status="no_data",
            error="profiler CSV missing",
        )
        mock_ctx = MagicMock()
        mock_ctx.monitor.result = trace
        mock_benchmark = MagicMock()
        mock_benchmark.run.return_value = bench_result
        mock_benchmark._perf_ctx = mock_ctx

        write_calls: list = []

        def _capture_write(*args, **kwargs):
            write_calls.append(call(*args, **kwargs))

        runner = CliRunner()
        with (
            patch(
                "winml.modelkit.commands.perf.PerfBenchmark",
                return_value=mock_benchmark,
            ),
            patch("winml.modelkit.commands.perf.display_console_report"),
            patch("winml.modelkit.commands.perf.write_json_report", side_effect=_capture_write),
        ):
            output_path = tmp_path / "perf_result.json"
            result = runner.invoke(
                perf,
                [
                    "-m",
                    "fake/model",
                    "--device",
                    "npu",
                    "--op-tracing",
                    "basic",
                    "-o",
                    str(output_path),
                ],
                obj={},
            )

        assert result.exit_code == 4, f"Expected exit 4, got {result.exit_code}: {result.output}"
        assert write_calls == [], (
            f"write_json_report must NOT be called when op-trace status='no_data' "
            f"(exit 4 path); got {len(write_calls)} call(s)"
        )


# ===========================================================================
# Hardware-gated CLI E2E (SC-1)
#
# PRD §10.5 / coreloop §8.4 mandate this test:
#   "test_cli_op_tracing_basic_on_qnn (skip if no QNN NPU): runs
#    wmk perf -m resnet50 --device npu --op-tracing basic, asserts CSV
#    produced, *_op_trace.json written, at least one operator entry."
#
# This is the only end-to-end proof that SC-1 holds: the headline
# invocation produces real per-operator trace data on a QNN NPU.
# The test is doubly-gated:
#   * QNNMonitor.is_available() — actual hardware/runtime probe.
#   * WINML_TEST_NPU=1 env var — explicit opt-in (matches existing
#     project pattern for NPU-bound tests).
# Without either, the test skips cleanly (Cardinal Rule 3 allows
# hardware-gated skipif).
# ===========================================================================


@pytest.mark.skipif(
    __import__("os").environ.get("WINML_TEST_NPU", "0") != "1",
    reason="Hardware-gated SC-1 test requires WINML_TEST_NPU=1 + QNN NPU",
)
def test_cli_op_tracing_basic_on_qnn(tmp_path):
    """SC-1 end-to-end: ``wmk perf --device npu --op-tracing basic`` on QNN.

    Hardware-gated. Must produce:
      * a profiling CSV under the monitor's output directory,
      * a ``*_op_trace.json`` next to the perf JSON output,
      * at least one operator entry, with ``status == "ok"``.

    A regression that silently falls back to CPU (the bug SC-1 explicitly
    targets — see PRD §3) would emit ``status == "no_data"`` here and
    ``test_no_data_status_exits_4`` would catch it logically. This test
    proves the happy path on real hardware.
    """
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    if not QNNMonitor.is_available():
        pytest.skip("QNN EP not available on this system")

    runner = CliRunner()
    output_path = tmp_path / "perf_result.json"
    result = runner.invoke(
        perf,
        [
            "-m",
            "microsoft/resnet-50",
            "--device",
            "npu",
            "--op-tracing",
            "basic",
            "--iterations",
            "10",
            "--warmup",
            "2",
            "-o",
            str(output_path),
        ],
        obj={},
        catch_exceptions=False,
    )

    assert result.exit_code == 0, (
        f"perf --op-tracing basic failed (exit {result.exit_code}):\n{result.output}"
    )

    # Per-op trace JSON written next to the perf output.
    trace_files = list(tmp_path.glob("*_op_trace.json"))
    assert trace_files, (
        f"Expected *_op_trace.json next to {output_path}; got: {list(tmp_path.iterdir())}"
    )

    import json

    trace_data = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert trace_data["status"] == "ok", (
        f"Expected status='ok' on real hardware, got {trace_data['status']!r} "
        f"with error={trace_data.get('error')!r}"
    )
    assert trace_data["operators"], (
        "Expected at least one operator entry; got 0. "
        "This is the canonical SC-1 silent-CPU-fallback signature."
    )
    # CSV path recorded in artifacts and present on disk.
    csv_path_str = trace_data["artifacts"].get("csv")
    assert csv_path_str, "Expected 'csv' key in artifacts"
    from pathlib import Path as _Path

    assert _Path(csv_path_str).is_file(), f"Expected profiling CSV at {csv_path_str}"
