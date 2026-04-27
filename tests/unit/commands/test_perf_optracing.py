# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the --op-tracing CLI option on winml perf and _resolve_ep_monitor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from click.testing import CliRunner


if TYPE_CHECKING:
    from pathlib import Path

from winml.modelkit.commands.perf import _resolve_ep_monitor, perf


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


class TestCliOpTracingDispatch:
    """CLI-level integration tests for --op-tracing dispatch (mocked benchmark)."""

    def test_onnx_input_with_op_tracing_fails_at_parse_time(self, tmp_path: Path):
        """--op-tracing on a .onnx input must fail BEFORE running the benchmark."""
        runner = CliRunner()
        onnx_file = tmp_path / "fake.onnx"
        onnx_file.write_bytes(b"")

        # Patch _run_onnx_benchmark to detect if it was called (it must NOT be).
        with patch(
            "winml.modelkit.commands.perf._run_onnx_benchmark",
        ) as mock_run:
            result = runner.invoke(
                perf,
                ["-m", str(onnx_file), "--op-tracing", "basic"],
                obj={},
            )

        assert result.exit_code != 0
        assert "not yet supported for direct ONNX" in result.output
        mock_run.assert_not_called()

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
        ):
            result = runner.invoke(
                perf,
                ["-m", "fake/model", "--device", "npu", "--op-tracing", "detail"],
                obj={},
            )

        assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
        assert "degraded" in result.output.lower() or "notice" in result.output.lower()


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
