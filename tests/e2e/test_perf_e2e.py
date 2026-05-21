# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for the perf CLI command.

A single ``_PerfBenchmarkSuite`` base class defines every test; each concrete
subclass overrides the ``model_arg`` fixture to point at a different model
source (a generated ONNX file or a HuggingFace model id). The perf command
uses @click.pass_context and requires obj={}.

Markers:
    e2e: Full end-to-end test

Running these tests:
    E2E tests are auto-skipped unless ``-m e2e`` is explicitly passed
    (see ``tests/e2e/conftest.py``). GPU / NPU / QNN tests additionally
    skip when the required hardware or EP is not available on the host.

    # Run the full file
    uv run pytest -m e2e tests/e2e/test_perf_e2e.py

    # Run a single test
    uv run pytest -m e2e tests/e2e/test_perf_e2e.py::TestPerfONNXDirect::test_benchmark_cpu

    # Verbose output (per-test pass/skip lines)
    uv run pytest -m e2e -v tests/e2e/test_perf_e2e.py
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from tests.e2e.require_ep import require_ep
from winml.modelkit.commands.perf import perf


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [pytest.mark.e2e]


# ===========================================================================
# Constants and Helpers
# ===========================================================================

CPU_EPS = ("cpu", "openvino")
NPU_EPS = ("qnn", "vitisai", "openvino")
GPU_EPS = ("dml", "nv_tensorrt_rtx", "migraphx", "openvino")


def _require_gpu() -> None:
    """Skip the current test unless a GPU is discoverable via PDH."""
    if sys.platform != "win32":
        pytest.skip("GPU discovery via PDH is Windows-only")
    from winml.modelkit.session.monitor._pdh import PdhPoller

    if not PdhPoller.is_gpu_available():
        pytest.skip("No GPU detected via PDH")


def _require_npu() -> None:
    """Skip the current test unless an NPU is discoverable via PDH."""
    if sys.platform != "win32":
        pytest.skip("NPU discovery via PDH is Windows-only")
    from winml.modelkit.session.monitor._pdh import PdhPoller

    if not PdhPoller.is_npu_available():
        pytest.skip("No NPU detected via PDH")


def _assert_hw_monitor_section(data: dict, device_kind: str) -> None:
    """Assert the ``hw_monitor`` section is present and well-formed.

    Checks the section emitted by HWMonitor when --monitor is passed:
    presence, ``device_kind`` match, a non-null ``adapter_luid``, and a
    positive ``mean_pct`` for the per-device utilization block.
    """
    assert "hw_monitor" in data, "hw_monitor section missing with --monitor"
    hw = data["hw_monitor"]
    if device_kind == "cpu":
        # For CPU, device_kind is None and adapter_luid is None
        assert hw["device_kind"] is None
        assert hw["adapter_luid"] is None
    else:
        assert hw["device_kind"] == device_kind
        assert hw["adapter_luid"] is not None
        assert hw[device_kind]["mean_pct"] > 0


def _build_perf_args(
    *,
    model_arg: str,
    output_file: Path,
    device: str | None = None,
    ep: str | None = None,
    module: str | None = None,
    monitor: bool = False,
    verbose: bool = False,
) -> list[str]:
    """Build the argv list passed to the perf CLI.

    Iterations are fixed by ``monitor``: 300 when monitoring (HWMonitor needs
    enough samples to observe utilization) and 3 otherwise (kept tiny for
    e2e speed). Warmup is always 1.
    """
    iterations = 300 if monitor else 3
    args: list[str] = [
        "-m",
        model_arg,
        "--iterations",
        str(iterations),
        "--warmup",
        "1",
        "-o",
        str(output_file),
    ]
    if device is not None:
        args += ["--device", device]
    if ep is not None:
        args += ["--ep", ep]
    if module is not None:
        args += ["--module", module]
    if monitor:
        args.append("--monitor")
    if verbose:
        args.append("--verbose")
    return args


def _assert_monitor_result(data: dict, *, device: str, device_kind: str | None = None, ep: str | None = None) -> None:
    """Assert a monitored perf run produced the expected device + hw_monitor data.

    Verifies the resolved ``device`` in ``benchmark_info``, that latency was
    measured, and delegates the hw_monitor checks to
    :func:`_assert_hw_monitor_section`. ``device_kind`` defaults to ``device``
    when not given (only differs for cases like VitisAI where ``--device`` and
    the monitored hardware diverge).
    """
    if device_kind is None:
        device_kind = device
    assert data["benchmark_info"]["device"] == device
    assert data["latency_ms"]["mean"] > 0
    if ep is not None:
        assert data["benchmark_info"]["ep"] == ep
    _assert_hw_monitor_section(data, device_kind)


# ===========================================================================
# Shared test suite
# ===========================================================================


class _PerfBenchmarkSuite:
    """Shared perf-CLI tests. Subclasses override ``model_arg`` fixture."""

    @pytest.fixture
    def model_arg(self) -> str:
        raise NotImplementedError("Subclasses must override model_arg fixture")

    def test_benchmark_cpu(self, tmp_path: Path, model_arg: str):
        """Benchmark on CPU with minimal iterations.

        Uses --device cpu --iterations 3 --warmup 1 for speed.
        Verifies JSON output file is created with expected schema.
        """
        output_file = tmp_path / "perf_result.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(model_arg=model_arg, output_file=output_file, device="cpu"),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        # Verify JSON output file exists and has expected structure
        assert output_file.exists(), f"Output file not created: {output_file}"
        data = json.loads(output_file.read_text())

        # Verify top-level schema
        assert "benchmark_info" in data
        assert "model_info" in data
        assert "latency_ms" in data
        assert "throughput" in data
        assert "raw_samples_ms" in data

        # Verify benchmark_info
        binfo = data["benchmark_info"]
        assert binfo["iterations"] == 3
        assert binfo["warmup"] == 1
        assert binfo["device"] == "cpu"

        # Verify latency stats are populated
        latency = data["latency_ms"]
        assert latency["mean"] > 0
        assert latency["min"] > 0
        assert latency["p50"] > 0

        # Verify model_info has input/output names
        minfo = data["model_info"]
        assert isinstance(minfo["input_names"], list)
        assert len(minfo["input_names"]) >= 1
        assert isinstance(minfo["output_names"], list)
        assert len(minfo["output_names"]) >= 1

        # Verify raw samples count matches iterations
        assert len(data["raw_samples_ms"]) == 3

    def test_benchmark_verbose(self, tmp_path: Path, model_arg: str):
        """Benchmark with --verbose should succeed and show debug output."""
        output_file = tmp_path / "verbose_result.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="cpu", verbose=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"
        assert output_file.exists()
        assert "Results saved to" in result.output

    def test_benchmark_cpu_monitor(self, tmp_path: Path, model_arg: str):
        """Benchmark on CPU with --monitor.

        Requires a real CPU discoverable via PDH. Verifies the JSON output
        contains the hw_monitor section produced by HWMonitor.
        """

        output_file = tmp_path / "perf_cpu_monitor.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="cpu", monitor=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists(), f"Output file not created: {output_file}"
        data = json.loads(output_file.read_text())
        _assert_monitor_result(data, device="cpu")

    def test_benchmark_gpu_monitor(self, tmp_path: Path, model_arg: str):
        """Benchmark on GPU with --monitor.

        Requires a real GPU discoverable via PDH. Verifies the JSON output
        contains the hw_monitor section produced by HWMonitor.
        """
        _require_gpu()

        output_file = tmp_path / "perf_gpu_monitor.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="gpu", monitor=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists(), f"Output file not created: {output_file}"
        data = json.loads(output_file.read_text())
        _assert_monitor_result(data, device="gpu")

    def test_benchmark_npu_monitor(self, tmp_path: Path, model_arg: str):
        """Benchmark on NPU with --monitor.

        Requires a real NPU discoverable via PDH. Verifies the JSON output
        contains the hw_monitor section produced by HWMonitor.
        """
        _require_npu()

        output_file = tmp_path / "perf_npu_monitor.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="npu", monitor=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists(), f"Output file not created: {output_file}"
        data = json.loads(output_file.read_text())
        _assert_monitor_result(data, device="npu")

    def test_benchmark_auto(self, tmp_path: Path, model_arg: str):
        """Benchmark with --device auto.

        Auto resolves to whatever is available on the host and should always
        succeed (CPU is the universal fallback).
        """
        output_file = tmp_path / "perf_auto.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(model_arg=model_arg, output_file=output_file, device="auto"),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["benchmark_info"]["device"] == "auto"
        # At least a non-cpu should exist and picked up
        assert data["benchmark_info"]["ep"] != "CPUExecutionProvider"
        assert data["latency_ms"]["mean"] > 0

    def test_benchmark_ep_qnn(self, tmp_path: Path, model_arg: str):
        """Benchmark with --ep qnn.

        Skipped if QNNExecutionProvider is not available on the host.
        """
        require_ep("qnn")

        output_file = tmp_path / "perf_qnn.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(model_arg=model_arg, output_file=output_file, ep="qnn"),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["benchmark_info"]["ep"] == "QNNExecutionProvider"
        assert data["latency_ms"]["mean"] > 0

    def test_benchmark_ep_qnn_device_gpu(self, tmp_path: Path, model_arg: str):
        """Benchmark with --ep qnn and --device gpu.

        --ep overrides the device-to-provider mapping, so the session should
        bind to QNN even though the requested device is GPU. Skipped if QNN
        or a GPU is unavailable on the host.
        """
        require_ep("qnn")
        _require_gpu()

        output_file = tmp_path / "perf_qnn_gpu.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(model_arg=model_arg, output_file=output_file, device="gpu", ep="qnn"),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["benchmark_info"]["device"] == "gpu"
        assert data["benchmark_info"]["ep"] == "QNNExecutionProvider"
        assert data["latency_ms"]["mean"] > 0


# ===========================================================================
# Concrete suites
# ===========================================================================


class TestPerfONNXDirect(_PerfBenchmarkSuite):
    """Benchmark a pre-exported ONNX file directly via WinMLSession."""

    @pytest.fixture
    def model_arg(self, onnx_model_path: Path) -> str:
        return str(onnx_model_path)


class TestPerfHuggingFace:
    """Benchmark a HuggingFace model by loading it via the perf command."""

    @pytest.fixture
    def model_arg(self) -> str:
        return "microsoft/resnet-50"

    @pytest.mark.parametrize("ep", NPU_EPS)
    def test_benchmark_ep_npu(self, ep: str, tmp_path: Path, model_arg: str):
        """Benchmark with --ep vitisai.

        Skipped if VitisAIExecutionProvider is not available on the host.
        """
        require_ep("vitisai")

        output_file = tmp_path / "perf_vitisai.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, ep="vitisai", monitor=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["benchmark_info"]["ep"] == "VitisAIExecutionProvider"
        _assert_monitor_result(data, device="npu")


# ===========================================================================
# Per-module benchmark
# ===========================================================================


class TestPerfModule:
    """Per-module benchmark via --module on a HuggingFace model."""

    def test_module_benchmark_cpu(self, tmp_path: Path):
        """Per-module benchmark on CPU for ResNetStage submodules of resnet-50."""
        output_file = tmp_path / "perf_module.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg="microsoft/resnet-50",
                output_file=output_file,
                device="cpu",
                module="ResNetStage",
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists(), f"Output file not created: {output_file}"
        data = json.loads(output_file.read_text())

        assert data["model_id"] == "microsoft/resnet-50"
        assert data["module_class"] == "ResNetStage"
        assert data["iterations"] == 3
        assert data["warmup"] == 1
        assert data["instance_count"] == 4
        assert len(data["instances"]) == data["instance_count"]
        for instance in data["instances"]:
            assert instance["mean_ms"] > 0

    def test_module_invalid_lists_available(self, tmp_path: Path):
        """Invalid --module should fail and list available module classes."""
        output_file = tmp_path / "perf_module_invalid.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg="microsoft/resnet-50",
                output_file=output_file,
                device="cpu",
                module="NotAValidModuleXyz",
            ),
            obj={},
            catch_exceptions=False,
        )

        assert result.exit_code != 0, "perf should fail for an invalid --module"
        assert "No modules matching 'NotAValidModuleXyz' found" in result.output
        assert "Available module class names in this model:" in result.output
        # The real ResNetStage class should appear in the available list.
        assert "ResNetStage" in result.output
        assert not output_file.exists(), "Output file should not be written on failure"
