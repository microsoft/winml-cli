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
# Helpers
# ===========================================================================


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
            [
                "-m",
                model_arg,
                "--device",
                "cpu",
                "--iterations",
                "3",
                "--warmup",
                "1",
                "-o",
                str(output_file),
            ],
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
            [
                "-m",
                model_arg,
                "--device",
                "cpu",
                "--iterations",
                "2",
                "--warmup",
                "1",
                "-o",
                str(output_file),
                "--verbose",
            ],
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"
        assert output_file.exists()
        assert "Results saved to" in result.output

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
            [
                "-m",
                model_arg,
                "--device",
                "gpu",
                "--iterations",
                "100",
                "--warmup",
                "1",
                "-o",
                str(output_file),
                "--monitor",
            ],
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists(), f"Output file not created: {output_file}"
        data = json.loads(output_file.read_text())

        assert data["benchmark_info"]["device"] == "gpu"
        assert data["latency_ms"]["mean"] > 0
        assert "hw_monitor" in data, "hw_monitor section missing with --monitor"
        assert data["hw_monitor"]["device_kind"] == "gpu"
        assert data["hw_monitor"]["adapter_luid"] is not None
        assert data["hw_monitor"]["gpu"]["mean_pct"] > 0

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
            [
                "-m",
                model_arg,
                "--device",
                "npu",
                "--iterations",
                "100",
                "--warmup",
                "1",
                "-o",
                str(output_file),
                "--monitor",
            ],
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists(), f"Output file not created: {output_file}"
        data = json.loads(output_file.read_text())

        assert data["benchmark_info"]["device"] == "npu"
        assert data["latency_ms"]["mean"] > 0
        assert "hw_monitor" in data, "hw_monitor section missing with --monitor"
        assert data["hw_monitor"]["device_kind"] == "npu"
        assert data["hw_monitor"]["adapter_luid"] is not None
        assert data["hw_monitor"]["npu"]["mean_pct"] > 0

    def test_benchmark_auto(self, tmp_path: Path, model_arg: str):
        """Benchmark with --device auto.

        Auto resolves to whatever is available on the host and should always
        succeed (CPU is the universal fallback).
        """
        output_file = tmp_path / "perf_auto.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            [
                "-m",
                model_arg,
                "--device",
                "auto",
                "--iterations",
                "3",
                "--warmup",
                "1",
                "-o",
                str(output_file),
            ],
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["benchmark_info"]["device"] == "auto"
        # At leat a non-cpu should exist and picked up
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
            [
                "-m",
                model_arg,
                "--ep",
                "qnn",
                "--iterations",
                "3",
                "--warmup",
                "1",
                "-o",
                str(output_file),
            ],
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
            [
                "-m",
                model_arg,
                "--device",
                "gpu",
                "--ep",
                "qnn",
                "--iterations",
                "3",
                "--warmup",
                "1",
                "-o",
                str(output_file),
            ],
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
    def model_arg(self, onnx_model_path: Path | None = None) -> str:
        if onnx_model_path is None:
            raise RuntimeError("Expected pytest to inject fixture 'onnx_model_path'")
        return str(onnx_model_path)


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
            [
                "-m",
                "microsoft/resnet-50",
                "--module",
                "ResNetStage",
                "--device",
                "cpu",
                "--iterations",
                "3",
                "--warmup",
                "1",
                "-o",
                str(output_file),
            ],
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
            [
                "-m",
                "microsoft/resnet-50",
                "--module",
                "NotAValidModuleXyz",
                "--device",
                "cpu",
                "--iterations",
                "3",
                "--warmup",
                "1",
                "-o",
                str(output_file),
            ],
            obj={},
            catch_exceptions=False,
        )

        assert result.exit_code != 0, "perf should fail for an invalid --module"
        assert "No modules matching 'NotAValidModuleXyz' found" in result.output
        assert "Available module class names in this model:" in result.output
        # The real ResNetStage class should appear in the available list.
        assert "ResNetStage" in result.output
        assert not output_file.exists(), "Output file should not be written on failure"
