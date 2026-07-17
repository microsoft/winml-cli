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

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import onnx
import pytest
from click.testing import CliRunner

from tests.e2e.require_ep import require_ep
from winml.modelkit.commands.perf import perf
from winml.modelkit.utils.constants import EP_ALIASES


pytestmark = [pytest.mark.e2e]


# ===========================================================================
# Constants and Helpers
# ===========================================================================

CPU_EPS = ("cpu", "openvino")
NPU_EPS = ("qnn", "vitisai", "openvino")
GPU_EPS = ("dml", "nv_tensorrt_rtx", "migraphx", "openvino", "qnn")
NON_CPU_EPS = ("qnn", "vitisai", "dml", "nv_tensorrt_rtx", "migraphx")


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


def _assert_hw_monitor_section(
    data: dict, device_kind: str, *, require_utilization: bool = True
) -> None:
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
        if require_utilization:
            assert hw[device_kind]["mean_pct"] > 0


def _build_perf_args(
    *,
    model_arg: str,
    output_file: Path,
    device: str | None = None,
    ep: str | None = None,
    module: str | None = None,
    monitor: bool = False,
    memory: bool | None = None,
    verbose: bool = False,
    no_skip_build: bool = False,
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
    if memory is True:
        args.append("--memory")
    elif memory is False:
        args.append("--no-memory")
    if verbose:
        args.append("--verbose")
    if no_skip_build:
        args.append("--no-skip-build")
    return args


def _assert_monitor_result(
    data: dict,
    *,
    device: str,
    device_kind: str | None = None,
    ep: str | None = None,
    require_utilization: bool = True,
) -> None:
    """Assert a monitored perf run produced the expected device + hw_monitor data.

    Verifies the resolved ``device`` in ``benchmark_info``, that latency was
    measured, and delegates the hw_monitor checks to
    :func:`_assert_hw_monitor_section`. ``device_kind`` defaults to ``device``
    when not given (only differs for cases like VitisAI where ``--device`` and
    the monitored hardware diverge). ``require_utilization`` is forwarded.
    """
    if device_kind is None:
        device_kind = device
    assert data["benchmark_info"]["device"] == device
    assert data["latency_ms"]["mean"] > 0
    if ep is not None:
        assert data["benchmark_info"]["ep"] == ep
    _assert_hw_monitor_section(data, device_kind, require_utilization=require_utilization)


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
        output_file = tmp_path / "perf_cpu.json"

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
        assert binfo["precision"] == "auto"

        # The real ONNX model ORT loaded is recorded and points at a file
        running_model = Path(binfo["running_model_path"])
        assert running_model.suffix == ".onnx"
        assert running_model.exists()

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
        assert minfo["precision"] == "fp32"

        # Verify raw samples count matches iterations
        assert len(data["raw_samples_ms"]) == 3

    def test_benchmark_cpu_verbose(self, tmp_path: Path, model_arg: str):
        """Benchmark with --verbose should succeed and show debug output."""
        output_file = tmp_path / "perf_cpu_verbose.json"

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

    def test_benchmark_cpu_build(self, tmp_path: Path, model_arg: str):
        """Benchmark with --no-skip-build should succeed and show debug output."""
        output_file = tmp_path / "perf_cpu_build.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="cpu", no_skip_build=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"
        assert output_file.exists()
        assert "Results saved to" in result.output

    def test_benchmark_cpu_memory(self, tmp_path: Path, model_arg: str):
        """Benchmark with --memory produces memory profile in JSON output.

        Verifies the JSON contains 'memory' section with RAM delta fields.
        """
        output_file = tmp_path / "perf_cpu_memory.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="cpu", memory=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())

        # Memory section must be present
        assert "memory" in data, f"No 'memory' key in JSON output. Keys: {list(data.keys())}"
        mem = data["memory"]

        # Required RAM fields
        assert "rss_baseline_mb" in mem
        assert "rss_after_compile_mb" in mem
        assert "rss_after_inference_mb" in mem
        assert "rss_model_load_delta_mb" in mem
        assert "rss_inference_delta_mb" in mem
        assert "rss_total_delta_mb" in mem

        # Values should be positive floats
        assert mem["rss_after_inference_mb"] > 0
        assert mem["rss_baseline_mb"] > 0

        # Console output should contain Memory section
        assert "Memory:" in result.output
        assert "RAM:" in result.output

    def test_benchmark_cpu_no_memory(self, tmp_path: Path, model_arg: str):
        """Benchmark with --no-memory omits memory profile from JSON output."""
        output_file = tmp_path / "perf_cpu_no_memory.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="cpu", memory=False
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())

        # Memory section must NOT be present
        assert "memory" not in data

        # Console output should NOT contain Memory section
        assert "Memory:" not in result.output

    def test_benchmark_npu_memory(self, tmp_path: Path, model_arg: str):
        """Benchmark on NPU with --memory produces VRAM fields.

        Verifies VRAM local/shared fields are present in JSON output.
        """
        _require_npu()
        output_file = tmp_path / "perf_npu_memory.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="npu", memory=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())

        assert "memory" in data
        mem = data["memory"]

        # RAM fields
        assert mem["rss_after_inference_mb"] > 0
        assert "rss_model_load_delta_mb" in mem

        # VRAM fields (NPU exposes device memory fields, but values depend on driver)
        assert "vram_local_after_inference_mb" in mem
        assert "vram_shared_after_inference_mb" in mem
        assert "vram_local_model_load_delta_mb" in mem
        assert "vram_shared_model_load_delta_mb" in mem
        assert "vram_local_inference_delta_mb" in mem
        assert "vram_shared_inference_delta_mb" in mem

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
        # Tiny synthetic fixture: below PDH utilization-publish floor.
        _assert_monitor_result(data, device="gpu", require_utilization=False)

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
        # Tiny synthetic fixture: below PDH utilization-publish floor.
        _assert_monitor_result(data, device="npu", require_utilization=False)

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
        # At least a non-cpu should exist and picked up
        assert data["benchmark_info"]["device"] in ("auto", "gpu", "npu")
        assert data["benchmark_info"]["ep"] != "CPUExecutionProvider"
        assert data["latency_ms"]["mean"] > 0

    @pytest.mark.parametrize("ep", NON_CPU_EPS)
    def test_benchmark_ep(self, ep: str, tmp_path: Path, model_arg: str):
        """Benchmark with --ep <ep>.

        Skipped if the specified ExecutionProvider is not available on the host.
        """
        require_ep(ep)

        output_file = tmp_path / f"perf_{ep}.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(model_arg=model_arg, output_file=output_file, ep=ep),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["benchmark_info"]["ep"] == EP_ALIASES[ep]
        assert data["benchmark_info"]["device"] in ("auto", "gpu", "npu"), "Expected a non-CPU EP"
        assert data["latency_ms"]["mean"] > 0

    @pytest.mark.parametrize("ep", CPU_EPS)
    def test_benchmark_ep_device_cpu(self, ep: str, tmp_path: Path, model_arg: str):
        """Benchmark with --ep <ep> and --device cpu.

        Skipped if the specified EP is unavailable on the host.
        """
        require_ep(ep)

        output_file = tmp_path / f"perf_{ep}_cpu.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="cpu", ep=ep, monitor=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        _assert_monitor_result(data, device="cpu", ep=EP_ALIASES[ep])

    @pytest.mark.parametrize("ep", GPU_EPS)
    def test_benchmark_ep_device_gpu(self, ep: str, tmp_path: Path, model_arg: str):
        """Benchmark with --ep <ep> and --device gpu.

        Skipped if the specified EP or a GPU is unavailable on the host.
        """
        require_ep(ep)
        _require_gpu()

        output_file = tmp_path / f"perf_{ep}_gpu.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="gpu", ep=ep, monitor=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        # Tiny synthetic fixture: below PDH utilization-publish floor.
        _assert_monitor_result(data, device="gpu", ep=EP_ALIASES[ep], require_utilization=False)

    @pytest.mark.parametrize("ep", NPU_EPS)
    def test_benchmark_ep_device_npu(self, ep: str, tmp_path: Path, model_arg: str):
        """Benchmark with --ep <ep> and --device npu.

        Skipped if the specified EP or a NPU is unavailable on the host.
        """
        require_ep(ep)
        _require_npu()

        output_file = tmp_path / f"perf_{ep}_npu.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="npu", ep=ep, monitor=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        # Tiny synthetic fixture: below PDH utilization-publish floor.
        _assert_monitor_result(data, device="npu", ep=EP_ALIASES[ep], require_utilization=False)


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

    @pytest.mark.parametrize("ep", CPU_EPS)
    def test_benchmark_ep_cpu(self, ep: str, tmp_path: Path, model_arg: str):
        """Benchmark with --ep <ep>."""
        require_ep(ep)

        output_file = tmp_path / f"perf_hf_{ep}_cpu.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="cpu", ep=ep, monitor=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["benchmark_info"]["ep"] == EP_ALIASES[ep]
        _assert_monitor_result(data, device="cpu")

    @pytest.mark.parametrize("ep", GPU_EPS)
    def test_benchmark_ep_gpu(self, ep: str, tmp_path: Path, model_arg: str):
        """Benchmark with --ep <ep>."""
        require_ep(ep)
        _require_gpu()

        output_file = tmp_path / f"perf_hf_{ep}_gpu.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="gpu", ep=ep, monitor=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["benchmark_info"]["ep"] == EP_ALIASES[ep]
        # Not all EPs bump PDH GPU-engine counters (OpenVINO routes via its own
        # compute path); validate structure only, not utilization magnitude.
        _assert_monitor_result(data, device="gpu", require_utilization=False)

    @pytest.mark.parametrize("ep", NPU_EPS)
    def test_benchmark_ep_npu(self, ep: str, tmp_path: Path, model_arg: str):
        """Benchmark with --ep <ep>."""
        require_ep(ep)
        _require_npu()

        output_file = tmp_path / f"perf_hf_{ep}_npu.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg=model_arg, output_file=output_file, device="npu", ep=ep, monitor=True
            ),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["benchmark_info"]["ep"] == EP_ALIASES[ep]
        # Not all EPs bump PDH NPU-engine counters reliably for short runs;
        # validate structure only, not utilization magnitude.
        _assert_monitor_result(data, device="npu", require_utilization=False)


# ===========================================================================
# Per-module benchmark
# ===========================================================================


class TestPerfModule:
    """Per-module benchmark via --module on a HuggingFace model."""

    def test_module_benchmark_cpu(self, tmp_path: Path):
        """Per-module benchmark on CPU for ResNetStage submodules of resnet-50."""
        output_file = tmp_path / "perf_module_cpu.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            _build_perf_args(
                model_arg="microsoft/resnet-50",
                output_file=output_file,
                ep="cpu",
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


# ===========================================================================
# Dynamic axes: --dynamic-axes re-exports with a symbolic batch dim.
# ===========================================================================


class TestPerfDynamicAxes:
    """``--dynamic-axes`` feeds the HF export so the benchmarked model is dynamic.

    ``--ignore-cache`` forces a fresh build in a throwaway folder so the cached
    static export isn't reused, and ``--no-skip-build`` guarantees the export
    actually runs. The benchmarked ResNet-50 then exposes a dynamic (``None``)
    batch axis instead of the static ``1``.
    """

    def test_dynamic_axes_cpu(self, tmp_path: Path):
        axes = tmp_path / "axes.json"
        axes.write_text(json.dumps({"pixel_values": {"0": "batch"}}))
        output_file = tmp_path / "perf_dynamic_axes.json"

        result = CliRunner().invoke(
            perf,
            [
                "-m",
                "microsoft/resnet-50",
                "--iterations",
                "3",
                "--warmup",
                "1",
                "-o",
                str(output_file),
                "--device",
                "cpu",
                "--dynamic-axes",
                str(axes),
                "--ignore-cache",
                "--no-skip-build",
                "--no-optimize",
                "--no-quant",
                "--no-memory",
            ],
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"
        assert "Dynamic axes:" in result.output
        assert output_file.exists()
        data = json.loads(output_file.read_text())

        # The benchmarked model exposes a dynamic (None) batch axis; the
        # remaining channel/spatial dims stay static.
        input_shapes = data["model_info"]["input_shapes"]
        assert input_shapes[0][0] is None, f"expected dynamic batch, got {input_shapes}"
        assert input_shapes[0][1:] == [3, 224, 224]
        assert data["latency_ms"]["mean"] > 0

        # The ONNX graph ORT actually loaded carries the symbolic batch dim.
        running = onnx.load(data["benchmark_info"]["running_model_path"])
        pixel_values = next(i for i in running.graph.input if i.name == "pixel_values")
        assert pixel_values.type.tensor_type.shape.dim[0].dim_param == "batch"


# ===========================================================================
# GenAI runtime (winml-genai): --device / --ep override
# ===========================================================================


def _genai_perf_args(
    *,
    bundle_dir: Path,
    output_file: Path,
    device: str | None = None,
    ep: str | None = None,
) -> list[str]:
    """Build argv for a fast winml-genai perf run against a tiny bundle.

    Kept deliberately small (2 iterations, 1 warmup, 4 new tokens) so the
    generation loop stays quick while still producing real timing samples.
    """
    args: list[str] = [
        "-m",
        str(bundle_dir),
        "--runtime",
        "winml-genai",
        "--iterations",
        "2",
        "--warmup",
        "1",
        "--max-new-tokens",
        "4",
        "-o",
        str(output_file),
    ]
    if device is not None:
        args += ["--device", device]
    if ep is not None:
        args += ["--ep", ep]
    return args


class TestPerfGenaiContract:
    """Contract for the winml-genai ``config`` sentinel — no bundle required.

    These lock the CLI surface that ``config`` is a winml-genai-only
    ``--device`` value: it is advertised in ``--help`` and rejected with a
    helpful message on the single-shot ONNX path. They run on any host under
    ``-m e2e`` (no genai stack, model download, or accelerator needed).
    """

    def test_help_lists_config_device(self):
        """``perf --help`` advertises ``config`` as a --device choice."""
        result = CliRunner().invoke(perf, ["--help"], obj={}, catch_exceptions=False)
        assert result.exit_code == 0
        assert "[config|auto|cpu|gpu|npu]" in result.output
        assert "winml-genai only" in result.output

    def test_onnx_rejects_device_config(self, tmp_path: Path, onnx_model_path: Path):
        """``--device config`` is rejected on the ONNX runtime (genai-only sentinel)."""
        output_file = tmp_path / "perf_reject.json"

        result = CliRunner().invoke(
            perf,
            _build_perf_args(
                model_arg=str(onnx_model_path), output_file=output_file, device="config"
            ),
            obj={},
            catch_exceptions=False,
        )

        assert result.exit_code == 2, f"expected UsageError exit 2, got {result.exit_code}"
        assert "--device config is only valid with --runtime winml-genai" in result.output
        assert not output_file.exists(), "no report should be written on rejection"


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.timeout(1800)
class TestPerfGenai:
    """Benchmark a real onnxruntime-genai bundle across --device / --ep overrides.

    A tiny Qwen3-0.6B int4 CPU bundle is built once per class via the
    onnxruntime-genai model builder (slow + network), then the perf CLI runs
    for each device/ep resolution. Skipped unless ``onnxruntime_genai`` /
    ``torch`` / ``transformers`` are importable and the build succeeds.

    Both ``config`` / CPU routing and the honest-reporting path are asserted:
    the builder emits a *flat* bundle (single ``model.onnx``, no
    ``decoder.pipeline``), so a hardware-EP override matches no stage and is
    reported as ``config`` rather than the requested EP (and still runs on CPU).
    The pipeline-rewrite override (skip-CPU, device-aware options) is covered by
    the GenaiSession unit tests.
    """

    @pytest.fixture(scope="class")
    def genai_bundle(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Build a tiny Qwen3-0.6B int4 CPU genai bundle once for the class."""
        for mod in ("onnxruntime_genai", "torch", "transformers"):
            if importlib.util.find_spec(mod) is None:
                pytest.skip(f"{mod} not installed; genai perf e2e needs the full LLM stack")

        out = tmp_path_factory.mktemp("genai_bundle") / "qwen3_0_6b_int4_cpu"
        cache = tmp_path_factory.mktemp("genai_hf_cache")
        cmd = [
            sys.executable,
            "-m",
            "onnxruntime_genai.models.builder",
            "-m",
            "Qwen/Qwen3-0.6B",
            "-o",
            str(out),
            "-p",
            "int4",
            "-e",
            "cpu",
            "-c",
            str(cache),
            "--extra_options",
            "hf_token=false",
        ]
        try:
            proc = subprocess.run(  # noqa: S603 -- trusted args (sys.executable + constants)
                cmd,
                capture_output=True,
                text=True,
                timeout=1500,
                check=False,
            )
        except subprocess.TimeoutExpired:
            proc = None

        if proc is None:
            pytest.skip("onnxruntime-genai model build timed out")
        elif proc.returncode != 0 or not (out / "genai_config.json").exists():
            pytest.skip(
                "onnxruntime-genai model build failed (network / auth / unsupported):\n"
                f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
            )
        return out

    def _run(
        self,
        bundle: Path,
        output_file: Path,
        *,
        device: str | None = None,
        ep: str | None = None,
    ) -> dict:
        """Invoke perf on the bundle and return the parsed JSON report."""
        result = CliRunner().invoke(
            perf,
            _genai_perf_args(bundle_dir=bundle, output_file=output_file, device=device, ep=ep),
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, (
            f"genai perf failed (exit {result.exit_code}):\n{result.output}"
        )
        assert output_file.exists(), f"report not written: {output_file}"
        data = json.loads(output_file.read_text())
        assert data["benchmark_info"]["runtime"] == "winml-genai"
        assert data["benchmark_info"]["generated_tokens"] > 0
        return data

    def test_default_respects_config(self, tmp_path: Path, genai_bundle: Path):
        """Omitting --device respects the bundle config (device == ep == config)."""
        data = self._run(genai_bundle, tmp_path / "genai_default.json")
        assert data["benchmark_info"]["device"] == "config"
        assert data["benchmark_info"]["ep"] == "config"

    def test_device_config_respects_config(self, tmp_path: Path, genai_bundle: Path):
        """--device config is the explicit form of the default (no override)."""
        data = self._run(genai_bundle, tmp_path / "genai_device_config.json", device="config")
        assert data["benchmark_info"]["device"] == "config"
        assert data["benchmark_info"]["ep"] == "config"

    def test_ep_cpu_overrides_ep_only(self, tmp_path: Path, genai_bundle: Path):
        """--ep cpu forces the EP but leaves device at the config default."""
        data = self._run(genai_bundle, tmp_path / "genai_ep_cpu.json", ep="cpu")
        assert data["benchmark_info"]["device"] == "config"
        assert data["benchmark_info"]["ep"] == "cpu"

    def test_device_cpu_overrides_device_and_ep(self, tmp_path: Path, genai_bundle: Path):
        """--device cpu forces both the device and the resolved EP to cpu."""
        data = self._run(genai_bundle, tmp_path / "genai_device_cpu.json", device="cpu")
        assert data["benchmark_info"]["device"] == "cpu"
        assert data["benchmark_info"]["ep"] == "cpu"

    def test_ep_qnn_on_flat_bundle_reports_config(self, tmp_path: Path, genai_bundle: Path):
        """A hardware-EP override matching no stage is reported as config.

        The flat CPU bundle has no ``decoder.pipeline`` for ``--ep qnn`` to
        rewrite, so the override takes no effect: nothing routes to QNN (so QNN
        never has to register — this runs on CPU-only CI), and the report says
        ``ep: config`` instead of falsely claiming ``qnn`` (comment #3).  The
        *requested* device is still echoed back.
        """
        data = self._run(genai_bundle, tmp_path / "genai_ep_qnn_flat.json", ep="qnn")
        assert data["benchmark_info"]["ep"] == "config"
        assert data["benchmark_info"]["device"] == "config"
