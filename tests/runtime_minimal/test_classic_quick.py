"""Quick sampling and ETW contracts without native libraries or ORT."""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from click import ClickException
from click.testing import CliRunner

from winml.modelkit.commands import _perf_runtime_classic as classic
from winml.modelkit.session.stats import PerfStats


@pytest.mark.parametrize("minimum,count", [(2.001, 20), (0, 20), (0, 0)])
def test_warmup_count_and_duration_excluded_from_host_and_gpu(monkeypatch, minimum, count):
    clock = [0.0]
    monkeypatch.setattr(classic.time, "perf_counter", lambda: clock[0])

    class Session:
        gpu_runs = 0

        @contextmanager
        def perf(self, warmup):
            self.stats = PerfStats(warmup=warmup)
            yield SimpleNamespace(stats=self.stats)

        def run(self, feeds):
            if len(self.stats.all_samples_ms) >= self.stats.warmup:
                self.gpu_runs += 1
            clock[0] += 0.1
            self.stats._samples.append(float(len(self.stats.all_samples_ms) + 1))
            return {"out": np.ones(1)}

    session = Session()
    stats, evidence = classic._run_iterations(
        session, {}, warmup=count, iterations=200, warmup_seconds=minimum
    )
    assert evidence["count"] >= count
    assert evidence["duration_ms"] >= minimum * 1000
    assert evidence["count"] == (21 if minimum else count)
    assert len(stats.samples_ms) == session.gpu_runs == 200
    assert stats.samples_ms[0] == evidence["count"] + 1


@pytest.mark.parametrize("failure", [None, "run", "finish", "incomplete"])
def test_capture_failure_never_publishes_host_as_gpu(tmp_path, monkeypatch, failure):
    from winml.modelkit.session import gpu_capture

    events = []

    class Capture:
        def __init__(self, path):
            pass

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            events.append("stop")

        def finish(self):
            assert events[-1] == "stop"
            if failure == "finish":
                raise RuntimeError("missing GPU events")
            if failure == "incomplete":
                return {"samples_ms": [], "valid_runs": 0}
            return {"samples_ms": [1.0], "valid_runs": 1, "source": "runtime-cgc-etw"}

    def execute(*args):
        if failure == "run":
            raise RuntimeError("session failed")
        return {"latency_ms": {"mean": 2.0}, "gpu_execution": {"coverage": "FULL"}}

    monkeypatch.setattr(gpu_capture, "GpuCapture", Capture)
    monkeypatch.setattr(classic, "_execute_session", execute)
    output = tmp_path / "report.json"
    params = {"etw_tracing": tmp_path / "trace", "output_format": "text", "iterations": 1}
    if failure:
        with pytest.raises((RuntimeError, ClickException)):
            classic._execute(params, tmp_path / "model.onnx", output, "model")
        assert not output.exists()
    else:
        classic._execute(params, tmp_path / "model.onnx", output, "model")
        import json

        result = json.loads(output.read_text())
        assert result["gpu_execution"]["source"] == "runtime-cgc-etw"
        assert result["runtime_execution_diagnostics"]["coverage"] == "FULL"
    assert events == ["enter", "stop"]


@pytest.mark.parametrize("suffix", [".onnx", ".mlir", ".mlirbc"])
def test_quick_adapter_options_reach_native_handler(tmp_path, monkeypatch, suffix):
    from winml.modelkit import onnx
    from winml.modelkit.cli import main
    from winml.modelkit.commands import perf

    monkeypatch.setattr(perf.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(perf.sys, "platform", "win32")
    monkeypatch.setattr(onnx, "get_external_data_files", lambda path: [])
    seen = []
    monkeypatch.setattr(classic, "_execute", lambda *args: seen.append(args[0]))
    model = tmp_path / ("model" + suffix)
    model.write_text("fixture")
    result = CliRunner().invoke(
        main,
        [
            "perf",
            "-m",
            str(model),
            "--runtime",
            "winml-runtime",
            "--backend",
            "cgc",
            "--device",
            "gpu",
            "--iterations",
            "200",
            "--warmup",
            "20",
            "--warmup-seconds",
            "2.001",
            "--etw-tracing",
            str(tmp_path / "trace"),
            "--no-memory",
            "--no-monitor",
            "--skip-build",
            "--no-compile",
            "-o",
            str(tmp_path / "result.json"),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert seen[0]["warmup_seconds"] == 2.001


def test_trace_cannot_contain_source(tmp_path, monkeypatch):
    from winml.modelkit.cli import main
    from winml.modelkit.commands import perf

    monkeypatch.setattr(perf.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(perf.sys, "platform", "win32")
    model = tmp_path / "model.mlir"
    model.write_text("fixture")
    result = CliRunner().invoke(
        main,
        [
            "perf",
            "-m",
            str(model),
            "--runtime",
            "winml-runtime",
            "--backend",
            "cgc",
            "--etw-tracing",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert model.read_text() == "fixture"


@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("backend", ["runtime", "cgep"])
def test_explicit_cli_luid_reaches_session_in_native_execution(
    tmp_path, monkeypatch, traced, backend
):
    import json

    from winml.modelkit import onnx, sysinfo
    from winml.modelkit import session as public_session
    from winml.modelkit.cli import main
    from winml.modelkit.commands import perf
    from winml.modelkit.session import gpu_capture, runtime_session

    luid = "0x00000001_0xFEDCBA98"
    requested = []
    events = []
    io = {
        "input_names": ["x"],
        "input_types": [np.dtype("float32")],
        "input_shapes": [[1]],
        "input_symbolic_shapes": [[None]],
        "value_ranges": {},
        "output_names": ["y"],
        "output_types": [np.dtype("float32")],
        "output_shapes": [[1]],
    }
    monkeypatch.setattr(perf.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(perf.sys, "platform", "win32")
    monkeypatch.setattr(onnx, "get_external_data_files", lambda path: [])
    monkeypatch.setattr(onnx, "get_io_config", lambda path: io)

    def adapter(value):
        requested.append(value)
        return SimpleNamespace(
            info=SimpleNamespace(luid=luid, name="fixture GPU"), close=lambda: None
        )

    class Session:
        io_config = io

        def __init__(self, path, **kwargs):
            self.execution_diagnostics = {"coverage": "FULL"}
            self.ep_name = "WinMLCGExecutionProvider"
            if backend == "runtime":
                assert kwargs["device_luid"] == 0x1FEDCBA98
            else:
                assert kwargs["ep_device"].device.adapter_luid == 0x1FEDCBA98
                options = kwargs["session_options"]()
                assert options.intra_op_num_threads == 0
                assert str(options.graph_optimization_level).endswith("ORT_ENABLE_ALL")

        def set_input_shapes(self, shapes):
            assert shapes == {"x": [1]}

        def compile(self):
            pass

        @contextmanager
        def perf(self, warmup):
            self.stats = PerfStats(warmup=warmup)
            yield SimpleNamespace(stats=self.stats)

        def run(self, inputs):
            self.stats._samples.append(1.0)
            return {"y": np.ones(1)}

        def close(self):
            events.append("session-close")

    class Capture:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            assert events[-1] == "session-close"
            events.append("capture-stop")

        def finish(self):
            assert events[-1] == "capture-stop"
            return {"samples_ms": [0.5, 0.5], "valid_runs": 2}

    monkeypatch.setattr(sysinfo, "open_gpu_adapter", adapter)
    monkeypatch.setattr(runtime_session, "WinMLRuntimeSession", Session)
    monkeypatch.setattr(gpu_capture, "GpuCapture", Capture)
    binding = SimpleNamespace(
        device=SimpleNamespace(adapter_luid=0x1FEDCBA98, ep_name="WinMLCGExecutionProvider")
    )

    def auto_device(target, *, device_luid):
        assert device_luid == luid
        return binding

    monkeypatch.setattr(public_session, "WinMLSession", Session)
    monkeypatch.setattr(public_session, "resolve_device", lambda target: target)
    monkeypatch.setattr(
        public_session,
        "WinMLEPRegistry",
        SimpleNamespace(instance=lambda: SimpleNamespace(auto_device=auto_device)),
    )
    model = tmp_path / "model.onnx"
    model.write_text("fixture")
    output = tmp_path / "report.json"
    args = [
        "perf",
        "-m",
        str(model),
        "--runtime",
        "winml-runtime",
        "--backend",
        "cgc",
        "--device",
        "gpu",
        "--device-luid",
        luid,
        "--warmup",
        "1",
        "--iterations",
        "2",
        "--no-memory",
        "-o",
        str(output),
    ]
    if traced:
        args += ["--etw-tracing", str(tmp_path / "trace")]
    if backend == "cgep":
        args[args.index("--runtime") + 1] = "winml-ort"
        position = args.index("--backend")
        del args[position : position + 2]
        args += ["--ep", "winmlcg"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert requested == [luid]
    assert json.loads(output.read_text())["benchmark_info"]["device_luid"] == 0x1FEDCBA98
