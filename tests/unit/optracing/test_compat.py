# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compatibility tests for the deprecated ``winml.modelkit.optracing`` surface."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
import warnings
from pathlib import Path

import pytest


def _capture_deprecation(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = call()
    return result, [warning for warning in caught if warning.category is DeprecationWarning]


def _reset_optracing_modules() -> None:
    for name in [
        name
        for name in sys.modules
        if name == "winml.modelkit.optracing" or name.startswith("winml.modelkit.optracing.")
    ]:
        sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("module_name", "attr_name", "expected_module", "expected_attr"),
    [
        (
            "winml.modelkit.optracing",
            "OpTraceResult",
            "winml.modelkit.session.monitor",
            "OpTraceResult",
        ),
        (
            "winml.modelkit.optracing",
            "OperatorMetrics",
            "winml.modelkit.session.monitor",
            "OperatorMetrics",
        ),
        (
            "winml.modelkit.optracing.result",
            "OpTraceResult",
            "winml.modelkit.session.monitor",
            "OpTraceResult",
        ),
        (
            "winml.modelkit.optracing.result",
            "OperatorMetrics",
            "winml.modelkit.session.monitor",
            "OperatorMetrics",
        ),
        (
            "winml.modelkit.optracing.report",
            "write_op_trace_json",
            "winml.modelkit.session.monitor",
            "write_op_trace_json",
        ),
    ],
)
def test_legacy_optracing_imports_reexport_current_symbols_with_caller_warning(
    module_name: str, attr_name: str, expected_module: str, expected_attr: str
) -> None:
    _reset_optracing_modules()
    compat_module = importlib.import_module(module_name)
    expected = getattr(importlib.import_module(expected_module), expected_attr)

    value, warning_records = _capture_deprecation(lambda: getattr(compat_module, attr_name))

    assert value is expected
    assert len(warning_records) == 1
    assert Path(warning_records[0].filename) == Path(__file__)


def test_legacy_display_report_keeps_old_default_top_n() -> None:
    _reset_optracing_modules()
    report_module = importlib.import_module("winml.modelkit.optracing.report")

    display, warning_records = _capture_deprecation(lambda: report_module.display_op_trace_report)

    assert callable(display)
    assert inspect.signature(display).parameters["top_n"].default == 15
    assert len(warning_records) == 1
    assert Path(warning_records[0].filename) == Path(__file__)


def test_legacy_from_import_keeps_top_level_symbols_available() -> None:
    _reset_optracing_modules()
    namespace: dict[str, object] = {}
    statement = compile(
        (
            "from winml.modelkit.optracing import "
            "OpTraceResult, OperatorMetrics, display_op_trace_report, "
            "write_op_trace_json, OpTracer, get_tracer, register_tracer"
        ),
        __file__,
        "exec",
    )

    _, warning_records = _capture_deprecation(
        lambda: exec(statement, namespace)  # noqa: S102
    )

    assert {
        "OpTraceResult",
        "OperatorMetrics",
        "display_op_trace_report",
        "write_op_trace_json",
        "OpTracer",
        "get_tracer",
        "register_tracer",
    }.issubset(namespace)
    assert len(warning_records) == 7
    assert all(Path(warning.filename) == Path(__file__) for warning in warning_records)


def test_legacy_registry_restores_builtin_qnn_default() -> None:
    _reset_optracing_modules()
    registry_module = importlib.import_module("winml.modelkit.optracing.registry")
    base_module = importlib.import_module("winml.modelkit.optracing.base")

    get_tracer, get_warnings = _capture_deprecation(lambda: registry_module.get_tracer)
    op_tracer, tracer_warnings = _capture_deprecation(lambda: base_module.OpTracer)
    tracer_class = get_tracer("QNNExecutionProvider", "basic")

    assert len(get_warnings) == 1
    assert len(tracer_warnings) == 1
    assert Path(get_warnings[0].filename) == Path(__file__)
    assert Path(tracer_warnings[0].filename) == Path(__file__)
    assert tracer_class is not None
    assert issubclass(tracer_class, op_tracer)


@pytest.mark.parametrize(
    "module_name",
    [
        "winml.modelkit.optracing.qnn",
        "winml.modelkit.optracing.qnn.profiler",
    ],
)
def test_legacy_qnn_profiler_import_warns_at_caller(module_name: str) -> None:
    _reset_optracing_modules()
    namespace: dict[str, object] = {}
    statement = compile(
        f"from {module_name} import QNNProfiler",
        __file__,
        "exec",
    )

    _, warning_records = _capture_deprecation(
        lambda: exec(statement, namespace)  # noqa: S102
    )

    profiler = namespace["QNNProfiler"]
    assert profiler.__name__ == "QNNProfiler"
    assert len(warning_records) == 1
    assert Path(warning_records[0].filename) == Path(__file__)


@pytest.mark.parametrize(
    ("module_name", "symbol_name"),
    [
        ("winml.modelkit.optracing.qnn.csv_parser", "parse_qnn_profiling_csv"),
        ("winml.modelkit.optracing.qnn.qhas_parser", "parse_qhas"),
        ("winml.modelkit.optracing.qnn.viewer", "find_qnn_sdk"),
        ("winml.modelkit.optracing.qnn.viewer", "run_qhas_viewer"),
    ],
)
def test_legacy_qnn_helper_imports_warn_at_caller(module_name: str, symbol_name: str) -> None:
    _reset_optracing_modules()
    namespace: dict[str, object] = {}
    statement = compile(
        f"from {module_name} import {symbol_name}",
        __file__,
        "exec",
    )

    _, warning_records = _capture_deprecation(
        lambda: exec(statement, namespace)  # noqa: S102
    )

    assert callable(namespace[symbol_name])
    assert len(warning_records) == 1
    assert Path(warning_records[0].filename) == Path(__file__)


def test_legacy_qnn_csv_parser_keeps_per_sample_result_shape(tmp_path) -> None:
    _reset_optracing_modules()
    parser_module = importlib.import_module("winml.modelkit.optracing.qnn.csv_parser")
    parser, _ = _capture_deprecation(lambda: parser_module.parse_qnn_profiling_csv)
    csv_path = tmp_path / "profiling.csv"
    csv_path.write_text(
        (
            "Msg Timestamp,Message,Time,Unit of Measurement,"
            "Timing Source,Event Level,Event Identifier\n"
            '0,ROOT,4,COUNT,HW,ROOT,"Number of HVX threads used"\n'
            '1,ROOT,100,CYCLES,HW,ROOT,"Accelerator (execute) time (cycles)"\n'
            '2,NODE,25,CYCLES,HW,SUB-EVENT,"Conv_token_1_2:OpId_7 (cycles)"\n'
            '3,ROOT,10,US,HW,ROOT,"Accelerator (execute) time"\n'
        ),
        encoding="utf-8",
    )

    parsed = parser(csv_path)

    assert parsed == [
        {
            "metadata": {
                "hvx_threads": 4,
                "accel_execute_cycles": 100,
                "accel_execute_us": 10,
            },
            "samples": [{"name": "Conv", "op_id": 7, "cycles": 25}],
        }
    ]


def test_legacy_qhas_parser_keeps_original_field_names() -> None:
    _reset_optracing_modules()
    parser_module = importlib.import_module("winml.modelkit.optracing.qnn.qhas_parser")
    parser, _ = _capture_deprecation(lambda: parser_module.parse_qhas)
    summary = {
        "time_us": 10,
        "graph_execute_us": 8,
        "inf_per_s": 100_000,
        "timeline_cycles": 100,
        "percent_utilization": 75,
        "total_dram_read": 1,
        "total_dram_write": 2,
        "total_vtcm_read": 3,
        "total_vtcm_write": 4,
        "peak_vtcm_alloc": 5,
        "qnn_nodes": 6,
        "htp_nodes": 7,
        "unique_qnn_ops": 8,
        "unique_htp_ops": 9,
    }
    qhas_data = {
        "data": {
            "htp_overall_summary": {"data": [summary]},
            "qnn_op_instances_nodes": {
                "data": [
                    {
                        "qnn_op": "/encoder/Conv_token_1_2",
                        "qnn_op_type": "Conv2d",
                        "cycles": 25,
                        "percent_active_cycles": 50,
                    }
                ]
            },
        }
    }

    parsed = parser(qhas_data)

    assert parsed["summary"] == {
        "time_us": 10,
        "graph_execute_us": 8,
        "inf_per_s": 100_000,
        "timeline_cycles": 100,
        "utilization_pct": 75,
        "total_dram_read": 1,
        "total_dram_write": 2,
        "total_vtcm_read": 3,
        "total_vtcm_write": 4,
        "peak_vtcm_alloc": 5,
        "qnn_nodes": 6,
        "htp_nodes": 7,
        "unique_qnn_ops": 8,
        "unique_htp_ops": 9,
    }
    assert parsed["operators"][0]["name"] == "/encoder/Conv_token_1_2"
    assert parsed["operators"][0]["op_path"] == "/encoder/Conv_token_1_2"
    assert parsed["operators"][0]["op_type"] == "Conv2d"


def test_legacy_qnn_basic_viewer_remains_available(tmp_path, monkeypatch) -> None:
    _reset_optracing_modules()
    viewer_module = importlib.import_module("winml.modelkit.optracing.qnn.viewer")
    run_basic_viewer, warning_records = _capture_deprecation(lambda: viewer_module.run_basic_viewer)
    executable = tmp_path / "sdk" / "bin" / "x64" / "qnn-profile-viewer.exe"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    qnn_log = tmp_path / "profiling_qnn.log"
    qnn_log.write_text("", encoding="utf-8")
    output = tmp_path / "profiling.csv"
    commands: list[list[str]] = []

    def _fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        commands.append(command)
        output.write_text("csv", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert run_basic_viewer(qnn_log, output, sdk_root=tmp_path / "sdk") == output
    assert commands == [
        [
            str(executable),
            "--input_log",
            str(qnn_log),
            "--output",
            str(output),
        ]
    ]
    assert len(warning_records) == 1
    assert Path(warning_records[0].filename) == Path(__file__)


def test_legacy_qnn_detail_compiles_epcontext_before_monitoring(tmp_path, monkeypatch) -> None:
    _reset_optracing_modules()
    profiler_module = importlib.import_module("winml.modelkit.optracing.qnn.profiler")
    onnx_module = importlib.import_module("winml.modelkit.onnx")
    session_module = importlib.import_module("winml.modelkit.session")
    monitor_module = importlib.import_module("winml.modelkit.session.monitor.qnn_monitor")
    profiler_class, _ = _capture_deprecation(lambda: profiler_module.QNNProfiler)
    events: list[str] = []
    ep_configs: list[object] = []
    result = object()

    class _Monitor:
        def __init__(self, **kwargs) -> None:
            self.result = result

    class _PerfContext:
        def __init__(self, monitor) -> None:
            self.monitor = monitor

        def __enter__(self):
            events.append("monitor")
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

    class _Session:
        running_model_path = tmp_path / "model_ctx.onnx"

        def __init__(self, *args, ep_config=None, **kwargs) -> None:
            ep_configs.append(ep_config)

        def compile(self) -> None:
            events.append("compile")

        def perf(self, *, warmup, monitor):
            return _PerfContext(monitor)

        def run(self, inputs) -> None:
            return None

    monkeypatch.setattr(onnx_module, "is_compiled_onnx", lambda path: True)
    monkeypatch.setattr(session_module, "WinMLSession", _Session)
    monkeypatch.setattr(monitor_module, "QNNMonitor", _Monitor)
    monkeypatch.setattr(profiler_class, "_resolve_inputs", lambda self, session: {})

    profiler = profiler_class(tmp_path / "model.onnx", output_dir=tmp_path, level="detail")

    assert profiler.run(iterations=1, warmup=0) is result
    assert events == ["compile", "monitor"]
    assert len(ep_configs) == 1
    assert ep_configs[0].enable_ep_context is True
    assert ep_configs[0].embed_context is False


def test_legacy_qnn_detail_rejects_compile_fallback_to_raw_model(tmp_path, monkeypatch) -> None:
    _reset_optracing_modules()
    profiler_module = importlib.import_module("winml.modelkit.optracing.qnn.profiler")
    onnx_module = importlib.import_module("winml.modelkit.onnx")
    session_module = importlib.import_module("winml.modelkit.session")
    monitor_module = importlib.import_module("winml.modelkit.session.monitor.qnn_monitor")
    profiler_class, _ = _capture_deprecation(lambda: profiler_module.QNNProfiler)
    raw_model = tmp_path / "model.onnx"
    events: list[str] = []
    result = object()

    class _Monitor:
        def __init__(self, **kwargs) -> None:
            self.result = result

    class _PerfContext:
        def __init__(self, monitor) -> None:
            self.monitor = monitor

        def __enter__(self):
            events.append("monitor")
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

    class _Session:
        running_model_path = raw_model

        def __init__(self, *args, **kwargs) -> None:
            return None

        def compile(self) -> None:
            events.append("compile")

        def perf(self, *, warmup, monitor):
            return _PerfContext(monitor)

        def run(self, inputs) -> None:
            return None

    monkeypatch.setattr(onnx_module, "is_compiled_onnx", lambda path: False)
    monkeypatch.setattr(session_module, "WinMLSession", _Session)
    monkeypatch.setattr(monitor_module, "QNNMonitor", _Monitor)
    monkeypatch.setattr(profiler_class, "_resolve_inputs", lambda self, session: {})
    profiler = profiler_class(raw_model, output_dir=tmp_path, level="detail")

    with pytest.raises(RuntimeError, match="EPContext"):
        profiler.run(iterations=0, warmup=0)

    assert events == ["compile"]


def test_legacy_tracer_registry_round_trips_with_substring_match() -> None:
    _reset_optracing_modules()
    base_module = importlib.import_module("winml.modelkit.optracing.base")
    registry_module = importlib.import_module("winml.modelkit.optracing.registry")
    current_monitor = importlib.import_module("winml.modelkit.session.monitor")

    op_tracer, tracer_warnings = _capture_deprecation(lambda: base_module.OpTracer)
    register_tracer, register_warnings = _capture_deprecation(
        lambda: registry_module.register_tracer
    )
    get_tracer, get_warnings = _capture_deprecation(lambda: registry_module.get_tracer)
    op_trace_result_cls = current_monitor.OpTraceResult

    class _CompatTracer(op_tracer):
        def run(self, iterations: int = 5, warmup: int = 2):
            return op_trace_result_cls(
                model=str(self.onnx_path),
                device="npu",
                tracing_level=self.level,
                status="ok",
            )

        def is_available(self) -> bool:
            return True

    for warning_records in (tracer_warnings, register_warnings, get_warnings):
        assert len(warning_records) == 1
        assert Path(warning_records[0].filename) == Path(__file__)

    register_tracer("UnitTestCompatEP", "detail", _CompatTracer)

    assert get_tracer("MyUnitTestCompatEPExecutionProvider", "detail") is _CompatTracer
    assert get_tracer("MyUnitTestCompatEPExecutionProvider", "basic") is None
