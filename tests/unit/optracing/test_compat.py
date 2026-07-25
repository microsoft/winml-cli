# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compatibility tests for the deprecated ``winml.modelkit.optracing`` surface."""

from __future__ import annotations

import importlib
import inspect
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
