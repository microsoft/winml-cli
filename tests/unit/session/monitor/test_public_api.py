# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Public API regression tests for ``winml.modelkit.session.monitor``."""

from __future__ import annotations


def test_monitor_package_reexports_tracing_types_and_report_helpers() -> None:
    from winml.modelkit.session.monitor import (
        OperatorMetrics,
        OpTraceResult,
        display_op_trace_report,
        write_op_trace_json,
    )
    from winml.modelkit.session.monitor.op_metrics import (
        OperatorMetrics as RawOperatorMetrics,
    )
    from winml.modelkit.session.monitor.op_metrics import (
        OpTraceResult as RawOpTraceResult,
    )
    from winml.modelkit.session.monitor.report import (
        display_op_trace_report as raw_display_op_trace_report,
    )
    from winml.modelkit.session.monitor.report import (
        write_op_trace_json as raw_write_op_trace_json,
    )

    assert OperatorMetrics is RawOperatorMetrics
    assert OpTraceResult is RawOpTraceResult
    assert display_op_trace_report is raw_display_op_trace_report
    assert write_op_trace_json is raw_write_op_trace_json


def test_openvino_monitor_is_exported_from_session_public_apis() -> None:
    from winml.modelkit.session import OpenVinoMonitor as SessionOpenVinoMonitor
    from winml.modelkit.session.monitor import OpenVinoMonitor

    monitor = OpenVinoMonitor()

    assert SessionOpenVinoMonitor is OpenVinoMonitor
    assert monitor.__enter__() is monitor
    assert monitor.__exit__(None, None, None) is None
    assert OpenVinoMonitor.is_available() is False
    assert monitor.to_dict() == {"ep": "OpenVINO", "device": "NPU", "status": "not_implemented"}
