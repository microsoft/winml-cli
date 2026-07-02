# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for OpenVINOMonitor — the OpenVINO EP per-op profiler."""

from __future__ import annotations

import csv
import json
import os
from unittest.mock import MagicMock, patch

import pytest


def _write_ov_csv(path: object, rows: list[list[str]]) -> None:
    """Write a synthetic OpenVINO profiling CSV at ``path``."""
    with open(path, "w", newline="", encoding="utf-8") as fh:  # type: ignore[arg-type]
        writer = csv.writer(fh)
        writer.writerow(["Layer Name", "Status", "Layer Type", "Real Time (us)", "Exec Type"])
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_is_available_no_openvino_returns_false():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    with (
        patch("onnxruntime.get_available_providers", return_value=[]),
        patch("onnxruntime.get_ep_devices", return_value=[]),
        patch("winml.modelkit.session.ep_registry.WinMLEPRegistry.instance"),
    ):
        assert OpenVINOMonitor.is_available() is False


def test_is_available_via_bundled_wheel():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    with patch(
        "onnxruntime.get_available_providers",
        return_value=["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    ):
        assert OpenVINOMonitor.is_available() is True


def test_is_available_via_winml_registry():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    fake_ep = MagicMock()
    fake_ep.ep_name = "OpenVINOExecutionProvider"
    with (
        patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]),
        patch("onnxruntime.get_ep_devices", return_value=[fake_ep]),
        patch("winml.modelkit.session.ep_registry.WinMLEPRegistry.instance"),
    ):
        assert OpenVINOMonitor.is_available() is True


def test_is_available_winml_path_failure_logs_warning(caplog, monkeypatch):
    """NFR-2: non-ImportError from WinMLEPRegistry.instance must log at WARNING."""
    import logging

    import onnxruntime as ort

    from winml.modelkit.session import ep_registry
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ort, "get_ep_devices", list)

    def _raises() -> None:
        raise RuntimeError("simulated WinML init failure")

    monkeypatch.setattr(
        ep_registry.WinMLEPRegistry, "instance", classmethod(lambda cls: _raises())
    )

    with caplog.at_level(logging.WARNING):
        assert OpenVINOMonitor.is_available() is False

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any(
        "WinML EP probe failed" in r.message and "RuntimeError" in r.message
        for r in warnings
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_invalid_device_raises():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    with pytest.raises(ValueError, match="device"):
        OpenVINOMonitor(device="TPU")


def test_level_only_basic():
    """OpenVINOMonitor raises ValueError for any level other than 'basic'."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    with pytest.raises(ValueError, match="level"):
        OpenVINOMonitor(level="detail")


def test_ctor_defaults(tmp_path):
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    m = OpenVINOMonitor(output_dir=tmp_path)
    assert m._level == "basic"
    assert m._device == "AUTO"
    assert m._output_dir == tmp_path
    assert m.result is None


def test_ctor_auto_tempdir():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    m = OpenVINOMonitor()
    assert m.output_dir.is_dir()
    assert m.output_dir.name.startswith("ov_profile_")


def test_output_dir_property_read_only(tmp_path):
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    m = OpenVINOMonitor(output_dir=tmp_path)
    with pytest.raises(AttributeError):
        m.output_dir = tmp_path / "other"  # type: ignore[misc]


def test_requires_session_teardown_is_false():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    assert OpenVINOMonitor.requires_session_teardown is False


def test_ep_name_is_openvino():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    assert OpenVINOMonitor.ep_name == "openvino"


# ---------------------------------------------------------------------------
# Provider options
# ---------------------------------------------------------------------------


def test_get_provider_options_merges_perf_count():
    """No caller load_config → inserts AUTO.PERF_COUNT=YES."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    m = OpenVINOMonitor(device="AUTO")
    opts = m.get_provider_options()
    assert "load_config" in opts
    lc = json.loads(opts["load_config"])
    assert lc == {"AUTO": {"PERF_COUNT": "YES"}}


def test_get_provider_options_preserves_caller_keys():
    """Caller's CPU.NUM_STREAMS=4 survives alongside owner-injected AUTO.PERF_COUNT=YES."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    caller_lc = json.dumps({"CPU": {"NUM_STREAMS": 4}})
    m = OpenVINOMonitor(device="AUTO", extra_provider_options={"load_config": caller_lc})
    opts = m.get_provider_options()
    lc = json.loads(opts["load_config"])
    assert lc["CPU"]["NUM_STREAMS"] == 4
    assert lc["AUTO"]["PERF_COUNT"] == "YES"


def test_get_provider_options_owner_enforced():
    """C-3: caller PERF_COUNT=NO is overridden to YES; other keys in device cfg are preserved."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    caller_lc = json.dumps({"AUTO": {"PERF_COUNT": "NO", "OTHER": "val"}})
    m = OpenVINOMonitor(device="AUTO", extra_provider_options={"load_config": caller_lc})
    opts = m.get_provider_options()
    lc = json.loads(opts["load_config"])
    assert lc["AUTO"]["PERF_COUNT"] == "YES"
    assert lc["AUTO"]["OTHER"] == "val"


def test_get_provider_options_idempotent():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    m = OpenVINOMonitor(device="CPU")
    assert m.get_provider_options() == m.get_provider_options()


def test_get_provider_options_device_cpu():
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    m = OpenVINOMonitor(device="CPU")
    lc = json.loads(m.get_provider_options()["load_config"])
    assert lc == {"CPU": {"PERF_COUNT": "YES"}}


# ---------------------------------------------------------------------------
# Context manager — env var lifecycle
# ---------------------------------------------------------------------------


def test_enter_sets_env_var_exit_restores(tmp_path):
    """Entering sets ORT_OPENVINO_PERF_COUNT; exiting restores prior value."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    sentinel = "/some/prior/path"
    os.environ["ORT_OPENVINO_PERF_COUNT"] = sentinel
    try:
        m = OpenVINOMonitor(output_dir=tmp_path)
        m.__enter__()
        assert os.environ["ORT_OPENVINO_PERF_COUNT"] == str(tmp_path)
        m.__exit__(None, None, None)
        assert os.environ.get("ORT_OPENVINO_PERF_COUNT") == sentinel
    finally:
        os.environ.pop("ORT_OPENVINO_PERF_COUNT", None)


def test_exit_when_env_was_unset_pops_key(tmp_path):
    """When the key was absent before __enter__, __exit__ removes it entirely."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    os.environ.pop("ORT_OPENVINO_PERF_COUNT", None)
    assert "ORT_OPENVINO_PERF_COUNT" not in os.environ

    m = OpenVINOMonitor(output_dir=tmp_path)
    m.__enter__()
    assert "ORT_OPENVINO_PERF_COUNT" in os.environ
    m.__exit__(None, None, None)
    assert "ORT_OPENVINO_PERF_COUNT" not in os.environ


def test_reentry_raises(tmp_path):
    """Entering an already-entered monitor raises RuntimeError."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    os.environ.pop("ORT_OPENVINO_PERF_COUNT", None)
    m = OpenVINOMonitor(output_dir=tmp_path)
    m.__enter__()
    try:
        with pytest.raises(RuntimeError, match="already entered"):
            m.__enter__()
    finally:
        m.__exit__(None, None, None)

    assert "ORT_OPENVINO_PERF_COUNT" not in os.environ


def test_exit_does_not_suppress_caller_exception(tmp_path):
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    os.environ.pop("ORT_OPENVINO_PERF_COUNT", None)
    m = OpenVINOMonitor(output_dir=tmp_path)
    m.__enter__()
    result = m.__exit__(RuntimeError, RuntimeError("test"), None)
    assert result is None or result is False


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def test_parse_csv_produces_operator_metrics(tmp_path):
    """Synthetic 2-row CSV yields 2 OperatorMetrics with correct names and durations."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    _write_ov_csv(
        tmp_path / "run_001.csv",
        [
            ["/model/conv1/Conv", "EXECUTED", "Convolution", "123.4", "CPU"],
            ["/model/relu1/Relu", "EXECUTED", "ReLU", "45.6", "CPU"],
        ],
    )

    os.environ.pop("ORT_OPENVINO_PERF_COUNT", None)
    m = OpenVINOMonitor(output_dir=tmp_path)
    m.__enter__()
    m.__exit__(None, None, None)

    assert m.result is not None
    assert m.result.status == "ok"
    assert len(m.result.operators) == 2

    by_path = {op.op_path: op for op in m.result.operators}
    assert "/model/conv1/Conv" in by_path
    assert "/model/relu1/Relu" in by_path
    assert abs(by_path["/model/conv1/Conv"].duration_us - 123.4) < 1e-6
    assert by_path["/model/conv1/Conv"].name == "Convolution"
    assert by_path["/model/relu1/Relu"].name == "ReLU"


def test_merges_multiple_csvs(tmp_path):
    """3 per-inference CSVs accumulate into one OperatorMetrics with 3 samples."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    for i in range(3):
        _write_ov_csv(
            tmp_path / f"run_{i:03d}.csv",
            [
                [
                    "/model/conv1/Conv",
                    "EXECUTED",
                    "Convolution",
                    str(100.0 + i * 10.0),
                    "CPU",
                ]
            ],
        )

    os.environ.pop("ORT_OPENVINO_PERF_COUNT", None)
    m = OpenVINOMonitor(output_dir=tmp_path)
    m.__enter__()
    m.__exit__(None, None, None)

    assert m.result is not None
    assert m.result.status == "ok"
    assert len(m.result.operators) == 1
    op = m.result.operators[0]
    assert op.sample_count == 3
    # (100 + 110 + 120) / 3 = 110.0
    assert abs(op.avg_us - 110.0) < 1e-6
    # CRIT-5B: duration_us must equal avg_us
    assert abs(op.duration_us - op.avg_us) < 1e-6
    assert m.result.num_samples == 3


def test_no_csv_produces_no_data_status(tmp_path):
    """Empty output directory yields status='no_data'."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    os.environ.pop("ORT_OPENVINO_PERF_COUNT", None)
    m = OpenVINOMonitor(output_dir=tmp_path)
    m.__enter__()
    m.__exit__(None, None, None)

    assert m.result is not None
    assert m.result.status == "no_data"


def test_result_before_exit_is_none(tmp_path):
    """result is None until __exit__ is called."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    m = OpenVINOMonitor(output_dir=tmp_path)
    assert m.result is None
    os.environ.pop("ORT_OPENVINO_PERF_COUNT", None)
    m.__enter__()
    assert m.result is None
    m.__exit__(None, None, None)


def test_onnx_op_types_override_layer_type(tmp_path):
    """When set_onnx_op_types provides a mapping, it wins over Layer Type (L1 > L2)."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    _write_ov_csv(
        tmp_path / "run_001.csv",
        [["/model/conv1/Conv", "EXECUTED", "Convolution", "50.0", "CPU"]],
    )

    os.environ.pop("ORT_OPENVINO_PERF_COUNT", None)
    m = OpenVINOMonitor(output_dir=tmp_path)
    m.set_onnx_op_types({"/model/conv1/Conv": "Conv"})
    m.__enter__()
    m.__exit__(None, None, None)

    assert m.result is not None
    assert m.result.status == "ok"
    assert m.result.operators[0].name == "Conv"


def test_percent_of_total_sums_to_100(tmp_path):
    """percent_of_total across all operators sums to 100.0."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    _write_ov_csv(
        tmp_path / "run_001.csv",
        [
            ["/op/A", "EXECUTED", "TypeA", "200.0", "CPU"],
            ["/op/B", "EXECUTED", "TypeB", "300.0", "CPU"],
            ["/op/C", "EXECUTED", "TypeC", "500.0", "CPU"],
        ],
    )

    os.environ.pop("ORT_OPENVINO_PERF_COUNT", None)
    m = OpenVINOMonitor(output_dir=tmp_path)
    m.__enter__()
    m.__exit__(None, None, None)

    assert m.result is not None
    total_pct = sum(op.percent_of_total for op in m.result.operators)
    assert abs(total_pct - 100.0) < 1e-6


def test_whitespace_in_headers_tolerated(tmp_path):
    """CSV headers with surrounding whitespace are normalized correctly."""
    from winml.modelkit.session.monitor.openvino_monitor import OpenVINOMonitor

    csv_path = tmp_path / "run_001.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(" Layer Name , Status , Layer Type , Real Time (us) , Exec Type \n")
        fh.write("/op/conv,EXECUTED,Convolution,77.0,CPU\n")

    os.environ.pop("ORT_OPENVINO_PERF_COUNT", None)
    m = OpenVINOMonitor(output_dir=tmp_path)
    m.__enter__()
    m.__exit__(None, None, None)

    assert m.result is not None
    assert m.result.status == "ok"
    assert len(m.result.operators) == 1
    assert m.result.operators[0].op_path == "/op/conv"
    assert abs(m.result.operators[0].duration_us - 77.0) < 1e-6
