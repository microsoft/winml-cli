# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for OpenVINO ONNX Runtime profile collection."""

from __future__ import annotations

import json

import pytest

from winml.modelkit.session.monitor.openvino_monitor import OpenVinoMonitor


def _parent_event(timestamp: float, duration: float = 100.0) -> dict:
    return {
        "name": "OpenVINOExecutionProvider_partition_kernel_time",
        "ts": timestamp,
        "dur": duration,
        "args": {"ep": "OpenVINOExecutionProvider"},
    }


def _operator_event(name: str, timestamp: float, duration: float) -> dict:
    return {
        "cat": "Kernel",
        "name": f"OV::{name}",
        "ts": timestamp,
        "dur": duration,
        "args": {
            "ov_status": "EXECUTED",
            "ov_node_type": "Convolution",
            "ep": "OpenVINOExecutionProvider",
            "parent_ort_event_name": "OpenVINOExecutionProvider_partition_kernel_time",
        },
    }


def test_configure_session_options_enables_profiling(tmp_path) -> None:
    monitor = OpenVinoMonitor(output_dir=tmp_path, device="cpu")

    class SessionOptions:
        enable_profiling = False
        profile_file_prefix = ""

    session_options = SessionOptions()
    monitor.configure_session_options(session_options)

    assert session_options.enable_profiling is True
    assert session_options.profile_file_prefix == str((tmp_path / "onnxruntime_profile").resolve())


@pytest.mark.parametrize(
    ("ort_version", "raises"),
    [("1.25.9", True), ("1.26.0", False), ("1.27.0.dev20260830", False)],
)
def test_validate_runtime_version(monkeypatch, ort_version: str, raises: bool) -> None:
    monkeypatch.setattr("onnxruntime.__version__", ort_version)

    if raises:
        with pytest.raises(RuntimeError, match="uv pip install"):
            OpenVinoMonitor.validate_runtime_version()
    else:
        OpenVinoMonitor.validate_runtime_version()


def test_parse_detailed_profile_excludes_warmup_and_removes_prefix(tmp_path) -> None:
    monitor = OpenVinoMonitor(output_dir=tmp_path, device="npu")
    monitor.set_perf_window(warmup=1, measured_iterations=2)
    profile_path = tmp_path / "onnxruntime_profile.json"
    profile_path.write_text(
        json.dumps(
            [
                _parent_event(0),
                _operator_event("/resnet/encoder/Conv", 10, 100),
                _parent_event(200),
                _operator_event("/resnet/encoder/Conv", 210, 4),
                _parent_event(400),
                _operator_event("/resnet/encoder/Conv", 410, 6),
            ]
        ),
        encoding="utf-8",
    )

    with monitor:
        pass

    assert monitor.result is not None
    assert monitor.result.status == "ok"
    assert monitor.result.num_samples == 2
    assert monitor.result.artifacts["profile"] == str(profile_path)
    assert len(monitor.result.operators) == 1
    operator = monitor.result.operators[0]
    assert operator.name == "Convolution"
    assert operator.op_path == "/resnet/encoder/Conv"
    assert operator.onnx_op_type is None
    assert operator.samples_us == [4.0, 6.0]
    assert operator.avg_us == 5.0
    assert monitor.result.summary["accel_execute_us"] == 10.0


def test_multiple_events_for_same_op_in_one_inference_are_one_sample(tmp_path) -> None:
    monitor = OpenVinoMonitor(output_dir=tmp_path)
    monitor.set_perf_window(warmup=0, measured_iterations=1)
    (tmp_path / "onnxruntime_profile.json").write_text(
        json.dumps(
            [
                _parent_event(0),
                _operator_event("/loop/Conv", 10, 4),
                _operator_event("/loop/Conv", 20, 6),
            ]
        ),
        encoding="utf-8",
    )

    with monitor:
        pass

    assert monitor.result is not None
    assert monitor.result.operators[0].samples_us == [10.0]


def test_fused_ep_node_only_requires_newer_openvino_ep(tmp_path) -> None:
    monitor = OpenVinoMonitor(output_dir=tmp_path)
    (tmp_path / "onnxruntime_profile.json").write_text(
        json.dumps([_parent_event(0)]),
        encoding="utf-8",
    )

    with monitor:
        pass

    assert monitor.result is not None
    assert monitor.result.status == "parse_failed"
    assert "OpenVINO EP 1.8.95.0 or newer" in (monitor.result.error or "")


@pytest.mark.parametrize("level", ["detail", "invalid"])
def test_rejects_unsupported_level(tmp_path, level: str) -> None:
    with pytest.raises(ValueError, match="only supports level 'basic'"):
        OpenVinoMonitor(level=level, output_dir=tmp_path)


def test_missing_profile_returns_no_data(tmp_path) -> None:
    monitor = OpenVinoMonitor(output_dir=tmp_path)

    with monitor:
        pass

    assert monitor.result is not None
    assert monitor.result.status == "no_data"
