# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for OpenVINO ONNX Runtime profile collection."""

from __future__ import annotations

import json

import pytest

from winml.modelkit.session.monitor.openvino_monitor import OpenVinoMonitor


def test_configure_session_options_enables_profiling(tmp_path) -> None:
    monitor = OpenVinoMonitor(output_dir=tmp_path, device="cpu")

    class SessionOptions:
        enable_profiling = False
        profile_file_prefix = ""

    session_options = SessionOptions()
    monitor.configure_session_options(session_options)

    assert session_options.enable_profiling is True
    assert session_options.profile_file_prefix == str((tmp_path / "onnxruntime_profile").resolve())


def test_parse_basic_profile(tmp_path) -> None:
    monitor = OpenVinoMonitor(output_dir=tmp_path, device="npu")
    monitor.set_onnx_op_types({"node_1": "Conv"})
    monitor.set_perf_window(warmup=1, measured_iterations=2)
    profile_path = tmp_path / "onnxruntime_profile.json"
    profile_path.write_text(
        json.dumps(
            [
                {
                    "name": "node_1_kernel_time",
                    "ts": 10,
                    "dur": 4.0,
                    "args": {"node_name": "node_1", "op_name": "ProviderConv"},
                },
                {
                    "name": "node_1_kernel_time",
                    "ts": 20,
                    "dur": 6.0,
                    "args": {"node_name": "node_1", "op_name": "ProviderConv"},
                },
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
    assert operator.name == "Conv"
    assert operator.samples_us == [4.0, 6.0]
    assert operator.avg_us == 5.0


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
