# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test CPU profiler and ORT-profile parser with mocked ORT (no model run)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from winml.modelkit.optracing.cpu.profile_parser import (
    parse_ort_profile,
    resolve_node_name,
)
from winml.modelkit.optracing.cpu.profiler import (
    CPUProfiler,
    _ort_type_to_numpy,
    _resolve_shape,
)


# =====================================================================
# Helpers: build a minimal ORT profiling JSON
# =====================================================================


def _node_event(name: str, dur: int, op_type: str, provider: str = "CPUExecutionProvider") -> dict:
    """A single ``cat == "Node"`` kernel-time event as ORT emits it."""
    return {
        "cat": "Node",
        "pid": 1,
        "tid": 1,
        "dur": dur,
        "ts": 0,
        "ph": "X",
        "name": f"{name}_kernel_time",
        "args": {"op_name": op_type, "provider": provider},
    }


def _write_profile(path: Path, runs: list[list[dict]]) -> None:
    """Flatten per-run event lists into one ORT profile JSON file."""
    events: list[dict] = [
        {"cat": "Session", "name": "session_initialization", "dur": 100, "ts": 0, "ph": "X"}
    ]
    for run in runs:
        events.extend(run)
    path.write_text(json.dumps(events), encoding="utf-8")


# =====================================================================
# resolve_node_name
# =====================================================================


def test_resolve_node_name_exact_match():
    name_to_type = {"/m/Conv": "Conv"}
    assert resolve_node_name(name_to_type, {}, "/m/Conv") == "/m/Conv"


def test_resolve_node_name_nchwc_suffix_mapped_via_output():
    # NCHWc-reordered kernel: profile name is the output name + "_nchwc".
    name_to_type = {"/m/Conv": "Conv"}
    output_to_name = {"/m/Relu_output_0": "/m/Conv"}
    resolved = resolve_node_name(name_to_type, output_to_name, "/m/Relu_output_0_nchwc")
    assert resolved == "/m/Conv"


def test_resolve_node_name_token_suffix_stripped():
    name_to_type = {"/m/Conv": "Conv"}
    assert resolve_node_name(name_to_type, {}, "/m/Conv_token_3") == "/m/Conv"


def test_resolve_node_name_output_fallback():
    output_to_name = {"out_0": "NodeA"}
    assert resolve_node_name({}, output_to_name, "out_0") == "NodeA"


def test_resolve_node_name_unknown_kept_as_is():
    # EP-inserted kernels (e.g. ReorderOutput) are not in the model graph.
    assert resolve_node_name({}, {}, "ReorderOutput") == "ReorderOutput"


# =====================================================================
# parse_ort_profile
# =====================================================================


def test_parse_aggregates_and_drops_warmup(tmp_path):
    profile = tmp_path / "onnxruntime_profile_x.json"
    name_to_type = {"/m/Conv": "Conv", "/m/Add": "Add"}
    output_to_name: dict[str, str] = {}

    # 1 warmup run (slow) + 2 measured runs (fast). Warmup must be dropped.
    runs = [
        [_node_event("/m/Conv", 1000, "Conv"), _node_event("/m/Add", 500, "Add")],  # warmup
        [_node_event("/m/Conv", 100, "Conv"), _node_event("/m/Add", 50, "Add")],
        [_node_event("/m/Conv", 200, "Conv"), _node_event("/m/Add", 30, "Add")],
    ]
    _write_profile(profile, runs)

    parsed = parse_ort_profile(profile, name_to_type, output_to_name, warmup=1, iterations=2)

    assert parsed["num_samples"] == 2
    ops = {op["op_path"]: op for op in parsed["operators"]}
    # Conv: mean(100, 200) = 150; Add: mean(50, 30) = 40 (warmup 1000/500 dropped).
    assert ops["/m/Conv"]["duration_us"] == 150.0
    assert ops["/m/Add"]["duration_us"] == 40.0
    assert ops["/m/Conv"]["name"] == "Conv"


def test_parse_sorts_by_duration_descending(tmp_path):
    profile = tmp_path / "onnxruntime_profile_x.json"
    name_to_type = {"/m/Conv": "Conv", "/m/Add": "Add"}
    runs = [
        [_node_event("/m/Add", 50, "Add"), _node_event("/m/Conv", 300, "Conv")],
    ]
    _write_profile(profile, runs)

    parsed = parse_ort_profile(profile, name_to_type, {}, warmup=0, iterations=1)

    assert [op["op_path"] for op in parsed["operators"]] == ["/m/Conv", "/m/Add"]


def test_parse_percent_of_total(tmp_path):
    profile = tmp_path / "onnxruntime_profile_x.json"
    name_to_type = {"/m/Conv": "Conv", "/m/Add": "Add"}
    runs = [
        [_node_event("/m/Conv", 75, "Conv"), _node_event("/m/Add", 25, "Add")],
    ]
    _write_profile(profile, runs)

    parsed = parse_ort_profile(profile, name_to_type, {}, warmup=0, iterations=1)
    ops = {op["op_path"]: op for op in parsed["operators"]}
    assert ops["/m/Conv"]["percent_of_total"] == 75.0
    assert ops["/m/Add"]["percent_of_total"] == 25.0


def test_parse_ignores_non_node_and_non_kernel_events(tmp_path):
    profile = tmp_path / "onnxruntime_profile_x.json"
    events = [
        {"cat": "Session", "name": "model_loading_uri", "dur": 10, "ph": "X"},
        {"cat": "Node", "name": "/m/Conv_fence_before", "dur": 5, "ph": "X", "args": {}},
        _node_event("/m/Conv", 100, "Conv"),
    ]
    profile.write_text(json.dumps(events), encoding="utf-8")

    parsed = parse_ort_profile(profile, {"/m/Conv": "Conv"}, {}, warmup=0, iterations=1)
    assert len(parsed["operators"]) == 1
    assert parsed["operators"][0]["op_path"] == "/m/Conv"


# =====================================================================
# Profiler: helpers
# =====================================================================


def test_ort_type_to_numpy_known_and_fallback():
    assert _ort_type_to_numpy("tensor(float)") == np.dtype("float32")
    assert _ort_type_to_numpy("tensor(int64)") == np.dtype("int64")
    assert _ort_type_to_numpy("tensor(bfloat16)") == np.dtype("float32")


def test_resolve_shape_symbolic():
    assert _resolve_shape(["batch", 3, None, -1]) == [1, 3, 1, 1]


def test_generate_inputs():
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "pixel_values"
    mock_input.shape = [1, 3, 224, 224]
    mock_input.type = "tensor(float)"
    mock_session.get_inputs.return_value = [mock_input]

    inputs = CPUProfiler._generate_inputs(mock_session)
    assert inputs["pixel_values"].shape == (1, 3, 224, 224)
    assert inputs["pixel_values"].dtype == np.float32


# =====================================================================
# Profiler: session options
# =====================================================================


def test_cpu_profiler_session_options_enable_profiling(tmp_path):
    profiler = CPUProfiler(Path("model.onnx"), output_dir=tmp_path, level="basic")
    mock_ort = MagicMock()
    mock_options = MagicMock()
    mock_ort.SessionOptions.return_value = mock_options

    options = profiler._build_session_options(mock_ort)

    assert options is mock_options
    assert mock_options.enable_profiling is True
    # Profiling file is placed under the output directory.
    assert mock_options.profile_file_prefix == str(tmp_path / "onnxruntime_profile")
    entries = {c.args[0]: c.args[1] for c in mock_options.add_session_config_entry.call_args_list}
    assert entries["session.intra_op.allow_spinning"] == "1"
    assert entries["session.inter_op.allow_spinning"] == "1"


# =====================================================================
# Profiler: result collection
# =====================================================================


def test_cpu_profiler_collect_results(tmp_path):
    profile = tmp_path / "onnxruntime_profile_x.json"
    runs = [
        [_node_event("/m/Conv", 100, "Conv"), _node_event("/m/Add", 40, "Add")],
    ]
    _write_profile(profile, runs)

    profiler = CPUProfiler(tmp_path / "model.onnx", output_dir=tmp_path, level="basic")

    with patch(
        "winml.modelkit.optracing.cpu.profiler.build_node_maps",
        return_value=({"/m/Conv": "Conv", "/m/Add": "Add"}, {}),
    ):
        result = profiler._collect_results(profile, iterations=1, warmup=0)

    assert result.model == "model.onnx"
    assert result.device == "cpu"
    assert result.ep == "CPUExecutionProvider"
    assert result.tracing_backend == "ort"
    assert len(result.operators) == 2
    assert result.operators[0].op_path == "/m/Conv"
    assert result.operators[0].name == "Conv"
    assert result.summary["num_operators"] == 2
    assert result.summary["total_op_us"] == 140.0
    assert result.artifacts["profile_json"] == str(profile)


def test_cpu_profiler_collect_results_missing_file(tmp_path):
    profiler = CPUProfiler(tmp_path / "model.onnx", output_dir=tmp_path, level="basic")
    result = profiler._collect_results(tmp_path / "nonexistent.json", iterations=1, warmup=0)
    assert result.num_samples == 0
    assert len(result.operators) == 0
    assert result.ep == "CPUExecutionProvider"


# =====================================================================
# Profiler: availability
# =====================================================================


def test_cpu_profiler_is_available():
    profiler = CPUProfiler(Path("model.onnx"), output_dir=Path("out"), level="basic")
    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
    with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
        assert profiler.is_available() is True
