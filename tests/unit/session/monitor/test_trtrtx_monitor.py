# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Generated trace coverage for TensorRT RTX monitor parsing and lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from onnx import helper

from winml.modelkit.commands.perf import _monitor_to_json_dict, _resolve_ep_monitor
from winml.modelkit.session import NvTensorRTRTXMonitor
from winml.modelkit.session.monitor import NvTensorRTRTXMonitor as PublicMonitor


def _generate_trace(contexts=1, runs=4, layers=3, repeats=2):
    durations = np.random.default_rng(17).uniform(1, 100, (contexts, runs, layers, repeats))
    names = [f"layer_{index}" for index in range(layers)]
    tids = np.random.default_rng(19).permutation(runs).tolist()
    events = []
    for pid in range(contexts):
        events.append({"ph": "M", "name": "process_name", "pid": pid})
        for index, tid in enumerate(tids):
            for layer, name in enumerate(names):
                events.extend(
                    {
                        "cat": "nv::trt::layer",
                        "ph": "X",
                        "pid": pid,
                        "tid": tid,
                        "name": name,
                        "dur": float(duration),
                    }
                    for duration in durations[pid, index, layer]
                )
    return events, durations, names


def _write_trace(monitor, payload):
    path = Path(monitor.get_provider_options()["nv_profiling_output_file"])
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_public_exports_and_provider_options(tmp_path):
    assert PublicMonitor is NvTensorRTRTXMonitor
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    options = monitor.get_provider_options()
    assert options["nv_enable_profiling"] == "1"
    assert Path(options["nv_profiling_output_file"]).parent == tmp_path.resolve()
    assert monitor.requires_session_teardown
    assert monitor.ep_name == "nvtensorrtrtx"
    assert monitor.result is None
    assert options == monitor.get_provider_options()
    assert options != NvTensorRTRTXMonitor(output_dir=tmp_path).get_provider_options()


@pytest.mark.parametrize("contexts", [1, 2])
@pytest.mark.parametrize("wrapped", [False, True])
def test_run_grouping_warmup_and_statistics(tmp_path, contexts, wrapped):
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    events, durations, names = _generate_trace(contexts=contexts)
    warmup, measured = 1, 2
    monitor.set_perf_window(warmup, measured)
    model_path = tmp_path / "model.onnx"
    monitor.set_onnx_model_path(model_path)
    profile = _write_trace(monitor, {"traceEvents": events} if wrapped else events)
    with monitor:
        pass

    result = monitor.result
    expected = durations.sum(axis=(0, 3))[warmup : warmup + measured]
    assert result.status == "ok"
    assert result.model == str(model_path)
    assert result.device == "gpu"
    assert result.num_samples == measured
    assert result.artifacts["profile"] == str(profile)
    assert result.summary["accel_execute_us"] == pytest.approx(expected.sum())
    assert [operator.op_path for operator in result.operators] == names
    for index, operator in enumerate(result.operators):
        samples = expected[:, index]
        assert operator.samples_us == pytest.approx(samples)
        assert operator.duration_us == pytest.approx(samples.mean())
        assert operator.p90_us == pytest.approx(np.quantile(samples, 0.9))
        assert operator.percent_of_total == pytest.approx(samples.sum() / expected.sum() * 100)
        assert result.statistics[operator.op_path]["count"] == measured
    assert _monitor_to_json_dict(monitor) == json.loads(result.to_json())


@pytest.mark.parametrize("separator", [r"]\u001f[ONNX Layer: ", "]\x1f[ONNX Layer: "])
def test_native_names_and_exact_type_enrichment(tmp_path, separator):
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    events, _, names = _generate_trace(layers=4)
    nodes = [
        helper.make_node("Relu", [f"value_{index}"], [f"value_{index + 1}"], name=f"node_{index}")
        for index in range(4)
    ]
    nodes[0].name = names[0]
    node_types = {node.name: node.op_type for node in nodes}
    monitor.set_onnx_op_types(node_types)
    for event in events:
        if event["ph"] != "X":
            continue
        index = names.index(event["name"])
        if index == 1:
            event["args"] = {"onnx_nodes": nodes[index].name}
        elif index == 2:
            event["args"] = {"onnx_nodes": separator.join(node.name for node in nodes[2:])}
    _write_trace(monitor, events)
    with monitor:
        pass

    assert [operator.op_path for operator in monitor.result.operators] == names
    serialized = json.loads(monitor.result.to_json())["operators"]
    for index, operator in enumerate(monitor.result.operators):
        expected_type = nodes[index].op_type if index < 2 else None
        assert operator.onnx_op_type == expected_type
        assert operator.name == (expected_type or ("Fused" if index == 2 else "Unknown"))
        source_nodes = [nodes[index]] if index < 2 else nodes[2:] if index == 2 else []
        expected_nodes = [{"name": node.name, "op_type": node.op_type} for node in source_nodes]
        assert operator.onnx_nodes == (expected_nodes or None)
        if source_nodes:
            assert serialized[index]["onnx_nodes"] == expected_nodes
        else:
            assert "onnx_nodes" not in serialized[index]
        if expected_type is None:
            assert "onnx_op_type" not in serialized[index]


def test_inconsistent_type_matches_are_not_attributed_to_one_node(tmp_path):
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    events, _, _ = _generate_trace(layers=1, repeats=1)
    nodes = [
        helper.make_node(op_type, ["input"], ["output"], name=f"node_{index}")
        for index, op_type in enumerate(("Relu", "Sigmoid"))
    ]
    monitor.set_onnx_op_types({node.name: node.op_type for node in nodes})
    layers = [event for event in events if event["ph"] == "X"]
    for index, event in enumerate(layers):
        event["args"] = {"onnx_nodes": nodes[index % len(nodes)].name}
    _write_trace(monitor, events)
    with monitor:
        pass
    assert monitor.result.operators[0].onnx_op_type is None
    assert monitor.result.operators[0].name == "Unknown"
    assert monitor.result.operators[0].onnx_nodes == [
        {"name": node.name, "op_type": node.op_type} for node in nodes
    ]


@pytest.mark.parametrize("separator", [r"]\u001f[ONNX Layer: ", "]\x1f[ONNX Layer: "])
@pytest.mark.parametrize("resolved_count", [0, 1, 2])
def test_fused_metadata_overrides_native_match_and_preserves_unresolved_nodes(
    tmp_path, separator, resolved_count
):
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    events, durations, names = _generate_trace(layers=1)
    nodes = [
        helper.make_node(op_type, [f"value_{index}"], [f"value_{index + 1}"], name=f"node_{index}")
        for index, op_type in enumerate(("MatMul", "Add"))
    ]
    node_types = {node.name: node.op_type for node in nodes[:resolved_count]}
    node_types[names[0]] = nodes[0].op_type
    monitor.set_onnx_op_types(node_types)
    for event in events:
        if event["ph"] == "X":
            event["args"] = {"onnx_nodes": separator.join(node.name for node in [*nodes, nodes[0]])}
    _write_trace(monitor, events)
    with monitor:
        pass
    operator = monitor.result.operators[0]
    assert operator.name == "Fused"
    assert operator.op_path == names[0]
    assert operator.onnx_op_type is None
    assert operator.onnx_nodes == [
        {"name": node.name, "op_type": node_types.get(node.name)} for node in nodes
    ]
    assert operator.samples_us == pytest.approx(durations.sum(axis=(0, 2, 3)))


def test_single_unresolved_source_keeps_name_with_null_type(tmp_path):
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    events, _, names = _generate_trace(layers=1)
    node = helper.make_node("Relu", ["input"], ["output"], name=f"{names[0]}_source")
    for event in events:
        if event["ph"] == "X":
            event["args"] = {"onnx_nodes": node.name}
    _write_trace(monitor, events)
    with monitor:
        pass
    operator = json.loads(monitor.result.to_json())["operators"][0]
    assert operator["name"] == "Unknown"
    assert operator["onnx_nodes"] == [{"name": node.name, "op_type": None}]
    assert "onnx_op_type" not in operator


def test_missing_and_stale_artifacts_are_not_reused(tmp_path):
    old_monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    events, _, _ = _generate_trace()
    old_path = _write_trace(old_monitor, events)
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    with monitor:
        pass
    assert monitor.result.status == "no_data"
    assert "TensorRT RTX EP 2.30.49 or newer" in monitor.result.error
    assert "nv_enable_profiling" in monitor.result.error
    assert old_path.exists()


@pytest.mark.parametrize("measured", [0, 2])
def test_no_layer_data_is_reported(tmp_path, measured):
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    monitor.set_perf_window(0, measured)
    events, _, _ = _generate_trace()
    _write_trace(monitor, [event for event in events if event["ph"] != "X"])
    with monitor:
        pass
    assert monitor.result.status == "no_data"
    assert monitor.result.error


def test_empty_measured_window_excludes_all_layers(tmp_path):
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    events, durations, _ = _generate_trace()
    monitor.set_perf_window(durations.shape[1], 0)
    _write_trace(monitor, events)
    with monitor:
        pass
    assert monitor.result.status == "no_data"
    assert monitor.result.num_samples == 0


def test_incomplete_context_is_a_parse_failure(tmp_path):
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    events, durations, _ = _generate_trace(contexts=2)
    monitor.set_perf_window(1, durations.shape[1])
    _write_trace(monitor, events)
    with monitor:
        pass
    assert monitor.result.status == "parse_failed"
    assert "expected at least" in monitor.result.error


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dur", float("nan")),
        ("dur", float("inf")),
        ("dur", -1),
        ("dur", True),
        ("pid", None),
        ("tid", []),
        ("name", ""),
        ("args", []),
        ("args", {"onnx_nodes": []}),
    ],
)
def test_malformed_layer_is_not_silently_dropped(tmp_path, field, value):
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    events, _, _ = _generate_trace()
    next(event for event in events if event["ph"] == "X")[field] = value
    path = _write_trace(monitor, events)
    with monitor:
        pass
    assert monitor.result.status == "parse_failed"
    assert monitor.result.error
    assert monitor.result.artifacts["profile"] == str(path)


@pytest.mark.parametrize("payload", [None, {}, {"traceEvents": {}}, [None]])
def test_invalid_trace_structure(tmp_path, payload):
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    _write_trace(monitor, payload)
    with monitor:
        pass
    assert monitor.result.status == "parse_failed"


def test_truncated_json_is_a_parse_failure(tmp_path):
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    events, _, _ = _generate_trace()
    path = _write_trace(monitor, events)
    path.write_text(json.dumps(events)[:-1], encoding="utf-8")
    with monitor:
        pass
    assert monitor.result.status == "parse_failed"


def test_monitor_cannot_be_reused(tmp_path):
    monitor = NvTensorRTRTXMonitor(output_dir=tmp_path)
    with monitor:
        pass
    with pytest.raises(RuntimeError, match="already entered"), monitor:
        pass


@pytest.mark.parametrize("warmup,measured", [(-1, 1), (1, -1)])
def test_invalid_perf_window(tmp_path, warmup, measured):
    with pytest.raises(ValueError, match="non-negative"):
        NvTensorRTRTXMonitor(output_dir=tmp_path).set_perf_window(warmup, measured)


def test_unsupported_level(tmp_path):
    with pytest.raises(ValueError, match="only supports level 'basic'"):
        NvTensorRTRTXMonitor(level="detail", output_dir=tmp_path)


@pytest.mark.parametrize(
    "ep", ["nv_tensorrt_rtx", "nvtensorrtrtx", "NvTensorRTRTXExecutionProvider", "NV_TENSORRT_RTX"]
)
@pytest.mark.parametrize("device", ["gpu", "GPU", "auto", None])
def test_perf_dispatch(tmp_path, monkeypatch, ep, device):
    monkeypatch.setattr(NvTensorRTRTXMonitor, "is_available", classmethod(lambda cls: True))
    monitor = _resolve_ep_monitor(ep, "basic", tmp_path, device)
    assert isinstance(monitor, NvTensorRTRTXMonitor)


@pytest.mark.parametrize(
    ("level", "device", "available", "error"),
    [
        ("detail", "gpu", True, "only level 'basic'"),
        ("basic", "cpu", True, "--device gpu"),
        ("basic", "npu", True, "--device gpu"),
        ("basic", "gpu", False, "TensorRT RTX is not available"),
    ],
)
def test_perf_dispatch_errors(tmp_path, monkeypatch, level, device, available, error):
    monkeypatch.setattr(NvTensorRTRTXMonitor, "is_available", classmethod(lambda cls: available))
    with pytest.raises(RuntimeError, match=error):
        _resolve_ep_monitor("nv_tensorrt_rtx", level, tmp_path, device)


@pytest.mark.parametrize("source", ["builtin", "plugin", "absent", "error"])
def test_availability(monkeypatch, source):
    import onnxruntime as ort

    from winml.modelkit.session import WinMLEPRegistry

    provider = "NvTensorRTRTXExecutionProvider"
    monkeypatch.setattr(WinMLEPRegistry, "instance", classmethod(lambda cls: None))
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: [provider] if source == "builtin" else []
    )

    def devices():
        if source == "error":
            raise RuntimeError("provider discovery failed")
        return [SimpleNamespace(ep_name=provider)] if source == "plugin" else []

    monkeypatch.setattr(ort, "get_ep_devices", devices)
    assert NvTensorRTRTXMonitor.is_available() == (source in ("builtin", "plugin"))
