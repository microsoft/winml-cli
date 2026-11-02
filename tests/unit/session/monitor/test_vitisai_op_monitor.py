# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Generated graph and trace coverage for AMD NPU operator profiling."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from winml.modelkit.commands.perf import _monitor_to_json_dict, _resolve_ep_monitor
from winml.modelkit.session import VitisAIOpMonitor
from winml.modelkit.session.monitor import VitisAIOpMonitor as PublicMonitor


_PROVIDER = "VitisAIExecutionProvider"


@pytest.fixture
def source_model(tmp_path):
    nodes = [
        helper.make_node("Relu", [f"value_{index}"], [f"value_{index + 1}"], name=f"node_{index}")
        for index in range(3)
    ]
    graph = helper.make_graph(
        nodes,
        "generated",
        [helper.make_tensor_value_info(nodes[0].input[0], TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info(nodes[-1].output[0], TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    path = tmp_path / "source.onnx"
    onnx.save(model, path)
    return path, nodes


def _generate_trace(nodes, partitions=1, runs=4, invocations=1, model_runs=True):
    durations = np.random.default_rng(42).uniform(
        1, 10, (runs, partitions, invocations, len(nodes), 2)
    )
    events = []
    for run in range(runs):
        run_start = run * 10000
        if model_runs:
            events.append({"cat": "Session", "name": "model_run", "ts": run_start, "dur": 9000})
        for partition in range(partitions):
            for invocation in range(invocations):
                start = run_start + partition * 2000 + invocation * 500
                events.append(
                    {
                        "cat": "Node",
                        "name": f"{_PROVIDER}_{partition}_kernel_time",
                        "pid": 1,
                        "ts": start,
                        "dur": 400,
                    }
                )
                for index, node in enumerate(nodes):
                    # Exercise both EP output suffix forms as well as direct node wrappers.
                    names = [
                        f"subgraph_({node.name})_fix",
                        f"subgraph_({node.output[0]}_vaip_{index})",
                        f"subgraph_({node.output[0]}_vaip_{index}_{partition})",
                    ]
                    for repeat, duration in enumerate(durations[run, partition, invocation, index]):
                        events.append(
                            {
                                "cat": "Kernel",
                                "name": names[index % len(names)],
                                "ts": start + index * 50 + repeat * 20 + 1,
                                "pid": 1,
                                "dur": float(duration),
                                "args": {"op_type": f"native_{node.op_type}"},
                            }
                        )
    np.random.default_rng(19).shuffle(events)
    return events, durations


def _write_trace(monitor, payload):
    options = SimpleNamespace(enable_profiling=False, profile_file_prefix="")
    monitor.configure_session_options(options)
    path = Path(f"{options.profile_file_prefix}_generated.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def monitor(tmp_path, source_model):
    path, nodes = source_model
    monitor = VitisAIOpMonitor(output_dir=tmp_path)
    monitor.set_onnx_model_path(path)
    monitor.set_onnx_op_types({node.name: node.op_type for node in nodes})
    return monitor


def _parse_trace(monitor, payload):
    _write_trace(monitor, payload)
    with monitor:
        pass
    assert monitor.result is not None
    return monitor.result


def test_exports_and_session_configuration(tmp_path):
    assert PublicMonitor is VitisAIOpMonitor
    options = SimpleNamespace(enable_profiling=False, profile_file_prefix="")
    monitor = VitisAIOpMonitor(output_dir=tmp_path)
    monitor.configure_session_options(options)
    assert options.enable_profiling
    assert Path(options.profile_file_prefix).parent == tmp_path.resolve()
    assert monitor.configures_session_options
    assert monitor.requires_session_teardown
    assert monitor.ep_name == "vitisai"
    assert monitor.get_provider_options() == {}
    assert monitor.result is None


@pytest.mark.parametrize(
    ("partitions", "model_runs", "wrapped", "invocations"),
    [
        (1, False, False, 1),
        (2, False, True, 1),
        (1, True, True, 1),
        (2, True, False, 1),
        (2, True, False, 2),
    ],
)
def test_warmup_mapping_partition_aggregation_and_statistics(
    monitor, source_model, partitions, model_runs, wrapped, invocations
):
    path, nodes = source_model
    warmup, measured = 1, 3
    events, durations = _generate_trace(
        nodes, partitions=partitions, model_runs=model_runs, invocations=invocations
    )
    monitor.set_perf_window(warmup, measured)
    result = _parse_trace(monitor, {"traceEvents": events} if wrapped else events)
    expected = durations.sum(axis=(1, 2, 4))[warmup:]
    assert result.status == "ok"
    assert result.model == str(path)
    assert result.device == "npu"
    assert result.ep == "vitisai"
    assert result.tracing_backend == "onnxruntime"
    assert result.num_samples == measured
    assert Path(result.artifacts["profile"]).is_file()
    assert result.summary["accel_execute_us"] == pytest.approx(expected.sum())
    assert [op.op_path for op in result.operators] == [node.name for node in nodes]
    for index, operator in enumerate(result.operators):
        samples = expected[:, index]
        assert operator.name == f"native_{nodes[index].op_type}"
        assert operator.onnx_op_type == nodes[index].op_type
        assert operator.samples_us == pytest.approx(samples)
        assert operator.avg_us == pytest.approx(samples.mean())
        assert operator.duration_us == operator.avg_us
        assert operator.p90_us == pytest.approx(np.quantile(samples, 0.9))
        assert operator.percent_of_total == pytest.approx(samples.sum() / expected.sum() * 100)
        assert result.statistics[operator.op_path]["count"] == measured
    assert _monitor_to_json_dict(monitor) == json.loads(result.to_json())


def test_variable_partition_invocations_use_explicit_run_boundaries(monitor, source_model):
    _, nodes = source_model
    events, durations = _generate_trace(nodes, invocations=2)
    parents = sorted(
        (event for event in events if event["cat"] == "Node"), key=lambda event: event["ts"]
    )
    removed = parents[3]
    events = [
        event
        for event in events
        if not (
            event["cat"] in ("Node", "Kernel")
            and removed["ts"] <= event["ts"] <= removed["ts"] + removed["dur"]
        )
    ]
    monitor.set_perf_window(1, durations.shape[0] - 1)
    result = _parse_trace(monitor, events)
    expected = durations.sum(axis=(1, 2, 4))
    expected[1] -= durations[1, 0, 1].sum(axis=1)
    assert result.status == "ok"
    for index, operator in enumerate(result.operators):
        assert operator.samples_us == pytest.approx(expected[1:, index])


def test_overlapping_parent_spans_do_not_double_count_kernels(monitor, source_model):
    events, _ = _generate_trace(source_model[1])
    parent = next(event for event in events if event["cat"] == "Node")
    events.append({**parent, "name": parent["name"].replace("_0_", "_1_")})
    result = _parse_trace(monitor, events)
    assert result.status == "parse_failed"
    assert "overlapping parent spans" in result.error
    assert not result.operators


def test_unrelated_kernels_and_cpu_nodes_are_not_attributed(monitor, source_model):
    _, nodes = source_model
    events, durations = _generate_trace(nodes)
    kernel = next(event for event in events if event["cat"] == "Kernel")
    events.extend(
        [
            {**kernel, "pid": kernel["pid"] + 1},
            {**kernel, "ts": max(event["ts"] for event in events) + 10000},
            {**kernel, "args": {"provider": "CPUExecutionProvider"}},
            {**kernel, "cat": "Node"},
        ]
    )
    result = _parse_trace(monitor, events)
    assert result.status == "ok"
    assert result.summary["accel_execute_us"] == pytest.approx(durations.sum())


def test_type_fallback_and_unresolved_names(monitor, source_model):
    _, nodes = source_model
    events, _ = _generate_trace(nodes)
    unknown_name = f"added_{nodes[-1].name}"
    for event in events:
        if event["cat"] == "Kernel":
            event["args"] = {}
            if nodes[-1].output[0] in event["name"]:
                event["name"] = unknown_name
    result = _parse_trace(monitor, events)
    assert result.status == "ok"
    for node, operator in zip(nodes, result.operators, strict=True):
        if operator.op_path == unknown_name:
            assert operator.name == "Unknown"
            assert "onnx_op_type" not in operator.to_dict()
        else:
            assert operator.name == node.op_type
            assert operator.onnx_op_type == node.op_type


def test_output_wrapper_can_resolve_a_node_name_directly(monitor, source_model):
    _, nodes = source_model
    events, durations = _generate_trace(nodes[:1])
    for event in events:
        if event["cat"] == "Kernel":
            event["name"] = f"subgraph_({nodes[0].name}_vaip_0)"
    operator = _parse_trace(monitor, events).operators[0]
    assert operator.op_path == nodes[0].name
    assert operator.onnx_op_type == nodes[0].op_type
    assert operator.total_us == pytest.approx(durations.sum())


def test_missing_metadata_is_logged_and_trace_remains_available(tmp_path, source_model, caplog):
    monitor = VitisAIOpMonitor(output_dir=tmp_path)
    monitor.set_onnx_model_path(tmp_path / "missing.onnx")
    events, _ = _generate_trace(source_model[1])
    result = _parse_trace(monitor, events)
    assert "Could not load VitisAI source graph metadata" in caplog.text
    assert result.status == "ok"
    assert all(operator.onnx_op_type is None for operator in result.operators)


def test_stale_artifact_is_not_reused(tmp_path, source_model):
    old_monitor = VitisAIOpMonitor(output_dir=tmp_path)
    events, _ = _generate_trace(source_model[1])
    old_path = _write_trace(old_monitor, events)
    monitor = VitisAIOpMonitor(output_dir=tmp_path)
    with monitor:
        pass
    assert monitor.result.status == "no_data"
    assert "2.1000.81" in monitor.result.error
    assert old_path.exists()


def test_fused_parent_only_reports_minimum_ep_version(monitor, source_model):
    events, _ = _generate_trace(source_model[1])
    result = _parse_trace(monitor, [event for event in events if event["cat"] != "Kernel"])
    assert result.status == "no_data"
    assert "VitisAI EP 2.1000.81 or newer" in result.error
    assert Path(result.artifacts["profile"]).is_file()


def test_zero_measured_runs_excludes_all_kernels(monitor, source_model):
    events, durations = _generate_trace(source_model[1])
    monitor.set_perf_window(durations.shape[0], 0)
    result = _parse_trace(monitor, events)
    assert result.status == "no_data"
    assert result.num_samples == 0
    assert not result.operators


@pytest.mark.parametrize(
    ("model_runs", "invocations", "extra_run", "message"),
    [
        (False, 1, 1, "Cannot align"),
        (True, 1, 1, "expected"),
        (False, 2, 0, "Cannot align"),
    ],
)
def test_unaligned_run_counts_are_rejected(
    monitor, source_model, model_runs, invocations, extra_run, message
):
    events, durations = _generate_trace(
        source_model[1], model_runs=model_runs, invocations=invocations
    )
    monitor.set_perf_window(1, durations.shape[0] - 1 + extra_run)
    result = _parse_trace(monitor, events)
    assert result.status == "parse_failed"
    assert message in result.error
    assert not result.operators


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dur", float("nan")),
        ("dur", float("inf")),
        ("dur", -1),
        ("dur", True),
        ("ts", None),
        ("name", ""),
        ("args", []),
        ("args", {"op_type": []}),
    ],
)
def test_malformed_kernel_reports_failure(monitor, source_model, field, value):
    events, _ = _generate_trace(source_model[1])
    next(event for event in events if event["cat"] == "Kernel")[field] = value
    result = _parse_trace(monitor, events)
    assert result.status == "parse_failed"
    assert result.error
    assert Path(result.artifacts["profile"]).is_file()


@pytest.mark.parametrize("payload", [None, {}, {"traceEvents": {}}, [None]])
def test_invalid_profile_structure(monitor, payload):
    assert _parse_trace(monitor, payload).status == "parse_failed"


def test_truncated_json(monitor, source_model):
    events, _ = _generate_trace(source_model[1])
    path = _write_trace(monitor, events)
    path.write_text(json.dumps(events)[:-1], encoding="utf-8")
    with monitor:
        pass
    assert monitor.result.status == "parse_failed"


def test_single_use_and_invalid_configuration(tmp_path):
    monitor = VitisAIOpMonitor(output_dir=tmp_path)
    with monitor:
        pass
    with pytest.raises(RuntimeError, match="already entered"), monitor:
        pass
    with pytest.raises(ValueError, match="only supports level 'basic'"):
        VitisAIOpMonitor(level="detail", output_dir=tmp_path)
    for warmup, measured in [(-1, 1), (1, -1)]:
        with pytest.raises(ValueError, match="non-negative"):
            monitor.set_perf_window(warmup, measured)


@pytest.mark.parametrize(
    ("ep", "device"),
    [
        ("vitisai", "npu"),
        ("VITISAI", "NPU"),
        ("VitisAIExecutionProvider", "auto"),
        ("vitisai", None),
    ],
)
def test_perf_dispatch(tmp_path, monkeypatch, ep, device):
    monkeypatch.setattr(VitisAIOpMonitor, "is_available", classmethod(lambda cls: True))
    assert isinstance(_resolve_ep_monitor(ep, "basic", tmp_path, device), VitisAIOpMonitor)


@pytest.mark.parametrize(
    ("level", "device", "available", "message"),
    [
        ("detail", "npu", True, "only level 'basic'"),
        ("basic", "cpu", True, "--device npu"),
        ("basic", "gpu", True, "--device npu"),
        ("basic", "npu", False, "VitisAI is not available"),
    ],
)
def test_dispatch_errors(tmp_path, monkeypatch, level, device, available, message):
    monkeypatch.setattr(VitisAIOpMonitor, "is_available", classmethod(lambda cls: available))
    with pytest.raises(RuntimeError, match=message):
        _resolve_ep_monitor("vitisai", level, tmp_path, device)


@pytest.mark.parametrize("source", ["builtin", "plugin", "absent", "error"])
def test_availability(monkeypatch, source):
    import onnxruntime as ort

    from winml.modelkit.session import WinMLEPRegistry

    monkeypatch.setattr(WinMLEPRegistry, "instance", classmethod(lambda cls: None))
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: [_PROVIDER] if source == "builtin" else []
    )

    def devices():
        if source == "error":
            raise RuntimeError("provider discovery failed")
        return [SimpleNamespace(ep_name=_PROVIDER)] if source == "plugin" else []

    monkeypatch.setattr(ort, "get_ep_devices", devices)
    assert VitisAIOpMonitor.is_available() == (source in ("builtin", "plugin"))


def test_real_ort_profile_is_flushed_on_session_teardown(monitor, source_model):
    import onnxruntime as ort

    options = ort.SessionOptions()
    monitor.configure_session_options(options)
    monitor.set_perf_window(1, 2)
    with monitor:
        session = ort.InferenceSession(
            str(source_model[0]), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = {
            entry.name: np.random.default_rng(42).standard_normal(entry.shape).astype(np.float32)
            for entry in session.get_inputs()
        }
        for _ in range(3):
            session.run(None, inputs)
        del session
    result = monitor.result
    assert result.status == "no_data"
    assert result.num_samples == 2
    events = json.loads(Path(result.artifacts["profile"]).read_text(encoding="utf-8"))
    assert sum(event.get("name") == "model_run" for event in events) == 3
