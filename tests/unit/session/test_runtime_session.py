# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the Windows ML Runtime inference backend."""

from __future__ import annotations

import json
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import click
import numpy as np
import pytest

from winml.modelkit.session.runtime_session import (
    WinMLRuntimeSession,
    _apply_io_metadata,
    _DXCoreAdapter,
    _numpy_dtype_for,
    _ResolvedRuntimeTarget,
    _shape_with_dynamic_dims,
    _stage_schema,
    _symbolic_dimensions_for_inputs,
)


class _NotSupportedError(Exception):
    pass


class _Schema:
    input_count = 1
    output_count = 1

    def input_desc(self, index: int) -> tuple[Any, list[int]]:
        assert index == 0
        return SimpleNamespace(name="FLOAT32"), [0, 3, 8, 8]

    def output_desc(self, index: int) -> tuple[Any, list[int]]:
        assert index == 0
        return SimpleNamespace(name="FLOAT32"), [0, 5]


class _Tensor:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def to_numpy(self) -> np.ndarray:
        return self._value


class _Stage:
    def __init__(self) -> None:
        self.execution_target = SimpleNamespace(kind=SimpleNamespace(name="GPU"))
        self.bound: dict[int, np.ndarray] = {}
        self.requested_outputs: set[int] = set()
        self.caller_owns_state_tensors = False

    def schema(self) -> _Schema:
        return _Schema()

    def ort_diagnostics(self) -> None:
        raise _NotSupportedError

    def bind_buffers(self, inputs, outputs) -> None:
        self.bound = dict(enumerate(inputs))

    def output(self, index: int) -> _Tensor:
        assert index == 0
        assert index in self.requested_outputs
        batch = self.bound[0].shape[0]
        return _Tensor(np.zeros((batch, 5), dtype=np.float32))

    def request_output(self, index: int) -> None:
        self.requested_outputs.add(index)

    def close(self) -> None:
        pass


class _Pipeline:
    def __init__(self) -> None:
        self.runs = 0

    def run(self) -> None:
        self.runs += 1

    def close(self) -> None:
        pass


class _Builder:
    def __init__(self, stage: _Stage, pipeline: _Pipeline) -> None:
        self.stage = stage
        self.pipeline = pipeline
        self.targets: list[Any] = []
        self.caller_owned_at_build: bool | None = None

    def add_model_stage(self, model: Any, target: Any = None) -> _Stage:
        self.targets.append(target)
        return self.stage

    def build(self) -> _Pipeline:
        self.caller_owned_at_build = self.stage.caller_owns_state_tensors
        return self.pipeline


class _Model:
    def ort_schema(self) -> None:
        raise _NotSupportedError

    def close(self) -> None:
        pass


class _Runtime:
    def __init__(self, stage: _Stage, pipeline: _Pipeline) -> None:
        self.builder = _Builder(stage, pipeline)

    def load_model(self, path: str) -> _Model:
        assert path.endswith(".mlir")
        return _Model()

    def create_pipeline_builder(self) -> _Builder:
        return self.builder

    def create_target_from_adapter(self, adapter: object) -> object:
        return adapter

    def close(self) -> None:
        pass


class _AdapterHandle:
    pointer = 123

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _mlir_ep_device() -> SimpleNamespace:
    return SimpleNamespace(
        device=SimpleNamespace(
            device_type="GPU",
            hardware_name="Test GPU",
            adapter_luid=456,
        )
    )


@pytest.fixture
def mlir_target(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, _AdapterHandle]:
    adapter = _AdapterHandle()

    def from_luid(cls: type[_DXCoreAdapter], luid: int) -> _AdapterHandle:
        assert luid == 456
        return adapter

    monkeypatch.setattr(_DXCoreAdapter, "from_luid", classmethod(from_luid))
    return _mlir_ep_device(), adapter


def test_onnx_cgc_reloads_compiled_model_without_io_counts() -> None:
    source_model = Mock()
    compiled_model = _Model()
    compiler = Mock()
    loaded_paths: list[str] = []

    def load_model(path: str) -> Any:
        loaded_paths.append(path)
        return source_model if len(loaded_paths) == 1 else compiled_model

    runtime = SimpleNamespace(load_model=load_model)
    target = _ResolvedRuntimeTarget(
        execution_target=SimpleNamespace(model_compiler=lambda: compiler),
        device_class="gpu",
    )
    session = WinMLRuntimeSession("model.onnx", ep_device=_mlir_ep_device(), backend="cgc")
    try:
        model, ort_schema, has_named_bindings = session._load_onnx_on_cgc(runtime, target)

        assert model is compiled_model
        assert ort_schema is source_model.ort_schema.return_value
        assert has_named_bindings is False
        assert loaded_paths[0] == str(session.running_model_path)
        assert len(loaded_paths) == 2
        compiler.compile_to_file.assert_called_once_with(source_model, loaded_paths[1])
        compiler.close.assert_called_once_with()
        source_model.schema.assert_not_called()
    finally:
        session.reset()


@pytest.mark.parametrize("fails", [False, True])
def test_cgc_compile_uses_public_symbolic_options_and_closes_them(monkeypatch, fails):
    io = {
        "input_names": ["x"],
        "input_shapes": [[None, 3]],
        "input_symbolic_shapes": [["batch", 3]],
    }
    monkeypatch.setattr("winml.modelkit.onnx.get_io_config", lambda _path: io)
    options = SimpleNamespace(
        supports_symbolic_dimensions=True, symbolic_dimensions={}, close=Mock()
    )
    source = Mock()
    compiler = Mock(create_options=Mock(return_value=options))
    if fails:
        compiler.compile_to_file.side_effect = RuntimeError("compile failed")
    runtime = Mock(load_model=Mock(side_effect=[source, _Model()]))
    target = _ResolvedRuntimeTarget(
        execution_target=SimpleNamespace(model_compiler=lambda: compiler), device_class="gpu"
    )
    session = WinMLRuntimeSession("source.onnx", ep_device=_mlir_ep_device(), backend="cgc")
    session.set_input_shapes({"x": (2, 3)})
    try:
        if fails:
            with pytest.raises(RuntimeError, match="compile failed"):
                session._load_onnx_on_cgc(runtime, target)
        else:
            session._load_onnx_on_cgc(runtime, target)
        assert options.symbolic_dimensions == {"batch": 2}
        assert compiler.compile_to_file.call_args.kwargs == {"options": options}
        options.close.assert_called_once_with()
        compiler.close.assert_called_once_with()
    finally:
        session.reset()


@pytest.mark.parametrize("shape", [(True, 3), (1.5, 3), (0, 3), (1 << 63, 3)])
def test_symbolic_dimensions_reject_invalid_extents(shape):
    io = {
        "input_names": ["x"],
        "input_shapes": [[None, 3]],
        "input_symbolic_shapes": [["batch", 3]],
    }
    with pytest.raises(click.ClickException):
        _symbolic_dimensions_for_inputs(io, {"x": shape})


def test_schema_helpers() -> None:
    assert _numpy_dtype_for(SimpleNamespace(name="FLOAT16")) == "float16"
    assert _shape_with_dynamic_dims([0, -1, (1 << 64) - 1, 4]) == [
        None,
        None,
        None,
        4,
    ]


def test_stage_schema_falls_back_to_projection_interface() -> None:
    interface = Mock()
    wrapped = Mock()
    stage = SimpleNamespace(interface=Mock())
    wr = SimpleNamespace(
        _b=SimpleNamespace(IWinMLStageSchema=interface),
        ModelSchema=Mock(return_value=wrapped),
    )

    assert _stage_schema(wr, stage) is wrapped
    stage.interface.QueryInterface.assert_called_once_with(interface)
    wr.ModelSchema.assert_called_once_with(
        stage.interface.QueryInterface.return_value, stage
    )


def test_persistent_compile_normalizes_dynamic_schema_dimensions(tmp_path, monkeypatch) -> None:
    source_schema = Mock()
    source_schema.inputs.return_value = [("float32", (0, 4))]
    source_schema.outputs.return_value = [("float32", (0, 4))]
    compiled_schema = Mock()
    compiled_schema.inputs.return_value = [("float32", ((1 << 64) - 1, 4))]
    compiled_schema.outputs.return_value = [("float32", ((1 << 64) - 1, 4))]
    source_model = Mock(schema=Mock(return_value=source_schema))
    compiled_model = Mock()
    options = SimpleNamespace(form=None, close=Mock())
    compiler = Mock(
        supports_form=Mock(return_value=True),
        create_options=Mock(return_value=options),
    )
    output = tmp_path / "output.mlir"
    compiler.compile_to_file.side_effect = lambda _model, path, **_kwargs: Path(
        path
    ).write_bytes(b"ML\xefRcompiled")
    target = Mock(model_compiler=Mock(return_value=compiler))
    resolved = _ResolvedRuntimeTarget(target, "gpu")
    stage = Mock(schema=Mock(return_value=compiled_schema))
    builder = Mock()
    builder.add_model_stage.return_value = stage
    runtime = Mock()
    runtime.load_model.side_effect = [source_model, compiled_model]
    runtime.create_pipeline_builder.return_value = builder
    wr = SimpleNamespace(
        Runtime=Mock(return_value=runtime),
        CompiledModelForm=SimpleNamespace(DEVICE_TARGETED="device-targeted"),
    )
    monkeypatch.setattr(
        "winml.modelkit.session.runtime_session.import_runtime", lambda: wr
    )
    session = WinMLRuntimeSession(
        tmp_path / "input.mlir", backend="cgc", device_luid=1
    )
    monkeypatch.setattr(session, "_resolve_target", lambda _runtime, _wr: resolved)

    session.compile_to_file(output)

    assert output.is_file()
    assert options.form == "device-targeted"
    compiler.compile_to_file.assert_called_once_with(
        source_model, str(output), options=options
    )
    compiled_schema.close.assert_called_once_with()
    with pytest.raises(click.ClickException):
        _numpy_dtype_for(SimpleNamespace(name="BFLOAT16"))


def test_io_metadata_replaces_ordinal_names(tmp_path: Path) -> None:
    model_path = tmp_path / "model.mlir"
    metadata_path = tmp_path / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "inputs": [{"name": "pixels", "index": 0}],
                "outputs": [{"name": "scores", "index": 0}],
            }
        )
    )
    io_config = {"input_names": ["input_0"], "output_names": ["output_0"]}

    _apply_io_metadata(io_config, model_path)

    assert io_config["input_names"] == ["pixels"]
    assert io_config["output_names"] == ["scores"]


def test_onnx_io_ranges_reach_tensor_comparison(tmp_path: Path) -> None:
    from onnx import TensorProto, helper, save_model

    from winml.modelkit.eval.tensor_similarity_evaluator import TensorSimilarityEvaluator
    from winml.modelkit.onnx import get_io_config

    model_path = tmp_path / "model.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["indices"], ["output"])],
        "input_ranges",
        [helper.make_tensor_value_info("indices", TensorProto.INT64, ["batch", 16])],
        [helper.make_tensor_value_info("output", TensorProto.INT64, ["batch", 16])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    helper.set_model_props(
        model,
        {"winml.io.inputs": json.dumps([{"name": "indices", "value_range": [0, 1]}])},
    )
    save_model(model, model_path)
    source_io = get_io_config(model_path)
    io_config = {
        "input_names": source_io["input_names"],
        "input_shapes": [[2, 16]],
        "input_types": ["int64"],
        "output_names": source_io["output_names"],
    }

    _apply_io_metadata(io_config, model_path)

    assert io_config["value_ranges"] == source_io["value_ranges"]
    assert io_config["input_shapes"] == [[2, 16]]
    assert io_config["input_types"] == ["int64"]
    evaluator = object.__new__(TensorSimilarityEvaluator)
    evaluator.model = SimpleNamespace(io_config=io_config)
    evaluator.config = SimpleNamespace(input_data=None, dataset=SimpleNamespace(samples=2, seed=42))

    dataset = evaluator.prepare_data()

    lower, upper = source_io["value_ranges"]["indices"]
    for sample in dataset:
        indices = sample["indices"]
        assert tuple(indices.shape) == tuple(io_config["input_shapes"][0])
        assert bool(((indices >= lower) & (indices < upper)).all())


@pytest.mark.parametrize(
    "failure_point",
    [
        "_load_mlir",
        "_load_onnx_on_cgc",
        "_load_onnx_on_ort",
        "create_pipeline_builder",
        "build",
        "_stage_diagnostics",
    ],
)
def test_build_failure_releases_adapter(monkeypatch, mlir_target, failure_point):
    ep_device, adapter = mlir_target
    close = Mock(wraps=adapter.close)
    monkeypatch.setattr(adapter, "close", close)
    runtime = _Runtime(_Stage(), _Pipeline())
    wr = SimpleNamespace(Runtime=lambda: runtime, NotSupportedError=_NotSupportedError)
    monkeypatch.setattr("winml.modelkit.session.runtime_session.import_runtime", lambda: wr)
    session = WinMLRuntimeSession("model.mlir", ep_device=ep_device, backend="cgc")
    if failure_point.startswith("_load_onnx"):
        session._is_mlir = False
        if failure_point == "_load_onnx_on_ort":
            session._backend = "ort"
            monkeypatch.setattr(
                session,
                "_resolve_target",
                lambda *_args: SimpleNamespace(
                    adapter=adapter,
                    execution_target=adapter.pointer,
                    provider_name=None,
                    device_class="gpu",
                ),
            )
    failure = RuntimeError("adapter cleanup probe")
    failing_call = Mock(side_effect=failure)
    if failure_point.startswith("_load_"):
        monkeypatch.setattr(session, failure_point, failing_call)
    elif failure_point == "_stage_diagnostics":
        monkeypatch.setattr(
            "winml.modelkit.session.runtime_session._stage_diagnostics",
            failing_call,
        )
    elif failure_point == "build":
        monkeypatch.setattr(runtime.builder, failure_point, failing_call)
    else:
        monkeypatch.setattr(runtime, failure_point, failing_call)
    with pytest.raises(RuntimeError, match="adapter cleanup probe") as raised:
        session.compile()
    assert raised.value is failure
    close.assert_called_once_with()
    assert session._adapter_handle is None
    assert session._built is False
    session.close()
    close.assert_called_once_with()


def test_run_requests_dynamic_outputs_before_each_execution(monkeypatch, mlir_target):
    stage = _Stage()
    pipeline = _Pipeline()
    runtime = _Runtime(stage, pipeline)
    wr = SimpleNamespace(Runtime=lambda: runtime, NotSupportedError=_NotSupportedError)
    monkeypatch.setattr("winml.modelkit.session.runtime_session.import_runtime", lambda: wr)
    names = ["sum", "product"]
    requests = []

    def request_output(index):
        requests.append(index)
        stage.requested_outputs.add(index)

    def execute():
        assert stage.requested_outputs == set(range(len(names)))
        pipeline.runs += 1

    def output(index):
        assert index in stage.requested_outputs
        data = stage.bound[0]
        return _Tensor(np.add(data, data) if index == 0 else np.multiply(data, data))

    monkeypatch.setattr(stage, "request_output", request_output)
    monkeypatch.setattr(stage, "output", output)
    monkeypatch.setattr(pipeline, "run", execute)
    session = WinMLRuntimeSession("model.mlir", ep_device=mlir_target[0], backend="cgc")
    try:
        session.compile()
        session._io_config["output_names"] = names
        session._io_config["output_shapes"] = [[None, 3, 8, 8]] * len(names)
        session._io_config["output_types"] = ["float32"] * len(names)
        random = np.random.default_rng(42)
        with session.perf(warmup=0):
            for _iteration in range(3):
                data = random.normal(size=(2, 3, 8, 8)).astype(np.float32)
                actual = session.run({"input_0": data})
                np.testing.assert_array_equal(actual["sum"], np.add(data, data))
                np.testing.assert_array_equal(actual["product"], np.multiply(data, data))
        assert requests == list(range(len(names))) * pipeline.runs
        assert pipeline.runs == 3
    finally:
        session.close()


@pytest.fixture
def timed_runtime(monkeypatch, mlir_target):
    stage = _Stage()
    pipeline = _Pipeline()
    runtime = _Runtime(stage, pipeline)
    wr = SimpleNamespace(Runtime=lambda: runtime, NotSupportedError=_NotSupportedError)
    monkeypatch.setattr("winml.modelkit.session.runtime_session.import_runtime", lambda: wr)
    monkeypatch.setattr(
        stage, "output", lambda index: _Tensor(np.negative(stage.bound[index]))
    )
    session = WinMLRuntimeSession("model.mlir", ep_device=mlir_target[0], backend="cgc")
    clock = SimpleNamespace(value=0.0, reads=[])
    calls = []
    phases = {
        "build": (runtime.builder, "build"),
        "prepare": (session, "_prepare_inputs"),
        "bind": (stage, "bind_buffers"),
        "request": (stage, "request_output"),
        "execute": (pipeline, "run"),
        "fetch": (stage, "output"),
        "numpy": (_Tensor, "to_numpy"),
    }
    durations = dict(
        zip(phases, np.random.default_rng(0).uniform(0.001, 0.02, len(phases)), strict=True)
    )

    def wrap(function, phase):
        def invoke(*args, **kwargs):
            calls.append(phase)
            clock.value += durations[phase]
            return function(*args, **kwargs)

        return invoke

    for phase, (owner, method) in phases.items():
        monkeypatch.setattr(owner, method, wrap(getattr(owner, method), phase))

    def timer():
        clock.reads.append(clock.value)
        return clock.value

    monkeypatch.setattr(
        "winml.modelkit.session.stats.time", SimpleNamespace(perf_counter=timer)
    )
    try:
        yield session, clock, calls, durations, phases
    finally:
        session.close()


def test_perf_times_feed_infer_fetch(timed_runtime):
    session, clock, calls, durations, _phases = timed_runtime
    timed_phases = ("bind", "request", "execute", "fetch", "numpy")
    expected_ms = sum(durations[phase] for phase in timed_phases) * 1000
    with session.perf(warmup=1) as ctx:
        rng = np.random.default_rng(42)
        for _ in range(3):
            values = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
            actual = session.run({"input_0": values})
            np.testing.assert_array_equal(actual["output_0"], np.negative(values))
            assert ctx.stats.all_samples_ms[-1] == pytest.approx(expected_ms)

    assert len(clock.reads) == 2 * ctx.stats.total_count
    assert ctx.stats.total_count == 3
    assert ctx.stats.count == 2
    assert ctx.stats.mean_ms == pytest.approx(expected_ms)
    assert calls.count("build") == 1
    assert calls.count("bind") == calls.count("execute")
    assert calls.count("request") == calls.count("execute")
    assert session.perf_stats is None


@pytest.mark.parametrize(
    "failure", ["prepare", "bind", "request", "execute", "fetch", "numpy"]
)
def test_perf_records_only_successful_feed_infer_fetch_calls(timed_runtime, monkeypatch, failure):
    session, _clock, _calls, _durations, phases = timed_runtime
    values = np.random.default_rng(42).standard_normal((2, 3, 8, 8)).astype(np.float32)
    with session.perf() as ctx:
        owner, method = phases[failure]
        with monkeypatch.context() as patch:
            patch.setattr(owner, method, Mock(side_effect=RuntimeError(failure)))
            with pytest.raises(RuntimeError, match=failure):
                session.run({"input_0": values})
        assert ctx.stats.total_count == 0

        actual = session.run({"input_0": values})
        np.testing.assert_array_equal(actual["output_0"], np.negative(values))
        assert ctx.stats.total_count == 1


def test_run_without_perf_does_not_read_clock(timed_runtime):
    session, clock, _calls, _durations, _phases = timed_runtime
    values = np.random.default_rng(42).standard_normal((2, 3, 8, 8)).astype(np.float32)

    actual = session.run({"input_0": values})

    np.testing.assert_array_equal(actual["output_0"], np.negative(values))
    assert clock.reads == []


@pytest.fixture
def host_runtime(timed_runtime, monkeypatch):
    session, clock, calls, durations, _phases = timed_runtime
    session.compile()
    session._io_config = {
        "input_names": ["x", "bias"],
        "input_types": ["float32", "float32"],
        "output_names": ["dynamic", "static", "scalar"],
        "output_shapes": [[None, 3], [2, 3], []],
        "output_types": ["float32", "float32", "float32"],
    }
    buffers = {}
    stage = session._stage
    durations["allocate"] = durations["bind"]
    durations["batch_bind"] = durations["bind"]
    empty = np.empty

    def allocate(*args, **kwargs):
        calls.append("allocate")
        clock.value += durations["allocate"]
        return empty(*args, **kwargs)

    def bind(inputs, outputs):
        calls.append("batch_bind")
        clock.value += durations["batch_bind"]
        buffers["inputs"] = inputs
        buffers["outputs"] = list(outputs)
        buffers["output_refs"] = [
            weakref.ref(output) if output is not None else None for output in outputs
        ]

    def execute():
        calls.append("execute")
        clock.value += durations["execute"]
        x, bias = buffers["inputs"]
        values = [x - bias, x + bias, np.sum(x)]
        buffers["values"] = values
        for target, value in zip(buffers["outputs"], values, strict=True):
            if target is not None:
                np.copyto(target, value)

    def fetch(index):
        assert buffers["outputs"][index] is None
        calls.append("fetch")
        clock.value += durations["fetch"]
        return _Tensor(buffers["values"][index].copy())

    def request(index):
        if buffers.get("outputs") and buffers["outputs"][index] is not None:
            buffers["outputs"][index] = None
            calls.append("release")
        else:
            calls.append("request")
        clock.value += durations["request"]

    monkeypatch.setattr(np, "empty", allocate)
    monkeypatch.setattr(stage, "bind_buffers", bind)
    monkeypatch.setattr(stage, "output", fetch)
    monkeypatch.setattr(stage, "request_output", request)
    monkeypatch.setattr(session._pipeline, "run", execute)
    return session, clock, calls, durations, buffers


@pytest.mark.parametrize("static_only", [False, True])
def test_host_buffers_preserve_outputs_and_time_complete_run(host_runtime, static_only):
    session, clock, calls, durations, buffers = host_runtime
    if static_only:
        session._io_config["output_shapes"][0] = [2, 3]
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2, 3)).astype(np.float32)
    bias = rng.normal(size=()).astype(np.float32)
    bound_count = 3 if static_only else 2
    expected_ms = 1000 * (
        bound_count * durations["allocate"] + durations["batch_bind"] + durations["execute"]
        + (0 if static_only else sum(durations[phase] for phase in ("request", "fetch", "numpy")))
    )
    with session.perf(warmup=1) as ctx:
        first = session.run({"bias": bias, "x": x})
        saved = {name: value.copy() for name, value in first.items()}
        assert buffers["inputs"][0] is x
        assert buffers["inputs"][1] is bias
        assert first["static"] is buffers["output_refs"][1]()
        assert first["scalar"] is buffers["output_refs"][2]()
        assert first["scalar"].shape == ()
        assert all(output is None for output in buffers["outputs"])
        assert clock.value - clock.reads[-1] == pytest.approx(bound_count * durations["request"])
        x *= -2
        second = session.run({"x": x, "bias": bias})
        np.testing.assert_array_equal(second["dynamic"], x - bias)
        np.testing.assert_array_equal(second["static"], x + bias)
        np.testing.assert_array_equal(second["scalar"], np.sum(x))
        assert list(second) == session.io_config["output_names"]
        assert ctx.stats.count == 1
        assert ctx.stats.all_samples_ms == pytest.approx([expected_ms, expected_ms])
        assert len(clock.reads) == 4
    session.close()
    for name, original in saved.items():
        np.testing.assert_array_equal(first[name], original)
        assert not np.shares_memory(first[name], second[name])
    assert "bind" not in calls
    assert calls.count("batch_bind") == 2
    assert calls.count("release") == 2 * bound_count


@pytest.mark.parametrize("failure", ["bind_buffers", "request_output", "run", "output"])
def test_host_buffer_failures_are_not_retried_or_recorded(host_runtime, monkeypatch, failure):
    session, _clock, calls, _durations, buffers = host_runtime
    x = np.random.default_rng(42).normal(size=(2, 3)).astype(np.float32)
    inputs = {"x": x, "bias": x.sum()}
    owner = session._pipeline if failure == "run" else session._stage
    with session.perf() as ctx:
        with monkeypatch.context() as patch:
            failing = Mock(side_effect=RuntimeError(failure))
            patch.setattr(owner, failure, failing)
            with pytest.raises(RuntimeError, match=failure):
                session.run(inputs)
            failing.assert_called_once()
        assert ctx.stats.total_count == 0
        assert all(output is None for output in buffers.get("outputs", []))
        session.run(inputs)
        assert ctx.stats.total_count == 1
    assert "bind" not in calls
    assert all(output is None for output in buffers["outputs"])


def test_run_releases_static_output_references(host_runtime):
    session, clock, _calls, _durations, buffers = host_runtime
    values = np.random.default_rng(42).normal(size=(2, 3)).astype(np.float32)
    outputs = session.run({"x": values, "bias": values.sum()})
    references = [weakref.ref(value) for value in outputs.values()]
    del outputs

    assert all(reference() is None for reference in references)
    assert all(output is None for output in buffers["outputs"])
    assert clock.reads == []


def test_output_release_failure_is_reported(host_runtime, monkeypatch):
    session, _clock, _calls, _durations, _buffers = host_runtime
    request_output = session._stage.request_output
    values = np.random.default_rng(42).normal(size=(2, 3)).astype(np.float32)

    def fail_release(index):
        if index == 1:
            raise RuntimeError("output release failed")
        request_output(index)

    with monkeypatch.context() as patch:
        patch.setattr(session._stage, "request_output", fail_release)
        with pytest.raises(RuntimeError, match="output release failed"):
            session.run({"x": values, "bias": values.sum()})
    request_output(1)
    request_output(2)


def test_mlir_session_builds_runs_and_resets(
    monkeypatch: pytest.MonkeyPatch,
    mlir_target: tuple[SimpleNamespace, _AdapterHandle],
) -> None:
    stage = _Stage()
    pipeline = _Pipeline()
    runtime = _Runtime(stage, pipeline)
    ep_device, adapter = mlir_target
    close = Mock(wraps=adapter.close)
    monkeypatch.setattr(adapter, "close", close)
    wr = SimpleNamespace(
        Runtime=lambda: runtime,
        NotSupportedError=_NotSupportedError,
    )
    monkeypatch.setattr(
        "winml.modelkit.session.runtime_session.import_runtime",
        lambda: wr,
    )

    session = WinMLRuntimeSession("model.mlir", ep_device=ep_device, backend="cgc")
    session.compile()
    session.compile()
    close.assert_not_called()
    assert session._adapter_handle is adapter
    assert runtime.builder.targets == [123]
    assert runtime.builder.caller_owned_at_build is False
    assert session.device == "gpu"
    assert session.io_config["input_names"] == ["input_0"]

    outputs = session.run({"input_0": np.zeros((2, 3, 8, 8), dtype=np.float64)})
    assert outputs["output_0"].shape == (2, 5)
    assert stage.bound[0].dtype == np.float32
    assert pipeline.runs == 1

    session.reset()
    assert session._pipeline is None
    assert adapter.closed is True
    close.assert_called_once_with()
    session.close()
    close.assert_called_once_with()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("shape", "transpose"),
    [((), False), ((1,), False), ((2, 3), False), ((2, 3), True)],
    ids=["scalar", "vector", "matrix", "noncontiguous"],
)
def test_prepare_inputs_preserves_shape(
    dtype: type,
    shape: tuple[int, ...],
    transpose: bool,
) -> None:
    values = np.random.default_rng(0).standard_normal(shape).astype(dtype)
    if transpose:
        values = values.T
    session = WinMLRuntimeSession("model.onnx", device="cpu", ep="cpu", backend="ort")
    try:
        session._io_config = {"input_names": ["input"], "input_types": [np.float32]}

        prepared = session._prepare_inputs({"input": values})["input"]

        assert prepared.shape == values.shape
        assert prepared.dtype == np.float32
        assert prepared.flags.c_contiguous
        np.testing.assert_array_equal(prepared, values.astype(np.float32))
    finally:
        session.close()


def test_mlir_session_requires_target_device() -> None:
    with pytest.raises(ValueError, match="ep_device or device_luid is required"):
        WinMLRuntimeSession("model.mlir", backend="cgc")


@pytest.mark.parametrize("model_suffix", [".mlir", ".onnx"])
@pytest.mark.parametrize("operation", ["compile", "run", "io_config"])
def test_cgc_target_rejects_dxcore_incompatible_device(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    model_suffix: str,
    operation: str,
) -> None:
    projected = SimpleNamespace(
        ep_name="DmlExecutionProvider",
        device_type="GPU",
        hardware_name="Projected GPU",
        adapter_luid=141027,
        ort_handle=object(),
    )
    physical = SimpleNamespace(
        ep_name="DmlExecutionProvider",
        device_type="GPU",
        hardware_name="Physical GPU",
        adapter_luid=95733,
    )
    physical_handle = object()
    adapter = _AdapterHandle()

    monkeypatch.setattr(
        "onnxruntime.get_ep_devices",
        lambda: [physical_handle],
    )
    monkeypatch.setattr(
        "winml.modelkit.session.ep_device.WinMLDevice",
        lambda handle: physical if handle is physical_handle else None,
    )

    attempted_luids = []
    native_error = click.ClickException("not visible to DXCore")

    def from_luid(cls: type[_DXCoreAdapter], luid: int) -> _AdapterHandle:
        attempted_luids.append(luid)
        if luid == 141027:
            raise native_error
        return adapter

    monkeypatch.setattr(_DXCoreAdapter, "from_luid", classmethod(from_luid))
    monkeypatch.setattr(
        "winml.modelkit.session.runtime_session.import_runtime",
        lambda: SimpleNamespace(Runtime=SimpleNamespace),
    )

    session = WinMLRuntimeSession(
        f"model{model_suffix}", ep_device=SimpleNamespace(device=projected), backend="cgc"
    )
    with pytest.raises(Exception) as exc_info:
        if operation == "compile":
            session.compile()
        elif operation == "run":
            session.run({"input": np.zeros((), dtype=np.float32)})
        else:
            _ = session.io_config

    assert isinstance(exc_info.value, click.ClickException)
    assert exc_info.value.__cause__ is native_error
    assert "Cannot resolve the selected device LUID" in str(exc_info.value)
    assert "winml sys" in str(exc_info.value)
    assert "--device-luid <LUID>" in str(exc_info.value)
    assert attempted_luids == [projected.adapter_luid]
    assert adapter.closed is False
    assert "CGC is falling back" not in caplog.text


@pytest.mark.parametrize("missing_luid", [False, True])
@pytest.mark.parametrize("model_suffix", [".mlir", ".onnx"])
def test_cgc_target_uses_selected_adapter(
    mlir_target: tuple[SimpleNamespace, _AdapterHandle],
    missing_luid: bool,
    model_suffix: str,
) -> None:
    ep_device, adapter = mlir_target
    device = ep_device.device
    runtime = _Runtime(_Stage(), _Pipeline())
    session = WinMLRuntimeSession(f"model{model_suffix}", ep_device=ep_device, backend="cgc")
    if missing_luid:
        device.adapter_luid = None
        with pytest.raises(Exception) as exc_info:
            session._resolve_target(runtime, SimpleNamespace())
        assert isinstance(exc_info.value, click.ClickException)
        assert exc_info.value.__cause__ is None
        assert "Cannot resolve the selected device LUID" in str(exc_info.value)
        assert "winml sys" in str(exc_info.value)
        assert "--device-luid <LUID>" in str(exc_info.value)
        assert adapter.closed is False
    else:
        resolved = session._resolve_target(runtime, SimpleNamespace())
        try:
            assert resolved.execution_target == adapter.pointer
            assert resolved.adapter is adapter
            assert adapter.closed is False
        finally:
            adapter.close()


def test_runtime_session_rejects_provider_options() -> None:
    with pytest.raises(click.ClickException, match="--ep-options"):
        WinMLRuntimeSession("model.onnx", provider_options={"key": "value"}, backend="ort")


def test_onnx_session_passes_resolved_ep_and_device_to_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("winml.modelkit.onnx.get_io_config", lambda _path: {"value_ranges": {}})
    stage = _Stage()
    pipeline = _Pipeline()

    class OrtModel(_Model):
        def ort_schema(self) -> None:
            return None

    class OrtRuntime(_Runtime):
        def __init__(self) -> None:
            super().__init__(stage, pipeline)
            self.target_args: tuple[str, Any, Any] | None = None

        def load_model(self, path: str) -> _Model:
            assert path.endswith(".onnx")
            return OrtModel()

        def create_ort_execution_target(
            self,
            provider_name: str,
            kind: Any,
            hardware_target: Any,
        ) -> object:
            self.target_args = (provider_name, kind, hardware_target)
            return object()

    runtime = OrtRuntime()
    gpu_kind = object()
    adapter = _AdapterHandle()
    wr = SimpleNamespace(
        Runtime=lambda: runtime,
        NotSupportedError=_NotSupportedError,
        ExecutionTargetKind=SimpleNamespace(GPU=gpu_kind),
    )
    ep_device = SimpleNamespace(
        device=SimpleNamespace(
            ep_name="NvTensorRTRTXExecutionProvider",
            device_type="GPU",
            hardware_name="Test GPU",
            adapter_luid=456,
        ),
        ep_short_name="nvtensorrtrtx",
        source_tag="winml-catalog",
    )
    monkeypatch.setattr(
        "winml.modelkit.session.runtime_session.import_runtime",
        lambda: wr,
    )
    monkeypatch.setattr(
        "winml.modelkit.session.runtime_session.resolve_provider_kind",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("concrete ep_device must bypass request-based resolution")
        ),
    )
    monkeypatch.setattr(
        _DXCoreAdapter,
        "from_luid",
        classmethod(lambda cls, luid: adapter if luid == 456 else None),
    )

    session = WinMLRuntimeSession("model.onnx", ep_device=ep_device, backend="ort")
    session.compile()

    assert runtime.target_args == (
        "NvTensorRTRTXExecutionProvider",
        gpu_kind,
        123,
    )
    assert runtime.builder.caller_owned_at_build is False
    assert runtime.builder.targets[0] is not None
    assert session.device == "gpu"
    assert session.requested_provider == "NvTensorRTRTXExecutionProvider"
    session.reset()
    assert adapter.closed is True


def test_onnx_session_without_ep_device_uses_request_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("winml.modelkit.onnx.get_io_config", lambda _path: {"value_ranges": {}})
    stage = _Stage()
    pipeline = _Pipeline()

    class OrtModel(_Model):
        def ort_schema(self) -> None:
            return None

    class OrtRuntime(_Runtime):
        def __init__(self) -> None:
            super().__init__(stage, pipeline)
            self.target_args: tuple[str, Any] | None = None

        def load_model(self, path: str) -> _Model:
            assert path.endswith(".onnx")
            return OrtModel()

        def create_ort_execution_target(self, provider_name: str, kind: Any) -> object:
            self.target_args = (provider_name, kind)
            return object()

    runtime = OrtRuntime()
    gpu_kind = object()
    wr = SimpleNamespace(
        Runtime=lambda: runtime,
        NotSupportedError=_NotSupportedError,
        ExecutionTargetKind=SimpleNamespace(GPU=gpu_kind),
    )
    calls: list[tuple[Any, ...]] = []

    def resolve(*args: Any) -> tuple[str, str]:
        calls.append(args)
        return "DmlExecutionProvider", "gpu"

    monkeypatch.setattr(
        "winml.modelkit.session.runtime_session.import_runtime",
        lambda: wr,
    )
    monkeypatch.setattr(
        "winml.modelkit.session.runtime_session.resolve_provider_kind",
        resolve,
    )

    session = WinMLRuntimeSession(
        "model.onnx",
        device="gpu",
        ep="dml",
        backend="ort",
    )
    session.compile()

    assert calls == [("gpu", "dml", None)]
    assert runtime.target_args == ("DmlExecutionProvider", gpu_kind)


def test_concurrent_compile_builds_once(
    monkeypatch: pytest.MonkeyPatch,
    mlir_target: tuple[SimpleNamespace, _AdapterHandle],
) -> None:
    stage = _Stage()
    pipeline = _Pipeline()
    build_entered = threading.Event()
    release_build = threading.Event()

    class BlockingBuilder(_Builder):
        builds = 0

        def build(self) -> _Pipeline:
            self.builds += 1
            build_entered.set()
            assert release_build.wait(timeout=5)
            return self.pipeline

    runtime = _Runtime(stage, pipeline)
    runtime.builder = BlockingBuilder(stage, pipeline)
    wr = SimpleNamespace(Runtime=lambda: runtime, NotSupportedError=_NotSupportedError)
    monkeypatch.setattr(
        "winml.modelkit.session.runtime_session.import_runtime",
        lambda: wr,
    )
    ep_device, _ = mlir_target
    session = WinMLRuntimeSession("model.mlir", ep_device=ep_device, backend="cgc")
    first = threading.Thread(target=session.compile)
    second = threading.Thread(target=session.compile)

    first.start()
    assert build_entered.wait(timeout=5)
    second.start()
    time.sleep(0.05)
    assert runtime.builder.builds == 1
    release_build.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert runtime.builder.builds == 1


def test_concurrent_runs_serialize_bind_run_read(
    monkeypatch: pytest.MonkeyPatch,
    mlir_target: tuple[SimpleNamespace, _AdapterHandle],
) -> None:
    run_entered = threading.Event()
    release_run = threading.Event()

    class TrackingStage(_Stage):
        def __init__(self) -> None:
            super().__init__()
            self.bind_count = 0

        def bind_buffers(self, inputs, outputs) -> None:
            self.bind_count += 1
            super().bind_buffers(inputs, outputs)

        def output(self, index: int) -> _Tensor:
            value = float(self.bound[index].flat[0])
            return _Tensor(np.full((1, 5), value, dtype=np.float32))

    class BlockingPipeline(_Pipeline):
        def run(self) -> None:
            self.runs += 1
            if self.runs == 1:
                run_entered.set()
                assert release_run.wait(timeout=5)

    stage = TrackingStage()
    pipeline = BlockingPipeline()
    runtime = _Runtime(stage, pipeline)
    wr = SimpleNamespace(Runtime=lambda: runtime, NotSupportedError=_NotSupportedError)
    monkeypatch.setattr(
        "winml.modelkit.session.runtime_session.import_runtime",
        lambda: wr,
    )
    ep_device, _ = mlir_target
    session = WinMLRuntimeSession("model.mlir", ep_device=ep_device, backend="cgc")
    outputs: dict[str, float] = {}

    def run(name: str, value: float) -> None:
        result = session.run({"input_0": np.full((1, 3, 8, 8), value, dtype=np.float32)})
        outputs[name] = float(result["output_0"][0, 0])

    first = threading.Thread(target=run, args=("first", 1.0))
    second = threading.Thread(target=run, args=("second", 2.0))
    first.start()
    assert run_entered.wait(timeout=5)
    second.start()
    time.sleep(0.05)
    assert stage.bind_count == 1
    release_run.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert outputs == {"first": 1.0, "second": 2.0}


def test_close_waits_for_active_run(
    monkeypatch: pytest.MonkeyPatch,
    mlir_target: tuple[SimpleNamespace, _AdapterHandle],
) -> None:
    run_entered = threading.Event()
    release_run = threading.Event()

    class BlockingPipeline(_Pipeline):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def run(self) -> None:
            run_entered.set()
            assert release_run.wait(timeout=5)

        def close(self) -> None:
            self.closed = True

    stage = _Stage()
    pipeline = BlockingPipeline()
    runtime = _Runtime(stage, pipeline)
    wr = SimpleNamespace(Runtime=lambda: runtime, NotSupportedError=_NotSupportedError)
    monkeypatch.setattr(
        "winml.modelkit.session.runtime_session.import_runtime",
        lambda: wr,
    )
    ep_device, _ = mlir_target
    session = WinMLRuntimeSession("model.mlir", ep_device=ep_device, backend="cgc")
    inference = threading.Thread(
        target=session.run,
        args=({"input_0": np.zeros((1, 3, 8, 8), dtype=np.float32)},),
    )
    teardown = threading.Thread(target=session.close)

    inference.start()
    assert run_entered.wait(timeout=5)
    teardown.start()
    time.sleep(0.05)
    assert pipeline.closed is False
    release_run.set()
    inference.join(timeout=5)
    teardown.join(timeout=5)

    assert not inference.is_alive()
    assert not teardown.is_alive()
    assert pipeline.closed is True


def test_perf_rejects_monitor_without_loading_runtime() -> None:
    session = WinMLRuntimeSession("model.mlir", ep_device=_mlir_ep_device(), backend="cgc")
    with pytest.raises(click.ClickException, match="monitor"), session.perf(monitor=object()):
        pass
