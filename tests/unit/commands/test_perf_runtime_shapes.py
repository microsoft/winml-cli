# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Concrete compilation shape handoff, with no native execution."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import click
import numpy as np
import onnx
import pytest

from winml.modelkit.commands.perf import (
    BenchmarkConfig,
    PerfBenchmark,
    _ort_options_for_dimensions,
    _runtime_input_shapes,
)


def _model(tmp_path: Path) -> Path:
    inputs = [
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, ["batch", 3])
        for name in ("left", "right")
    ]
    output = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, ["batch", 3])
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Add", ["left", "right"], ["out"])], "shapes", inputs, [output]
        )
    )
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    return path


def test_npz_headers_override_shape_config_without_loading_tensors(tmp_path, monkeypatch):
    model = _model(tmp_path)
    before = model.read_bytes()
    data = tmp_path / "inputs.npz"
    np.savez(data, left=np.zeros((2, 3)), right=np.zeros((2, 3)))
    monkeypatch.setattr(np, "load", Mock(side_effect=AssertionError("No tensor allocation")))
    shapes, dimensions = _runtime_input_shapes(model, data, {"batch": 99}, 42)
    assert shapes == {"left": (2, 3), "right": (2, 3)}
    assert dimensions == {"batch": 2}
    assert model.read_bytes() == before


def test_shape_config_resolves_symbols_without_inputs(tmp_path):
    shapes, dimensions = _runtime_input_shapes(_model(tmp_path), None, {"batch": 4}, 1)
    assert shapes == {"left": (4, 3), "right": (4, 3)}
    assert dimensions == {"batch": 4}


@pytest.mark.parametrize("right_shape", [(3, 3), (2, 4), (2, 3, 1)])
def test_npz_rejects_symbol_conflict_static_mismatch_and_rank(tmp_path, right_shape):
    data = tmp_path / "inputs.npz"
    np.savez(data, left=np.zeros((2, 3)), right=np.zeros(right_shape))
    with pytest.raises(click.ClickException):
        _runtime_input_shapes(_model(tmp_path), data, None, 1)


def test_ort_options_are_fresh_and_receive_dimensions_before_use(monkeypatch):
    import onnxruntime as ort

    created = []

    def create():
        value = SimpleNamespace(add_free_dimension_override_by_name=Mock())
        created.append(value)
        return value

    monkeypatch.setattr(ort, "SessionOptions", create)
    first = _ort_options_for_dimensions({"batch": 2})
    second = _ort_options_for_dimensions({"batch": 2})
    assert first is not second
    for options in created:
        options.add_free_dimension_override_by_name.assert_called_once_with("batch", 2)


@pytest.mark.parametrize("runtime", ["winml-runtime", "winml-ort"])
def test_perf_hands_shapes_to_runtime_before_compilation(tmp_path, monkeypatch, runtime):
    from winml.modelkit.models import WinMLAutoModel

    model_path = _model(tmp_path)
    data = tmp_path / "inputs.npz"
    np.savez(data, left=np.zeros((2, 3)), right=np.zeros((2, 3)))
    config = BenchmarkConfig(
        model_id=str(model_path),
        runtime=runtime,
        backend="cgc" if runtime == "winml-runtime" else None,
        input_data=data,
        device="gpu",
    )
    benchmark = PerfBenchmark(config)
    benchmark._ep_device = SimpleNamespace(
        device=SimpleNamespace(ep_name="WinMLCGExecutionProvider")
    )
    monkeypatch.setattr(benchmark, "_resolve_device_ep", lambda: None)
    session = SimpleNamespace(set_input_shapes=Mock(), compile=Mock())
    wrapper = SimpleNamespace(_session=session)
    options = object()
    configure = Mock(return_value=options)
    monkeypatch.setattr("winml.modelkit.commands.perf._ort_options_for_dimensions", configure)

    def construct(**kwargs):
        if runtime == "winml-ort":
            assert kwargs["session_options"]() is options
            configure.assert_called_once_with({"batch": 2})
        else:
            assert "session_options" not in kwargs
        return wrapper

    monkeypatch.setattr(WinMLAutoModel, "from_onnx", construct)
    benchmark._load_model()
    if runtime == "winml-runtime":
        session.set_input_shapes.assert_called_once_with({"left": (2, 3), "right": (2, 3)})
    else:
        session.set_input_shapes.assert_not_called()
    session.compile.assert_not_called()
