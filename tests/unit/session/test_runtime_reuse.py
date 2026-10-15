"""Reusable CPU buffers must preserve changing inputs and output ownership."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from winml.modelkit.session.runtime_session import WinMLRuntimeSession


def test_reuse_updates_input_and_keeps_previous_outputs_owned():
    session = WinMLRuntimeSession("test.onnx", backend="ort")
    session._backend = "cgc"
    session._built = True
    session._io_config = {
        "input_names": ["x"],
        "input_shapes": [[2]],
        "input_types": ["float32"],
        "output_names": ["y"],
        "output_shapes": [[2]],
        "output_types": ["float32"],
    }
    tensors = []

    def create(array, *args, **kwargs):
        tensor = SimpleNamespace(array=array, close=Mock())
        tensors.append(tensor)
        return tensor

    def owned(dtype, shape):
        tensor = create(np.zeros(shape, dtype=np.float32))
        tensor.copy_from = lambda source: np.copyto(tensor.array, source.array)
        return tensor

    session._wr = SimpleNamespace(
        TensorDataType={"FLOAT32": 1}, TensorAccessMode=SimpleNamespace(READWRITE=2)
    )
    session._runtime = SimpleNamespace(tensor_from_buffer=create, create_tensor=owned, close=Mock())
    session._stage = SimpleNamespace(bind_input=Mock(), bind_output=Mock(), close=Mock())
    session._pipeline = SimpleNamespace(
        run=lambda: np.multiply(tensors[1].array, 2, out=tensors[2].array), close=Mock()
    )
    session.prepare_reusable_io({"x": np.array([1, 2], dtype=np.float32)})
    first = session.run({"x": np.array([1, 2], dtype=np.float32)})["y"]
    second = session.run({"x": np.array([3, 4], dtype=np.float32)})["y"]
    np.testing.assert_array_equal(first, [2, 4])
    np.testing.assert_array_equal(second, [6, 8])
    assert len(tensors) == 3
    session._stage.bind_input.assert_called_once()
    session._stage.bind_output.assert_called_once()
    with pytest.raises(ValueError, match="shape/dtype changed"):
        session.run({"x": np.zeros(3, dtype=np.float32)})
    session.reset()
    assert session._reusable_io is None
    for tensor in tensors:
        tensor.close.assert_called_once()


@pytest.mark.skipif(
    not os.environ.get("WINML_REUSE_GPU_MODEL"), reason="Requires Runtime GPU artifacts"
)
def test_reused_runtime_gpu_observes_changed_inputs():
    import onnxruntime as ort

    from winml.modelkit.commands.perf import BenchmarkConfig, PerfBenchmark

    benchmark = PerfBenchmark(
        BenchmarkConfig(
            model_id=os.environ["WINML_REUSE_GPU_MODEL"],
            runtime="winml-runtime",
            backend="cgc",
            device="gpu",
            device_luid=os.environ["WINML_REUSE_GPU_LUID"],
        )
    )
    benchmark._load_model()
    session = benchmark._single._session
    reference = ort.InferenceSession(
        os.environ["WINML_REUSE_GPU_ONNX"], providers=["CPUExecutionProvider"]
    )
    with np.load(os.environ["WINML_REUSE_GPU_INPUT"]) as archive:
        inputs = {name: archive[name] for name in archive.files}
    try:
        session.prepare_reusable_io(inputs)
        previous = None
        for scale in (1.0, 0.5, 0.0):
            changed = {name: value * scale for name, value in inputs.items()}
            actual = session.run(changed)
            expected = reference.run(None, changed)
            for name, value in zip(session.io_config["output_names"], expected, strict=True):
                np.testing.assert_allclose(actual[name], value, rtol=1e-3, atol=1e-4)
            if previous is not None:
                assert any(not np.array_equal(previous[name], actual[name]) for name in actual)
            previous = actual
    finally:
        session.close()
