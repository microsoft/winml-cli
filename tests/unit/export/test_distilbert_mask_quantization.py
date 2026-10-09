# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""DistilBERT mask constants must remain safe for static activation quantization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from winml.modelkit.config import generate_build_config
from winml.modelkit.optim import optimize_onnx
from winml.modelkit.quant import StaticPass, WinMLQuantizationConfig


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _available_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "winml.modelkit.sysinfo.hardware.get_available_devices", lambda: ["npu", "cpu"]
    )


class _Reader:
    def __init__(self, inputs: dict[str, np.ndarray]) -> None:
        self.inputs = inputs
        self.rewind()

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self.samples, None)

    def rewind(self) -> None:
        self.samples = iter([self.inputs])


@pytest.mark.parametrize(
    "override,expected",
    [(None, True), ({"optim": {"clamp_constant_values": False}}, False)],
)
def test_distilbert_mask_policy_respects_user_override(override, expected: bool) -> None:
    config = generate_build_config(
        model_type="distilbert",
        task="fill-mask",
        device="npu",
        ep="openvino",
        precision="w8a16",
        override=override,
    )
    assert config.optim.get("clamp_constant_values") is expected
    assert config.quant.activation_type == "uint16"
    assert config.quant.weight_type == "uint8"


@pytest.mark.parametrize("additive", [False, True], ids=["where", "additive"])
def test_distilbert_quantization_preserves_masked_probabilities(
    tmp_path: Path, additive: bool
) -> None:
    # TF4 masks scores with Where; TF5 adds a shared, precomputed mask.
    sentinel = numpy_helper.from_array(np.array(np.finfo(np.float32).min), "sentinel")
    nodes = [helper.make_node("Constant", [], ["blocked"], value=sentinel)]
    if additive:
        nodes.extend(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["zero"],
                    value=numpy_helper.from_array(np.array(0, dtype=np.float32)),
                ),
                helper.make_node("Where", ["keep", "zero", "blocked"], ["mask"]),
                helper.make_node("Add", ["scores", "mask"], ["masked"]),
            ]
        )
    else:
        nodes.append(helper.make_node("Where", ["keep", "scores", "blocked"], ["masked"]))
    nodes.append(helper.make_node("Softmax", ["masked"], ["probabilities"], axis=-1))
    graph = helper.make_graph(
        nodes,
        "masked_attention",
        [
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("keep", TensorProto.BOOL, [2, 4]),
        ],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [2, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    inputs = {
        "scores": np.random.default_rng(42).normal(size=(2, 4)).astype(np.float32),
        "keep": np.array([[True, True, False, False], [True, False, True, False]]),
    }
    original = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    expected = original.run(None, inputs)[0]
    config = generate_build_config(
        model_type="distilbert", task="fill-mask", device="npu", ep="openvino", precision="w8a16"
    )
    optimized_path = tmp_path / "optimized.onnx"
    optimized = optimize_onnx(model, optimized_path, **config.optim.to_dict())
    optimized_session = ort.InferenceSession(
        optimized.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    np.testing.assert_allclose(optimized_session.run(None, inputs)[0], expected, atol=1e-7)
    quantized_path = tmp_path / "quantized.onnx"
    quant = WinMLQuantizationConfig(
        activation_type="uint16", weight_type="uint8", calibration_data=_Reader(inputs)
    )
    result = StaticPass(quant).run(optimized_path, quantized_path, use_external_data=False)
    assert result.success, result.errors
    quantized = onnx.load(quantized_path)
    initializers = {i.name: numpy_helper.to_array(i) for i in quantized.graph.initializer}
    scales = [
        initializers[n.input[1]]
        for n in quantized.graph.node
        if n.op_type in {"QuantizeLinear", "DequantizeLinear"}
    ]
    assert scales, "Static activation quantization must still run"
    # ORT uses scale=1 for an all-zero constant; the mask range stays below 1.
    assert all(
        np.isfinite(scale).all() and (scale > 0).all() and (scale <= 1).all() for scale in scales
    )
    session = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, inputs)[0]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=0.01, rtol=0)
    np.testing.assert_allclose(actual[~inputs["keep"]], 0, atol=1e-6)
