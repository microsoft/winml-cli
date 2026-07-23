# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for Conv/Add/BatchNormalization folding patterns."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from winml.modelkit.optim.pipes.rewrite import RewritePipe
from winml.modelkit.pattern import (
    AddConvBatchNormalizationPattern,
    ConvAddBatchNormalizationPattern,
    FoldedConvAddPattern,
)


def _value_info(name: str, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _build_model(
    *,
    static_first: bool = False,
    with_bias: bool = False,
    training_mode: int = 0,
    dynamic_scale: bool = False,
    public_conv_output: bool = False,
    variance: np.ndarray | None = None,
) -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    rng = np.random.default_rng(29)
    channels = 3
    initializers = {
        "weight": rng.normal(size=(channels, 2, 1, 1)).astype(np.float32),
        "static": rng.normal(size=(1, channels, 1, 1)).astype(np.float32),
        "scale": rng.uniform(0.5, 1.5, size=channels).astype(np.float32),
        "beta": rng.normal(size=channels).astype(np.float32),
        "mean": rng.normal(size=channels).astype(np.float32),
        "variance": (
            rng.uniform(0.5, 1.5, size=channels).astype(np.float32)
            if variance is None
            else variance
        ),
    }
    conv_inputs = ["x", "weight"]
    if with_bias:
        initializers["conv_bias"] = rng.normal(size=channels).astype(np.float32)
        conv_inputs.append("conv_bias")

    add_inputs = ["conv_out", "static"]
    if static_first:
        add_inputs.reverse()
    nodes = [
        helper.make_node(
            "Conv",
            conv_inputs,
            ["conv_out"],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        ),
        helper.make_node("Add", add_inputs, ["add_out"]),
        helper.make_node(
            "BatchNormalization",
            ["add_out", "scale", "beta", "mean", "variance"],
            ["y"],
            epsilon=0.01,
            training_mode=training_mode,
        ),
    ]
    graph_inputs = [_value_info("x", [1, 2, 2, 2])]
    if dynamic_scale:
        initializers.pop("scale")
        graph_inputs.append(_value_info("scale", [channels]))

    outputs = [_value_info("y", [1, channels, 2, 2])]
    if public_conv_output:
        outputs.append(_value_info("conv_out", [1, channels, 2, 2]))
    graph = helper.make_graph(
        nodes,
        "conv_add_batch_norm",
        graph_inputs,
        outputs,
        initializer=[numpy_helper.from_array(value, name) for name, value in initializers.items()],
        value_info=[
            _value_info("conv_out", [1, channels, 2, 2]),
            _value_info("add_out", [1, channels, 2, 2]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    feeds = {"x": rng.normal(size=(1, 2, 2, 2)).astype(np.float32)}
    if dynamic_scale:
        feeds["scale"] = rng.uniform(0.5, 1.5, size=channels).astype(np.float32)
    return model, feeds


def _fold(model: onnx.ModelProto) -> onnx.ModelProto:
    config = RewritePipe.build_config(conv_add_batch_normalization_folding=True)
    return RewritePipe().process(model, config)


def _run(
    model: onnx.ModelProto,
    feeds: dict[str, np.ndarray],
) -> list[np.ndarray]:
    return ReferenceEvaluator(model).run(None, feeds)


def test_rewrite_capability_is_registered() -> None:
    config = RewritePipe.build_config(conv_add_batch_normalization_folding=True)
    assert len(config.rules) == 2
    assert {rule.source.pattern_id for rule in config.rules} == {
        "SUBGRAPH/Conv-Add-Batch-NormalizationPattern"
    }
    assert {rule.target.pattern_id for rule in config.rules} == {"SUBGRAPH/FoldedConvAddPattern"}
    assert isinstance(config.rules[0].source, ConvAddBatchNormalizationPattern)
    assert isinstance(config.rules[1].source, AddConvBatchNormalizationPattern)
    assert all(isinstance(rule.target, FoldedConvAddPattern) for rule in config.rules)


def test_both_add_orders_and_optional_bias_are_equivalent() -> None:
    for static_first in (False, True):
        for with_bias in (False, True):
            model, feeds = _build_model(
                static_first=static_first,
                with_bias=with_bias,
            )
            expected = _run(model, feeds)
            transformed = _fold(model)

            onnx.checker.check_model(transformed)
            assert [node.op_type for node in transformed.graph.node] == ["Conv", "Add"]
            assert transformed.graph.node[-1].output == ["y"]
            assert not any(node.op_type == "BatchNormalization" for node in transformed.graph.node)
            actual = _run(transformed, feeds)
            np.testing.assert_allclose(actual[0], expected[0], rtol=3e-5, atol=3e-5)


def test_training_dynamic_parameters_and_public_intermediate_are_rejected() -> None:
    cases = (
        {"training_mode": 1},
        {"dynamic_scale": True},
        {"public_conv_output": True},
    )
    for kwargs in cases:
        model, _ = _build_model(**kwargs)
        transformed = _fold(model)
        assert any(node.op_type == "BatchNormalization" for node in transformed.graph.node)


def test_invalid_variance_is_rejected() -> None:
    model, _ = _build_model(variance=np.asarray([-0.02, 0.5, 1.0], dtype=np.float32))
    transformed = _fold(model)
    assert any(node.op_type == "BatchNormalization" for node in transformed.graph.node)
