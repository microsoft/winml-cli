# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np
import pytest
from onnx import (
    ModelProto,
    NodeProto,
    TensorProto,
    checker,
    helper,
    numpy_helper,
    version_converter,
)
from onnx.reference import ReferenceEvaluator

from winml.modelkit.optim import OptimizationError, Optimizer
from winml.modelkit.optim.pipes import (
    PIPES,
    CGIRRewritePipe,
    CGIRRewritePipeConfig,
    RewritePipe,
)
from winml.modelkit.pattern import PatternMatcher


if TYPE_CHECKING:
    from collections.abc import Callable


def test_initializer_shape_without_value_info():
    from winml.modelkit.pattern.cgc.cgc_constant_folding import _ConstantParameters

    weights = np.random.default_rng(42).normal(size=(2, 3)).astype(np.float32)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Shape", ["weights"], ["shape"])],
            "initializer_shape",
            [],
            [helper.make_tensor_value_info("shape", TensorProto.INT64, [weights.ndim])],
            [numpy_helper.from_array(weights, "weights")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )
    checker.check_model(model)
    expected = ReferenceEvaluator(model).run(None, {})[0]
    actual = _ConstantParameters(model, static_shapes=True).evaluate("shape", set(), set())
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("domain", ["", "com.microsoft"])
@pytest.mark.parametrize("scale_shape", [(), (1,)])
@pytest.mark.parametrize("zero_mode", ["scalar", "vector", "omitted", "empty"])
def test_normalize_int32_dq(domain, scale_shape, zero_mode):
    import onnxruntime as ort

    from winml.modelkit.optim import WinMLOptimizationConfig
    from winml.modelkit.pattern.cgc import normalize_int32_dq

    random = np.random.default_rng(42)
    data = random.integers(-1000, 1000, size=(8,), dtype=np.int32)
    scale = random.uniform(0.01, 0.1, size=scale_shape).astype(np.float32)
    initializers = [numpy_helper.from_array(data, "data"), numpy_helper.from_array(scale, "scale")]
    node_inputs = ["data", "scale"]
    if zero_mode in ("scalar", "vector"):
        zero = np.zeros(() if zero_mode == "scalar" else (1,), dtype=np.int32)
        initializers.append(numpy_helper.from_array(zero, "zero"))
        node_inputs.append("zero")
    elif zero_mode == "empty":
        node_inputs.append("")
    model = helper.make_model(helper.make_graph(
        [helper.make_node("DequantizeLinear", node_inputs, ["result"], domain=domain)],
        "dq", [], [helper.make_tensor_value_info("result", TensorProto.FLOAT, data.shape),
                   helper.make_tensor_value_info("scale", TensorProto.FLOAT, scale.shape)],
        initializers,
    ), opset_imports=[helper.make_opsetid("", 18)] + (
        [helper.make_opsetid(domain, 1)] if domain else []
    ), ir_version=10)
    original = model.SerializeToString()
    result = CGIRRewritePipe().process(model, CGIRRewritePipe.build_config(normalize_int32_dq=True))
    checker.check_model(result)
    assert model.SerializeToString() == original
    assert result.graph.node[0].domain == domain
    assert len(result.graph.node[0].input) == 2
    assert normalize_int32_dq(result) is result
    assert WinMLOptimizationConfig.for_cgc()["normalize_int32_dq"]
    options = ort.SessionOptions()
    options.log_severity_level = 3
    reference = ort.InferenceSession(original, options, providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(
        result.SerializeToString(), options, providers=["CPUExecutionProvider"],
    )
    for actual, expected in zip(candidate.run(None, {}), reference.run(None, {}), strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "guard", ["scale_input", "zero_input", "nonzero", "dtype", "domain", "per_axis"],
)
def test_normalize_int32_dq_preserves_unsupported(guard):
    from winml.modelkit.pattern.cgc import normalize_int32_dq

    random = np.random.default_rng(42)
    data = random.integers(1, 10, size=8, dtype=np.int32 if guard != "dtype" else np.uint8)
    scale = random.uniform(0.01, 0.1, size=8 if guard == "per_axis" else 1).astype(np.float32)
    zero = np.zeros((), dtype=data.dtype)
    if guard == "nonzero":
        zero = random.integers(1, 10, size=(), dtype=data.dtype)
    values = {"data": data, "scale": scale, "zero": zero}
    inputs = [helper.make_tensor_value_info(name, numpy_helper.from_array(values[name]).data_type,
                                          values[name].shape)
              for name in values if guard == name + "_input"]
    model = helper.make_model(helper.make_graph(
        [helper.make_node("DequantizeLinear", list(values), ["result"],
                          domain="custom" if guard == "domain" else "")],
        "guard", inputs, [helper.make_tensor_value_info("result", TensorProto.FLOAT, data.shape)],
        [numpy_helper.from_array(value, name) for name, value in values.items()],
    ), opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    assert normalize_int32_dq(model) is model


@pytest.mark.parametrize("opset", [11, 17, 18])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("has_shape", [False, True])
def test_cgc_constant_folding_pad_parameters(opset, shared, has_shape):
    from winml.modelkit.pattern.cgc import cgc_constant_folding

    seed = np.random.default_rng(42)
    widths = seed.integers(0, 3, size=4, dtype=np.int32)
    nodes = [
        helper.make_node("Cast", ["widths"], ["pads"], to=TensorProto.INT64),
        helper.make_node("Pad", ["source", "pads"], ["result"]),
    ]
    outputs = [helper.make_tensor_value_info("result", TensorProto.FLOAT, [None, None])]
    if has_shape:
        nodes.append(helper.make_node("Shape", ["result"], ["shape"]))
        outputs.append(helper.make_tensor_value_info("shape", TensorProto.INT64, [2]))
    if shared:
        outputs.append(helper.make_tensor_value_info("pads", TensorProto.INT64, [4]))
    model = helper.make_model(
        helper.make_graph(
            nodes, "constant_pad",
            [helper.make_tensor_value_info("source", TensorProto.FLOAT, [2, 3])],
            outputs, [numpy_helper.from_array(widths, "widths")],
        ),
        opset_imports=[helper.make_opsetid("", opset)], ir_version=10,
    )
    original = model.SerializeToString()
    feeds = {"source": seed.normal(size=(2, 3)).astype(np.float32)}
    expected = ReferenceEvaluator(model).run(None, feeds)
    result = CGIRRewritePipe().process(
        model, CGIRRewritePipe.build_config(cgc_constant_folding=True),
    )
    checker.check_model(result)
    assert model.SerializeToString() == original
    assert list(result.graph.input) == list(model.graph.input)
    assert [(value.name, value.type.tensor_type.elem_type) for value in result.graph.output] == [
        (value.name, value.type.tensor_type.elem_type) for value in model.graph.output
    ]
    if has_shape:
        for output, reference in zip(result.graph.output, expected, strict=True):
            assert tuple(dim.dim_value for dim in output.type.tensor_type.shape.dim) == (
                reference.shape
            )
    assert (result is model) == (not has_shape)
    assert sum(node.op_type == "Cast" for node in result.graph.node) == int(not has_shape)
    assert cgc_constant_folding(result) is result
    for actual, reference in zip(
        ReferenceEvaluator(result).run(None, feeds), expected, strict=True,
    ):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("opset", [16, 20])
@pytest.mark.parametrize("align", [0, 1])
@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("spatial,output_hw", [
    ((4, 7), (5, 6)), ((1, 5), (3, 1)), ((5, 1), (1, 4)), ((1, 1), (2, 3)),
])
def test_gridsample_to_gather(opset, align, fp16, dynamic, spatial, output_hw):
    import onnxruntime as ort

    dtype = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    batch_dim = "batch" if dynamic else 2
    model = helper.make_model(helper.make_graph([
        helper.make_node("GridSample", ["X", "grid"], ["Y"],
                         mode="linear" if opset >= 20 else "bilinear",
                         padding_mode="zeros", align_corners=align),
    ], "sampling", [
        helper.make_tensor_value_info("X", dtype, [batch_dim, 3, *spatial]),
        helper.make_tensor_value_info("grid", dtype, [batch_dim, *output_hw, 2]),
    ], [helper.make_tensor_value_info("Y", dtype, [batch_dim, 3, *output_hw])]),
        opset_imports=[helper.make_opsetid("", opset)], ir_version=10)
    original = model.SerializeToString()
    result = Optimizer().optimize(model, gridsample_to_gather=True)
    checker.check_model(result, full_check=True)
    assert model.SerializeToString() == original
    assert not any(node.op_type == "GridSample" for node in result.graph.node)
    assert sum(node.op_type == "GatherND" for node in result.graph.node) == 4
    ranks = {value.name: len(value.type.tensor_type.shape.dim)
             for value in result.graph.value_info}
    if dynamic:
        assert sum(node.op_type == "Range" for node in result.graph.node) == 1
    for node in result.graph.node:
        if node.op_type == "GatherND":
            assert {attr.name: helper.get_attribute_value(attr)
                    for attr in node.attribute}["batch_dims"] == 0
            assert all(ranks[name] == 3 for name in [*node.input, *node.output])
    rng = np.random.default_rng(42)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    reference = ort.InferenceSession(original, options, providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(
        result.SerializeToString(), options, providers=["CPUExecutionProvider"],
    )
    for batch in ([1, 2, 3] if dynamic else [2]):
        feeds = {"X": rng.normal(size=(batch, 3, *spatial)).astype(
                     np.float16 if fp16 else np.float32),
                 "grid": rng.uniform(-3, 3, size=(batch, *output_hw, 2)).astype(
                     np.float16 if fp16 else np.float32)}
        feeds["grid"].reshape(-1, 2)[:3] = np.linspace(-1, 1, 3)[:, None]
        expected = reference.run(None, feeds)
        actual = candidate.run(None, feeds)
        np.testing.assert_allclose(actual[0], expected[0], atol=2e-3 if fp16 else 2e-6, rtol=2e-3)


@pytest.mark.parametrize("mode,padding", [("nearest", "zeros"), ("linear", "border"),
                                         ("cubic", "reflection")])
def test_gridsample_to_gather_preserves_unsupported_modes(mode, padding):
    model = helper.make_model(helper.make_graph([
        helper.make_node("GridSample", ["X", "grid"], ["Y"], mode=mode, padding_mode=padding),
    ], "sampling", [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 7]),
        helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 5, 6, 2]),
    ], [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 5, 6])]),
        opset_imports=[helper.make_opsetid("", 20)], ir_version=10)
    result = CGIRRewritePipe().process(
        model, CGIRRewritePipe.build_config(gridsample_to_gather=True),
    )
    assert result is model


def test_cgc_constant_folding_is_enabled_only_by_cgc_defaults():
    from winml.modelkit.optim import WinMLOptimizationConfig

    assert not CGIRRewritePipe.build_config().rules
    assert WinMLOptimizationConfig.for_cgc()["cgc_constant_folding"] is True
    assert len(CGIRRewritePipe.build_config(cgc_constant_folding=True).rules) == 1


@pytest.mark.parametrize("dynamic", [False, True])
def test_cgc_constant_folding_static_shape_chain(dynamic):
    from winml.modelkit.pattern.cgc import cgc_constant_folding

    shape = ["batch" if dynamic else 2, 3]
    model = helper.make_model(helper.make_graph([
        helper.make_node("Shape", ["source"], ["shape"]),
        helper.make_node("Gather", ["shape", "axis"], ["batch"], axis=0),
        helper.make_node("Unsqueeze", ["batch", "axes"], ["batch_vector"]),
        helper.make_node("Concat", ["batch_vector", "tail"], ["target"], axis=0),
        helper.make_node("Reshape", ["source", "target"], ["reshaped"]),
        helper.make_node("Cast", ["reshaped"], ["result"], to=TensorProto.FLOAT),
    ], "shape_chain", [helper.make_tensor_value_info("source", TensorProto.FLOAT16, shape)],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, shape)], [
            numpy_helper.from_array(np.asarray(0, np.int64), "axis"),
            numpy_helper.from_array(np.asarray([0], np.int64), "axes"),
            numpy_helper.from_array(np.asarray([-1], np.int64), "tail"),
        ]), opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    original = model.SerializeToString()
    result = CGIRRewritePipe().process(
        model, CGIRRewritePipe.build_config(cgc_constant_folding=True),
    )
    assert original == model.SerializeToString()
    assert any(node.op_type == "Shape" for node in result.graph.node) == dynamic
    assert sum(node.op_type == "Cast" for node in result.graph.node) == 1
    assert (result is model) == dynamic
    assert cgc_constant_folding(result) is result
    for batch in ([1, 4] if dynamic else [2]):
        feeds = {"source": np.random.default_rng(42).normal(size=(batch, 3)).astype(np.float16)}
        np.testing.assert_array_equal(
            ReferenceEvaluator(result).run(None, feeds)[0],
            ReferenceEvaluator(model).run(None, feeds)[0],
        )


@pytest.mark.parametrize("start,end", [(1, 3), (-2, 100), (2, 1)])
def test_cgc_constant_folding_partial_shape(start, end):
    from winml.modelkit.pattern.cgc import cgc_constant_folding

    shape = ["batch", 3, 4]
    output_rank = len(shape[start:end])
    model = helper.make_model(helper.make_graph([
        helper.make_node("Shape", ["source"], ["result"], start=start, end=end),
    ], "partial_shape", [helper.make_tensor_value_info("source", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("result", TensorProto.INT64, [output_rank])]),
        opset_imports=[helper.make_opsetid("", 15)], ir_version=10)
    result = cgc_constant_folding(model)
    assert result.graph.node[0].op_type == "Constant"
    for batch in [1, 2]:
        feeds = {"source": np.random.default_rng(42).normal(size=(batch, 3, 4)).astype(np.float32)}
        np.testing.assert_array_equal(
            ReferenceEvaluator(result).run(None, feeds)[0],
            ReferenceEvaluator(model).run(None, feeds)[0],
        )


def test_cgc_constant_folding_rejects_broadcast_allocation(monkeypatch):
    from winml.modelkit.pattern.cgc.cgc_constant_folding import _ConstantParameters

    folding_module = import_module("winml.modelkit.pattern.cgc.cgc_constant_folding")
    monkeypatch.setattr(folding_module, "_MAX_ELEMENTS", 32)
    model = helper.make_model(helper.make_graph([
        helper.make_node("Add", ["rows", "columns"], ["result"]),
    ], "broadcast", [], [helper.make_tensor_value_info("result", TensorProto.INT64, [8, 8])], [
        numpy_helper.from_array(np.arange(8, dtype=np.int64).reshape(8, 1), "rows"),
        numpy_helper.from_array(np.arange(8, dtype=np.int64).reshape(1, 8), "columns"),
    ]), opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    with pytest.raises(ValueError, match="Broadcast allocation budget"):
        _ConstantParameters(model, static_shapes=True).evaluate("result", set(), set())


@pytest.mark.parametrize("source_op", ["Relu", "PRelu"])
@pytest.mark.parametrize("model_barrier", [False, True])
def test_cgir_reuses_matcher_without_skipping_rules(source_op, model_barrier, monkeypatch):
    import winml.modelkit.optim.pipes.cgir_rewrite as cgir_module

    prepared = []
    matched = []

    class TrackingMatcher(PatternMatcher):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            prepared.append(self)

        def match(self):
            matched.append(tuple(self.patterns))
            return super().match()

    monkeypatch.setattr(cgir_module, "PatternMatcher", TrackingMatcher)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(
                source_op, ["source", "slope"] if source_op == "PRelu" else ["source"],
                ["result"],
            )],
            "matcher_reuse",
            [helper.make_tensor_value_info("source", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("result", TensorProto.FLOAT, [2, 3])],
            [numpy_helper.from_array(
                np.random.default_rng(42).uniform(size=()).astype(np.float32), "slope",
            )],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    resize_rule = CGIRRewritePipe.build_config(omit_empty_resize_inputs=True).rules[0]
    prelu_rule = CGIRRewritePipe.build_config(prelu_to_relu=True).rules[0]
    rules = [resize_rule]
    if model_barrier:
        rules.extend(CGIRRewritePipe.build_config(deduplicate_opset_imports=True).rules)
    rules.extend([prelu_rule, resize_rule])
    original = model.SerializeToString()
    feeds = {"source": np.exp(np.random.default_rng(42).normal(size=(2, 3))).astype(np.float32)}
    expected = ReferenceEvaluator(model).run(None, feeds)

    result = CGIRRewritePipe().process(model, CGIRRewritePipeConfig(rules=rules))

    assert len(prepared) == 1 + int(model_barrier) + int(source_op == "PRelu")
    expected_patterns = [resize_rule.source.__name__, prelu_rule.source.__name__]
    if source_op == "PRelu":
        expected_patterns.append(prelu_rule.source.__name__)
    expected_patterns.append(resize_rule.source.__name__)
    assert matched == [(name,) for name in expected_patterns]
    assert model.SerializeToString() == original
    checker.check_model(result)
    for actual, reference in zip(
        ReferenceEvaluator(result).run(None, feeds), expected, strict=True,
    ):
        np.testing.assert_allclose(actual, reference)


@pytest.mark.parametrize("capability", [
    "log-to-reduce-log-sum",
    "materialize-initializer-parameters",
    "fold-scalar-initializer-casts",
    "eliminate-identity",
    "fold-constant-pad-pads",
])
def test_retired_cgir_rules_are_not_registered(capability):
    from click.testing import CliRunner

    from winml.modelkit.commands.optimize import optimize
    from winml.modelkit.optim import WinMLOptimizationConfig, get_all_capabilities

    assert capability not in get_all_capabilities()
    assert capability not in CGIRRewritePipe.capabilities
    assert capability.replace("-", "_") not in WinMLOptimizationConfig.for_cgc()
    result = CliRunner().invoke(optimize, ["--help"])
    assert result.exit_code == 0
    assert f"--enable-{capability}" not in result.output
    assert f"--disable-{capability}" not in result.output


def _make_resize_model(
    *,
    opset: int,
    roi: np.ndarray,
    scales: np.ndarray,
    use_sizes: bool,
    cast_inputs: bool = True,
    resize_count: int = 1,
) -> ModelProto:
    nodes: list[NodeProto] = []
    initializers = [
        numpy_helper.from_array(roi, "roi_source"),
        numpy_helper.from_array(scales, "scales_source"),
    ]
    roi_name = "roi_source"
    scales_name = "scales_source"
    if cast_inputs:
        nodes.extend(
            [
                helper.make_node("Cast", [roi_name], ["roi"], to=TensorProto.FLOAT),
                helper.make_node(
                    "Cast",
                    [scales_name],
                    ["scales"],
                    to=TensorProto.FLOAT,
                ),
            ]
        )
        roi_name = "roi"
        scales_name = "scales"

    outputs = []
    for index in range(resize_count):
        node_inputs = ["X", roi_name, scales_name]
        if use_sizes:
            sizes_name = f"sizes_{index}"
            initializers.append(
                numpy_helper.from_array(
                    np.array([1, 1, 4, 4], dtype=np.int64),
                    sizes_name,
                )
            )
            node_inputs.append(sizes_name)
        output_name = f"Y_{index}"
        nodes.append(
            helper.make_node(
                "Resize",
                node_inputs,
                [output_name],
                name=f"resize_{index}",
                mode="nearest",
            )
        )
        outputs.append(
            helper.make_tensor_value_info(
                output_name,
                TensorProto.FLOAT,
                [1, 1, 4, 4],
            )
        )

    graph = helper.make_graph(
        nodes,
        "resize",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 2, 2])],
        outputs,
        initializers,
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=11,
    )


def _make_tile_model(
    repeats: np.ndarray,
    *,
    initializer_backed: bool,
) -> ModelProto:
    input_shape = [2, 3]
    output_shape = [
        dimension * int(repeat)
        for dimension, repeat in zip(input_shape, repeats, strict=True)
    ]
    repeats_tensor = numpy_helper.from_array(repeats, "repeats")
    nodes = []
    initializers = []
    if initializer_backed:
        initializers.append(repeats_tensor)
    else:
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                ["repeats"],
                name="repeats_constant",
                value=repeats_tensor,
            )
        )
    nodes.append(helper.make_node("Tile", ["X", "repeats"], ["Y"], name="tile"))
    return helper.make_model(
        helper.make_graph(
            nodes,
            "tile",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
            initializer=initializers,
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=11,
    )






def _empty_float() -> np.ndarray:
    return np.empty((0,), dtype=np.float32)


def _nonempty_roi() -> np.ndarray:
    return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)


def _effective_scales() -> np.ndarray:
    return np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)


def _enabled_config() -> CGIRRewritePipeConfig:
    return CGIRRewritePipe.build_config(omit_empty_resize_inputs=True)


def _resize_nodes(model: ModelProto) -> list[NodeProto]:
    return [node for node in model.graph.node if node.op_type == "Resize"]


def _default_opset(model: ModelProto) -> int:
    return next(
        int(opset.version)
        for opset in model.opset_import
        if opset.domain in {"", "ai.onnx"}
    )


def test_omit_empty_resize_inputs_is_disabled_by_default() -> None:
    config = CGIRRewritePipe.build_config()

    assert config.rules == []
    assert not CGIRRewritePipe.should_process(config)
    assert CGIRRewritePipe.capabilities["omit-empty-resize-inputs"].default is False




def test_disable_graph_optimization_does_not_enable_rules_implicitly() -> None:
    config = CGIRRewritePipe.build_config(ort_graph_optimization=False)

    assert config.rules == []
    assert CGIRRewritePipe.should_process(config) is False


def test_omit_empty_resize_inputs_is_owned_only_by_cgir_rewrite_pipe() -> None:
    assert "omit-empty-resize-inputs" in CGIRRewritePipe.capabilities
    assert "omit-empty-resize-inputs" not in RewritePipe.capabilities
    assert PIPES[0] is CGIRRewritePipe






def test_omit_empty_resize_inputs_requires_explicit_enable() -> None:
    config = _enabled_config()

    assert CGIRRewritePipe.should_process(config)
    assert len(config.rules) == 1


@pytest.mark.parametrize("opset", [11, 12])
def test_omit_empty_resize_inputs_updates_legacy_opset_and_rewrites(opset: int) -> None:
    model = _make_resize_model(
        opset=opset,
        roi=_empty_float(),
        scales=_empty_float(),
        use_sizes=True,
    )

    result = CGIRRewritePipe().process(model, _enabled_config())

    assert _default_opset(model) == opset
    assert list(_resize_nodes(model)[0].input) == ["X", "roi", "scales", "sizes_0"]
    assert _default_opset(result) == 13
    assert list(_resize_nodes(result)[0].input) == ["X", "", "", "sizes_0"]
    checker.check_model(result)


def test_omit_empty_resize_inputs_rewrites_opset_13_without_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_resize_model(
        opset=13,
        roi=_empty_float(),
        scales=_empty_float(),
        use_sizes=True,
    )

    def unexpected_conversion(*_args: object, **_kwargs: object) -> ModelProto:
        pytest.fail("opset 13 model must not be converted")

    monkeypatch.setattr(
        "winml.modelkit.optim.pipes.cgir_rewrite.version_converter.convert_version",
        unexpected_conversion,
    )

    result = CGIRRewritePipe().process(model, _enabled_config())

    assert _default_opset(result) == 13
    assert list(_resize_nodes(result)[0].input) == ["X", "", "", "sizes_0"]


def test_omit_empty_resize_inputs_rewrites_all_matches() -> None:
    model = _make_resize_model(
        opset=11,
        roi=_empty_float(),
        scales=_empty_float(),
        use_sizes=True,
        resize_count=3,
    )

    result = CGIRRewritePipe().process(model, _enabled_config())

    assert [
        list(node.input)
        for node in _resize_nodes(result)
    ] == [
        ["X", "", "", "sizes_0"],
        ["X", "", "", "sizes_1"],
        ["X", "", "", "sizes_2"],
    ]


def test_omit_empty_resize_inputs_preserves_effective_scales() -> None:
    model = _make_resize_model(
        opset=13,
        roi=_empty_float(),
        scales=_effective_scales(),
        use_sizes=False,
    )

    result = CGIRRewritePipe().process(model, _enabled_config())

    assert list(_resize_nodes(result)[0].input) == ["X", "", "scales"]


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: _make_resize_model(
            opset=11,
            roi=_nonempty_roi(),
            scales=_empty_float(),
            use_sizes=True,
        ),
        lambda: _make_resize_model(
            opset=11,
            roi=_nonempty_roi(),
            scales=_effective_scales(),
            use_sizes=True,
        ),
        lambda: _make_resize_model(
            opset=13,
            roi=_nonempty_roi(),
            scales=_effective_scales(),
            use_sizes=False,
        ),
    ],
    ids=["nonempty-roi", "scales-and-sizes", "no-empty-inputs"],
)
def test_omit_empty_resize_inputs_does_not_rewrite_nonmatches(
    model_factory: Callable[[], ModelProto],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = model_factory()
    original = model.SerializeToString()

    def unexpected_conversion(*_args: object, **_kwargs: object) -> ModelProto:
        pytest.fail("non-matching model must not be converted")

    monkeypatch.setattr(
        "winml.modelkit.optim.pipes.cgir_rewrite.version_converter.convert_version",
        unexpected_conversion,
    )

    result = CGIRRewritePipe().process(model, _enabled_config())

    assert result is model
    assert result.SerializeToString() == original


def test_omit_empty_resize_inputs_does_not_rewrite_omitted_inputs() -> None:
    model = _make_resize_model(
        opset=13,
        roi=_empty_float(),
        scales=_empty_float(),
        use_sizes=True,
    )
    resize = _resize_nodes(model)[0]
    resize.input[1] = ""
    resize.input[2] = ""
    original = model.SerializeToString()

    result = CGIRRewritePipe().process(model, _enabled_config())

    assert result is model
    assert result.SerializeToString() == original


def test_omit_empty_resize_inputs_converts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_resize_model(
        opset=11,
        roi=_empty_float(),
        scales=_empty_float(),
        use_sizes=True,
        resize_count=2,
    )
    calls: list[int] = []
    convert_version = version_converter.convert_version

    def recorded_conversion(
        source: ModelProto,
        target_opset: int,
    ) -> ModelProto:
        calls.append(target_opset)
        return convert_version(source, target_opset)

    monkeypatch.setattr(
        "winml.modelkit.optim.pipes.cgir_rewrite.version_converter.convert_version",
        recorded_conversion,
    )

    CGIRRewritePipe().process(model, _enabled_config())

    assert calls == [13]


def test_omit_empty_resize_inputs_surfaces_conversion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_resize_model(
        opset=11,
        roi=_empty_float(),
        scales=_empty_float(),
        use_sizes=True,
    )
    original = model.SerializeToString()

    def failed_conversion(*_args: object, **_kwargs: object) -> ModelProto:
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(
        "winml.modelkit.optim.pipes.cgir_rewrite.version_converter.convert_version",
        failed_conversion,
    )

    with pytest.raises(OptimizationError, match="conversion failed"):
        CGIRRewritePipe().process(model, _enabled_config())

    assert model.SerializeToString() == original


def test_optimizer_leaves_resize_unchanged_by_default() -> None:
    model = _make_resize_model(
        opset=11,
        roi=_empty_float(),
        scales=_empty_float(),
        use_sizes=True,
    )

    result = Optimizer().optimize(model)

    assert _default_opset(result) == 11
    resize_inputs = list(_resize_nodes(result)[0].input)
    assert resize_inputs[0] == "X"
    assert resize_inputs[1]
    assert resize_inputs[2]
    assert resize_inputs[3] == "sizes_0"


def test_optimizer_applies_explicit_cgir_resize_rewrite() -> None:
    model = _make_resize_model(
        opset=11,
        roi=_empty_float(),
        scales=_empty_float(),
        use_sizes=True,
    )

    result = Optimizer().optimize(model, omit_empty_resize_inputs=True)

    assert _default_opset(result) == 13
    assert list(_resize_nodes(result)[0].input) == ["X", "", "", "sizes_0"]
    checker.check_model(result)
















def test_deduplicate_opset_imports_is_explicit_and_cgir_only() -> None:
    capability = "deduplicate-opset-imports"
    assert CGIRRewritePipe.capabilities[capability].default is False
    assert capability not in RewritePipe.capabilities
    model = _make_tile_model(np.ones(2, dtype=np.int64), initializer_backed=True)
    model.opset_import.append(model.opset_import[0])
    config = CGIRRewritePipe.build_config(ort_graph_optimization=False)
    assert CGIRRewritePipe().process(model, config) is model
    enabled = CGIRRewritePipe.build_config(deduplicate_opset_imports=True)
    assert CGIRRewritePipe.should_process(enabled)
    assert len(enabled.rules) == 1


def test_deduplicate_opset_imports_preserves_model_content_and_domain_order() -> None:
    model = _make_tile_model(np.ones(2, dtype=np.int64), initializer_backed=True)
    model.opset_import.extend(
        [
            helper.make_opsetid("example.first", 1),
            model.opset_import[0],
            helper.make_opsetid("example.second", 1),
            helper.make_opsetid("example.first", 1),
        ]
    )
    model.functions.append(
        helper.make_function(
            "example.first", "PassThrough", ["X"], ["Y"],
            [helper.make_node("Identity", ["X"], ["Y"])],
            [helper.make_opsetid("", 18)],
        )
    )
    original = model.SerializeToString()
    expected_imports = list(dict.fromkeys(
        (opset.domain, opset.version) for opset in model.opset_import
    ))
    result = CGIRRewritePipe().process(
        model, CGIRRewritePipe.build_config(deduplicate_opset_imports=True)
    )
    assert [(opset.domain, opset.version) for opset in result.opset_import] == expected_imports
    assert result.graph.SerializeToString() == model.graph.SerializeToString()
    without_imports = ModelProto()
    without_imports.CopyFrom(result)
    del without_imports.opset_import[:]
    without_imports.opset_import.extend(model.opset_import)
    assert without_imports.SerializeToString() == original
    assert model.SerializeToString() == original
    checker.check_model(result)
    config = CGIRRewritePipe.build_config(deduplicate_opset_imports=True)
    assert CGIRRewritePipe().process(result, config) is result


@pytest.mark.parametrize("domain", ["", "example.custom"])
def test_deduplicate_opset_imports_rejects_conflicts_without_mutation(domain: str) -> None:
    model = _make_tile_model(np.ones(2, dtype=np.int64), initializer_backed=True)
    model.opset_import.extend(
        [helper.make_opsetid(domain, 18), helper.make_opsetid(domain, 19)]
    )
    original = model.SerializeToString()
    with pytest.raises(OptimizationError, match="Conflicting opset imports"):
        CGIRRewritePipe().process(
            model, CGIRRewritePipe.build_config(deduplicate_opset_imports=True)
        )
    assert model.SerializeToString() == original


def test_deduplicate_opset_imports_precedes_version_conversion() -> None:
    model = _make_resize_model(
        opset=11, roi=_empty_float(), scales=_empty_float(), use_sizes=True
    )
    model.opset_import.append(model.opset_import[0])
    original = model.SerializeToString()
    result = CGIRRewritePipe().process(
        model,
        CGIRRewritePipe.build_config(
            omit_empty_resize_inputs=True, deduplicate_opset_imports=True
        ),
    )
    assert [opset.version for opset in result.opset_import if opset.domain == ""] == [13]
    assert list(_resize_nodes(result)[0].input) == ["X", "", "", "sizes_0"]
    assert model.SerializeToString() == original
    checker.check_model(result)


def test_optimizer_applies_opset_deduplication_without_export() -> None:
    model = _make_tile_model(np.ones(2, dtype=np.int64), initializer_backed=True)
    model.opset_import.append(model.opset_import[0])
    result = Optimizer().optimize(
        model, ort_graph_optimization=False, deduplicate_opset_imports=True
    )
    assert len(result.opset_import) == 1
    assert _default_opset(result) == _default_opset(model)
