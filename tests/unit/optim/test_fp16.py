# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""FP16 conversion utility tests.

Tests for winml.modelkit.optim.fp16.convert_to_fp16 which converts
FP32 ONNX models to FP16 precision.

Following Cardinal Rules:
- CARDINAL RULE #1: No hardcoded model architectures
- CARDINAL RULE #2: All tests use pytest with code-generated results
- CARDINAL RULE #3: Tests must run and pass
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
from google.protobuf.message import EncodeError
from onnx import (
    AttributeProto,
    GraphProto,
    ModelProto,
    SparseTensorProto,
    TensorProto,
    checker,
    helper,
    numpy_helper,
    shape_inference,
)

from winml.modelkit.quant.fp16 import convert_to_fp16


if TYPE_CHECKING:
    import pytest


# =============================================================================
# HELPERS
# =============================================================================


def _build_simple_fp32_model() -> ModelProto:
    """Build a simple FP32 model: out = x + weight."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    weight = numpy_helper.from_array(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), "weight")
    add = helper.make_node("Add", ["x", "weight"], ["out"], name="add")
    graph = helper.make_graph([add], "simple", [x], [out], [weight])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_multi_op_fp32_model() -> ModelProto:
    """Build a model with multiple ops: out = Relu(x + weight)."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    weight = numpy_helper.from_array(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), "weight")
    add = helper.make_node("Add", ["x", "weight"], ["add_out"], name="add")
    relu = helper.make_node("Relu", ["add_out"], ["out"], name="relu")
    graph = helper.make_graph([add, relu], "multi_op", [x], [out], [weight])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_scalar_float_attribute_model() -> ModelProto:
    """Build a FLOAT-producing scalar attribute ORT leaves unchanged."""
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [])
    constant = helper.make_node(
        "Constant",
        [],
        ["output"],
        name="constant",
        value_float=3.5,
    )
    graph = helper.make_graph([constant], "scalar_float_attribute", [], [output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_sparse_float_attribute_model() -> ModelProto:
    """Build a sparse FLOAT tensor attribute ORT leaves unchanged."""
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])
    values = numpy_helper.from_array(np.array([3.5], dtype=np.float32), "values")
    indices = numpy_helper.from_array(np.array([[1]], dtype=np.int64), "indices")
    sparse = helper.make_sparse_tensor(values, indices, [2])
    constant = helper.make_node(
        "Constant",
        [],
        ["output"],
        name="constant",
        sparse_value=sparse,
    )
    graph = helper.make_graph([constant], "sparse_float_attribute", [], [output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_external_float_attribute_model(*, clear_data: bool) -> ModelProto:
    """Build a FLOAT tensor attribute carrying external-data metadata."""
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    value = numpy_helper.from_array(np.array([3.5], dtype=np.float32), "value")
    if clear_data:
        value.ClearField("raw_data")
        del value.float_data[:]
    value.data_location = TensorProto.EXTERNAL
    location = value.external_data.add()
    location.key = "location"
    location.value = "value.bin"
    length = value.external_data.add()
    length.key = "length"
    length.value = "4"
    constant = helper.make_node("Constant", [], ["output"], name="constant", value=value)
    graph = helper.make_graph([constant], "external_float_attribute", [], [output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_nested_external_int_name_collision_model() -> ModelProto:
    """Reuse a selected FLOAT weight name for nested external INT data."""
    model = _build_simple_fp32_model()
    weight = model.graph.initializer[0]
    weight.data_location = TensorProto.EXTERNAL
    weight_location = weight.external_data.add()
    weight_location.key = "location"
    weight_location.value = "weight.bin"
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    integer_output = helper.make_tensor_value_info("integer_output", TensorProto.INT64, [1])

    def _branch(name: str, initializer_name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.INT64, [1])
        initializer = numpy_helper.from_array(np.array([7], dtype=np.int64), initializer_name)
        if name == "then":
            initializer.ClearField("raw_data")
            del initializer.int32_data[:]
            del initializer.int64_data[:]
            initializer.data_location = TensorProto.EXTERNAL
            location = initializer.external_data.add()
            location.key = "location"
            location.value = "integer.bin"
        identity = helper.make_node(
            "Identity",
            [initializer_name],
            [branch_output.name],
            name=f"{name}_identity",
        )
        return helper.make_graph([identity], name, [], [branch_output], [initializer])

    conditional = helper.make_node(
        "If",
        ["condition"],
        ["integer_output"],
        name="if",
        then_branch=_branch("then", "weight"),
        else_branch=_branch("else", "other"),
    )
    model.graph.input.append(condition)
    model.graph.output.append(integer_output)
    model.graph.node.append(conditional)
    model.graph.name = "nested_external_int_name_collision"
    return model


def _build_omitted_optional_output_model() -> ModelProto:
    """Build optional inputs and outputs that use the empty sentinel."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    dropped = helper.make_tensor_value_info("dropped", TensorProto.FLOAT, [2])
    clipped = helper.make_tensor_value_info("clipped", TensorProto.FLOAT, [2])
    drop = helper.make_node("Dropout", ["x"], ["dropped", ""], name="drop")
    clip = helper.make_node("Clip", ["x", "", ""], ["clipped"], name="clip")
    graph = helper.make_graph(
        [drop, clip],
        "omitted_optional_output",
        [x],
        [dropped, clipped],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_initializer_backed_output_model() -> ModelProto:
    """Build a graph whose output is supplied directly by an initializer."""
    out = helper.make_tensor_value_info("constant_output", TensorProto.FLOAT, [1, 2])
    value = numpy_helper.from_array(
        np.array([[1.0001, 2.0003]], dtype=np.float32), "constant_output"
    )
    graph = helper.make_graph([], "initializer_output", [], [out], [value])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_shared_initializer_output_model() -> ModelProto:
    """Build a graph where an initializer is both an output and a node input."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    shared = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    value = numpy_helper.from_array(np.array([[1.0, 2.0]], dtype=np.float32), "shared")
    add = helper.make_node("Add", ["x", "shared"], ["y"], name="add")
    graph = helper.make_graph([add], "shared_initializer", [x], [shared, y], [value])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_overridable_shared_initializer_output_model() -> ModelProto:
    """Build a graph input/output initializer that can be overridden by callers."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    shared = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    value = numpy_helper.from_array(np.array([[1.0, 2.0]], dtype=np.float32), "shared")
    add = helper.make_node("Add", ["x", "shared"], ["y"], name="add")
    graph = helper.make_graph([add], "overridable_shared", [x, shared], [shared, y], [value])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_nested_initializer_output_model() -> ModelProto:
    """Build an If whose branch outputs are supplied by initializers."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    def _branch(name: str, output_name: str, value: float):
        branch_output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1])
        initializer = numpy_helper.from_array(np.array([value], dtype=np.float32), output_name)
        return helper.make_graph([], name, [], [branch_output], [initializer])

    node = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="if",
        then_branch=_branch("then", "then_output", 1.0),
        else_branch=_branch("else", "else_output", 2.0),
    )
    graph = helper.make_graph([node], "nested_initializer", [condition], [output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_nested_consumed_initializer_output_model() -> ModelProto:
    """Build an If whose branch initializer outputs are also consumed locally."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    def _branch(name: str, value: float) -> GraphProto:
        initializer_name = f"{name}_value"
        branch_output = helper.make_tensor_value_info(initializer_name, TensorProto.FLOAT, [1])
        initializer = numpy_helper.from_array(np.array([value], dtype=np.float32), initializer_name)
        identity = helper.make_node(
            "Identity", [initializer_name], [f"{name}_used"], name=f"{name}_identity"
        )
        return helper.make_graph([identity], name, [], [branch_output], [initializer])

    node = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="if",
        then_branch=_branch("then", 1.0),
        else_branch=_branch("else", 2.0),
    )
    graph = helper.make_graph([node], "nested_consumed_initializer", [condition], [output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_nested_consumed_initializer_output_with_top_level_input_collision_model() -> ModelProto:
    """Build nested local initializer consumers shadowing a kept top-level input."""
    same = helper.make_tensor_value_info("same", TensorProto.FLOAT, [1])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    def _branch(name: str, value: float, *, collides: bool) -> GraphProto:
        output_name = "same" if collides else f"{name}_value"
        branch_output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1])
        initializer = numpy_helper.from_array(np.array([value], dtype=np.float32), output_name)
        identity = helper.make_node(
            "Identity", [output_name], [f"{name}_used"], name=f"{name}_identity"
        )
        return helper.make_graph([identity], name, [], [branch_output], [initializer])

    node = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="if",
        then_branch=_branch("then", 1.0, collides=True),
        else_branch=_branch("else", 2.0, collides=False),
    )
    graph = helper.make_graph([node], "nested_input_collision", [same, condition], [output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_fp32_identity_with_fp16_nested_initializer_model() -> ModelProto:
    """Build FP32 top-level I/O with only nested FP16 floating initializers."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    branch_output = helper.make_tensor_value_info("branch_value", TensorProto.FLOAT16, [1])
    nested_output = helper.make_tensor_value_info("nested_output", TensorProto.FLOAT16, [1])
    initializer = numpy_helper.from_array(np.array([1.0], dtype=np.float16), "branch_value")
    branch = helper.make_graph([], "branch", [], [branch_output], [initializer])
    identity = helper.make_node("Identity", ["x"], ["y"], name="identity")
    if_node = helper.make_node(
        "If",
        ["condition"],
        ["nested_output"],
        name="if",
        then_branch=branch,
        else_branch=branch,
    )
    graph = helper.make_graph(
        [identity, if_node],
        "fp32_top_level_with_fp16_nested_initializer",
        [x, condition],
        [y, nested_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_sparse_shape_reshape_model() -> ModelProto:
    """Build a Reshape graph that consumes a sparse INT64 shape initializer."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])
    sparse_shape = SparseTensorProto()
    sparse_shape.values.CopyFrom(numpy_helper.from_array(np.array([2, 2], dtype=np.int64)))
    sparse_shape.indices.CopyFrom(numpy_helper.from_array(np.array([[0], [1]], dtype=np.int64)))
    sparse_shape.dims.extend([2])
    sparse_shape.values.name = "shape"
    reshape = helper.make_node("Reshape", ["x", "shape"], ["output"], name="reshape")
    graph = helper.make_graph([reshape], "sparse_shape", [x], [output])
    graph.sparse_initializer.append(sparse_shape)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_sparse_float_add_model() -> ModelProto:
    """Build an Add graph that consumes a sparse FLOAT initializer."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])
    sparse_weight = SparseTensorProto()
    sparse_weight.values.CopyFrom(numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32)))
    sparse_weight.indices.CopyFrom(numpy_helper.from_array(np.array([[0], [1]], dtype=np.int64)))
    sparse_weight.dims.extend([2])
    sparse_weight.values.name = "weight"
    add = helper.make_node("Add", ["x", "weight"], ["output"], name="add")
    graph = helper.make_graph([add], "sparse_float_add", [x], [output])
    graph.sparse_initializer.append(sparse_weight)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_sparse_float_add_model_with_value_info() -> ModelProto:
    """Build an Add graph with sparse FLOAT initializer metadata."""
    model = _build_sparse_float_add_model()
    model.graph.value_info.append(
        helper.make_sparse_tensor_value_info("weight", TensorProto.FLOAT, [2])
    )
    return model


def _build_sparse_float_add_model_with_tensor_value_info() -> ModelProto:
    """Build an Add graph with tensor metadata for a sparse FLOAT initializer."""
    model = _build_sparse_float_add_model()
    model.graph.value_info.append(helper.make_tensor_value_info("weight", TensorProto.FLOAT, [2]))
    return model


def _build_retained_float_initializer_metadata_model() -> ModelProto:
    """Build an always-FLOAT initializer with convertible tensor metadata."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
        "scales",
    )
    resize = helper.make_node(
        "Resize",
        ["x", "", "scales"],
        ["output"],
        name="resize",
        mode="nearest",
    )
    graph = helper.make_graph(
        [resize],
        "retained_float_initializer_metadata",
        [x],
        [output],
        [scales],
        value_info=[helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_sparse_float_add_model_with_io_metadata() -> ModelProto:
    """Build an Add graph whose sparse FLOAT initializer is also graph sparse I/O."""
    model = _build_sparse_float_add_model()
    model.graph.input.append(helper.make_sparse_tensor_value_info("weight", TensorProto.FLOAT, [2]))
    model.graph.output.append(
        helper.make_sparse_tensor_value_info("weight", TensorProto.FLOAT, [2])
    )
    return model


def _build_sparse_float_add_model_with_tensor_io_metadata() -> ModelProto:
    """Build an Add graph whose sparse FLOAT initializer is also graph tensor I/O."""
    model = _build_sparse_float_add_model()
    model.graph.input.append(helper.make_tensor_value_info("weight", TensorProto.FLOAT, [2]))
    model.graph.output.append(helper.make_tensor_value_info("weight", TensorProto.FLOAT, [2]))
    return model


def _build_direct_sparse_float_output_model() -> ModelProto:
    """Build a graph with a direct sparse FLOAT initializer output."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    dense_output = helper.make_tensor_value_info("dense_output", TensorProto.FLOAT, [2])
    sparse_output = helper.make_sparse_tensor_value_info("sparse_output", TensorProto.FLOAT, [2])
    sparse_value = SparseTensorProto()
    sparse_value.values.CopyFrom(numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32)))
    sparse_value.indices.CopyFrom(numpy_helper.from_array(np.array([[0], [1]], dtype=np.int64)))
    sparse_value.dims.extend([2])
    sparse_value.values.name = "sparse_output"
    identity = helper.make_node("Identity", ["x"], ["dense_output"], name="identity")
    graph = helper.make_graph(
        [identity], "direct_sparse_output", [x], [dense_output, sparse_output]
    )
    graph.sparse_initializer.append(sparse_value)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_direct_sparse_float_tensor_output_model() -> ModelProto:
    """Build a graph with a direct sparse FLOAT initializer and tensor output metadata."""
    model = _build_direct_sparse_float_output_model()
    sparse_output = next(value for value in model.graph.output if value.name == "sparse_output")
    sparse_output.CopyFrom(helper.make_tensor_value_info("sparse_output", TensorProto.FLOAT, [2]))
    return model


def _build_if_with_direct_sparse_float_outputs_model() -> ModelProto:
    """Build an If whose branches return direct sparse FLOAT initializers."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    output = helper.make_sparse_tensor_value_info("output", TensorProto.FLOAT, [2])

    def _branch(name: str, values: list[float]) -> GraphProto:
        output_name = f"{name}_output"
        branch_output = helper.make_sparse_tensor_value_info(output_name, TensorProto.FLOAT, [2])
        sparse_value = SparseTensorProto()
        sparse_value.values.CopyFrom(numpy_helper.from_array(np.array(values, dtype=np.float32)))
        sparse_value.indices.CopyFrom(numpy_helper.from_array(np.array([[0], [1]], dtype=np.int64)))
        sparse_value.dims.extend([2])
        sparse_value.values.name = output_name
        graph = helper.make_graph([], name, [], [branch_output])
        graph.sparse_initializer.append(sparse_value)
        return graph

    conditional = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="if",
        then_branch=_branch("then", [1.0, 2.0]),
        else_branch=_branch("else", [3.0, 4.0]),
    )
    graph = helper.make_graph([conditional], "nested_sparse_outputs", [condition], [output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_deep_if_with_direct_sparse_float_outputs_model() -> ModelProto:
    """Build two nested If levels ending in direct sparse FLOAT outputs."""
    outer_condition = helper.make_tensor_value_info("outer_condition", TensorProto.BOOL, [])
    inner_condition = helper.make_tensor_value_info("inner_condition", TensorProto.BOOL, [])
    output = helper.make_sparse_tensor_value_info("output", TensorProto.FLOAT, [2])

    def _leaf(name: str, values: list[float]) -> GraphProto:
        output_name = f"{name}_output"
        branch_output = helper.make_sparse_tensor_value_info(output_name, TensorProto.FLOAT, [2])
        sparse_value = SparseTensorProto()
        sparse_value.values.CopyFrom(
            numpy_helper.from_array(np.array(values, dtype=np.float32), output_name)
        )
        sparse_value.indices.CopyFrom(numpy_helper.from_array(np.array([[0], [1]], dtype=np.int64)))
        sparse_value.dims.extend([2])
        graph = helper.make_graph([], name, [], [branch_output])
        graph.sparse_initializer.append(sparse_value)
        return graph

    def _outer_branch(name: str, offset: float) -> GraphProto:
        branch_output = helper.make_sparse_tensor_value_info(
            f"{name}_output", TensorProto.FLOAT, [2]
        )
        inner = helper.make_node(
            "If",
            ["inner_condition"],
            [branch_output.name],
            name=f"{name}_if",
            then_branch=_leaf(f"{name}_then", [offset + 1.0, offset + 2.0]),
            else_branch=_leaf(f"{name}_else", [offset + 3.0, offset + 4.0]),
        )
        return helper.make_graph([inner], name, [], [branch_output])

    outer = helper.make_node(
        "If",
        ["outer_condition"],
        ["output"],
        name="outer_if",
        then_branch=_outer_branch("outer_then", 0.0),
        else_branch=_outer_branch("outer_else", 4.0),
    )
    graph = helper.make_graph(
        [outer],
        "deep_nested_sparse_outputs",
        [outer_condition, inner_condition],
        [output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _iter_attribute_graphs(model: ModelProto) -> list[GraphProto]:
    """Return nested graphs stored on node attributes."""
    graphs: list[GraphProto] = []
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.g.name:
                graphs.append(attribute.g)
            graphs.extend(attribute.graphs)
    return graphs


def _mark_initializers_as_external(graph: GraphProto, *, clear_data: bool) -> None:
    """Mark all graph initializers as external, optionally without resident bytes."""
    for initializer in graph.initializer:
        if clear_data:
            initializer.ClearField("raw_data")
            del initializer.float_data[:]
        initializer.data_location = TensorProto.EXTERNAL
        location = initializer.external_data.add()
        location.key = "location"
        location.value = f"{initializer.name}.bin"


def _build_lexically_captured_initializer_output_model() -> ModelProto:
    """Build an output initializer captured only by nested If branches."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    shared = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    initializer = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "shared")

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        identity = helper.make_node(
            "Identity", ["shared"], [f"{name}_output"], name=f"{name}_identity"
        )
        return helper.make_graph([identity], name, [], [branch_output])

    node = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    graph = helper.make_graph(
        [node], "lexical_capture", [condition], [shared, output], [initializer]
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_initializer_output_name_collision_model() -> ModelProto:
    """Build a legal graph with a name that collides with ORT's generated alias."""
    existing = helper.make_tensor_value_info("graph_output_cast_0", TensorProto.FLOAT, [1])
    constant_output = helper.make_tensor_value_info("constant_output", TensorProto.FLOAT, [1])
    result = helper.make_tensor_value_info("result", TensorProto.FLOAT, [1])
    initializer = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "constant_output")
    identity = helper.make_node("Identity", ["graph_output_cast_0"], ["result"], name="identity")
    graph = helper.make_graph(
        [identity], "name_collision", [existing], [constant_output, result], [initializer]
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_initializer_output_node_name_collision_model() -> ModelProto:
    """Build a graph with a user node named like ORT's output Cast node."""
    model = _build_initializer_backed_output_model()
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    model.graph.input.append(x)
    model.graph.output.append(y)
    model.graph.node.append(helper.make_node("Identity", ["x"], ["y"], name="graph_output_cast0"))
    return model


def _build_nested_node_name_collision_model() -> ModelProto:
    """Build a nested user node named like ORT's top-level output Cast."""
    model = _build_initializer_backed_output_model()
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    nested_output = helper.make_tensor_value_info("nested_output", TensorProto.FLOAT, [1])

    def _branch(name: str, value: float) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        initializer = numpy_helper.from_array(np.array([value], dtype=np.float32), f"{name}_value")
        identity = helper.make_node(
            "Identity",
            [f"{name}_value"],
            [f"{name}_output"],
            name="graph_output_cast0" if name == "then" else f"{name}_identity",
        )
        return helper.make_graph([identity], name, [], [branch_output], [initializer])

    node = helper.make_node(
        "If",
        ["condition"],
        ["nested_output"],
        name="if",
        then_branch=_branch("then", 1.0),
        else_branch=_branch("else", 2.0),
    )
    model.graph.input.append(condition)
    model.graph.output.append(nested_output)
    model.graph.node.append(node)
    return model


def _build_regular_output_nested_node_name_collision_model() -> ModelProto:
    """Build a normal output whose generated Cast name collides in a nested graph."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    nested_output = helper.make_tensor_value_info("nested_output", TensorProto.FLOAT, [1])

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        identity = helper.make_node(
            "Identity",
            ["x"],
            [f"{name}_output"],
            name="graph_output_cast0" if name == "then" else f"{name}_identity",
        )
        return helper.make_graph([identity], name, [], [branch_output])

    identity = helper.make_node("Identity", ["x"], ["output"], name="identity")
    conditional = helper.make_node(
        "If",
        ["condition"],
        ["nested_output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    graph = helper.make_graph(
        [identity, conditional],
        "regular_output_nested_node_collision",
        [x, condition],
        [output, nested_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_neutral_nested_node_name_collision_model() -> ModelProto:
    """Build a generated-name collision on a conversion-neutral INT node."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    integer = helper.make_tensor_value_info("integer", TensorProto.INT64, [1])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    nested_output = helper.make_tensor_value_info("nested_output", TensorProto.INT64, [1])

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.INT64, [1])
        identity = helper.make_node(
            "Identity",
            ["integer"],
            [branch_output.name],
            name=("graph_output_cast0" if name == "then" else f"{name}_identity"),
        )
        return helper.make_graph([identity], name, [], [branch_output])

    identity = helper.make_node("Identity", ["x"], ["output"], name="identity")
    conditional = helper.make_node(
        "If",
        ["condition"],
        ["nested_output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    graph = helper.make_graph(
        [identity, conditional],
        "neutral_nested_node_name_collision",
        [x, integer, condition],
        [output, nested_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_neutral_shadowed_keep_io_collision_model() -> ModelProto:
    """Shadow kept FLOAT I/O only inside a skipped neutral INT node."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    integer = helper.make_tensor_value_info("integer", TensorProto.INT64, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    loop_output = helper.make_tensor_value_info("loop_output", TensorProto.INT64, [1])
    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    body_state = helper.make_tensor_value_info("x", TensorProto.INT64, [1])
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.INT64, [1])
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["body_condition"],
                ["body_condition_out"],
                name="condition_feedback",
            ),
            helper.make_node(
                "Identity",
                ["x"],
                ["body_state_out"],
                name="graph_input_cast0",
            ),
        ],
        "body",
        [iteration, body_condition, body_state],
        [body_condition_out, body_state_out],
    )
    identity = helper.make_node("Identity", ["x"], ["output"], name="identity")
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "integer"],
        ["loop_output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [identity, loop],
        "neutral_shadowed_keep_io_collision",
        [x, trip_count, condition, integer],
        [output, loop_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_inferred_output_nested_node_name_collision_model() -> ModelProto:
    """Build an inferred FLOAT output whose generated Cast name collides."""
    model = _build_regular_output_nested_node_name_collision_model()
    model.graph.output[0].CopyFrom(helper.make_empty_tensor_value_info("output"))
    return model


def _build_nested_local_generated_tensor_alias_model() -> ModelProto:
    """Build a Loop with a local input matching a top-level generated tensor alias."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    loop_state = helper.make_tensor_value_info("loop_state", TensorProto.FLOAT, [1])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    loop_output = helper.make_tensor_value_info("loop_output", TensorProto.FLOAT, [1])

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    local_state = helper.make_tensor_value_info("graph_output_cast_0", TensorProto.FLOAT, [1])
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.FLOAT, [1])
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity", ["body_condition"], ["body_condition_out"], name="body_condition"
            ),
            helper.make_node(
                "Identity",
                ["graph_output_cast_0"],
                ["body_state_out"],
                name="body_state",
            ),
        ],
        "body",
        [iteration, body_condition, local_state],
        [body_condition_out, body_state_out],
    )
    identity = helper.make_node("Identity", ["x"], ["output"], name="identity")
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "loop_state"],
        ["loop_output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [identity, loop],
        "nested_local_generated_alias",
        [trip_count, condition, loop_state, x],
        [output, loop_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_kept_input_free_capture_value_info_model() -> ModelProto:
    """Annotate a free kept-input capture with nested value_info."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        identity = helper.make_node(
            "Identity",
            ["x"],
            [branch_output.name],
            name=f"{name}_identity",
        )
        return helper.make_graph(
            [identity],
            name,
            [],
            [branch_output],
            value_info=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        )

    conditional = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    graph = helper.make_graph(
        [conditional],
        "kept_input_free_capture_value_info",
        [x, condition],
        [output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_nested_blocked_generated_tensor_alias_model() -> ModelProto:
    """Build a blocked nested node that consumes a generated tensor alias locally."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    loop_state = helper.make_tensor_value_info("loop_state", TensorProto.INT64, [1])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    loop_output = helper.make_tensor_value_info("loop_output", TensorProto.INT64, [1])

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    local_state = helper.make_tensor_value_info("graph_output_cast_0", TensorProto.INT64, [1])
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.INT64, [1])
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["body_condition"],
                ["body_condition_out"],
                name="body_condition",
            ),
            helper.make_node(
                "Identity",
                ["graph_output_cast_0"],
                ["body_state_out"],
                name="body_state",
            ),
        ],
        "body",
        [iteration, body_condition, local_state],
        [body_condition_out, body_state_out],
    )
    relu = helper.make_node("Relu", ["x"], ["output"], name="relu")
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "loop_state"],
        ["loop_output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [relu, loop],
        "nested_blocked_generated_alias",
        [trip_count, condition, loop_state, x],
        [output, loop_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_nested_mixed_generated_tensor_alias_model() -> ModelProto:
    """Build a mixed-type nested node that consumes a generated tensor alias locally."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    loop_state = helper.make_tensor_value_info("loop_state", TensorProto.FLOAT, [1])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    loop_output = helper.make_tensor_value_info("loop_output", TensorProto.FLOAT, [1])

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    local_scales = helper.make_tensor_value_info("graph_output_cast_0", TensorProto.FLOAT, [1])
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.FLOAT, [1])
    resize_input = helper.make_tensor("resize_input", TensorProto.FLOAT, [1], [2.0])
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["body_condition"],
                ["body_condition_out"],
                name="body_condition",
            ),
            helper.make_node(
                "Resize",
                ["resize_input", "", "graph_output_cast_0"],
                ["body_state_out"],
                name="body_resize",
            ),
        ],
        "body",
        [iteration, body_condition, local_scales],
        [body_condition_out, body_state_out],
        initializer=[resize_input],
    )
    relu = helper.make_node("Relu", ["x"], ["output"], name="relu")
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "loop_state"],
        ["loop_output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [relu, loop],
        "nested_mixed_generated_alias",
        [trip_count, condition, loop_state, x],
        [output, loop_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_inferred_nested_mixed_name_collision_model() -> ModelProto:
    """Build a mixed nested input collision visible only after shape inference."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    loop_scale = helper.make_tensor_value_info("loop_scale", TensorProto.FLOAT, [1])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    loop_output = helper.make_tensor_value_info("loop_output", TensorProto.FLOAT, [None])

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    local_scale = helper.make_empty_tensor_value_info("same")
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.FLOAT, [None])
    resize_input = helper.make_tensor("resize_input", TensorProto.FLOAT, [1], [2.0])
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["body_condition"],
                ["body_condition_out"],
                name="body_condition",
            ),
            helper.make_node(
                "Resize",
                ["resize_input", "", "same"],
                ["body_state_out"],
                name="body_resize",
            ),
        ],
        "body",
        [iteration, body_condition, local_scale],
        [body_condition_out, body_state_out],
        initializer=[resize_input],
    )
    identity = helper.make_node("Identity", ["x"], ["same"], name="top_identity")
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "loop_scale"],
        ["loop_output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [identity, loop],
        "inferred_nested_mixed_collision",
        [trip_count, condition, loop_scale, x],
        [loop_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_top_blocked_global_value_info_collision_model() -> ModelProto:
    """Build a top blocked input colliding with nested FLOAT metadata."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    same = helper.make_tensor_value_info("same", TensorProto.INT64, [1])
    loop_state = helper.make_tensor_value_info("loop_state", TensorProto.FLOAT, [1])
    integer_output = helper.make_tensor_value_info("integer_output", TensorProto.INT64, [1])
    loop_output = helper.make_tensor_value_info("loop_output", TensorProto.FLOAT, [1])

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    local_same = helper.make_tensor_value_info("same", TensorProto.FLOAT, [1])
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.FLOAT, [1])
    body = helper.make_graph(
        [
            helper.make_node(
                "And",
                ["body_condition", "body_condition"],
                ["body_condition_out"],
                name="body_condition",
            ),
            helper.make_node("Relu", ["same"], ["body_state_out"], name="body_relu"),
        ],
        "body",
        [iteration, body_condition, local_same],
        [body_condition_out, body_state_out],
    )
    blocked = helper.make_node("Identity", ["same"], ["integer_output"])
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "loop_state"],
        ["loop_output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [blocked, loop],
        "top_blocked_global_collision",
        [trip_count, condition, same, loop_state],
        [integer_output, loop_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_duplicate_late_cast_alias_model() -> ModelProto:
    """Build two unnamed blocked nodes that generate identical late Cast aliases."""
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1])
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [1])
    d = helper.make_tensor_value_info("d", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["a"], ["b"]),
            helper.make_node("Identity", ["c"], ["d"]),
        ],
        "duplicate_late_cast_alias",
        [a, c],
        [b, d],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_late_cast_node_name_collision_model() -> ModelProto:
    """Build a blocked node whose generated Cast node name already exists."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    other = helper.make_tensor_value_info("other", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    other_output = helper.make_tensor_value_info("other_output", TensorProto.FLOAT, [1])
    blocked = helper.make_node("Abs", ["x"], ["output"], name="blocked")
    existing = helper.make_node(
        "Identity",
        ["other"],
        ["other_output"],
        name="blocked_input_cast0",
    )
    graph = helper.make_graph(
        [blocked, existing],
        "late_cast_node_name_collision",
        [x, other],
        [output, other_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_blocked_subgraph_float_capture_model() -> ModelProto:
    """Build blocked branches that capture a converted outer FLOAT value."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    relu_output = helper.make_tensor_value_info("relu_output", TensorProto.FLOAT, [1])
    if_output = helper.make_tensor_value_info("if_output", TensorProto.FLOAT, [1])

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        identity = helper.make_node(
            "Identity", ["x"], [branch_output.name], name=f"{name}_identity"
        )
        return helper.make_graph([identity], name, [], [branch_output])

    relu = helper.make_node("Relu", ["x"], ["relu_output"], name="relu")
    conditional = helper.make_node(
        "If",
        ["condition"],
        ["if_output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    graph = helper.make_graph(
        [relu, conditional],
        "blocked_subgraph_float_capture",
        [x, condition],
        [relu_output, if_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_untyped_blocked_subgraph_capture_model() -> ModelProto:
    """Build a blocked capture whose FLOAT producer has no declared metadata."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    relu_output = helper.make_tensor_value_info("relu_output", TensorProto.FLOAT, [1])
    if_output = helper.make_tensor_value_info("if_output", TensorProto.FLOAT, [1])

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        identity = helper.make_node(
            "Identity", ["x"], [branch_output.name], name=f"{name}_identity"
        )
        return helper.make_graph([identity], name, [], [branch_output])

    constant = helper.make_node(
        "Constant",
        [],
        ["x"],
        name="constant",
        value=helper.make_tensor("value", TensorProto.FLOAT, [1], [2.0]),
    )
    relu = helper.make_node("Relu", ["x"], ["relu_output"], name="relu")
    conditional = helper.make_node(
        "If",
        ["condition"],
        ["if_output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    graph = helper.make_graph(
        [constant, relu, conditional],
        "untyped_blocked_subgraph_capture",
        [condition],
        [relu_output, if_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_uninferred_custom_op_blocked_capture_model() -> ModelProto:
    """Build a custom type-preserving output omitted by successful inference."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    if_output = helper.make_tensor_value_info("if_output", TensorProto.FLOAT, [1])

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        identity = helper.make_node(
            "Identity", ["y"], [branch_output.name], name=f"{name}_identity"
        )
        return helper.make_graph([identity], name, [], [branch_output])

    gelu = helper.make_node("Gelu", ["x"], ["y"], name="gelu", domain="com.microsoft")
    conditional = helper.make_node(
        "If",
        ["condition"],
        ["if_output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    graph = helper.make_graph(
        [gelu, conditional],
        "uninferred_custom_op_blocked_capture",
        [x, condition],
        [if_output],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _build_missing_metadata_blocked_edge_model() -> ModelProto:
    """Build an uninferred custom edge that can cross a blocked-node boundary."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    bias = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "bias")
    gelu = helper.make_node("Gelu", ["x"], ["hidden"], name="gelu", domain="com.microsoft")
    add = helper.make_node("Add", ["hidden", "bias"], ["output"], name="add")
    graph = helper.make_graph(
        [gelu, add],
        "missing_metadata_blocked_edge",
        [x],
        [output],
        [bias],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _build_existing_fp16_boundary_model() -> ModelProto:
    """Build an uninferred blocked output with an explicit FP16 boundary."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1])
    gelu = helper.make_node("Gelu", ["x"], ["hidden"], name="gelu", domain="com.microsoft")
    boundary = helper.make_node(
        "Cast", ["hidden"], ["output"], name="boundary", to=TensorProto.FLOAT16
    )
    graph = helper.make_graph(
        [gelu, boundary],
        "existing_fp16_boundary",
        [x],
        [output],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _build_missing_metadata_equal_consumer_model() -> ModelProto:
    """Build a blocked uninferred output coupled to a converted sibling input."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    other = helper.make_tensor_value_info("other", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [1])
    gelu = helper.make_node("Gelu", ["x"], ["hidden"], name="gelu", domain="com.microsoft")
    equal = helper.make_node("Equal", ["hidden", "other"], ["output"], name="equal")
    graph = helper.make_graph(
        [gelu, equal],
        "missing_metadata_equal_consumer",
        [x, other],
        [output],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _build_missing_metadata_sequence_consumer_model() -> ModelProto:
    """Build a blocked uninferred tensor feeding a FLOAT sequence output."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, [1])
    gelu = helper.make_node("Gelu", ["x"], ["hidden"], name="gelu", domain="com.microsoft")
    sequence = helper.make_node("SequenceConstruct", ["hidden"], ["output"], name="sequence")
    graph = helper.make_graph(
        [gelu, sequence],
        "missing_metadata_sequence_consumer",
        [x],
        [output],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _build_missing_metadata_sequence_map_consumer_model() -> ModelProto:
    """Build a blocked tensor feeding a SequenceMap additional input."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    sequence = helper.make_tensor_sequence_value_info("sequence", TensorProto.FLOAT, [1])
    output = helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, [1])
    element = helper.make_tensor_value_info("element", TensorProto.FLOAT, [1])
    additional = helper.make_tensor_value_info("additional", TensorProto.FLOAT, [1])
    mapped = helper.make_tensor_value_info("mapped", TensorProto.FLOAT, [1])
    body = helper.make_graph(
        [helper.make_node("Add", ["element", "additional"], ["mapped"])],
        "body",
        [element, additional],
        [mapped],
    )
    gelu = helper.make_node("Gelu", ["x"], ["hidden"], name="gelu", domain="com.microsoft")
    sequence_map = helper.make_node(
        "SequenceMap",
        ["sequence", "hidden"],
        ["output"],
        name="sequence_map",
        body=body,
    )
    graph = helper.make_graph(
        [gelu, sequence_map],
        "missing_metadata_sequence_map_consumer",
        [x, sequence],
        [output],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _build_int_sequence_map_consumer_model() -> ModelProto:
    """Build a blocked uninferred INT output used by SequenceMap."""
    text = helper.make_tensor_value_info("text", TensorProto.STRING, [1])
    sequence = helper.make_tensor_sequence_value_info("sequence", TensorProto.INT32, [1])
    output = helper.make_tensor_sequence_value_info("output", TensorProto.INT32, [1])
    element = helper.make_tensor_value_info("element", TensorProto.INT32, [1])
    additional = helper.make_tensor_value_info("additional", TensorProto.INT32, [1])
    mapped = helper.make_tensor_value_info("mapped", TensorProto.INT32, [1])
    body = helper.make_graph(
        [helper.make_node("Add", ["element", "additional"], ["mapped"])],
        "body",
        [element, additional],
        [mapped],
    )
    murmur = helper.make_node(
        "MurmurHash3",
        ["text"],
        ["hidden"],
        name="murmur",
        domain="com.microsoft",
        positive=0,
    )
    sequence_map = helper.make_node(
        "SequenceMap",
        ["sequence", "hidden"],
        ["output"],
        name="sequence_map",
        body=body,
    )
    graph = helper.make_graph(
        [murmur, sequence_map],
        "int_sequence_map_consumer",
        [text, sequence],
        [output],
        value_info=[helper.make_tensor_value_info("hidden", TensorProto.INT32, [1])],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _build_int_identity_consumer_model() -> ModelProto:
    """Build a blocked INT output consumed by a same-type schema edge."""
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 2, 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 2])
    selected = helper.make_tensor_value_info("selected", TensorProto.INT64, [None, 3])
    output = helper.make_tensor_value_info("output", TensorProto.INT64, [None, 3])
    max_output = numpy_helper.from_array(np.array(2, dtype=np.int64), "max_output")
    nms = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output"],
        ["selected"],
        name="nms",
    )
    identity = helper.make_node("Identity", ["selected"], ["output"], name="identity")
    graph = helper.make_graph(
        [nms, identity],
        "int_identity_consumer",
        [boxes, scores],
        [output],
        [max_output],
        value_info=[selected],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_uninferred_int_identity_consumer_model() -> ModelProto:
    """Build an uninferred INT output with a concrete same-type consumer output."""
    text = helper.make_tensor_value_info("text", TensorProto.STRING, [1])
    output = helper.make_tensor_value_info("output", TensorProto.INT32, [1])
    murmur = helper.make_node(
        "MurmurHash3",
        ["text"],
        ["hidden"],
        name="murmur",
        domain="com.microsoft",
        positive=0,
    )
    identity = helper.make_node("Identity", ["hidden"], ["output"], name="identity")
    graph = helper.make_graph(
        [murmur, identity],
        "uninferred_int_identity_consumer",
        [text],
        [output],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _build_function_int_consumer_model() -> ModelProto:
    """Build a concrete INT edge consumed by a local function without a schema."""
    text = helper.make_tensor_value_info("text", TensorProto.STRING, [1])
    hidden = helper.make_tensor_value_info("hidden", TensorProto.INT32, [1])
    output = helper.make_tensor_value_info("output", TensorProto.INT32, [1])
    murmur = helper.make_node(
        "MurmurHash3",
        ["text"],
        ["hidden"],
        name="murmur",
        domain="com.microsoft",
        positive=0,
    )
    function_node = helper.make_node(
        "IntIdentity",
        ["hidden"],
        ["output"],
        name="function",
        domain="local.test",
    )
    graph = helper.make_graph(
        [murmur, function_node],
        "function_int_consumer",
        [text],
        [output],
        value_info=[hidden],
    )
    function = helper.make_function(
        "local.test",
        "IntIdentity",
        ["input"],
        ["output"],
        [helper.make_node("Identity", ["input"], ["output"])],
        [helper.make_opsetid("", 17)],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
            helper.make_opsetid("local.test", 1),
        ],
    )
    model.functions.append(function)
    return model


def _build_function_with_unrelated_graph_attribute_model() -> ModelProto:
    """Build a local function carrying an unrelated graph-valued attribute."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    attribute_input = helper.make_tensor_value_info("attribute_input", TensorProto.INT32, [1])
    attribute_output = helper.make_tensor_value_info("attribute_input", TensorProto.INT32, [1])
    attribute_graph = helper.make_graph(
        [],
        "unused_attribute",
        [attribute_input],
        [attribute_output],
    )
    gelu = helper.make_node(
        "Gelu",
        ["x"],
        ["hidden"],
        name="gelu",
        domain="com.microsoft",
    )
    function_node = helper.make_node(
        "FloatIdentity",
        ["hidden"],
        ["output"],
        name="function",
        domain="local.test",
        unused_graph=attribute_graph,
    )
    graph = helper.make_graph(
        [gelu, function_node],
        "function_with_unrelated_graph_attribute",
        [x],
        [output],
    )
    function = helper.make_function(
        "local.test",
        "FloatIdentity",
        ["input"],
        ["output"],
        [helper.make_node("Identity", ["input"], ["output"])],
        [helper.make_opsetid("", 17)],
        attributes=["unused_graph"],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
            helper.make_opsetid("local.test", 1),
        ],
    )
    model.functions.append(function)
    return model


def _build_function_with_float_constant_model() -> ModelProto:
    """Build a local function whose concrete FLOAT body is not converted by ORT."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    one = numpy_helper.from_array(np.array([1.0], dtype=np.float32))
    function = helper.make_function(
        "local.test",
        "AddOne",
        ["input"],
        ["function_output"],
        [
            helper.make_node("Constant", [], ["one"], name="one", value=one),
            helper.make_node(
                "Add",
                ["input", "one"],
                ["function_output"],
                name="add",
            ),
        ],
        [helper.make_opsetid("", 17)],
        value_info=[helper.make_tensor_value_info("one", TensorProto.FLOAT, [1])],
    )
    invocation = helper.make_node(
        "AddOne",
        ["x"],
        ["output"],
        name="function",
        domain="local.test",
    )
    graph = helper.make_graph(
        [invocation],
        "function_with_float_constant",
        [x],
        [output],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("local.test", 1),
        ],
    )
    model.functions.append(function)
    return model


def _build_function_with_contrib_float_constant_model() -> ModelProto:
    """Build a concrete FLOAT function body ONNX inference cannot validate."""
    model = _build_function_with_float_constant_model()
    function = model.functions[0]
    function.node[1].CopyFrom(
        helper.make_node(
            "Gelu",
            ["one"],
            ["function_output"],
            name="gelu",
            domain="com.microsoft",
        )
    )
    function.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    model.graph.name = "function_with_contrib_float_constant"
    return model


def _build_function_with_scalar_contrib_float_model() -> ModelProto:
    """Build scalar FLOAT storage before an uninferred function-body op."""
    model = _build_function_with_contrib_float_constant_model()
    function = model.functions[0]
    function.node[0].CopyFrom(
        helper.make_node(
            "Constant",
            [],
            ["one"],
            name="one",
            value_float=1.0,
        )
    )
    del function.value_info[:]
    model.graph.name = "function_with_scalar_contrib_float"
    return model


def _build_always_float_function_graph_attribute_model() -> ModelProto:
    """Build an always-float op type with a graph attribute ORT skips."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    local_input = helper.make_tensor_value_info("local_input", TensorProto.FLOAT, [1])
    local_output = helper.make_tensor_value_info("local_output", TensorProto.FLOAT, [1])
    attribute_graph = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["local_input"],
                ["local_output"],
                name="local_identity",
            )
        ],
        "unused_attribute",
        [local_input],
        [local_output],
    )
    function_node = helper.make_node(
        "GroupNorm",
        ["x"],
        ["output"],
        name="function",
        domain="local.test",
        unused_graph=attribute_graph,
    )
    graph = helper.make_graph(
        [function_node],
        "always_float_function_graph_attribute",
        [x],
        [output],
    )
    function = helper.make_function(
        "local.test",
        "GroupNorm",
        ["input"],
        ["output"],
        [helper.make_node("Identity", ["input"], ["output"])],
        [helper.make_opsetid("", 17)],
        attributes=["unused_graph"],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("local.test", 1),
        ],
    )
    model.functions.append(function)
    return model


def _build_always_float_function_capture_model() -> ModelProto:
    """Build executed skipped attributes that capture a converted initializer."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    converted = helper.make_tensor_value_info("converted", TensorProto.FLOAT, [1])
    selected = helper.make_tensor_value_info("selected", TensorProto.FLOAT, [1])
    shared = numpy_helper.from_array(np.array([2.0], dtype=np.float32), "shared")

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        identity = helper.make_node(
            "Identity",
            ["shared"],
            [branch_output.name],
            name=f"{name}_identity",
        )
        return helper.make_graph(
            [identity],
            name,
            [],
            [branch_output],
            value_info=[helper.make_tensor_value_info("shared", TensorProto.FLOAT, [1])],
        )

    function_if = helper.make_node("If", ["condition"], ["output"], name="if")
    then_branch = helper.make_attribute_ref("then_branch", AttributeProto.GRAPH)
    then_branch.ref_attr_name = "then_branch"
    else_branch = helper.make_attribute_ref("else_branch", AttributeProto.GRAPH)
    else_branch.ref_attr_name = "else_branch"
    function_if.attribute.extend([then_branch, else_branch])
    function = helper.make_function(
        "local.test",
        "GroupNorm",
        ["condition"],
        ["output"],
        [function_if],
        [helper.make_opsetid("", 17)],
        attributes=["then_branch", "else_branch"],
    )
    function_node = helper.make_node(
        "GroupNorm",
        ["condition"],
        ["selected"],
        name="function",
        domain="local.test",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    identity = helper.make_node("Identity", ["shared"], ["converted"], name="identity")
    graph = helper.make_graph(
        [identity, function_node],
        "always_float_function_capture",
        [condition],
        [converted, selected],
        [shared],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("local.test", 1),
        ],
    )
    model.functions.append(function)
    return model


def _build_always_float_nested_function_capture_model() -> ModelProto:
    """Nest referenced invocation attributes inside a function body graph."""
    model = _build_always_float_function_capture_model()
    function = model.functions[0]
    invocation = next(node for node in model.graph.node if node.name == "function")
    for attribute in invocation.attribute:
        if attribute.type != AttributeProto.GRAPH:
            continue
        attribute.g.node[0].CopyFrom(
            helper.make_node(
                "IsNaN",
                ["shared"],
                [attribute.g.output[0].name],
                name=f"{attribute.g.name}_is_nan",
            )
        )
        attribute.g.output[0].type.tensor_type.elem_type = TensorProto.BOOL
    model.graph.output[1].type.tensor_type.elem_type = TensorProto.BOOL

    def _outer_branch(name: str) -> GraphProto:
        branch_output = helper.make_empty_tensor_value_info(f"{name}_output")
        inner_if = helper.make_node(
            "If",
            ["condition"],
            [branch_output.name],
            name=f"{name}_inner_if",
        )
        then_branch = helper.make_attribute_ref("then_branch", AttributeProto.GRAPH)
        then_branch.ref_attr_name = "then_branch"
        else_branch = helper.make_attribute_ref("else_branch", AttributeProto.GRAPH)
        else_branch.ref_attr_name = "else_branch"
        inner_if.attribute.extend([then_branch, else_branch])
        return helper.make_graph([inner_if], name, [], [branch_output])

    function.node[0].CopyFrom(
        helper.make_node(
            "If",
            ["condition"],
            ["output"],
            name="outer_if",
            then_branch=_outer_branch("outer_then"),
            else_branch=_outer_branch("outer_else"),
        )
    )
    model.graph.name = "always_float_nested_function_capture"
    return model


def _build_always_float_default_function_capture_model() -> ModelProto:
    """Supply executed function graph attributes through default values."""
    model = _build_always_float_function_capture_model()
    function = model.functions[0]
    function_node = next(node for node in model.graph.node if node.name == "function")
    function.attribute_proto.extend(function_node.attribute)
    del function_node.attribute[:]
    model.graph.name = "always_float_default_function_capture"
    return model


def _build_always_float_unused_function_capture_model() -> ModelProto:
    """Build an unused function attribute that captures converted FLOAT."""
    model = _build_always_float_function_graph_attribute_model()
    attribute_graph = next(
        attribute.g
        for attribute in model.graph.node[0].attribute
        if attribute.type == AttributeProto.GRAPH
    )
    del attribute_graph.input[:]
    attribute_graph.node[0].input[0] = "shared"
    attribute_graph.value_info.append(
        helper.make_tensor_value_info("shared", TensorProto.FLOAT, [1])
    )
    shared = numpy_helper.from_array(np.array([3.0], dtype=np.float32), "shared")
    converted = helper.make_tensor_value_info("converted", TensorProto.FLOAT, [1])
    model.graph.initializer.append(shared)
    model.graph.node.append(helper.make_node("Relu", ["shared"], ["converted"], name="relu"))
    model.graph.output.append(converted)
    model.graph.name = "always_float_unused_function_capture"
    return model


def _build_skipped_duplicate_initializer_model() -> ModelProto:
    """Build duplicate FLOAT names where ORT skips the nested scope."""
    model = _build_always_float_function_graph_attribute_model()
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), "duplicate")
    )
    attribute_graph = next(
        attribute.g
        for attribute in model.graph.node[0].attribute
        if attribute.type == AttributeProto.GRAPH
    )
    attribute_graph.initializer.append(
        numpy_helper.from_array(np.array([2.0], dtype=np.float32), "duplicate")
    )
    model.graph.name = "skipped_duplicate_initializer"
    return model


def _build_scan8_int_state_model() -> ModelProto:
    """Build Scan-8 with a fixed input before heterogeneous variadic inputs."""
    text = helper.make_tensor_value_info("text", TensorProto.STRING, [1])
    scan_input = helper.make_tensor_value_info("scan_input", TensorProto.FLOAT, [1, 2])
    final_state = helper.make_tensor_value_info("final_state", TensorProto.INT32, [1])
    scan_output = helper.make_tensor_value_info("scan_output", TensorProto.FLOAT, [1, 2])
    body_state = helper.make_tensor_value_info("body_state", TensorProto.INT32, [])
    body_input = helper.make_tensor_value_info("body_input", TensorProto.FLOAT, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.INT32, [])
    body_output = helper.make_tensor_value_info("body_output", TensorProto.FLOAT, [])
    body = helper.make_graph(
        [
            helper.make_node("Identity", ["body_state"], ["body_state_out"]),
            helper.make_node("Identity", ["body_input"], ["body_output"]),
        ],
        "body",
        [body_state, body_input],
        [body_state_out, body_output],
    )
    murmur = helper.make_node(
        "MurmurHash3",
        ["text"],
        ["hidden"],
        name="murmur",
        domain="com.microsoft",
        positive=0,
    )
    scan = helper.make_node(
        "Scan",
        ["", "hidden", "scan_input"],
        ["final_state", "scan_output"],
        name="scan",
        body=body,
        num_scan_inputs=1,
    )
    graph = helper.make_graph(
        [murmur, scan],
        "scan8_int_state",
        [text, scan_input],
        [final_state, scan_output],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 8),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _build_scan8_float_state_model() -> ModelProto:
    """Build Scan-8 with an uninferred FLOAT state requiring a boundary."""
    model = _build_scan8_int_state_model()
    model.graph.node[0].CopyFrom(
        helper.make_node(
            "Gelu",
            ["x"],
            ["hidden"],
            name="gelu",
            domain="com.microsoft",
        )
    )
    model.graph.input[0].CopyFrom(helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]))
    model.graph.output[0].CopyFrom(
        helper.make_tensor_value_info("final_state", TensorProto.FLOAT, [1])
    )
    body = next(
        attribute.g for attribute in model.graph.node[1].attribute if attribute.name == "body"
    )
    body.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
    body.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
    model.graph.name = "scan8_float_state"
    return model


def _build_scan8_without_state_model() -> ModelProto:
    """Build Scan-8 whose only variadic input is a scan input."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    output = helper.make_tensor_value_info("output", TensorProto.INT32, [1, 2])
    body_input = helper.make_empty_tensor_value_info("body_input")
    body_output = helper.make_tensor_value_info("body_output", TensorProto.INT32, [])
    bias = numpy_helper.from_array(np.array(1.0, dtype=np.float32), "bias")
    body = helper.make_graph(
        [
            helper.make_node("Add", ["body_input", "bias"], ["added"]),
            helper.make_node(
                "Cast",
                ["added"],
                ["body_output"],
                to=TensorProto.INT32,
            ),
        ],
        "body",
        [body_input],
        [body_output],
        [bias],
        value_info=[helper.make_tensor_value_info("bias", TensorProto.FLOAT, [])],
    )
    gelu = helper.make_node(
        "Gelu",
        ["x"],
        ["hidden"],
        name="gelu",
        domain="com.microsoft",
    )
    scan = helper.make_node(
        "Scan",
        ["", "hidden"],
        ["output"],
        name="scan",
        body=body,
        num_scan_inputs=1,
    )
    graph = helper.make_graph(
        [gelu, scan],
        "scan8_without_state",
        [x],
        [output],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 8),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _build_loop_float_state_model() -> ModelProto:
    """Build a Loop with an untyped formal and declared FLOAT feedback."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    body_state = helper.make_empty_tensor_value_info("body_state")
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.FLOAT, [1])
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["body_condition"],
                ["body_condition_out"],
            ),
            helper.make_node("Identity", ["body_state"], ["body_state_out"]),
        ],
        "body",
        [iteration, body_condition, body_state],
        [body_condition_out, body_state_out],
    )
    gelu = helper.make_node(
        "Gelu",
        ["x"],
        ["hidden"],
        name="gelu",
        domain="com.microsoft",
    )
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "hidden"],
        ["output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [gelu, loop],
        "loop_float_state",
        [trip_count, condition, x],
        [output],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 11),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _build_optional_float_blocked_capture_model() -> ModelProto:
    """Build a blocked branch that captures an Optional wrapping converted FLOAT."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    if_output = helper.make_tensor_value_info("if_output", TensorProto.FLOAT, [1])
    optional_info = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )
    opt = helper.make_value_info("opt", optional_info)

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        get_element = helper.make_node(
            "OptionalGetElement",
            ["opt"],
            [branch_output.name],
            name=f"{name}_get_element",
        )
        return helper.make_graph([get_element], name, [], [branch_output])

    optional = helper.make_node("Optional", ["x"], ["opt"], name="optional")
    conditional = helper.make_node(
        "If",
        ["condition"],
        ["if_output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    graph = helper.make_graph(
        [optional, conditional],
        "optional_float_blocked_capture",
        [x, condition],
        [if_output],
        value_info=[opt],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_blocked_float_optional_output_model() -> ModelProto:
    """Build a blocked FLOAT output wrapped by an unchanged Optional."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    optional_type = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )
    output = helper.make_value_info("output", optional_type)
    gelu = helper.make_node(
        "Gelu",
        ["x"],
        ["hidden"],
        name="gelu",
        domain="com.microsoft",
    )
    optional = helper.make_node("Optional", ["hidden"], ["output"], name="optional")
    graph = helper.make_graph(
        [gelu, optional],
        "blocked_float_optional_output",
        [x],
        [output],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _build_float_optional_output_model() -> ModelProto:
    """Build converted FLOAT wrapped by unchanged Optional metadata."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    optional_type = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )
    output = helper.make_value_info("output", optional_type)
    optional = helper.make_node("Optional", ["x"], ["output"], name="optional")
    graph = helper.make_graph([optional], "float_optional_output", [x], [output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _build_blocked_tensor_optional_consumer_model() -> ModelProto:
    """Build an inferred blocked tensor output wrapped by an Optional."""
    model = _build_blocked_float_optional_output_model()
    model.graph.node[0].CopyFrom(helper.make_node("Abs", ["x"], ["hidden"], name="abs"))
    model.graph.name = "blocked_tensor_optional_consumer"
    return model


def _build_blocked_float_optional_capture_model() -> ModelProto:
    """Build a blocked child capturing an Optional with an FP32 payload."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    optional_type = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )
    optional_info = helper.make_value_info("optional", optional_type)

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        get_element = helper.make_node(
            "OptionalGetElement",
            ["optional"],
            [branch_output.name],
            name=f"{name}_get_element",
        )
        return helper.make_graph([get_element], name, [], [branch_output])

    gelu = helper.make_node(
        "Gelu",
        ["x"],
        ["hidden"],
        name="gelu",
        domain="com.microsoft",
    )
    optional = helper.make_node("Optional", ["hidden"], ["optional"], name="optional")
    conditional = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    graph = helper.make_graph(
        [gelu, optional, conditional],
        "blocked_float_optional_capture",
        [x, condition],
        [output],
        value_info=[optional_info],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )


def _build_empty_optional_blocked_capture_model() -> ModelProto:
    """Build a blocked branch capturing a source-less empty Optional."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [])
    converted = helper.make_tensor_value_info("converted", TensorProto.FLOAT, [1])
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_info = helper.make_value_info("optional", optional_type)

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.BOOL, [])
        has_element = helper.make_node(
            "OptionalHasElement",
            ["optional"],
            [branch_output.name],
            name=f"{name}_has_element",
        )
        return helper.make_graph([has_element], name, [], [branch_output])

    make_optional = helper.make_node(
        "Optional", [], ["optional"], name="optional", type=tensor_type
    )
    conditional = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    relu = helper.make_node("Relu", ["x"], ["converted"], name="relu")
    graph = helper.make_graph(
        [make_optional, conditional, relu],
        "empty_optional_blocked_capture",
        [condition, x],
        [output, converted],
        value_info=[optional_info],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _build_nested_optional_input_blocked_capture_model() -> ModelProto:
    """Build a blocked capture of a nested Optional formal input."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [])
    optional_type = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    body_optional = helper.make_value_info("body_optional", optional_type)
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_optional_out = helper.make_value_info("body_optional", optional_type)

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.BOOL, [])
        has_element = helper.make_node(
            "OptionalHasElement",
            ["body_optional"],
            [branch_output.name],
            name=f"{name}_has_element",
        )
        return helper.make_graph([has_element], name, [], [branch_output])

    body_if = helper.make_node(
        "If",
        ["body_condition"],
        ["body_if_output"],
        name="body_if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["body_condition"],
                ["body_condition_out"],
                name="body_condition",
            ),
            body_if,
        ],
        "body",
        [iteration, body_condition, body_optional],
        [body_condition_out, body_optional_out],
        value_info=[helper.make_tensor_value_info("body_if_output", TensorProto.BOOL, [])],
    )
    make_optional = helper.make_node("Optional", ["x"], ["optional"], name="make_optional")
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "optional"],
        ["final_optional"],
        name="loop",
        body=body,
    )
    get_output = helper.make_node(
        "OptionalHasElement", ["final_optional"], ["output"], name="get_output"
    )
    graph = helper.make_graph(
        [make_optional, loop, get_output],
        "nested_optional_input_blocked_capture",
        [trip_count, condition, x],
        [output],
        value_info=[
            helper.make_value_info("optional", optional_type),
            helper.make_value_info("final_optional", optional_type),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _build_top_optional_input_blocked_capture_model() -> ModelProto:
    """Build a nested blocked capture sourced from an unchanged top Optional input."""
    model = _build_nested_optional_input_blocked_capture_model()
    optional_type = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )
    del model.graph.node[0]
    del model.graph.input[2]
    model.graph.input.append(helper.make_value_info("optional", optional_type))
    retained = [value for value in model.graph.value_info if value.name != "optional"]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained)
    model.graph.name = "top_optional_input_blocked_capture"
    return model


def _build_identity_optional_feedback_blocked_capture_model() -> ModelProto:
    """Return an unchanged Optional state through a differently named Identity."""
    model = _build_top_optional_input_blocked_capture_model()
    optional_type = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )
    loop = model.graph.node[0]
    body = next(attribute.g for attribute in loop.attribute if attribute.g.name)
    body.output[1].CopyFrom(helper.make_value_info("body_optional_out", optional_type))
    body.node.append(
        helper.make_node(
            "Identity",
            ["body_optional"],
            ["body_optional_out"],
            name="optional_feedback",
        )
    )
    model.graph.name = "identity_optional_feedback_blocked_capture"
    return model


def _build_rewrapped_optional_feedback_blocked_capture_model() -> ModelProto:
    """Rewrap a converted FLOAT element as Optional loop feedback."""
    model = _build_top_optional_input_blocked_capture_model()
    optional_type = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )
    loop = model.graph.node[0]
    body = next(attribute.g for attribute in loop.attribute if attribute.g.name)
    body.output[1].CopyFrom(helper.make_value_info("body_optional_out", optional_type))
    body.node.extend(
        [
            helper.make_node(
                "OptionalGetElement",
                ["body_optional"],
                ["feedback_element"],
                name="unwrap_feedback",
            ),
            helper.make_node(
                "Optional",
                ["feedback_element"],
                ["body_optional_out"],
                name="rewrap_feedback",
            ),
        ]
    )
    body.value_info.append(
        helper.make_tensor_value_info("feedback_element", TensorProto.FLOAT, [1])
    )
    model.graph.name = "rewrapped_optional_feedback_blocked_capture"
    return model


def _build_loop_optional_output_blocked_capture_model() -> ModelProto:
    """Pass an Optional through Loop before a blocked capture."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [])
    converted = helper.make_tensor_value_info("converted", TensorProto.FLOAT, [1])
    optional_type = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )
    optional = helper.make_value_info("optional", optional_type)
    final_optional = helper.make_value_info("final_optional", optional_type)
    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    body_optional = helper.make_value_info("body_optional", optional_type)
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_optional_out = helper.make_value_info("body_optional_out", optional_type)
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["body_condition"],
                ["body_condition_out"],
                name="condition_feedback",
            ),
            helper.make_node(
                "Identity",
                ["body_optional"],
                ["body_optional_out"],
                name="optional_feedback",
            ),
        ],
        "body",
        [iteration, body_condition, body_optional],
        [body_condition_out, body_optional_out],
    )

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.BOOL, [])
        has_element = helper.make_node(
            "OptionalHasElement",
            ["final_optional"],
            [branch_output.name],
            name=f"{name}_has_element",
        )
        return helper.make_graph([has_element], name, [], [branch_output])

    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "optional"],
        ["final_optional"],
        name="loop",
        body=body,
    )
    conditional = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    relu = helper.make_node("Relu", ["x"], ["converted"], name="relu")
    graph = helper.make_graph(
        [loop, conditional, relu],
        "loop_optional_output_blocked_capture",
        [trip_count, condition, optional, x],
        [output, converted],
        value_info=[final_optional],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _build_optional_identity_blocked_capture_model() -> ModelProto:
    """Build an unchanged Optional pass-through captured by a blocked child."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    optional_type = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )
    optional = helper.make_value_info("optional", optional_type)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [])
    converted = helper.make_tensor_value_info("converted", TensorProto.FLOAT, [1])
    passed_optional = helper.make_value_info("passed_optional", optional_type)

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.BOOL, [])
        has_element = helper.make_node(
            "OptionalHasElement",
            ["passed_optional"],
            [branch_output.name],
            name=f"{name}_has_element",
        )
        return helper.make_graph([has_element], name, [], [branch_output])

    identity = helper.make_node(
        "Identity",
        ["optional"],
        ["passed_optional"],
        name="identity",
    )
    conditional = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    relu = helper.make_node("Relu", ["x"], ["converted"], name="relu")
    graph = helper.make_graph(
        [identity, conditional, relu],
        "optional_identity_blocked_capture",
        [condition, optional, x],
        [output, converted],
        value_info=[passed_optional],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _build_long_optional_identity_chain_model(
    length: int = 1010,
) -> ModelProto:
    """Build a long precision-preserving Optional producer chain."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    optional_type = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )
    optional = helper.make_value_info("optional", optional_type)
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [])
    nodes = []
    value_info = []
    source = "optional"
    for index in range(length):
        target = f"optional_{index}"
        nodes.append(
            helper.make_node(
                "Identity",
                [source],
                [target],
                name=f"identity_{index}",
            )
        )
        value_info.append(helper.make_value_info(target, optional_type))
        source = target

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.BOOL, [])
        has_element = helper.make_node(
            "OptionalHasElement",
            [source],
            [branch_output.name],
            name=f"{name}_has_element",
        )
        return helper.make_graph([has_element], name, [], [branch_output])

    nodes.append(
        helper.make_node(
            "If",
            ["condition"],
            ["output"],
            name="if",
            then_branch=_branch("then"),
            else_branch=_branch("else"),
        )
    )
    graph = helper.make_graph(
        nodes,
        "long_optional_identity_chain",
        [condition, optional],
        [output],
        value_info=value_info,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _build_free_optional_identity_blocked_capture_model() -> ModelProto:
    """Build a nested Identity sourced from a converted outer Optional."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    state = helper.make_tensor_value_info("state", TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [])
    optional_type = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )
    optional = helper.make_value_info("optional", optional_type)

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    body_state = helper.make_tensor_value_info("body_state", TensorProto.BOOL, [])
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.BOOL, [])
    passed_optional = helper.make_value_info("passed_optional", optional_type)

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.BOOL, [])
        has_element = helper.make_node(
            "OptionalHasElement",
            ["passed_optional"],
            [branch_output.name],
            name=f"{name}_has_element",
        )
        return helper.make_graph([has_element], name, [], [branch_output])

    body_if = helper.make_node(
        "If",
        ["body_state"],
        ["body_state_out"],
        name="body_if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["optional"],
                ["passed_optional"],
                name="optional_identity",
            ),
            helper.make_node(
                "Identity",
                ["body_condition"],
                ["body_condition_out"],
                name="condition_identity",
            ),
            body_if,
        ],
        "body",
        [iteration, body_condition, body_state],
        [body_condition_out, body_state_out],
        value_info=[passed_optional],
    )
    make_optional = helper.make_node("Optional", ["x"], ["optional"], name="make_optional")
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "state"],
        ["output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [make_optional, loop],
        "free_optional_identity_blocked_capture",
        [trip_count, condition, state, x],
        [output],
        value_info=[optional],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _build_optional_feedback_blocked_capture_model() -> ModelProto:
    """Build a nested Optional input changed by a later loop-carried value."""
    model = _build_top_optional_input_blocked_capture_model()
    optional_type = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )
    model.graph.input.append(helper.make_tensor_value_info("feedback", TensorProto.FLOAT, [1]))
    loop = model.graph.node[0]
    body = next(attribute.g for attribute in loop.attribute if attribute.g.name)
    body.output[1].CopyFrom(helper.make_value_info("body_optional_out", optional_type))
    body.node.append(
        helper.make_node(
            "Optional",
            ["feedback"],
            ["body_optional_out"],
            name="make_feedback_optional",
        )
    )
    model.graph.name = "optional_feedback_blocked_capture"
    return model


def _build_mispositioned_optional_feedback_model() -> ModelProto:
    """Build two loop states with unchanged binding returned in the wrong slot."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    optional_type = helper.make_optional_type_proto(
        helper.make_tensor_type_proto(TensorProto.FLOAT, [1])
    )
    optional_a = helper.make_value_info("optional_a", optional_type)
    optional_b = helper.make_value_info("optional_b", optional_type)
    feedback = helper.make_tensor_value_info("feedback", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.BOOL, [])

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    body_a = helper.make_value_info("body_a", optional_type)
    body_b = helper.make_value_info("body_b", optional_type)
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    converted_a = helper.make_value_info("converted_a", optional_type)
    returned_b = helper.make_value_info("body_a", optional_type)

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.BOOL, [])
        has_element = helper.make_node(
            "OptionalHasElement",
            ["body_a"],
            [branch_output.name],
            name=f"{name}_has_element",
        )
        return helper.make_graph([has_element], name, [], [branch_output])

    body_if = helper.make_node(
        "If",
        ["body_condition"],
        ["body_if_output"],
        name="body_if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity",
                ["body_condition"],
                ["body_condition_out"],
                name="body_condition",
            ),
            body_if,
            helper.make_node(
                "Optional",
                ["feedback"],
                ["converted_a"],
                name="make_converted_a",
            ),
        ],
        "body",
        [iteration, body_condition, body_a, body_b],
        [body_condition_out, converted_a, returned_b],
        value_info=[helper.make_tensor_value_info("body_if_output", TensorProto.BOOL, [])],
    )
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "optional_a", "optional_b"],
        ["final_a", "final_b"],
        name="loop",
        body=body,
    )
    has_output = helper.make_node("OptionalHasElement", ["final_a"], ["output"], name="has_output")
    graph = helper.make_graph(
        [loop, has_output],
        "mispositioned_optional_feedback",
        [trip_count, condition, optional_a, optional_b, feedback],
        [output],
        value_info=[
            helper.make_value_info("final_a", optional_type),
            helper.make_value_info("final_b", optional_type),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _build_scan_mispositioned_optional_feedback_model() -> ModelProto:
    """Add a scan output to the wrong-slot optional feedback model."""
    model = _build_mispositioned_optional_feedback_model()
    loop = model.graph.node[0]
    loop.output.append("scan_output")
    body = next(attribute.g for attribute in loop.attribute if attribute.g.name)
    body.output.append(helper.make_tensor_value_info("body_if_output", TensorProto.BOOL, []))
    model.graph.name = "scan_mispositioned_optional_feedback"
    return model


def _build_scan_top_optional_input_blocked_capture_model() -> ModelProto:
    """Add a scan output without changing the pass-through loop state."""
    model = _build_top_optional_input_blocked_capture_model()
    loop = model.graph.node[0]
    loop.output.append("scan_output")
    body = next(attribute.g for attribute in loop.attribute if attribute.g.name)
    body.output.append(helper.make_tensor_value_info("body_if_output", TensorProto.BOOL, []))
    model.graph.name = "scan_top_optional_input_blocked_capture"
    return model


def _build_blocked_float_sequence_input_model() -> ModelProto:
    """Build a blocked node consuming a sequence of FLOAT tensors."""
    sequence = helper.make_tensor_sequence_value_info("sequence", TensorProto.FLOAT, [1])
    length = helper.make_tensor_value_info("length", TensorProto.INT64, [])
    node = helper.make_node("SequenceLength", ["sequence"], ["length"], name="sequence_length")
    graph = helper.make_graph([node], "blocked_float_sequence_input", [sequence], [length])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_float_sequence_io_model() -> ModelProto:
    """Build a FLOAT tensor sequence exposed directly as graph I/O."""
    sequence_input = helper.make_tensor_sequence_value_info("sequence", TensorProto.FLOAT, [1])
    sequence_output = helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, [1])
    identity = helper.make_node("Identity", ["sequence"], ["output"], name="identity")
    graph = helper.make_graph(
        [identity],
        "float_sequence_io",
        [sequence_input],
        [sequence_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_overridable_blocked_initializer_model() -> ModelProto:
    """Build a graph-input initializer consumed only by a blocked node."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    initializer = numpy_helper.from_array(np.array([2.0], dtype=np.float32), "x")
    blocked = helper.make_node("Identity", ["x"], ["output"], name="blocked")
    graph = helper.make_graph(
        [blocked], "overridable_blocked_initializer", [x], [output], [initializer]
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_duplicate_float_output_name_model() -> ModelProto:
    """Build a legal graph that exposes one FLOAT value through two outputs."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    first = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    second = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    identity = helper.make_node("Identity", ["x"], ["y"], name="identity")
    graph = helper.make_graph([identity], "duplicate_float_output", [x], [first, second])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_shared_float_input_output_model() -> ModelProto:
    """Build a direct FLOAT pass-through with one public input/output name."""
    graph_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    graph_output = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    graph = helper.make_graph([], "shared_float_io", [graph_input], [graph_output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_nested_shadowed_initializer_output_model() -> ModelProto:
    """Build legal nested outputs that shadow an outer initializer name."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    outer = numpy_helper.from_array(np.array([9.0], dtype=np.float32), "same")

    def _branch(name: str, value: float) -> GraphProto:
        branch_output = helper.make_tensor_value_info("same", TensorProto.FLOAT, [1])
        initializer = numpy_helper.from_array(np.array([value], dtype=np.float32), "same")
        return helper.make_graph([], name, [], [branch_output], [initializer])

    node = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="if",
        then_branch=_branch("then", 1.0),
        else_branch=_branch("else", 2.0),
    )
    graph = helper.make_graph([node], "shadow", [condition], [output], [outer])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_blocked_if_duplicate_local_initializer_model() -> ModelProto:
    """Build blocked If branches with duplicate local FLOAT initializer names."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    def _branch(name: str, value: float) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        initializer = numpy_helper.from_array(np.array([value], dtype=np.float32), "same")
        identity = helper.make_node(
            "Identity", ["same"], [f"{name}_output"], name=f"{name}_identity"
        )
        return helper.make_graph([identity], name, [], [branch_output], [initializer])

    node = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="if",
        then_branch=_branch("then", 1.0),
        else_branch=_branch("else", 2.0),
    )
    graph = helper.make_graph([node], "blocked_if_duplicate_local", [condition], [output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_blocked_if_shadowed_output_initializer_model() -> ModelProto:
    """Build a top-level output initializer shadowed by blocked branch-local values."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    shared = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [1])
    initializer = numpy_helper.from_array(np.array([9.0], dtype=np.float32), "shared")

    def _branch(name: str, value: float) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        local = numpy_helper.from_array(np.array([value], dtype=np.float32), "shared")
        identity = helper.make_node(
            "Identity", ["shared"], [f"{name}_output"], name=f"{name}_identity"
        )
        return helper.make_graph([identity], name, [], [branch_output], [local])

    node = helper.make_node(
        "If",
        ["condition"],
        ["if_output"],
        name="if",
        then_branch=_branch("then", 1.0),
        else_branch=_branch("else", 2.0),
    )
    graph = helper.make_graph([node], "blocked_if_shadowed", [condition], [shared], [initializer])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_traversed_shadowed_output_initializer_model() -> ModelProto:
    """Build traversed Loop body names that shadow a top-level output initializer."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    loop_carried = helper.make_tensor_value_info("loop_carried", TensorProto.FLOAT, [1])
    shared = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [1])
    loop_output = helper.make_tensor_value_info("loop_output", TensorProto.FLOAT, [1])
    initializer = numpy_helper.from_array(np.array([9.0], dtype=np.float32), "shared")

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    local_shared = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [1])
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_value_out = helper.make_tensor_value_info("body_value_out", TensorProto.FLOAT, [1])
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity", ["body_condition"], ["body_condition_out"], name="body_condition"
            ),
            helper.make_node("Identity", ["shared"], ["body_value_out"], name="body_value"),
        ],
        "body",
        [iteration, body_condition, local_shared],
        [body_condition_out, body_value_out],
    )
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "loop_carried"],
        ["loop_output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [loop],
        "traversed_shadowed",
        [trip_count, condition, loop_carried],
        [shared, loop_output],
        [initializer],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_loop_state_input_shadowing_output_initializer_model() -> ModelProto:
    """Build a Loop whose local state input shadows a top-level output initializer."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    loop_state = helper.make_tensor_value_info("loop_state", TensorProto.FLOAT, [1])
    same = helper.make_tensor_value_info("same", TensorProto.FLOAT, [1])
    loop_output = helper.make_tensor_value_info("loop_output", TensorProto.FLOAT, [1])
    initializer = numpy_helper.from_array(np.array([9.0], dtype=np.float32), "same")

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    body_state = helper.make_tensor_value_info("same", TensorProto.FLOAT, [1])
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.FLOAT, [1])
    body_initializer = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "body_state_out")
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity", ["body_condition"], ["body_condition_out"], name="body_condition"
            )
        ],
        "body",
        [iteration, body_condition, body_state],
        [body_condition_out, body_state_out],
        [body_initializer],
    )
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "loop_state"],
        ["loop_output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [loop],
        "loop_state_input_shadowing",
        [trip_count, condition, loop_state],
        [same, loop_output],
        [initializer],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_loop_value_info_shadowing_output_initializer_model() -> ModelProto:
    """Build a Loop whose local value_info shadows a top-level output initializer."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    loop_state = helper.make_tensor_value_info("loop_state", TensorProto.FLOAT, [1])
    same = helper.make_tensor_value_info("same", TensorProto.FLOAT, [1])
    loop_output = helper.make_tensor_value_info("loop_output", TensorProto.FLOAT, [1])
    initializer = numpy_helper.from_array(np.array([9.0], dtype=np.float32), "same")

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    body_state = helper.make_tensor_value_info("body_state", TensorProto.FLOAT, [1])
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.FLOAT, [1])
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity", ["body_condition"], ["body_condition_out"], name="body_condition"
            ),
            helper.make_node("Identity", ["body_state"], ["body_state_out"], name="body_state"),
        ],
        "body",
        [iteration, body_condition, body_state],
        [body_condition_out, body_state_out],
    )
    body.value_info.append(helper.make_tensor_value_info("same", TensorProto.FLOAT, [1]))
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "loop_state"],
        ["loop_output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [loop],
        "loop_value_info_shadowing",
        [trip_count, condition, loop_state],
        [same, loop_output],
        [initializer],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_nested_output_initializer_with_top_level_io_name_collision_model() -> ModelProto:
    """Build a nested direct initializer output matching an unrelated top-level input."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    loop_state = helper.make_tensor_value_info("loop_state", TensorProto.FLOAT, [1])
    same = helper.make_tensor_value_info("same", TensorProto.FLOAT, [1])
    loop_output = helper.make_tensor_value_info("loop_output", TensorProto.FLOAT, [1])

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    body_state = helper.make_tensor_value_info("body_state", TensorProto.FLOAT, [1])
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_output = helper.make_tensor_value_info("same", TensorProto.FLOAT, [1])
    body_initializer = numpy_helper.from_array(np.array([7.0], dtype=np.float32), "same")
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity", ["body_condition"], ["body_condition_out"], name="body_condition"
            )
        ],
        "body",
        [iteration, body_condition, body_state],
        [body_condition_out, body_output],
        [body_initializer],
    )
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "loop_state"],
        ["loop_output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [loop],
        "nested_io_name_collision",
        [trip_count, condition, loop_state, same],
        [loop_output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_nested_initializer_then_mapped_free_capture_model() -> ModelProto:
    """Build a nested initializer followed by a mapped capture of kept top-level I/O."""
    model = _build_nested_output_initializer_with_top_level_io_name_collision_model()
    captured_output = helper.make_tensor_value_info("captured_output", TensorProto.FLOAT, [1])

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        identity = helper.make_node(
            "Identity", ["same"], [f"{name}_output"], name=f"{name}_identity"
        )
        return helper.make_graph([identity], name, [], [branch_output])

    conditional = helper.make_node(
        "If",
        ["condition"],
        ["captured_output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    model.graph.node.append(conditional)
    model.graph.output.append(captured_output)
    return model


def _build_mapped_capture_shadowed_by_generated_alias_model() -> ModelProto:
    """Build a mapped free capture whose target alias has a nested local binding."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    def _branch(name: str, *, shadow_alias: bool) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        identity = helper.make_node("Identity", ["x"], [f"{name}_output"], name=f"{name}_identity")
        initializers = (
            [numpy_helper.from_array(np.array([9.0], dtype=np.float32), "graph_input_cast_0")]
            if shadow_alias
            else []
        )
        return helper.make_graph([identity], name, [], [branch_output], initializer=initializers)

    conditional = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        name="if",
        then_branch=_branch("then", shadow_alias=True),
        else_branch=_branch("else", shadow_alias=False),
    )
    graph = helper.make_graph(
        [conditional],
        "mapped_capture_shadowed_by_alias",
        [x, condition],
        [output],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_nested_blocked_ordinary_name_collision_model() -> ModelProto:
    """Build nested blocked input shadowing an earlier global value-info name."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    same = helper.make_tensor_value_info("same", TensorProto.FLOAT, [])
    selected = helper.make_tensor_value_info("selected", TensorProto.INT64, [None, 3])

    def _branch(name: str, threshold_name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(
            f"{name}_selected", TensorProto.INT64, [None, 3]
        )
        boxes = numpy_helper.from_array(
            np.array([[[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]], dtype=np.float32),
            f"{name}_boxes",
        )
        scores = numpy_helper.from_array(
            np.array([[[0.9, 0.8]]], dtype=np.float32), f"{name}_scores"
        )
        max_output = numpy_helper.from_array(np.array(2, dtype=np.int64), f"{name}_max_output")
        iou_threshold = numpy_helper.from_array(
            np.array(0.5, dtype=np.float32), f"{name}_iou_threshold"
        )
        score_threshold = numpy_helper.from_array(np.array(0.85, dtype=np.float32), threshold_name)
        nms = helper.make_node(
            "NonMaxSuppression",
            [
                boxes.name,
                scores.name,
                max_output.name,
                iou_threshold.name,
                score_threshold.name,
            ],
            [branch_output.name],
            name=f"{name}_nms",
        )
        return helper.make_graph(
            [nms],
            name,
            [],
            [branch_output],
            initializer=[
                boxes,
                scores,
                max_output,
                iou_threshold,
                score_threshold,
            ],
        )

    conditional = helper.make_node(
        "If",
        ["condition"],
        ["selected"],
        name="if",
        then_branch=_branch("then", "same"),
        else_branch=_branch("else", "else_score_threshold"),
    )
    graph = helper.make_graph(
        [conditional],
        "nested_blocked_ordinary_collision",
        [condition, same],
        [selected],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_blocked_if_free_capture_output_initializer_model() -> ModelProto:
    """Build a blocked If branch that free-captures a top-level output initializer."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    shared = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [1])
    if_output = helper.make_tensor_value_info("if_output", TensorProto.FLOAT, [1])
    initializer = numpy_helper.from_array(np.array([9.0], dtype=np.float32), "shared")

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        identity = helper.make_node(
            "Identity", ["shared"], [f"{name}_output"], name=f"{name}_identity"
        )
        return helper.make_graph([identity], name, [], [branch_output])

    node = helper.make_node(
        "If",
        ["condition"],
        ["if_output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    graph = helper.make_graph(
        [node],
        "blocked_if_free_capture",
        [condition],
        [shared, if_output],
        [initializer],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_blocked_if_shadowed_free_capture_model() -> ModelProto:
    """Build blocked branches that capture a local shadow, not the top-level initializer."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    loop_state = helper.make_tensor_value_info("loop_state", TensorProto.FLOAT, [1])
    shared = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [1])
    loop_output = helper.make_tensor_value_info("loop_output", TensorProto.FLOAT, [1])
    top_initializer = numpy_helper.from_array(np.array([9.0], dtype=np.float32), "shared")

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    body_state = helper.make_tensor_value_info("body_state", TensorProto.FLOAT, [1])
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.FLOAT, [1])
    local_shared = numpy_helper.from_array(np.array([7.0], dtype=np.float16), "shared")

    def _branch(name: str) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT16, [1])
        identity = helper.make_node(
            "Identity", ["shared"], [f"{name}_output"], name=f"{name}_identity"
        )
        return helper.make_graph([identity], name, [], [branch_output])

    blocked_if = helper.make_node(
        "If",
        ["body_condition"],
        ["if_output"],
        name="if",
        then_branch=_branch("then"),
        else_branch=_branch("else"),
    )
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity", ["body_condition"], ["body_condition_out"], name="body_condition"
            ),
            helper.make_node("Identity", ["body_state"], ["body_state_out"], name="body_state"),
            blocked_if,
        ],
        "body",
        [iteration, body_condition, body_state],
        [body_condition_out, body_state_out],
        [local_shared],
    )
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "loop_state"],
        ["loop_output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [loop],
        "blocked_shadowed_capture",
        [trip_count, condition, loop_state],
        [shared, loop_output],
        [top_initializer],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_duplicate_non_output_initializer_name_model() -> ModelProto:
    """Build duplicate initializer names across scopes without initializer-backed outputs."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    top_output = helper.make_tensor_value_info("top_output", TensorProto.FLOAT, [1])
    nested_output = helper.make_tensor_value_info("nested_output", TensorProto.FLOAT, [1])
    outer = numpy_helper.from_array(np.array([9.0], dtype=np.float32), "same")
    top_identity = helper.make_node("Identity", ["same"], ["top_output"], name="top_identity")

    def _branch(name: str, value: float) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        initializer = numpy_helper.from_array(np.array([value], dtype=np.float32), "same")
        identity = helper.make_node(
            "Identity", ["same"], [f"{name}_output"], name=f"{name}_identity"
        )
        return helper.make_graph([identity], name, [], [branch_output], [initializer])

    node = helper.make_node(
        "If",
        ["condition"],
        ["nested_output"],
        name="if",
        then_branch=_branch("then", 1.0),
        else_branch=_branch("else", 2.0),
    )
    graph = helper.make_graph(
        [top_identity, node],
        "duplicate_non_output",
        [condition],
        [top_output, nested_output],
        [outer],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_blocked_initializer_consumer_model() -> ModelProto:
    """Build an output initializer consumed only by an FP32-blocked node."""
    shared = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [1])
    copied = helper.make_tensor_value_info("copied", TensorProto.FLOAT, [1])
    initializer = numpy_helper.from_array(np.array([1.0001], dtype=np.float32), "shared")
    identity = helper.make_node("Identity", ["shared"], ["copied"], name="identity")
    graph = helper.make_graph([identity], "blocked_consumer", [], [shared, copied], [initializer])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_blocked_initializer_with_nested_input_shadow_model() -> ModelProto:
    """Build a blocked initializer consumer plus an unrelated nested local shadow."""
    trip_count = helper.make_tensor_value_info("trip_count", TensorProto.INT64, [])
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    loop_state = helper.make_tensor_value_info("loop_state", TensorProto.FLOAT, [1])
    blocked_output = helper.make_tensor_value_info("blocked_output", TensorProto.FLOAT, [1])
    loop_output = helper.make_tensor_value_info("loop_output", TensorProto.FLOAT, [1])
    initializer = numpy_helper.from_array(np.array([1.0001], dtype=np.float32), "shared")

    iteration = helper.make_tensor_value_info("iteration", TensorProto.INT64, [])
    body_condition = helper.make_tensor_value_info("body_condition", TensorProto.BOOL, [])
    local_shared = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [1])
    body_condition_out = helper.make_tensor_value_info("body_condition_out", TensorProto.BOOL, [])
    body_state_out = helper.make_tensor_value_info("body_state_out", TensorProto.FLOAT, [1])
    body = helper.make_graph(
        [
            helper.make_node(
                "Identity", ["body_condition"], ["body_condition_out"], name="body_condition"
            ),
            helper.make_node("Relu", ["shared"], ["body_state_out"], name="body_relu"),
        ],
        "body",
        [iteration, body_condition, local_shared],
        [body_condition_out, body_state_out],
    )
    blocked = helper.make_node("Abs", ["shared"], ["blocked_output"], name="blocked_abs")
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "loop_state"],
        ["loop_output"],
        name="loop",
        body=body,
    )
    graph = helper.make_graph(
        [blocked, loop],
        "blocked_initializer_nested_shadow",
        [trip_count, condition, loop_state],
        [blocked_output, loop_output],
        [initializer],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _build_unused_initializer_with_nested_input_shadow_model() -> ModelProto:
    """Build an unused initializer shadowed by a consumed nested formal."""
    model = _build_blocked_initializer_with_nested_input_shadow_model()
    del model.graph.node[0]
    model.graph.output[0].CopyFrom(model.graph.output[1])
    del model.graph.output[1:]
    model.graph.name = "unused_initializer_nested_shadow"
    return model


def _build_non_float_initializer_before_nested_float_initializer_model() -> ModelProto:
    """Build an earlier non-FLOAT binding sharing a later nested FLOAT initializer name."""
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    integer_output = helper.make_tensor_value_info("integer_output", TensorProto.INT64, [1])
    nested_output = helper.make_tensor_value_info("nested_output", TensorProto.FLOAT, [1])
    integer_initializer = numpy_helper.from_array(np.array([7], dtype=np.int64), "same")

    def _branch(name: str, initializer_name: str, value: float) -> GraphProto:
        branch_output = helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])
        initializer = numpy_helper.from_array(np.array([value], dtype=np.float32), initializer_name)
        identity = helper.make_node(
            "Identity", [initializer_name], [f"{name}_output"], name=f"{name}_identity"
        )
        return helper.make_graph([identity], name, [], [branch_output], [initializer])

    integer_identity = helper.make_node(
        "Identity", ["same"], ["integer_output"], name="integer_identity"
    )
    conditional = helper.make_node(
        "If",
        ["condition"],
        ["nested_output"],
        name="if",
        then_branch=_branch("then", "same", 1.5),
        else_branch=_branch("else", "other", 2.5),
    )
    graph = helper.make_graph(
        [integer_identity, conditional],
        "initializer_registration_order",
        [condition],
        [integer_output, nested_output],
        [integer_initializer],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


# =============================================================================
# CONVERT_TO_FP16 TESTS
# =============================================================================


class TestConvertToFP16:
    """Test convert_to_fp16 utility function."""

    def test_converts_weights_to_fp16(self) -> None:
        """FP16 conversion converts float32 initializers to float16."""
        model = _build_simple_fp32_model()
        result = convert_to_fp16(model)

        has_fp16 = any(init.data_type == TensorProto.FLOAT16 for init in result.graph.initializer)
        assert has_fp16, "Expected at least one FP16 initializer after conversion"

    def test_success_mutates_and_returns_input_model(self) -> None:
        """A successful conversion preserves the wrapper's in-place API."""
        model = _build_simple_fp32_model()

        result = convert_to_fp16(model)

        assert result is model
        assert any(
            initializer.data_type == TensorProto.FLOAT16 for initializer in model.graph.initializer
        )

    def test_exact_node_exclusion_preserves_only_the_selected_cast(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        token_ids = helper.make_tensor_value_info("token_ids", TensorProto.INT32, [1])
        selected_output = helper.make_tensor_value_info("selected_out", TensorProto.FLOAT, [1])
        neighbor_output = helper.make_tensor_value_info("neighbor_out", TensorProto.FLOAT, [1])
        embeddings_output = helper.make_tensor_value_info("embeddings_out", TensorProto.FLOAT, [1])
        selected = helper.make_node(
            "Cast",
            ["x"],
            ["selected_out"],
            name="InsertedPrecisionFreeCast_selected",
            to=TensorProto.FLOAT,
        )
        neighbor = helper.make_node(
            "Cast",
            ["x"],
            ["neighbor_out"],
            name="InsertedPrecisionFreeCast_neighbor",
            to=TensorProto.FLOAT,
        )
        embeddings_cast = helper.make_node(
            "Cast",
            ["token_ids"],
            ["embeddings_out"],
            name="InsertedPrecisionFreeCast_/deberta/embeddings/Cast_output_0",
            to=TensorProto.FLOAT,
        )
        model = helper.make_model(
            helper.make_graph(
                [selected, neighbor, embeddings_cast],
                "named_cast_boundaries",
                [x, token_ids],
                [selected_output, neighbor_output, embeddings_output],
            ),
            opset_imports=[helper.make_opsetid("", 17)],
        )

        result = convert_to_fp16(
            model,
            keep_io_types=False,
            node_block_list=["InsertedPrecisionFreeCast_selected"],
        )
        cast_targets = {
            node.name: next(attribute.i for attribute in node.attribute if attribute.name == "to")
            for node in result.graph.node
            if node.op_type == "Cast" and node.name.startswith("InsertedPrecisionFreeCast_")
        }

        assert cast_targets["InsertedPrecisionFreeCast_selected"] == TensorProto.FLOAT
        assert cast_targets["InsertedPrecisionFreeCast_neighbor"] == TensorProto.FLOAT16
        assert (
            cast_targets["InsertedPrecisionFreeCast_/deberta/embeddings/Cast_output_0"]
            == TensorProto.FLOAT16
        )
        checker.check_model(result)
        shape_inference.infer_shapes(result, check_type=True, strict_mode=True)
        ort.InferenceSession(result.SerializeToString(), providers=["CPUExecutionProvider"])

    def test_unknown_node_exclusion_has_no_heuristic_effect(self) -> None:
        original = _build_simple_fp32_model()
        expected = convert_to_fp16(ModelProto.FromString(original.SerializeToString()))
        actual = convert_to_fp16(original, node_block_list=["unknown_model_specific_prefix"])

        assert actual.SerializeToString() == expected.SerializeToString()

    def test_scalar_float_tensor_attribute_is_rejected(self) -> None:
        """ORT cannot convert scalar FLOAT storage in tensor attributes."""
        model = _build_scalar_float_attribute_model()
        original = model.SerializeToString()

        checker.check_model(model)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {})
        assert output.dtype == np.float32

        with np.testing.assert_raises_regex(RuntimeError, "incompatible FP16 types"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_sparse_float_tensor_attribute_is_rejected(self) -> None:
        """ORT cannot convert sparse FLOAT storage in tensor attributes."""
        model = _build_sparse_float_attribute_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, check_type=True, strict_mode=True)

        with np.testing.assert_raises_regex(RuntimeError, "incompatible FP16 types"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_loaded_external_tensor_attribute_is_internalized(
        self,
    ) -> None:
        """Resident tensor attributes drop stale external metadata."""
        model = _build_external_float_attribute_model(clear_data=False)

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        value = result.graph.node[0].attribute[0].t
        assert value.data_type == TensorProto.FLOAT16
        assert value.data_location == TensorProto.DEFAULT
        assert not value.external_data
        assert len(value.raw_data) == 2
        checker.check_model(result)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {})
        assert output.dtype == np.float16

    def test_unloaded_external_tensor_attribute_is_rejected(
        self,
    ) -> None:
        """Selected tensor attributes require resident external data."""
        model = _build_external_float_attribute_model(clear_data=True)
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "unloaded external data"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_omitted_optional_output_is_not_a_binding(self) -> None:
        """Empty optional output and input sentinels are never connected."""
        model = _build_omitted_optional_output_model()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["Dropout"],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        dropped, clipped = session.run(None, {"x": np.array([1.0, 2.0], dtype=np.float16)})
        assert dropped.dtype == np.float16
        assert clipped.dtype == np.float16

    def test_default_keeps_io_types(self) -> None:
        """Default keep_io_types=True preserves FP32 model I/O."""
        model = _build_simple_fp32_model()
        result = convert_to_fp16(model, keep_io_types=True)

        for inp in result.graph.input:
            assert inp.type.tensor_type.elem_type == TensorProto.FLOAT
        for outp in result.graph.output:
            assert outp.type.tensor_type.elem_type == TensorProto.FLOAT

    def test_float_sequence_io_is_rejected_when_io_types_are_kept(self) -> None:
        """Unsupported FLOAT container I/O is not silently converted to FP16."""
        model = _build_float_sequence_io_model()
        original = model.SerializeToString()

        checker.check_model(model)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {"sequence": [np.array([2.0], dtype=np.float32)]},
        )
        assert output[0].dtype == np.float32

        with np.testing.assert_raises_regex(RuntimeError, "FLOAT container graph I/O"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_float_optional_input_is_preserved_when_io_types_are_kept(self) -> None:
        """Container I/O that ORT leaves unchanged remains supported."""
        model = _build_top_optional_input_blocked_capture_model()

        result = convert_to_fp16(
            model,
            keep_io_types=True,
            op_block_list=["If"],
        )

        optional_input = next(value for value in result.graph.input if value.name == "optional")
        assert (
            optional_input.type.optional_type.elem_type.tensor_type.elem_type == TensorProto.FLOAT
        )
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "optional": np.array([2.0], dtype=np.float32),
            },
        )
        assert output == np.array(True)

    def test_keep_io_types_false_converts_io(self) -> None:
        """With keep_io_types=False, model I/O becomes FP16."""
        model = _build_simple_fp32_model()
        result = convert_to_fp16(model, keep_io_types=False)

        for inp in result.graph.input:
            assert inp.type.tensor_type.elem_type == TensorProto.FLOAT16
        for outp in result.graph.output:
            assert outp.type.tensor_type.elem_type == TensorProto.FLOAT16

    def test_initializer_backed_output_stays_fp32_when_io_types_are_kept(self) -> None:
        """An initializer graph output remains valid FP32 when preserving I/O."""
        model = _build_initializer_backed_output_model()

        result = convert_to_fp16(model, keep_io_types=True)

        output = result.graph.output[0]
        assert output.type.tensor_type.elem_type == TensorProto.FLOAT
        assert any(
            initializer.data_type == TensorProto.FLOAT for initializer in result.graph.initializer
        )
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        np.testing.assert_array_equal(
            session.run(None, {})[0],
            np.array([[1.0001, 2.0003]], dtype=np.float32),
        )

    def test_initializer_backed_output_converts_data_when_io_types_are_not_kept(self) -> None:
        """A pure-FP16 output converts its backing initializer as well as its type."""
        model = _build_initializer_backed_output_model()

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        output = result.graph.output[0]
        initializer = result.graph.initializer[0]
        assert output.type.tensor_type.elem_type == TensorProto.FLOAT16
        assert initializer.data_type == TensorProto.FLOAT16
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_initializer_backed_output_with_fp16_consumer_converts_to_fp16(self) -> None:
        """Pure-FP16 conversion allows an output initializer consumed by FP16-capable nodes."""
        model = _build_shared_initializer_output_model()

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        assert all(
            output.type.tensor_type.elem_type == TensorProto.FLOAT16
            for output in result.graph.output
        )
        assert result.graph.initializer[0].data_type == TensorProto.FLOAT16
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        shared, y = session.run(None, {"x": np.array([[3.0, 4.0]], dtype=np.float16)})
        np.testing.assert_array_equal(shared, np.array([[1.0, 2.0]], dtype=np.float16))
        np.testing.assert_array_equal(y, np.array([[4.0, 6.0]], dtype=np.float16))

    def test_overridable_initializer_output_is_rejected_when_io_types_are_not_kept(
        self,
    ) -> None:
        """Graph input/output initializer aliases are rejected before ORT optimizer crashes."""
        model = _build_overridable_shared_initializer_output_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "also a graph input"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_scan8_scan_output_is_not_feedback_type_evidence(self) -> None:
        """A scan output cannot type an unrelated untyped scan input."""
        model = _build_scan8_without_state_model()
        original = model.SerializeToString()

        checker.check_model(model)
        with np.testing.assert_raises_regex(RuntimeError, "missing FLOAT metadata"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=["Gelu"],
            )
        assert model.SerializeToString() == original

    def test_nested_initializer_output_with_fp16_consumer_converts_to_fp16(self) -> None:
        """Nested initializer outputs are allowed when ORT converts them consistently."""
        model = _build_nested_consumed_initializer_output_model()

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        np.testing.assert_array_equal(
            session.run(None, {"condition": np.array(True)})[0],
            np.array([1.0], dtype=np.float16),
        )

    def test_fp16_initializers_do_not_skip_fp32_graph_conversion(self) -> None:
        """FP16 initializers alone do not prove graph I/O and nodes are already FP16."""
        model = _build_fp32_identity_with_fp16_nested_initializer_model()

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        assert result.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT16
        assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_sparse_initializer_inputs_are_available_to_topological_sort(self) -> None:
        """Sorting after conversion treats sparse initializer names as available values."""
        model = _build_sparse_shape_reshape_model()

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_sparse_float_initializer_converts_with_fp16_consumers(self) -> None:
        """Sparse FLOAT initializers are converted when their consumers become FP16."""
        model = _build_sparse_float_add_model()

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16
        assert result.graph.sparse_initializer[0].values.data_type == TensorProto.FLOAT16
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        np.testing.assert_array_equal(
            session.run(None, {"x": np.array([3.0, 4.0], dtype=np.float16)})[0],
            np.array([4.0, 6.0], dtype=np.float16),
        )

    def test_sparse_float_initializer_value_info_converts_with_values(self) -> None:
        """Sparse FLOAT value_info metadata is kept consistent with converted values."""
        model = _build_sparse_float_add_model_with_value_info()

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        sparse_info = next(value for value in result.graph.value_info if value.name == "weight")
        assert sparse_info.type.sparse_tensor_type.elem_type == TensorProto.FLOAT16
        assert result.graph.sparse_initializer[0].values.data_type == TensorProto.FLOAT16
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_sparse_float_initializer_tensor_value_info_is_rejected_before_mutation(
        self,
    ) -> None:
        """Tensor metadata for sparse FLOAT initializers is rejected before ORT fails."""
        model = _build_sparse_float_add_model_with_tensor_value_info()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "sparse initializer metadata"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_sparse_float_initializer_io_alias_is_rejected_when_io_types_are_not_kept(
        self,
    ) -> None:
        """Sparse graph input/output initializer aliases are rejected before mutation."""
        model = _build_sparse_float_add_model_with_io_metadata()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "also sparse graph input"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_sparse_float_initializer_tensor_io_alias_is_rejected_before_mutation(
        self,
    ) -> None:
        """Tensor graph input/output sparse initializer metadata is rejected pre-ORT."""
        model = _build_sparse_float_add_model_with_tensor_io_metadata()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "sparse initializer metadata"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_retained_float_initializer_metadata_is_rejected(self) -> None:
        """Converted metadata cannot describe an initializer retained in FP32."""
        model = _build_retained_float_initializer_metadata_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "x": np.array(
                    [[[[1.0, 2.0], [3.0, 4.0]]]],
                    dtype=np.float32,
                )
            },
        )
        assert output.dtype == np.float32

        with np.testing.assert_raises_regex(RuntimeError, "initializer metadata"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=[],
            )
        assert model.SerializeToString() == original

    def test_sparse_float_initializer_io_metadata_is_rejected_when_io_is_kept(self) -> None:
        """Sparse graph I/O dtypes are not silently changed when keep_io_types=True."""
        model = _build_sparse_float_add_model_with_io_metadata()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "sparse graph I/O"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_sparse_float_initializer_tensor_io_metadata_is_rejected_before_keep_io(
        self,
    ) -> None:
        """Tensor graph I/O sparse initializer metadata is rejected before ORT fails."""
        model = _build_sparse_float_add_model_with_tensor_io_metadata()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "sparse initializer metadata"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_direct_sparse_float_output_converts_when_io_types_are_not_kept(self) -> None:
        """Direct sparse FLOAT outputs convert without requiring node consumers."""
        model = _build_direct_sparse_float_output_model()

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        sparse_output = next(
            value for value in result.graph.output if value.name == "sparse_output"
        )
        assert sparse_output.type.sparse_tensor_type.elem_type == TensorProto.FLOAT16
        assert result.graph.sparse_initializer[0].values.data_type == TensorProto.FLOAT16
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_direct_sparse_float_tensor_output_is_rejected_before_mutation(
        self,
    ) -> None:
        """Direct sparse FLOAT outputs with tensor metadata are rejected pre-ORT."""
        model = _build_direct_sparse_float_tensor_output_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "sparse initializer metadata"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_nested_direct_sparse_float_outputs_are_rejected_before_mutation(
        self,
    ) -> None:
        """Nested sparse outputs are rejected when parent edge types cannot be repaired."""
        model = _build_if_with_direct_sparse_float_outputs_model()
        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "nested sparse graph output"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_nested_direct_sparse_float_outputs_are_preserved_when_io_is_kept(
        self,
    ) -> None:
        """A kept top-level sparse output leaves its nested sparse edges in FLOAT."""
        model = _build_if_with_direct_sparse_float_outputs_model()

        result = convert_to_fp16(model, keep_io_types=True, op_block_list=[])

        assert result.graph.output[0].type.sparse_tensor_type.elem_type == TensorProto.FLOAT
        for graph in _iter_attribute_graphs(result):
            assert graph.output[0].type.sparse_tensor_type.elem_type == TensorProto.FLOAT
            assert graph.sparse_initializer[0].values.data_type == TensorProto.FLOAT
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_deep_nested_sparse_outputs_are_preserved_when_io_is_kept(
        self,
    ) -> None:
        """A kept sparse edge remains FLOAT through every enclosing graph."""
        model = _build_deep_if_with_direct_sparse_float_outputs_model()

        result = convert_to_fp16(model, keep_io_types=True, op_block_list=[])

        assert result.graph.output[0].type.sparse_tensor_type.elem_type == TensorProto.FLOAT
        for graph in _iter_attribute_graphs(result):
            assert graph.output[0].type.sparse_tensor_type.elem_type == TensorProto.FLOAT
            for sparse in graph.sparse_initializer:
                assert sparse.values.data_type == TensorProto.FLOAT
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_direct_sparse_float_output_with_fp32_consumer_is_rejected(self) -> None:
        """Direct sparse FP16 outputs are mixed uses when blocked consumers stay FP32."""
        model = _build_sparse_float_add_model_with_io_metadata()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "both FP16 and FP32"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["Add"])
        assert model.SerializeToString() == original

    def test_unloaded_nested_external_initializer_output_is_rejected_before_mutation(
        self,
    ) -> None:
        """Unloaded nested external backing data is rejected before dtype metadata changes."""
        model = _build_nested_consumed_initializer_output_model()
        for graph in _iter_attribute_graphs(model):
            _mark_initializers_as_external(graph, clear_data=True)
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(
            RuntimeError,
            "load external weights before FP16 conversion",
        ):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_loaded_nested_external_initializer_output_is_internalized(self) -> None:
        """Loaded nested external output data no longer points to stale sidecars."""
        model = _build_nested_consumed_initializer_output_model()
        for graph in _iter_attribute_graphs(model):
            _mark_initializers_as_external(graph, clear_data=False)

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        for graph in _iter_attribute_graphs(result):
            for initializer in graph.initializer:
                assert initializer.data_type == TensorProto.FLOAT16
                assert initializer.data_location == TensorProto.DEFAULT
                assert not initializer.external_data

    def test_non_float_external_initializer_output_metadata_is_preserved(self) -> None:
        """Non-FLOAT direct output initializers are outside the FP16 repair path."""
        model = _build_initializer_backed_output_model()
        int_output = helper.make_tensor_value_info("int_output", TensorProto.INT64, [1])
        int_value = numpy_helper.from_array(np.array([7], dtype=np.int64), "int_output")
        int_value.ClearField("raw_data")
        int_value.data_location = TensorProto.EXTERNAL
        location = int_value.external_data.add()
        location.key = "location"
        location.value = "int_output.bin"
        model.graph.output.append(int_output)
        model.graph.initializer.append(int_value)

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        int_initializer = next(
            initializer
            for initializer in result.graph.initializer
            if initializer.name == "int_output"
        )
        assert int_initializer.data_type == TensorProto.INT64
        assert int_initializer.data_location == TensorProto.EXTERNAL
        assert [(entry.key, entry.value) for entry in int_initializer.external_data] == [
            ("location", "int_output.bin")
        ]

    def test_multiple_initializer_backed_outputs_are_all_converted(self) -> None:
        """Every initializer-backed output is repaired independently."""
        model = _build_initializer_backed_output_model()
        second_output = helper.make_tensor_value_info("second_output", TensorProto.FLOAT, [1, 2])
        second_value = numpy_helper.from_array(
            np.array([[3.0, 4.0]], dtype=np.float32), "second_output"
        )
        model.graph.output.append(second_output)
        model.graph.initializer.append(second_value)

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        assert all(
            output.type.tensor_type.elem_type == TensorProto.FLOAT16
            for output in result.graph.output
        )
        assert all(
            initializer.data_type == TensorProto.FLOAT16 for initializer in result.graph.initializer
        )
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_multiple_initializer_outputs_keep_exact_fp32_values(self) -> None:
        """Removing several orphan Casts preserves every FP32 model output."""
        model = _build_initializer_backed_output_model()
        second_output = helper.make_tensor_value_info("second_output", TensorProto.FLOAT, [1, 2])
        second_value = numpy_helper.from_array(
            np.array([[3.0005, 4.0007]], dtype=np.float32), "second_output"
        )
        model.graph.output.append(second_output)
        model.graph.initializer.append(second_value)

        result = convert_to_fp16(model, keep_io_types=True)

        checker.check_model(result)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        first, second = session.run(None, {})
        np.testing.assert_array_equal(first, np.array([[1.0001, 2.0003]], dtype=np.float32))
        np.testing.assert_array_equal(second, np.array([[3.0005, 4.0007]], dtype=np.float32))

    def test_overridable_initializer_output_is_rejected_before_mutation(self) -> None:
        """A graph-input initializer keeps its caller override semantics."""
        model = _build_initializer_backed_output_model()
        model.graph.input.append(
            helper.make_tensor_value_info("constant_output", TensorProto.FLOAT, [1, 2])
        )
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "also a graph input"):
            convert_to_fp16(model, keep_io_types=True)

        assert model.SerializeToString() == original

    def test_collision_uses_original_mixed_output_index(self) -> None:
        """ORT Cast collision checks count preceding non-FLOAT outputs."""
        model = _build_initializer_backed_output_model()
        int_output = helper.make_tensor_value_info("int_output", TensorProto.INT64, [1])
        int_value = numpy_helper.from_array(np.array([1], dtype=np.int64), "int_output")
        model.graph.output.insert(0, int_output)
        model.graph.initializer.append(int_value)
        model.graph.input.append(
            helper.make_tensor_value_info("graph_output_cast_1", TensorProto.FLOAT, [1])
        )
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "graph_output_cast_1"):
            convert_to_fp16(model, keep_io_types=True)

        assert model.SerializeToString() == original

    def test_unloaded_external_initializer_output_is_rejected(self) -> None:
        """Conversion refuses external backing data that was not loaded."""
        model = _build_initializer_backed_output_model()
        initializer = model.graph.initializer[0]
        initializer.ClearField("raw_data")
        initializer.data_location = TensorProto.EXTERNAL
        location = initializer.external_data.add()
        location.key = "location"
        location.value = "weights.data"

        original_output_type = model.graph.output[0].type.tensor_type.elem_type
        original_initializer_type = initializer.data_type

        with np.testing.assert_raises_regex(
            RuntimeError,
            "load external weights before FP16 conversion",
        ):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        assert model.graph.output[0].type.tensor_type.elem_type == original_output_type
        assert model.graph.initializer[0].data_type == original_initializer_type

    def test_loaded_external_initializer_output_is_converted(self) -> None:
        """Resident tensor bytes are valid even if external metadata remains."""
        model = _build_initializer_backed_output_model()
        initializer = model.graph.initializer[0]
        initializer.data_location = TensorProto.EXTERNAL
        location = initializer.external_data.add()
        location.key = "location"
        location.value = "weights.data"

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16
        assert result.graph.initializer[0].data_type == TensorProto.FLOAT16

    def test_loaded_external_initializer_output_is_internalized_when_io_is_kept(self) -> None:
        """Resident output data no longer points to a stale sidecar after repair."""
        model = _build_initializer_backed_output_model()
        initializer = model.graph.initializer[0]
        initializer.data_location = TensorProto.EXTERNAL
        location = initializer.external_data.add()
        location.key = "location"
        location.value = "weights.data"

        result = convert_to_fp16(model, keep_io_types=True)

        repaired = result.graph.initializer[0]
        assert repaired.data_location == TensorProto.DEFAULT
        assert not repaired.external_data
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        np.testing.assert_array_equal(
            session.run(None, {})[0],
            np.array([[1.0001, 2.0003]], dtype=np.float32),
        )

    def test_loaded_external_weight_is_internalized_before_conversion(
        self,
    ) -> None:
        """Resident external FLOAT weights convert without stale sidecars."""
        model = _build_simple_fp32_model()
        _mark_initializers_as_external(model.graph, clear_data=False)

        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=[],
        )

        initializer = result.graph.initializer[0]
        assert initializer.data_type == TensorProto.FLOAT16
        assert initializer.data_location == TensorProto.DEFAULT
        assert not initializer.external_data
        checker.check_model(result)
        shape_inference.infer_shapes(result, check_type=True, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {"x": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16)},
        )
        assert output.dtype == np.float16

    def test_unloaded_external_weight_is_rejected_before_mutation(
        self,
    ) -> None:
        """Selected external FLOAT weights require resident tensor data."""
        model = _build_simple_fp32_model()
        _mark_initializers_as_external(model.graph, clear_data=True)
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "unloaded external data"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=[],
            )
        assert model.SerializeToString() == original

    def test_external_non_float_name_collision_is_not_selected(
        self,
    ) -> None:
        """Global FLOAT selection does not internalize same-named INT data."""
        model = _build_nested_external_int_name_collision_model()

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        assert result.graph.initializer[0].data_type == TensorProto.FLOAT16
        then_graph = next(
            attribute.g
            for attribute in result.graph.node[-1].attribute
            if attribute.g.name == "then"
        )
        integer = then_graph.initializer[0]
        assert integer.data_type == TensorProto.INT64
        assert integer.data_location == TensorProto.EXTERNAL
        assert integer.external_data

    def test_shared_initializer_output_is_rejected_before_mutation(self) -> None:
        """Shared initializer-output semantics are rejected before mutation."""
        model = _build_shared_initializer_output_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "has internal consumers"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_nested_initializer_outputs_are_converted_when_io_types_are_kept(self) -> None:
        """Nested direct output initializers are repaired in traversed graphs."""
        model = _build_nested_initializer_output_model()

        result = convert_to_fp16(model, keep_io_types=True, op_block_list=[])

        assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        for graph in _iter_attribute_graphs(result):
            assert all(
                output.type.tensor_type.elem_type == TensorProto.FLOAT16 for output in graph.output
            )
            assert all(
                initializer.data_type == TensorProto.FLOAT16 for initializer in graph.initializer
            )
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        np.testing.assert_array_equal(
            session.run(None, {"condition": np.array(True)})[0],
            np.array([1.0], dtype=np.float32),
        )

    def test_lexical_nested_consumers_are_rejected_before_mutation(self) -> None:
        """Lexically shared output initializers are rejected before mutation."""
        model = _build_lexically_captured_initializer_output_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "has internal consumers"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_generated_tensor_name_collision_is_rejected_before_mutation(self) -> None:
        """Repair allocates a fresh alias instead of duplicating an existing name."""
        model = _build_initializer_output_name_collision_model()

        original = model.SerializeToString()
        with np.testing.assert_raises_regex(RuntimeError, "existing names collide"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_generated_cast_node_name_collision_is_rejected_before_mutation(self) -> None:
        """A user node occupying ORT's deterministic Cast name fails safely."""
        model = _build_initializer_output_node_name_collision_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "existing names collide"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])

        assert model.SerializeToString() == original

    def test_nested_generated_node_name_collision_is_rejected_before_mutation(self) -> None:
        """Nested nodes also participate in ORT's global generated-name set."""
        model = _build_nested_node_name_collision_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "existing names collide"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])

        assert model.SerializeToString() == original

    def test_regular_output_nested_node_name_collision_is_rejected_before_mutation(
        self,
    ) -> None:
        """Generated I/O Cast names are reserved even without initializer outputs."""
        model = _build_regular_output_nested_node_name_collision_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "graph_output_cast0"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_neutral_nested_node_name_collision_is_allowed(self) -> None:
        """Skipping an already-concrete INT node cannot alter precision."""
        model = _build_neutral_nested_node_name_collision_model()

        checker.check_model(model)
        shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
        result = convert_to_fp16(model, keep_io_types=True, op_block_list=[])

        checker.check_model(result)
        shape_inference.infer_shapes(result, check_type=True, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output, nested_output = session.run(
            None,
            {
                "x": np.array([2.0], dtype=np.float32),
                "integer": np.array([7], dtype=np.int64),
                "condition": np.array(True),
            },
        )
        np.testing.assert_array_equal(output, np.array([2.0], dtype=np.float32))
        np.testing.assert_array_equal(nested_output, np.array([7], dtype=np.int64))

    def test_neutral_collision_can_use_shadowed_int_formal(
        self,
    ) -> None:
        """A skipped neutral node leaves its local INT formal untouched."""
        model = _build_neutral_shadowed_keep_io_collision_model()

        checker.check_model(model)
        shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
        result = convert_to_fp16(model, keep_io_types=True, op_block_list=[])

        checker.check_model(result)
        shape_inference.infer_shapes(result, check_type=True, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output, loop_output = session.run(
            None,
            {
                "x": np.array([2.0], dtype=np.float32),
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "integer": np.array([7], dtype=np.int64),
            },
        )
        np.testing.assert_array_equal(output, np.array([2.0], dtype=np.float32))
        np.testing.assert_array_equal(loop_output, np.array([7], dtype=np.int64))

    def test_inferred_output_nested_node_name_collision_is_rejected_before_mutation(
        self,
    ) -> None:
        """Generated Cast reservations use the same inferred I/O types as ORT."""
        model = _build_inferred_output_nested_node_name_collision_model()
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        assert inferred.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        checker.check_model(inferred)
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "graph_output_cast0"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_nested_local_generated_tensor_alias_does_not_collide(self) -> None:
        """A nested local binding may legally shadow a generated top-level alias."""
        model = _build_nested_local_generated_tensor_alias_model()

        result = convert_to_fp16(model, keep_io_types=True, op_block_list=[])

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output, loop_output = session.run(
            None,
            {
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "loop_state": np.array([2.0], dtype=np.float32),
                "x": np.array([3.0], dtype=np.float32),
            },
        )
        np.testing.assert_array_equal(output, np.array([3.0], dtype=np.float32))
        np.testing.assert_array_equal(loop_output, np.array([2.0], dtype=np.float32))

    def test_nested_blocked_generated_tensor_alias_is_rejected_before_mutation(
        self,
    ) -> None:
        """ORT's blocked-node lookup is global even for a nested local binding."""
        model = _build_nested_blocked_generated_tensor_alias_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "blocked or mixed-type nested node"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=["Identity"])
        assert model.SerializeToString() == original

    def test_nested_mixed_generated_tensor_alias_is_rejected_before_mutation(
        self,
    ) -> None:
        """ORT's mixed-input lookup is global even for a nested local binding."""
        model = _build_nested_mixed_generated_tensor_alias_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "blocked or mixed-type nested node"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_inferred_nested_mixed_name_collision_is_rejected_before_mutation(
        self,
    ) -> None:
        """Late lookup safety uses the same shape inference metadata as ORT."""
        model = _build_inferred_nested_mixed_name_collision_model()
        original = model.SerializeToString()

        checker.check_model(model)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (loop_output,) = session.run(
            None,
            {
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "loop_scale": np.array([2.0], dtype=np.float32),
                "x": np.array([3.0], dtype=np.float32),
            },
        )
        assert loop_output.shape == (2,)

        with np.testing.assert_raises_regex(RuntimeError, "global value-info"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_top_blocked_global_value_info_collision_is_rejected(self) -> None:
        """Top-level blocked nodes also use ORT's global metadata lookup."""
        model = _build_top_blocked_global_value_info_collision_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)

        with np.testing.assert_raises_regex(RuntimeError, "global value-info"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["Identity"])
        assert model.SerializeToString() == original

    def test_duplicate_late_cast_alias_is_rejected_before_mutation(self) -> None:
        """Blocked nodes must not allocate the same deterministic Cast tensor name."""
        model = _build_duplicate_late_cast_alias_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)

        with np.testing.assert_raises_regex(RuntimeError, "late Cast tensor"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["Identity"])
        assert model.SerializeToString() == original

    def test_late_cast_node_name_collision_is_rejected_before_mutation(self) -> None:
        """Generated late Cast node names must be unique in the top-level graph."""
        model = _build_late_cast_node_name_collision_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)

        with np.testing.assert_raises_regex(RuntimeError, "late Cast node"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["Abs"])
        assert model.SerializeToString() == original

    def test_blocked_subgraph_float_capture_is_rejected_before_mutation(self) -> None:
        """Skipped branch graphs cannot keep FLOAT captures that become FP16."""
        model = _build_blocked_subgraph_float_capture_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)

        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["If"])
        assert model.SerializeToString() == original

    def test_untyped_blocked_capture_is_rejected_when_inference_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inference fallback conservatively rejects unresolved node captures."""
        model = _build_untyped_blocked_subgraph_capture_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)

        def fail_inference(*args: object, **kwargs: object) -> None:
            raise EncodeError("simulated serialization failure")

        monkeypatch.setattr(shape_inference, "infer_shapes", fail_inference)
        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["If"])
        assert model.SerializeToString() == original

    def test_uninferred_custom_op_blocked_capture_is_rejected(self) -> None:
        """Successful but incomplete inference still requires conservative safety."""
        model = _build_uninferred_custom_op_blocked_capture_model()
        original = model.SerializeToString()

        checker.check_model(model)
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        assert all(value.name != "y" for value in inferred.graph.value_info)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "x": np.array([1.0], dtype=np.float32),
                "condition": np.array(True),
            },
        )
        assert output.dtype == np.float32

        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["If"])
        assert model.SerializeToString() == original

    def test_nested_value_info_can_annotate_kept_input_capture(
        self,
    ) -> None:
        """value_info alone does not create a nested lexical binding."""
        model = _build_kept_input_free_capture_value_info_model()

        checker.check_model(model)
        shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
        result = convert_to_fp16(
            model,
            keep_io_types=True,
            op_block_list=[],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, check_type=True, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "x": np.array([2.0], dtype=np.float32),
                "condition": np.array(True),
            },
        )
        np.testing.assert_array_equal(output, np.array([2.0], dtype=np.float32))

    def test_empty_metadata_blocked_capture_is_rejected(self) -> None:
        """Name-only metadata does not prove a captured producer stays non-FLOAT."""
        model = _build_uninferred_custom_op_blocked_capture_model()
        model.graph.value_info.append(helper.make_empty_tensor_value_info("y"))
        original = model.SerializeToString()

        checker.check_model(model)
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        y = next(value for value in inferred.graph.value_info if value.name == "y")
        assert y.type.tensor_type.elem_type == TensorProto.UNDEFINED

        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["If"])
        assert model.SerializeToString() == original

    def test_missing_metadata_blocked_input_is_rejected(self) -> None:
        """A blocked input needs metadata to receive an FP16-to-FLOAT boundary Cast."""
        model = _build_missing_metadata_blocked_edge_model()
        original = model.SerializeToString()

        checker.check_model(model)
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        assert all(value.name != "hidden" for value in inferred.graph.value_info)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"x": np.array([1.0], dtype=np.float32)})
        assert output.dtype == np.float32

        with np.testing.assert_raises_regex(RuntimeError, "missing FLOAT metadata"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["Add"])
        assert model.SerializeToString() == original

    def test_missing_metadata_blocked_output_is_rejected(self) -> None:
        """A blocked output needs metadata to receive a FLOAT-to-FP16 boundary Cast."""
        model = _build_missing_metadata_blocked_edge_model()
        original = model.SerializeToString()

        checker.check_model(model)
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        assert all(value.name != "hidden" for value in inferred.graph.value_info)

        with np.testing.assert_raises_regex(RuntimeError, "missing FLOAT metadata"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["Gelu"])
        assert model.SerializeToString() == original

    def test_existing_fp16_boundary_does_not_require_metadata(self) -> None:
        """An explicit type boundary can safely consume a blocked FLOAT output."""
        model = _build_existing_fp16_boundary_model()

        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["Gelu"],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"x": np.array([1.0], dtype=np.float16)})
        assert output.dtype == np.float16

    def test_missing_metadata_equal_sibling_coupling_is_rejected(self) -> None:
        """A same-type sibling input still requires an FP16 output boundary."""
        model = _build_missing_metadata_equal_consumer_model()
        original = model.SerializeToString()

        checker.check_model(model)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "x": np.array([1.0], dtype=np.float32),
                "other": np.array([1.0], dtype=np.float32),
            },
        )
        assert output.dtype == np.bool_

        with np.testing.assert_raises_regex(RuntimeError, "missing FLOAT metadata"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["Gelu"])
        assert model.SerializeToString() == original

    def test_missing_metadata_sequence_coupling_is_rejected(self) -> None:
        """A tensor input and sequence output can share payload precision."""
        model = _build_missing_metadata_sequence_consumer_model()
        original = model.SerializeToString()

        checker.check_model(model)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"x": np.array([1.0], dtype=np.float32)})
        assert output[0].dtype == np.float32

        with np.testing.assert_raises_regex(RuntimeError, "missing FLOAT metadata"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["Gelu"])
        assert model.SerializeToString() == original

    def test_missing_metadata_sequence_map_coupling_is_rejected(self) -> None:
        """Partially overlapping schema constraints can still couple payload precision."""
        model = _build_missing_metadata_sequence_map_consumer_model()
        original = model.SerializeToString()

        checker.check_model(model)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "x": np.array([1.0], dtype=np.float32),
                "sequence": [np.array([2.0], dtype=np.float32)],
            },
        )
        assert output[0].dtype == np.float32

        with np.testing.assert_raises_regex(RuntimeError, "missing FLOAT metadata"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["Gelu"])
        assert model.SerializeToString() == original

    def test_int_sequence_map_consumer_does_not_require_float_boundary(self) -> None:
        """Concrete child formal types prove an uninferred edge is non-FLOAT."""
        model = _build_int_sequence_map_consumer_model()

        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["MurmurHash3"],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "text": np.array(["hello"]),
                "sequence": [np.array([1], dtype=np.int32)],
            },
        )
        assert output[0].dtype == np.int32

    def test_int_same_type_consumer_does_not_require_float_boundary(self) -> None:
        """Concrete non-FLOAT edges override same-type schema relationships."""
        model = _build_int_identity_consumer_model()

        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["NonMaxSuppression"],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "boxes": np.array(
                    [
                        [
                            [0.0, 0.0, 1.0, 1.0],
                            [2.0, 2.0, 3.0, 3.0],
                        ]
                    ],
                    dtype=np.float16,
                ),
                "scores": np.array([[[0.9, 0.8]]], dtype=np.float16),
            },
        )
        assert output.dtype == np.int64

    def test_concrete_consumer_output_proves_uninferred_input_is_int(self) -> None:
        """A concrete same-type output can prove an uninferred input is non-FLOAT."""
        model = _build_uninferred_int_identity_consumer_model()

        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["MurmurHash3"],
        )

        checker.check_model(result)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"text": np.array(["hello"])})
        assert output.dtype == np.int32

    def test_concrete_function_input_is_int_without_registered_schema(self) -> None:
        """Concrete non-FLOAT evidence is sufficient for a local function input."""
        model = _build_function_int_consumer_model()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["MurmurHash3"],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"text": np.array(["hello"])})
        assert output.dtype == np.int32

    def test_unconverted_float_constant_in_local_function_is_rejected(
        self,
    ) -> None:
        """ORT does not visit concrete FLOAT values in FunctionProto bodies."""
        model = _build_function_with_float_constant_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"x": np.array([3.0], dtype=np.float32)})
        np.testing.assert_array_equal(output, np.array([4.0], dtype=np.float32))

        with np.testing.assert_raises_regex(RuntimeError, "local function"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=[],
            )
        assert model.SerializeToString() == original

    def test_unconverted_contrib_float_function_is_rejected(self) -> None:
        """Concrete FLOAT function bodies cannot rely on ONNX-only inference."""
        model = _build_function_with_contrib_float_constant_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"x": np.array([3.0], dtype=np.float32)})
        assert output.dtype == np.float32

        with np.testing.assert_raises_regex(RuntimeError, "local function"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=[],
            )
        assert model.SerializeToString() == original

    def test_unconverted_scalar_contrib_float_function_is_rejected(
        self,
    ) -> None:
        """Scalar FLOAT storage in an uninferred function stays FP32."""
        model = _build_function_with_scalar_contrib_float_model()
        original = model.SerializeToString()

        checker.check_model(model)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"x": np.array([3.0], dtype=np.float32)})
        assert output.dtype == np.float32

        with np.testing.assert_raises_regex(RuntimeError, "local function"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_unrelated_graph_attribute_does_not_supply_input_type(self) -> None:
        """Graph attributes require schema-proven input alignment."""
        model = _build_function_with_unrelated_graph_attribute_model()
        original = model.SerializeToString()

        checker.check_model(model)
        with np.testing.assert_raises_regex(RuntimeError, "missing FLOAT metadata"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=["Gelu"],
            )
        assert model.SerializeToString() == original

    def test_always_float_op_graph_attribute_is_not_traversed(self) -> None:
        """Preflight mirrors ORT when an always-float op owns a graph attribute."""
        model = _build_always_float_function_graph_attribute_model()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["Identity"],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"x": np.array([2.0], dtype=np.float16)})
        assert output.dtype == np.float16

    def test_scan8_child_formal_proves_variadic_state_is_int(self) -> None:
        """A fixed node-only input does not shift heterogeneous child formals."""
        model = _build_scan8_int_state_model()

        checker.check_model(model)
        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["MurmurHash3"],
        )
        checker.check_model(result)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        final_state, scan_output = session.run(
            None,
            {
                "text": np.array(["hello"]),
                "scan_input": np.array([[1.0, 2.0]], dtype=np.float16),
            },
        )
        assert final_state.dtype == np.int32
        assert scan_output.dtype == np.float16

    def test_scan8_float_state_requires_precision_boundary_metadata(self) -> None:
        """A FLOAT child formal proves positional state coupling."""
        model = _build_scan8_float_state_model()
        original = model.SerializeToString()

        checker.check_model(model)
        with np.testing.assert_raises_regex(RuntimeError, "missing FLOAT metadata"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=["Gelu"],
            )
        assert model.SerializeToString() == original

    def test_loop_feedback_output_proves_float_state_coupling(self) -> None:
        """A positionally aligned FLOAT feedback output requires a boundary."""
        model = _build_loop_float_state_model()
        original = model.SerializeToString()

        checker.check_model(model)
        with np.testing.assert_raises_regex(RuntimeError, "missing FLOAT metadata"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=["Gelu"],
            )
        assert model.SerializeToString() == original

    def test_metadata_free_int_input_to_blocked_node_is_preserved(self) -> None:
        """Concrete consumer output evidence prevents a spurious FLOAT Cast."""
        model = _build_uninferred_int_identity_consumer_model()

        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["Identity"],
        )

        checker.check_model(result)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"text": np.array(["hello"])})
        assert output.dtype == np.int32

    def test_optional_float_blocked_capture_is_rejected(self) -> None:
        """A blocked child cannot retain FLOAT for an Optional that becomes FP16."""
        model = _build_optional_float_blocked_capture_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "x": np.array([2.0], dtype=np.float32),
                "condition": np.array(True),
            },
        )
        np.testing.assert_array_equal(output, np.array([2.0], dtype=np.float32))

        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["If"])
        assert model.SerializeToString() == original

    def test_kept_input_optional_producer_uses_generated_fp16_alias(self) -> None:
        """Producer tracing follows the FP16 side of a kept input Cast."""
        model = _build_optional_float_blocked_capture_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(
                model,
                keep_io_types=True,
                op_block_list=["If"],
            )
        assert model.SerializeToString() == original

    def test_blocked_float_can_feed_unconverted_optional_output(self) -> None:
        """Optional metadata does not force its FLOAT payload to FP16."""
        model = _build_blocked_float_optional_output_model()

        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["Gelu"],
        )

        checker.check_model(result)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"x": np.array([2.0], dtype=np.float16)})
        assert output.dtype == np.float32

    def test_converted_float_cannot_feed_unconverted_optional_output(
        self,
    ) -> None:
        """Optional payload declarations must match converted tensor inputs."""
        model = _build_float_optional_output_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"x": np.array([2.0], dtype=np.float32)})
        assert output.dtype == np.float32

        with np.testing.assert_raises_regex(RuntimeError, "incompatible FP16 types"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_blocked_tensor_cast_cannot_feed_unconverted_optional(self) -> None:
        """A late output Cast cannot change an Optional payload silently."""
        model = _build_blocked_tensor_optional_consumer_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        with np.testing.assert_raises_regex(RuntimeError, "container payload declaration"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=["Abs"],
            )
        assert model.SerializeToString() == original

    def test_blocked_float_optional_capture_remains_fp32(self) -> None:
        """A blocked child may capture an Optional whose payload stays FP32."""
        model = _build_blocked_float_optional_capture_model()

        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["Gelu", "If"],
        )

        checker.check_model(result)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "x": np.array([2.0], dtype=np.float16),
                "condition": np.array(True),
            },
        )
        assert output.dtype == np.float16

    def test_empty_optional_blocked_capture_remains_source_less(self) -> None:
        """An input-less Optional has no FLOAT producer to convert."""
        model = _build_empty_optional_blocked_capture_model()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["If"],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output, converted = session.run(
            None,
            {
                "condition": np.array(True),
                "x": np.array([2.0], dtype=np.float16),
            },
        )
        assert output == np.array(False)
        assert converted.dtype == np.float16

    def test_loop_optional_output_preserves_top_input_precision(self) -> None:
        """A graph-bearing producer can pass through unchanged Optional state."""
        model = _build_loop_optional_output_blocked_capture_model()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["If"],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output, converted = session.run(
            None,
            {
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "optional": np.array([2.0], dtype=np.float32),
                "x": np.array([3.0], dtype=np.float16),
            },
        )
        assert output == np.array(True)
        assert converted.dtype == np.float16

    def test_always_float_executed_attribute_capture_is_rejected(self) -> None:
        """Skipped executed attributes cannot retain a converted capture."""
        model = _build_always_float_function_capture_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        converted, selected = session.run(None, {"condition": np.array(True)})
        assert converted.dtype == np.float32
        assert selected.dtype == np.float32

        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=[],
            )
        assert model.SerializeToString() == original

    def test_always_float_nested_attribute_capture_is_rejected(self) -> None:
        """Function attribute references inside body graphs still execute."""
        model = _build_always_float_nested_function_capture_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        converted, selected = session.run(None, {"condition": np.array(True)})
        assert converted.dtype == np.float32
        assert selected.dtype == np.bool_

        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=[],
            )
        assert model.SerializeToString() == original

    def test_always_float_default_attribute_capture_is_rejected(self) -> None:
        """Default graph attributes execute when invocation values are absent."""
        model = _build_always_float_default_function_capture_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        converted, selected = session.run(None, {"condition": np.array(True)})
        assert converted.dtype == np.float32
        assert selected.dtype == np.float32

        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=[],
            )
        assert model.SerializeToString() == original

    def test_always_float_unused_attribute_capture_is_ignored(self) -> None:
        """Only local-function attributes referenced by its body execute."""
        model = _build_always_float_unused_function_capture_model()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=[],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output, converted = session.run(None, {"x": np.array([2.0], dtype=np.float16)})
        assert output.dtype == np.float16
        assert converted.dtype == np.float16

    def test_skipped_scope_duplicate_initializer_is_allowed(self) -> None:
        """Initializer name checks visit only scopes ORT traverses."""
        model = _build_skipped_duplicate_initializer_model()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=[],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"x": np.array([2.0], dtype=np.float16)})
        assert output.dtype == np.float16

    def test_nested_optional_input_blocked_capture_is_rejected(self) -> None:
        """A nested FLOAT container input may receive a converted parent value."""
        model = _build_nested_optional_input_blocked_capture_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "x": np.array([2.0], dtype=np.float32),
            },
        )
        assert output == np.array(True)

        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["If"])
        assert model.SerializeToString() == original

    def test_top_optional_input_nested_blocked_capture_is_preserved(self) -> None:
        """A top Optional input remains FLOAT when carried into a nested graph."""
        model = _build_top_optional_input_blocked_capture_model()

        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["If"],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "optional": np.array([2.0], dtype=np.float32),
            },
        )
        assert output == np.array(True)

    def test_identity_optional_feedback_is_preserved(self) -> None:
        """A type-preserving producer may rename unchanged feedback."""
        model = _build_identity_optional_feedback_blocked_capture_model()

        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["If"],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "optional": np.array([2.0], dtype=np.float32),
            },
        )
        assert output == np.array(True)

    def test_rewrapped_optional_feedback_conversion_is_rejected(self) -> None:
        """Converted tensor intermediates change rewrapped feedback payloads."""
        model = _build_rewrapped_optional_feedback_blocked_capture_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=["If"],
            )
        assert model.SerializeToString() == original

    def test_optional_identity_blocked_capture_is_preserved(self) -> None:
        """An unchanged Optional pass-through stays independent of FP16 tensors."""
        model = _build_optional_identity_blocked_capture_model()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["If"],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output, converted = session.run(
            None,
            {
                "condition": np.array(True),
                "optional": np.array([2.0], dtype=np.float32),
                "x": np.array([3.0], dtype=np.float16),
            },
        )
        assert output == np.array(True)
        assert converted.dtype == np.float16

    def test_long_optional_identity_chain_is_iterative(self) -> None:
        """Long precision-preserving chains do not consume Python stack."""
        model = _build_long_optional_identity_chain_model()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["If"],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "condition": np.array(True),
                "optional": np.array([2.0], dtype=np.float32),
            },
        )
        assert output == np.array(True)

    def test_free_optional_identity_uses_outer_precision_source(self) -> None:
        """Producer tracing resolves free inputs to their lexical owner."""
        model = _build_free_optional_identity_blocked_capture_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "state": np.array(True),
                "x": np.array([2.0], dtype=np.float32),
            },
        )
        assert output == np.array(True)

        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=["If"],
            )
        assert model.SerializeToString() == original

    def test_optional_loop_feedback_conversion_is_rejected(self) -> None:
        """Later loop-carried values must be included in nested input analysis."""
        model = _build_optional_feedback_blocked_capture_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "trip_count": np.array(2, dtype=np.int64),
                "condition": np.array(True),
                "optional": np.array([2.0], dtype=np.float32),
                "feedback": np.array([3.0], dtype=np.float32),
            },
        )
        assert output == np.array(True)

        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["If"])
        assert model.SerializeToString() == original

    def test_mispositioned_optional_feedback_is_rejected(self) -> None:
        """A same-named output in another state slot is not unchanged feedback."""
        model = _build_mispositioned_optional_feedback_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "trip_count": np.array(2, dtype=np.int64),
                "condition": np.array(True),
                "optional_a": np.array([1.0], dtype=np.float32),
                "optional_b": np.array([2.0], dtype=np.float32),
                "feedback": np.array([3.0], dtype=np.float32),
            },
        )
        assert output == np.array(True)

        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["If"])
        assert model.SerializeToString() == original

    def test_scan_output_does_not_shift_optional_feedback_slot(self) -> None:
        """Scan outputs cannot change loop-state feedback alignment."""
        model = _build_scan_mispositioned_optional_feedback_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)

        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["If"])
        assert model.SerializeToString() == original

    def test_scan_output_preserves_unchanged_optional_feedback_slot(self) -> None:
        """Scan outputs do not hide an unchanged loop-carried binding."""
        model = _build_scan_top_optional_input_blocked_capture_model()

        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=["If"],
        )

        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "optional": np.array([2.0], dtype=np.float32),
            },
        )
        assert output == np.array(True)

    def test_blocked_float_sequence_input_is_rejected_before_mutation(self) -> None:
        """ORT's late tensor Cast cannot consume a sequence value."""
        model = _build_blocked_float_sequence_input_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (length,) = session.run(None, {"sequence": [np.array([1.0], dtype=np.float32)]})
        assert length == 1

        with np.testing.assert_raises_regex(RuntimeError, "non-tensor"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=["SequenceLength"],
            )
        assert model.SerializeToString() == original

    def test_overridable_blocked_initializer_is_rejected_before_mutation(self) -> None:
        """A converted graph input cannot retain a FLOAT default initializer."""
        model = _build_overridable_blocked_initializer_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)

        with np.testing.assert_raises_regex(RuntimeError, "initializer-backed graph input"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["Identity"])
        assert model.SerializeToString() == original

    def test_duplicate_float_output_name_is_rejected_before_mutation(self) -> None:
        """ORT cannot map repeated kept outputs to two generated Cast aliases."""
        model = _build_duplicate_float_output_name_model()
        original = model.SerializeToString()

        checker.check_model(model)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        first, second = session.run(None, {"x": np.array([3.0], dtype=np.float32)})
        np.testing.assert_array_equal(first, second)

        with np.testing.assert_raises_regex(RuntimeError, "repeated FLOAT output"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_shared_float_input_output_name_is_rejected_before_mutation(self) -> None:
        """ORT overwrites keep-I/O mappings when one name is both input and output."""
        model = _build_shared_float_input_output_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(None, {"x": np.array([3.0], dtype=np.float32)})
        np.testing.assert_array_equal(output, np.array([3.0], dtype=np.float32))

        with np.testing.assert_raises_regex(RuntimeError, "both input and output"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_nested_initializer_output_shadowing_is_rejected_before_mutation(self) -> None:
        """Nested shadowing is rejected before ORT's global initializer map."""
        model = _build_nested_shadowed_initializer_output_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "duplicate FLOAT initializer names"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_duplicate_non_output_initializers_are_rejected_before_mutation(self) -> None:
        """Duplicate FLOAT initializer names are rejected before ORT mutates the graph."""
        model = _build_duplicate_non_output_initializer_name_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "duplicate FLOAT initializer names"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_duplicate_initializers_under_blocked_if_are_not_rejected(self) -> None:
        """Duplicate local initializers under blocked nodes are not traversed by ORT."""
        model = _build_blocked_if_duplicate_local_initializer_model()

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=["If"])

        assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_blocked_shadowed_consumers_do_not_make_output_initializer_shared(self) -> None:
        """Blocked branch-local values do not consume a top-level output initializer."""
        model = _build_blocked_if_shadowed_output_initializer_model()

        result = convert_to_fp16(model, keep_io_types=True, op_block_list=["If"])

        assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
        assert result.graph.initializer[0].data_type == TensorProto.FLOAT
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        np.testing.assert_array_equal(
            session.run(None, {"condition": np.array(True)})[0],
            np.array([9.0], dtype=np.float32),
        )

    def test_traversed_shadowed_output_initializer_references_are_rejected(self) -> None:
        """Traversed nested refs with the same name hit ORT's global I/O name mapping."""
        model = _build_traversed_shadowed_output_initializer_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "scope-aware"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_nested_input_shadowing_initializer_output_is_rejected_before_mutation(
        self,
    ) -> None:
        """Traversed nested input declarations can hit ORT's global I/O name mapping."""
        model = _build_loop_state_input_shadowing_output_initializer_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "scope-aware"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_nested_value_info_does_not_shadow_initializer_output(
        self,
    ) -> None:
        """A value_info annotation alone does not create a local binding."""
        model = _build_loop_value_info_shadowing_output_initializer_model()

        result = convert_to_fp16(model, keep_io_types=True, op_block_list=[])

        checker.check_model(result)
        shape_inference.infer_shapes(result, check_type=True, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        same, loop_output = session.run(
            None,
            {
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "loop_state": np.array([2.0], dtype=np.float32),
            },
        )
        np.testing.assert_array_equal(same, np.array([9.0], dtype=np.float32))
        np.testing.assert_array_equal(loop_output, np.array([2.0], dtype=np.float32))

    def test_nested_shadowed_top_level_input_references_are_rejected(self) -> None:
        """Top-level input casts are also globally mapped by ORT keep_io_types."""
        model = _build_nested_consumed_initializer_output_with_top_level_input_collision_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "scope-aware"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_nested_output_initializer_matching_kept_top_level_io_is_repaired(self) -> None:
        """Nested direct output repair overrides ORT's global top-level I/O skip name."""
        model = _build_nested_output_initializer_with_top_level_io_name_collision_model()

        result = convert_to_fp16(model, keep_io_types=True, op_block_list=[])

        [body] = _iter_attribute_graphs(result)
        assert body.output[1].type.tensor_type.elem_type == TensorProto.FLOAT16
        assert body.initializer[0].data_type == TensorProto.FLOAT16
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_mapped_free_capture_does_not_bind_to_earlier_nested_initializer(self) -> None:
        """ORT tracks the mapped top-level capture separately from a nested initializer."""
        model = _build_nested_initializer_then_mapped_free_capture_model()

        result = convert_to_fp16(model, keep_io_types=True, op_block_list=[])

        assert result.graph.output[-1].type.tensor_type.elem_type == TensorProto.FLOAT
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_mapped_capture_shadowed_by_generated_alias_is_rejected(self) -> None:
        """A nested binding must not intercept a generated top-level I/O alias."""
        model = _build_mapped_capture_shadowed_by_generated_alias_model()
        original = model.SerializeToString()

        checker.check_model(model)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (output,) = session.run(
            None,
            {
                "x": np.array([3.0], dtype=np.float32),
                "condition": np.array(True),
            },
        )
        np.testing.assert_array_equal(output, np.array([3.0], dtype=np.float32))

        with np.testing.assert_raises_regex(RuntimeError, "generated alias"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_nested_blocked_ordinary_name_collision_is_rejected(self) -> None:
        """A blocked nested input must retain its local lexical binding."""
        model = _build_nested_blocked_ordinary_name_collision_model()
        original = model.SerializeToString()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (selected,) = session.run(
            None,
            {
                "condition": np.array(True),
                "same": np.array(0.0, dtype=np.float32),
            },
        )
        assert selected.shape == (1, 3)

        with np.testing.assert_raises_regex(RuntimeError, "global value-info"):
            convert_to_fp16(
                model,
                keep_io_types=False,
                op_block_list=["NonMaxSuppression"],
            )
        assert model.SerializeToString() == original

    def test_blocked_free_capture_prevents_pure_fp16_initializer_repair(self) -> None:
        """Blocked subgraphs can still consume outer initializers at runtime."""
        model = _build_blocked_if_free_capture_output_initializer_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "blocked subgraph"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["If"])
        assert model.SerializeToString() == original

    def test_blocked_free_capture_of_shadowed_value_does_not_reject_outer_initializer(self) -> None:
        """Blocked descendants that capture local shadows do not consume outer initializers."""
        model = _build_blocked_if_shadowed_free_capture_model()

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=["If"])

        assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16
        assert result.graph.initializer[0].data_type == TensorProto.FLOAT16
        [body] = _iter_attribute_graphs(result)
        assert body.initializer[0].data_type == TensorProto.FLOAT16
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_unconsumed_nested_initializer_outputs_are_converted_to_fp16(self) -> None:
        """Unconsumed nested direct output initializers are repaired after conversion."""
        model = _build_nested_initializer_output_model()
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([1.0], dtype=np.float16), "top_level_fp16")
        )

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        assert result.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT16
        for graph in _iter_attribute_graphs(result):
            assert all(
                output.type.tensor_type.elem_type == TensorProto.FLOAT16 for output in graph.output
            )
            assert all(
                initializer.data_type == TensorProto.FLOAT16 for initializer in graph.initializer
            )
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_blocked_fp32_consumer_does_not_round_trip_through_fp16(self) -> None:
        """Blocked consumers are rejected instead of silently losing precision."""
        model = _build_blocked_initializer_consumer_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "initializer metadata"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["Identity"])
        assert model.SerializeToString() == original

    def test_nested_local_input_cannot_change_outer_initializer_precision(self) -> None:
        """ORT's global initializer tracker must not treat local shadows as consumers."""
        model = _build_blocked_initializer_with_nested_input_shadow_model()
        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(
            RuntimeError, "initializer tracking is not scope-aware"
        ):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["Abs"])
        assert model.SerializeToString() == original

    def test_harmless_initializer_consumer_misattribution_is_allowed(self) -> None:
        """A false FP16 consumer is harmless when the initializer already converts."""
        model = _build_blocked_initializer_with_nested_input_shadow_model()
        model.graph.node[0].op_type = "Relu"
        model.graph.node[0].name = "top_relu"

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        assert result.graph.initializer[0].data_type == TensorProto.FLOAT16
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        blocked_output, loop_output = session.run(
            None,
            {
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "loop_state": np.array([2.0], dtype=np.float16),
            },
        )
        np.testing.assert_allclose(blocked_output, np.array([1.0], dtype=np.float16), atol=1e-3)
        np.testing.assert_array_equal(loop_output, np.array([2.0], dtype=np.float16))

    def test_unused_initializer_shadow_misattribution_is_allowed(self) -> None:
        """An unobservable initializer may be converted by a local name collision."""
        model = _build_unused_initializer_with_nested_input_shadow_model()

        checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)
        result = convert_to_fp16(
            model,
            keep_io_types=False,
            op_block_list=[],
        )

        assert result.graph.initializer[0].data_type == TensorProto.FLOAT16
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (loop_output,) = session.run(
            None,
            {
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "loop_state": np.array([2.0], dtype=np.float16),
            },
        )
        np.testing.assert_array_equal(loop_output, np.array([2.0], dtype=np.float16))

    def test_output_repair_makes_initializer_misattribution_harmless(self) -> None:
        """A direct output repair can independently require initializer conversion."""
        model = _build_loop_state_input_shadowing_output_initializer_model()
        [body] = _iter_attribute_graphs(model)
        del body.initializer[:]
        body.node.append(helper.make_node("Relu", ["same"], ["body_state_out"], name="body_state"))

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        assert result.graph.initializer[0].data_type == TensorProto.FLOAT16
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        same, loop_output = session.run(
            None,
            {
                "trip_count": np.array(1, dtype=np.int64),
                "condition": np.array(True),
                "loop_state": np.array([2.0], dtype=np.float16),
            },
        )
        np.testing.assert_array_equal(same, np.array([9.0], dtype=np.float16))
        np.testing.assert_array_equal(loop_output, np.array([2.0], dtype=np.float16))

    def test_later_nested_float_initializer_does_not_rebind_earlier_non_float_input(
        self,
    ) -> None:
        """Initializer tracking follows ORT registration order across graph scopes."""
        model = _build_non_float_initializer_before_nested_float_initializer_model()

        result = convert_to_fp16(model, keep_io_types=False, op_block_list=[])

        assert result.graph.initializer[0].data_type == TensorProto.INT64
        same = next(
            initializer
            for graph in _iter_attribute_graphs(result)
            for initializer in graph.initializer
            if initializer.name == "same"
        )
        assert same.data_type == TensorProto.FLOAT16
        checker.check_model(result)
        session = ort.InferenceSession(
            result.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        integer_output, nested_output = session.run(None, {"condition": np.array(True)})
        np.testing.assert_array_equal(integer_output, np.array([7], dtype=np.int64))
        np.testing.assert_array_equal(nested_output, np.array([1.5], dtype=np.float16))

    def test_initializer_output_repair_preserves_unrelated_casts(self) -> None:
        """Repair removes only ORT's orphan output Cast, not user graph Casts."""
        model = _build_initializer_backed_output_model()
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        cast_output = helper.make_tensor_value_info("cast_output", TensorProto.INT32, [1])
        model.graph.input.append(x)
        model.graph.output.append(cast_output)
        model.graph.node.append(
            helper.make_node(
                "Cast",
                ["x"],
                ["cast_output"],
                name="user_cast",
                to=TensorProto.INT32,
            )
        )

        result = convert_to_fp16(model, keep_io_types=True)

        checker.check_model(result)
        assert any(node.name == "user_cast" for node in result.graph.node)

    def test_preserves_model_structure(self) -> None:
        """FP16 conversion preserves graph structure (node count diff ≤ 2)."""
        model = _build_multi_op_fp32_model()
        original_count = len(model.graph.node)
        result = convert_to_fp16(model, keep_io_types=True)
        converted_count = len(result.graph.node)

        assert converted_count - original_count <= 2, (
            f"Node count changed from {original_count} to {converted_count}, "
            f"difference {converted_count - original_count} exceeds threshold of 2"
        )

    def test_op_block_list_keeps_ops_in_fp32(self) -> None:
        """Ops in block list should remain operating on FP32 data."""
        model = _build_multi_op_fp32_model()
        result = convert_to_fp16(model, op_block_list=["Relu"])

        op_types = [n.op_type for n in result.graph.node]
        assert "Cast" in op_types, "Expected Cast nodes for blocked ops"

    def test_none_op_block_list_uses_ort_defaults(self) -> None:
        """When op_block_list is None, ORT uses its DEFAULT_OP_BLOCK_LIST."""
        model = _build_simple_fp32_model()
        # Should not raise — ORT applies its default safety list
        result = convert_to_fp16(model, op_block_list=None)
        assert result is not None

    def test_preserves_already_fp16_model_without_casts(self) -> None:
        """Already-FP16 graph I/O and initializers remain FP16 without extra Casts."""
        # Build a model with FP16 initializers directly
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 4])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT16, [1, 4])
        weight_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16)
        weight = numpy_helper.from_array(weight_data, "weight")
        add = helper.make_node("Add", ["x", "weight"], ["out"], name="add")
        graph = helper.make_graph([add], "fp16_model", [x], [out], [weight])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        result = convert_to_fp16(model)

        assert all(
            value.type.tensor_type.elem_type == TensorProto.FLOAT16
            for value in (*result.graph.input, *result.graph.output)
        )
        assert all(
            initializer.data_type == TensorProto.FLOAT16 for initializer in result.graph.initializer
        )
        assert all(node.op_type != "Cast" for node in result.graph.node)
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)

    def test_preserves_fp16_model_with_int_initializers_without_casts(self) -> None:
        """FP16 graph with INT64 shape initializers remains FP16 without extra Casts."""
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 4])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT16, [1, 4])
        weight_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16)
        weight = numpy_helper.from_array(weight_data, "weight")
        # INT64 initializer (e.g., shape tensor) — should be ignored by skip logic
        shape_tensor = numpy_helper.from_array(np.array([1, 4], dtype=np.int64), "shape")
        add = helper.make_node("Add", ["x", "weight"], ["out"], name="add")
        graph = helper.make_graph([add], "fp16_mixed", [x], [out], [weight, shape_tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        result = convert_to_fp16(model)

        assert all(
            value.type.tensor_type.elem_type == TensorProto.FLOAT16
            for value in (*result.graph.input, *result.graph.output)
        )
        assert any(
            initializer.name == "shape" and initializer.data_type == TensorProto.INT64
            for initializer in result.graph.initializer
        )
        assert all(node.op_type != "Cast" for node in result.graph.node)
        checker.check_model(result)
        shape_inference.infer_shapes(result, strict_mode=True)
