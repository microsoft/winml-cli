# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
import pytest
from onnx import StringStringEntryProto, TensorProto, checker, helper, save

from winml.modelkit.quant.fp16 import convert_to_fp16


if TYPE_CHECKING:
    from pathlib import Path

    from onnx import ModelProto, NodeProto


HIERARCHY_KEY = "winml.hierarchy.tag"


def _tag(node: NodeProto, value: str) -> NodeProto:
    node.metadata_props.append(StringStringEntryProto(key=HIERARCHY_KEY, value=value))
    return node


def _repeated_max_model() -> ModelProto:
    nodes = [
        _tag(
            helper.make_node("Max", ["x", "limit"], ["first"], name="first_max"),
            "/Net/First",
        ),
        _tag(
            helper.make_node("Max", ["first", "limit"], ["output"], name="second_max"),
            "/Net/Second",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "repeated_unnamed_nodes",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        [helper.make_tensor("limit", TensorProto.FLOAT, [1], [0.0])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def test_repeated_unnamed_blocked_nodes_produce_valid_model(tmp_path: Path) -> None:
    converted = convert_to_fp16(_repeated_max_model(), keep_io_types=False)

    checker.check_model(converted)
    output_path = tmp_path / "converted.onnx"
    save(converted, output_path)
    ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])


def test_conversion_is_deterministic_and_preserves_output_alias() -> None:
    first = convert_to_fp16(_repeated_max_model(), keep_io_types=False)
    second = convert_to_fp16(_repeated_max_model(), keep_io_types=False)

    first_names = [node.name for node in first.graph.node]
    assert first_names == [node.name for node in second.graph.node]
    assert len(first_names) == len(set(first_names))
    assert [output.name for output in first.graph.output] == ["output"]


def test_conversion_preserves_hierarchy_and_blocked_initializer_precision() -> None:
    converted = convert_to_fp16(_repeated_max_model(), keep_io_types=False)

    assert all(
        initializer.data_type == TensorProto.FLOAT for initializer in converted.graph.initializer
    )
    tags = [
        {prop.key: prop.value for prop in node.metadata_props}.get(HIERARCHY_KEY)
        for node in converted.graph.node
    ]
    assert tags
    assert all(tag in {"/Net/First", "/Net/Second"} for tag in tags)


def test_repeated_subgraphs_with_unsafe_global_value_info_are_rejected() -> None:
    branches = []
    for name in ("then", "else"):
        max_node = _tag(helper.make_node("Max", ["x", "x"], [f"{name}_output"]), f"/Net/{name}")
        branches.append(
            helper.make_graph(
                [max_node],
                name,
                [],
                [helper.make_tensor_value_info(f"{name}_output", TensorProto.FLOAT, [1])],
            )
        )
    if_node = helper.make_node(
        "If",
        ["condition"],
        ["output"],
        then_branch=branches[0],
        else_branch=branches[1],
        name="choose",
    )
    graph = helper.make_graph(
        [if_node],
        "nested",
        [
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    with pytest.raises(RuntimeError, match="global value-info"):
        convert_to_fp16(model, keep_io_types=False)


def test_local_function_nodes_preserve_unnamed_structure() -> None:
    functions = []
    calls = []
    for index in range(2):
        function_name = f"LocalMax{index}"
        functions.append(
            helper.make_function(
                "local",
                function_name,
                ["x"],
                ["y"],
                [helper.make_node("Max", ["x", "x"], ["y"])],
                [helper.make_opsetid("", 18)],
            )
        )
        calls.append(
            helper.make_node(
                function_name,
                ["x" if index == 0 else "first"],
                ["first" if index == 0 else "output"],
                domain="local",
            )
        )
    graph = helper.make_graph(
        calls,
        "functions",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(
        graph,
        functions=functions,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("local", 1)],
    )

    converted = convert_to_fp16(model, keep_io_types=False)

    checker.check_model(converted)
    names = [node.name for node in converted.graph.node]
    names.extend(node.name for function in converted.functions for node in function.node)
    assert all(not name for name in names)


def test_explicit_float_precision_boundary_remains_valid(tmp_path: Path) -> None:
    boundary = helper.make_node(
        "Cast",
        ["x"],
        ["float_x"],
        name="InsertedPrecisionFreeCast_x",
        to=TensorProto.FLOAT,
        doc_string="cast node to cast from float16 to float32 on cpu",
    )
    maximum = helper.make_node("Max", ["float_x", "zero"], ["output"])
    graph = helper.make_graph(
        [boundary, maximum],
        "precision_boundary",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        [helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    converted = convert_to_fp16(model, keep_io_types=False)

    checker.check_model(converted)
    assert all(
        initializer.data_type == TensorProto.FLOAT for initializer in converted.graph.initializer
    )
    converted_boundary = next(
        node for node in converted.graph.node if node.name == "InsertedPrecisionFreeCast_x"
    )
    target = next(
        attribute.i for attribute in converted_boundary.attribute if attribute.name == "to"
    )
    assert target == TensorProto.FLOAT
    assert converted_boundary.input == ["x"]
    assert converted_boundary.output == ["InsertedPrecisionFreeCast_x_output_cast_0"]
    boundary_output_types = {
        value.type.tensor_type.elem_type
        for value in converted.graph.value_info
        if value.name == converted_boundary.output[0]
    }
    assert boundary_output_types == {TensorProto.FLOAT}
    return_cast = next(
        node
        for node in converted.graph.node
        if node.name == "InsertedPrecisionFreeCast_x_output_cast0"
    )
    return_target = next(
        attribute.i for attribute in return_cast.attribute if attribute.name == "to"
    )
    assert return_cast.input == converted_boundary.output
    assert return_cast.output == ["float_x"]
    assert return_target == TensorProto.FLOAT16
    assert next(node for node in converted.graph.node if node.op_type == "Max").name == (
        "winml_fp16_unnamed_0"
    )
    output_path = tmp_path / "precision_boundary.onnx"
    save(converted, output_path)
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    (output,) = session.run(None, {"x": np.array([-1.0], dtype=np.float16)})
    np.testing.assert_array_equal(output, np.array([0.0], dtype=np.float32))


@pytest.mark.parametrize(
    ("op_type", "input_type", "output_type", "attributes"),
    [
        pytest.param("Identity", TensorProto.FLOAT16, TensorProto.FLOAT16, {}, id="wrong-op"),
        pytest.param(
            "Cast",
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            {"to": TensorProto.FLOAT16},
            id="wrong-direction",
        ),
    ],
)
def test_malformed_precision_free_boundary_is_rejected_before_mutation(
    op_type: str,
    input_type: int,
    output_type: int,
    attributes: dict[str, int],
) -> None:
    boundary = helper.make_node(
        op_type,
        ["x"],
        ["boundary_output"],
        name="InsertedPrecisionFreeCast_x",
        doc_string="cast node to cast from float16 to float32 on cpu",
        **attributes,
    )
    graph = helper.make_graph(
        [boundary],
        "malformed_precision_boundary",
        [helper.make_tensor_value_info("x", input_type, [1])],
        [helper.make_tensor_value_info("boundary_output", output_type, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    original = model.SerializeToString()

    with pytest.raises(RuntimeError, match="Malformed FP16-to-FLOAT precision boundary"):
        convert_to_fp16(model, keep_io_types=False)
    assert model.SerializeToString() == original
