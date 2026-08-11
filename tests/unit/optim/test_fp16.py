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

import numpy as np
import onnxruntime as ort
from onnx import GraphProto, ModelProto, TensorProto, checker, helper, numpy_helper, shape_inference

from winml.modelkit.quant.fp16 import convert_to_fp16


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

    def test_default_keeps_io_types(self) -> None:
        """Default keep_io_types=True preserves FP32 model I/O."""
        model = _build_simple_fp32_model()
        result = convert_to_fp16(model, keep_io_types=True)

        for inp in result.graph.input:
            assert inp.type.tensor_type.elem_type == TensorProto.FLOAT
        for outp in result.graph.output:
            assert outp.type.tensor_type.elem_type == TensorProto.FLOAT

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

    def test_shared_initializer_output_is_rejected_before_mutation(self) -> None:
        """Shared initializer-output semantics are rejected before mutation."""
        model = _build_shared_initializer_output_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "has internal consumers"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

    def test_nested_initializer_outputs_are_rejected_before_mutation(self) -> None:
        """Nested initializer-output semantics are rejected before mutation."""
        model = _build_nested_initializer_output_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "mismatched types"):
            convert_to_fp16(model, keep_io_types=True, op_block_list=[])
        assert model.SerializeToString() == original

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

    def test_nested_fp32_initializer_prevents_already_fp16_shortcut(self) -> None:
        """Nested FP32 outputs are rejected before the already-FP16 shortcut."""
        model = _build_nested_initializer_output_model()
        model.graph.initializer.append(
            numpy_helper.from_array(np.array([1.0], dtype=np.float16), "top_level_fp16")
        )

        original = model.SerializeToString()
        with np.testing.assert_raises_regex(RuntimeError, "mismatched types"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=[])
        assert model.SerializeToString() == original

    def test_blocked_fp32_consumer_does_not_round_trip_through_fp16(self) -> None:
        """Blocked consumers are rejected instead of silently losing precision."""
        model = _build_blocked_initializer_consumer_model()
        original = model.SerializeToString()

        with np.testing.assert_raises_regex(RuntimeError, "mismatched types"):
            convert_to_fp16(model, keep_io_types=False, op_block_list=["Identity"])
        assert model.SerializeToString() == original

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

    def test_skips_already_fp16_model(self) -> None:
        """If all floating-point initializers are already FP16, conversion is skipped."""
        # Build a model with FP16 initializers directly
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 4])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT16, [1, 4])
        weight_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16)
        weight = numpy_helper.from_array(weight_data, "weight")
        add = helper.make_node("Add", ["x", "weight"], ["out"], name="add")
        graph = helper.make_graph([add], "fp16_model", [x], [out], [weight])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        original_nodes = len(model.graph.node)
        result = convert_to_fp16(model)

        # Should return the same model unchanged (no Cast nodes inserted)
        assert len(result.graph.node) == original_nodes
        assert result is model

    def test_skips_fp16_model_with_int_initializers(self) -> None:
        """FP16 model with non-float initializers (e.g. INT64 shapes) should still skip."""
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 4])
        out = helper.make_tensor_value_info("out", TensorProto.FLOAT16, [1, 4])
        weight_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16)
        weight = numpy_helper.from_array(weight_data, "weight")
        # INT64 initializer (e.g., shape tensor) — should be ignored by skip logic
        shape_tensor = numpy_helper.from_array(np.array([1, 4], dtype=np.int64), "shape")
        add = helper.make_node("Add", ["x", "weight"], ["out"], name="add")
        graph = helper.make_graph([add], "fp16_mixed", [x], [out], [weight, shape_tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        original_nodes = len(model.graph.node)
        result = convert_to_fp16(model)

        assert len(result.graph.node) == original_nodes
        assert result is model
