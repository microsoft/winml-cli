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
from onnx import ModelProto, TensorProto, helper, numpy_helper

from winml.modelkit.optim import convert_to_fp16


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
