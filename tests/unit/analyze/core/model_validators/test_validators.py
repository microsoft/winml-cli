# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for model validators.

Tests cover:
- ConstantFoldingValidator detection and recommendations
- ShapeInferenceValidator detection and recommendations
- ModelValidatorManager operation and integration
"""

from __future__ import annotations

import pytest
from onnx import TensorProto, helper

from tests.unit.test_helpers import stable_test_node_keys as _stable_test_node_keys
from winml.modelkit.analyze import ONNXModel, RuntimeTestResult
from winml.modelkit.analyze.core.model_validators import (
    ConstantFoldingValidator,
    ModelValidatorManager,
)
from winml.modelkit.analyze.models.runtime_checks import (  # Testing internal implementation
    NodeTag,
    PatternRuntime,
)
from winml.modelkit.pattern import (
    OperatorPattern,
    PatternMatchResult,
    PatternType,
    SkeletonMatchResult,
)


def create_onnx_model_wrapper(model_proto):
    """Helper to create ONNXModel from onnx.ModelProto."""
    return ONNXModel.from_onnx_model(model_proto, model_path="test.onnx")


def create_runtime_result_with_tags(
    pattern_id: str, node_name: str, op_type: str, tags: list[NodeTag]
) -> PatternRuntime:
    """Helper to create PatternRuntime with node tags."""
    # Create a simple OperatorPattern
    pattern = OperatorPattern(
        pattern_id=pattern_id,
        pattern_type=PatternType.OPERATOR,
        namespace="ai.onnx",
        op_type=op_type,
    )

    # Create a mock node proto for testing
    from onnx import helper

    node_proto = helper.make_node(op_type, ["input"], ["output"], name=node_name)

    # Create SkeletonMatchResult
    skeleton_result = SkeletonMatchResult(
        pattern=pattern,
        matched_nodes=[node_proto],
        matched_node_keys=_stable_test_node_keys([node_proto]),
        matcher=None,
    )

    # Create PatternMatchResult
    pattern_match = PatternMatchResult(
        skeleton_match_result=skeleton_result,
        schema_input_to_value={},
        schema_output_to_value={},
        type_param_to_type={},
    )

    return PatternRuntime(
        pattern_id=pattern_id,
        result=RuntimeTestResult(
            compile=True,
            run=True,
            no_data=False,
            node_tags=tags,
        ),
        alternatives=[],
        pattern_match=pattern_match,
    )


class TestConstantFoldingValidator:
    """Test constant folding validator."""

    def test_detect_constant_only_nodes(self):
        """Test detection of nodes with all constant inputs."""
        # Create model: Constant_A, Constant_B -> Add -> Output
        const_a = helper.make_node(
            "Constant",
            [],
            ["a"],
            value=helper.make_tensor("const_a", TensorProto.FLOAT, [1], [1.0]),
        )
        const_b = helper.make_node(
            "Constant",
            [],
            ["b"],
            value=helper.make_tensor("const_b", TensorProto.FLOAT, [1], [2.0]),
        )
        add = helper.make_node("Add", ["a", "b"], ["output"], name="add_node")

        graph = helper.make_graph(
            [const_a, const_b, add],
            "test",
            [],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        )
        model_proto = helper.make_model(graph)
        model = create_onnx_model_wrapper(model_proto)

        # Create runtime result with ALL_INPUTS_CONSTANT tag
        op_runtime_results = [
            create_runtime_result_with_tags(
                "OP/ai.onnx/Add", "add_node", "Add", [NodeTag.ALL_INPUTS_CONSTANT]
            )
        ]

        validator = ConstantFoldingValidator(model, op_runtime_results=op_runtime_results)
        info = validator.validate()

        assert info is not None
        assert info.pattern_id == "MODEL/ConstantFolding"
        assert len(info.actions) == 1
        # Check that explanation mentions constant-only nodes
        assert "constant inputs" in info.explanation
        # Check that details contain tool recommendations (JSON format)
        assert "winml optimize" in info.actions[0].details

    def test_no_constant_only_nodes(self):
        """Test that models without constant-only nodes return None."""
        # Create model: Input -> Relu -> Output (no constant-only nodes)
        relu = helper.make_node("Relu", ["input"], ["output"], name="relu_node")

        graph = helper.make_graph(
            [relu],
            "test",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
        )
        model_proto = helper.make_model(graph)
        model = create_onnx_model_wrapper(model_proto)

        # Create runtime result WITHOUT ALL_INPUTS_CONSTANT tag
        op_runtime_results = [
            create_runtime_result_with_tags("OP/ai.onnx/Relu", "relu_node", "Relu", [])
        ]

        validator = ConstantFoldingValidator(model, op_runtime_results=op_runtime_results)
        info = validator.validate()

        assert info is None

    def test_constant_folding_with_initializer(self):
        """Test detection with initializer inputs."""
        # Create model with initializer as constant
        const_tensor = helper.make_tensor("weight", TensorProto.FLOAT, [2, 2], [1, 2, 3, 4])

        relu = helper.make_node("Relu", ["weight"], ["output"], name="relu_node")

        graph = helper.make_graph(
            [relu],
            "test",
            [],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 2])],
            initializer=[const_tensor],
        )
        model_proto = helper.make_model(graph)
        model = create_onnx_model_wrapper(model_proto)

        # Create runtime result with ALL_INPUTS_CONSTANT tag
        op_runtime_results = [
            create_runtime_result_with_tags(
                "OP/ai.onnx/Relu", "relu_node", "Relu", [NodeTag.ALL_INPUTS_CONSTANT]
            )
        ]

        validator = ConstantFoldingValidator(model, op_runtime_results=op_runtime_results)
        info = validator.validate()

        assert info is not None
        # Check explanation mentions constant inputs
        assert "constant inputs" in info.explanation
        # Check details contain tool recommendations
        assert "winml optimize" in info.actions[0].details

    def test_explanation_contains_node_count(self):
        """Test that explanation mentions correct node count."""
        # Create two constant-only nodes with explicit Constant ops
        const_a = helper.make_node(
            "Constant",
            [],
            ["a"],
            value=helper.make_tensor("const_a", TensorProto.FLOAT, [1], [1.0]),
        )
        const_b = helper.make_node(
            "Constant",
            [],
            ["b"],
            value=helper.make_tensor("const_b", TensorProto.FLOAT, [1], [2.0]),
        )
        const_c = helper.make_node(
            "Constant",
            [],
            ["c"],
            value=helper.make_tensor("const_c", TensorProto.FLOAT, [1], [3.0]),
        )
        add = helper.make_node("Add", ["a", "b"], ["output1"], name="add_node")
        mul = helper.make_node("Mul", ["c", "c"], ["output2"], name="mul_node")

        graph = helper.make_graph(
            [const_a, const_b, const_c, add, mul],
            "test",
            [],
            [
                helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1]),
                helper.make_tensor_value_info("output2", TensorProto.FLOAT, [1]),
            ],
        )
        model_proto = helper.make_model(graph)
        model = create_onnx_model_wrapper(model_proto)

        # Create runtime results with ALL_INPUTS_CONSTANT tag for both Add and Mul
        op_runtime_results = [
            create_runtime_result_with_tags(
                "OP/ai.onnx/Add", "add_node", "Add", [NodeTag.ALL_INPUTS_CONSTANT]
            ),
            create_runtime_result_with_tags(
                "OP/ai.onnx/Mul", "mul_node", "Mul", [NodeTag.ALL_INPUTS_CONSTANT]
            ),
        ]

        validator = ConstantFoldingValidator(model, op_runtime_results=op_runtime_results)
        info = validator.validate()

        assert info is not None
        # Both Add and Mul have all constant inputs, so we expect 2 nodes
        assert "2 node(s)" in info.explanation


class TestModelValidatorManager:
    """Test model validator manager."""

    def test_get_available_validators(self):
        """Test that available validators are listed."""
        validators = ModelValidatorManager.get_available_validators()

        assert "constant_folding" in validators
        assert "shape_inference" in validators
        assert len(validators) >= 2

    def test_run_all_validators(self):
        """Test running all validators on a model."""
        # Create model with constant-only nodes
        const_a = helper.make_node(
            "Constant",
            [],
            ["a"],
            value=helper.make_tensor("const_a", TensorProto.FLOAT, [1], [1.0]),
        )
        const_b = helper.make_node(
            "Constant",
            [],
            ["b"],
            value=helper.make_tensor("const_b", TensorProto.FLOAT, [1], [2.0]),
        )
        add = helper.make_node("Add", ["a", "b"], ["output"], name="add_node")

        graph = helper.make_graph(
            [const_a, const_b, add],
            "test",
            [],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        )
        model_proto = helper.make_model(graph)
        model = create_onnx_model_wrapper(model_proto)

        # Create runtime result with ALL_INPUTS_CONSTANT tag
        op_runtime_results = [
            create_runtime_result_with_tags(
                "OP/ai.onnx/Add", "add_node", "Add", [NodeTag.ALL_INPUTS_CONSTANT]
            )
        ]

        manager = ModelValidatorManager(
            model, op_runtime_results=op_runtime_results, device="NPU", ep="QNNExecutionProvider"
        )
        information = manager.run_all_validators()

        # Should find at least constant folding issue
        assert len(information) >= 1
        pattern_ids = [info.pattern_id for info in information]
        assert "MODEL/ConstantFolding" in pattern_ids

    def test_selective_validators(self):
        """Test running only specific validators."""
        const_a = helper.make_node(
            "Constant",
            [],
            ["a"],
            value=helper.make_tensor("const_a", TensorProto.FLOAT, [1], [1.0]),
        )
        const_b = helper.make_node(
            "Constant",
            [],
            ["b"],
            value=helper.make_tensor("const_b", TensorProto.FLOAT, [1], [2.0]),
        )
        add = helper.make_node("Add", ["a", "b"], ["output"], name="add_node")

        graph = helper.make_graph(
            [const_a, const_b, add],
            "test",
            [],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        )
        model_proto = helper.make_model(graph)
        model = create_onnx_model_wrapper(model_proto)

        # Create runtime result with ALL_INPUTS_CONSTANT tag
        op_runtime_results = [
            create_runtime_result_with_tags(
                "OP/ai.onnx/Add", "add_node", "Add", [NodeTag.ALL_INPUTS_CONSTANT]
            )
        ]

        # Enable only constant folding
        manager = ModelValidatorManager(
            model,
            enabled_validators=["constant_folding"],
            op_runtime_results=op_runtime_results,
            device="NPU",
            ep="QNNExecutionProvider",
        )

        # Should have exactly one validator
        assert len(manager.validators) == 1
        assert manager.validators[0].validator_name == "ConstantFoldingValidator"

        # Run validators
        information = manager.run_all_validators()
        assert len(information) >= 1

    def test_invalid_model_proto_raises_error(self):
        """Test that invalid model raises AttributeError when trying to get model."""
        # Passing None should fail when trying to call get_model()
        with pytest.raises(AttributeError):
            ModelValidatorManager(None, device="NPU", ep="QNNExecutionProvider")  # type: ignore

        # Passing a string should fail when trying to call get_model()
        with pytest.raises(AttributeError):
            ModelValidatorManager("not a model", device="NPU", ep="QNNExecutionProvider")  # type: ignore

    def test_unknown_validator_logs_warning(self, caplog):
        """Test that unknown validator names are handled gracefully."""
        import logging

        relu = helper.make_node("Relu", ["input"], ["output"])
        graph = helper.make_graph(
            [relu],
            "test",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
        )
        model_proto = helper.make_model(graph)
        model = create_onnx_model_wrapper(model_proto)

        # Create empty runtime results
        op_runtime_results = []

        # Unknown validator should be logged but not cause error
        with caplog.at_level(logging.WARNING):
            manager = ModelValidatorManager(
                model,
                enabled_validators=["unknown_validator", "constant_folding"],
                op_runtime_results=op_runtime_results,
                device="NPU",
                ep="QNNExecutionProvider",
            )

        # Should only have constant_folding validator (unknown ones are skipped)
        assert len(manager.validators) == 1
        assert manager.validators[0].validator_name == "ConstantFoldingValidator"
        assert "Unknown validator" in caplog.text


def _make_batched_const_matmul_proto(const_rank: int = 3):
    """Model: data [2,3,4] @ W(const) [2,4,5] -> out [2,3,5]."""
    import numpy as np
    from onnx import numpy_helper

    w_shape = [2, 4, 5] if const_rank == 3 else [4, 5]
    w = numpy_helper.from_array(np.zeros(w_shape, dtype=np.float32), "W")
    matmul = helper.make_node("MatMul", ["data", "W"], ["out"], name="batched_matmul")
    graph = helper.make_graph(
        [matmul],
        "batched_const_matmul",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 3, 5])],
        initializer=[w],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestBatchedConstMatMulValidator:
    """OpenVINO-GPU batched constant MatMul detector."""

    def _validate(self, proto, ep, device):
        from winml.modelkit.analyze.core.model_validators import BatchedConstMatMulValidator

        model = create_onnx_model_wrapper(proto)
        return BatchedConstMatMulValidator(model, ep=ep, device=device).validate()

    def test_detects_for_openvino_gpu(self):
        """Emits a GraphOptimization action enabling the surgery for OV GPU."""
        info = self._validate(
            _make_batched_const_matmul_proto(), "OpenVINOExecutionProvider", "GPU"
        )
        assert info is not None
        assert info.pattern_id == "MODEL/BatchedConstantMatMul"
        items = info.actions[0].action_items
        assert items[0].type == "GraphOptimization"
        assert items[0].optimization_options == {"untie-constant-batched-matmul": True}

    def test_skipped_for_openvino_npu(self):
        """Device-gated: NPU is unaffected."""
        assert (
            self._validate(_make_batched_const_matmul_proto(), "OpenVINOExecutionProvider", "NPU")
            is None
        )

    def test_skipped_for_non_intel_gpu(self):
        """IHV-gated: a non-Intel GPU EP is unaffected."""
        info = self._validate(_make_batched_const_matmul_proto(), "DmlExecutionProvider", "GPU")
        assert info is None

    def test_skipped_for_two_dim_constant(self):
        """Rank-2 constant gemm compiles on OV GPU; not flagged."""
        info = self._validate(
            _make_batched_const_matmul_proto(const_rank=2), "OpenVINOExecutionProvider", "GPU"
        )
        assert info is None

    def test_manager_wires_validator_for_openvino_gpu(self):
        """Manager constructs the validator and surfaces the action for OV GPU."""
        model = create_onnx_model_wrapper(_make_batched_const_matmul_proto())
        manager = ModelValidatorManager(model, device="GPU", ep="OpenVINOExecutionProvider")
        names = [v.validator_name for v in manager.validators]
        assert "BatchedConstMatMulValidator" in names
        infos = manager.run_all_validators()
        assert any(i.pattern_id == "MODEL/BatchedConstantMatMul" for i in infos)
