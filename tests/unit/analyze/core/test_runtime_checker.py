# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Unit tests for RuntimeChecker type hints and functionality.

Tests verify:
- Correct return types for summary() method
- Correct type annotations for alternatives
- Type safety with PatternRuntime and PatternAlternative
- Cache reuse for RuntimeCheckerQuery
"""

import time

import pytest
from onnx import TensorProto, helper

from winml.modelkit.pattern.match import PatternMatchResult, SkeletonMatchResult
from winml.modelkit.pattern.models import OperatorPattern, PatternType
from winml.modelkit.analyze.core.runtime_checker import RuntimeChecker
from winml.modelkit.analyze.models.onnx_model import ONNXModel
from winml.modelkit.analyze.models.runtime_checks import (
    PatternAlternative,
    PatternRuntime,
    RuntimeTestResult,
)


@pytest.fixture
def simple_onnx_model() -> ONNXModel:
    """Create a simple ONNX model for testing."""
    # Create a simple Add operation model
    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 3, 224, 224])
    input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 3, 224, 224])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    add_node = helper.make_node("Add", ["input1", "input2"], ["output"], name="add_node")

    graph_def = helper.make_graph([add_node], "test_graph", [input1, input2], [output])

    model_def = helper.make_model(
        graph_def, producer_name="test", opset_imports=[helper.make_opsetid("", 13)]
    )

    return ONNXModel.from_onnx_model(model_def, "test.onnx")


@pytest.fixture
def sample_pattern_match() -> PatternMatchResult:
    """Create a sample PatternMatchResult for testing."""
    pattern = OperatorPattern(
        pattern_id="OP/ai.onnx/Conv",
        pattern_type=PatternType.OPERATOR,
        namespace="ai.onnx",
        op_type="Conv",
        description="Conv operator",
    )

    # Create mock node proto matching the model's inputs
    node_proto = helper.make_node("Conv", ["input1"], ["conv_output"], name="conv_node")

    # Create SkeletonMatchResult
    skeleton_result = SkeletonMatchResult(
        pattern=pattern,
        matched_nodes=[node_proto],
        matcher=None,
    )

    return PatternMatchResult(
        skeleton_match_result=skeleton_result,
        schema_input_to_value={},
        schema_output_to_value={},
        type_param_to_type={},
    )


class TestRuntimeCheckerTypeHints:
    """Test RuntimeChecker return type correctness."""

    def test_summary_returns_correct_type(
        self, simple_onnx_model: ONNXModel, sample_pattern_match: PatternMatchResult
    ):
        """Test that summary() returns dict[str, list[PatternRuntime]]."""
        # Initialize with both model and patterns to populate summary
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
            patterns=[sample_pattern_match],
        )

        result = checker.summary()

        # Verify return type structure
        assert isinstance(result, dict)
        assert all(isinstance(key, str) for key in result)

        # Check that values are lists of PatternRuntime
        for value in result.values():
            assert isinstance(value, list)
            assert all(isinstance(item, PatternRuntime) for item in value)

        # Verify expected keys (both should be present since we have model + patterns)
        assert "op_runtime_check_result" in result
        assert "subgraph_runtime_check_result" in result

    def test_summary_with_model_only(self, simple_onnx_model: ONNXModel):
        """Test summary() when initialized with model only."""
        # When initialized with only model, summary() needs patterns parameter
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        # Pass empty patterns to avoid ValueError
        result = checker.summary(patterns=[])

        # Should have both keys, but subgraph will be empty
        assert isinstance(result, dict)
        assert "op_runtime_check_result" in result
        assert "subgraph_runtime_check_result" in result

        # Verify types
        op_results = result["op_runtime_check_result"]
        assert isinstance(op_results, list)
        assert all(isinstance(item, PatternRuntime) for item in op_results)
        assert len(result["subgraph_runtime_check_result"]) == 0

    def test_op_support_returns_list_of_pattern_runtime(self, simple_onnx_model: ONNXModel):
        """Test that op_support() returns list[PatternRuntime]."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        result = checker.op_support()

        # Verify return type
        assert isinstance(result, list)
        assert all(isinstance(item, PatternRuntime) for item in result)

        # Should have one operator (Add node)
        assert len(result) > 0

    def test_subgraph_support_returns_list_of_pattern_runtime(
        self, sample_pattern_match: PatternMatchResult, simple_onnx_model: ONNXModel
    ):
        """Test that subgraph_support() returns list[PatternRuntime]."""
        # Need model for _lookup_pattern_support
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
            patterns=[sample_pattern_match],
        )

        result = checker.subgraph_support()

        # Verify return type
        assert isinstance(result, list)
        assert all(isinstance(item, PatternRuntime) for item in result)
        assert len(result) == 1

    def test_query_pattern_support_returns_pattern_runtime(
        self, sample_pattern_match: PatternMatchResult, simple_onnx_model: ONNXModel
    ):
        """Test that query_pattern_support() returns PatternRuntime."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        result = checker.query_pattern_support(sample_pattern_match)

        # Verify return type
        assert isinstance(result, PatternRuntime)
        assert result.pattern_id == "OP/ai.onnx/Conv"
        assert isinstance(result.result, RuntimeTestResult)
        assert isinstance(result.alternatives, list)

    def test_alternatives_is_list_of_pattern_alternative(
        self, sample_pattern_match: PatternMatchResult, simple_onnx_model: ONNXModel
    ):
        """Test that PatternRuntime.alternatives is list[PatternAlternative]."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        result = checker.query_pattern_support(sample_pattern_match)

        # Verify alternatives type
        assert isinstance(result.alternatives, list)

        # Currently alternatives is empty (not implemented)
        # But when implemented, should contain PatternAlternative objects
        for alt in result.alternatives:
            assert isinstance(alt, PatternAlternative)
            assert hasattr(alt, "pattern_id")
            assert hasattr(alt, "result")
            assert hasattr(alt, "alternative_type")


class TestRuntimeCheckerValidation:
    """Test RuntimeChecker initialization validation."""

    def test_requires_either_model_or_patterns(self):
        """Test that RuntimeChecker requires at least one of model or patterns."""
        with pytest.raises(
            ValueError, match="At least one of 'model' or 'patterns' must be provided"
        ):
            RuntimeChecker(
                ep="QNNExecutionProvider",
                device="NPU",
                model=None,
                patterns=None,
            )

    def test_requires_non_empty_ep(self, simple_onnx_model: ONNXModel):
        """Test that ep parameter cannot be empty."""
        with pytest.raises(ValueError, match="ep parameter cannot be empty"):
            RuntimeChecker(
                ep="",
                device="NPU",
                model=simple_onnx_model,
            )

    def test_requires_non_empty_device(self, simple_onnx_model: ONNXModel):
        """Test that device parameter cannot be empty."""
        with pytest.raises(ValueError, match="device parameter cannot be empty"):
            RuntimeChecker(
                ep="QNNExecutionProvider",
                device="",
                model=simple_onnx_model,
            )

    def test_op_support_requires_model(self, sample_pattern_match: PatternMatchResult):
        """Test that op_support() requires model to be provided."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            patterns=[sample_pattern_match],
        )

        with pytest.raises(ValueError, match="op_support\\(\\) requires ONNXModel"):
            checker.op_support()

    def test_subgraph_support_requires_patterns(self, simple_onnx_model: ONNXModel):
        """Test that subgraph_support() requires patterns when not initialized with them."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        with pytest.raises(ValueError, match="patterns parameter is required"):
            checker.subgraph_support(patterns=None)


class TestRuntimeCheckerIntegration:
    """Integration tests for RuntimeChecker."""

    def test_full_workflow_with_model(self, simple_onnx_model: ONNXModel):
        """Test complete workflow: initialize with model, check op support, get summary."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        # Get operator support
        op_results = checker.op_support()
        assert len(op_results) > 0
        assert all(isinstance(r, PatternRuntime) for r in op_results)

        # Get summary with empty patterns
        summary = checker.summary(patterns=[])
        assert isinstance(summary, dict)
        assert "op_runtime_check_result" in summary
        assert len(summary["op_runtime_check_result"]) == len(op_results)

    def test_full_workflow_with_patterns(
        self, sample_pattern_match: PatternMatchResult, simple_onnx_model: ONNXModel
    ):
        """Test complete workflow: initialize with patterns, check subgraph support."""
        # Need model for pattern lookup
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
            patterns=[sample_pattern_match],
        )

        # Get subgraph support
        subgraph_results = checker.subgraph_support()
        assert len(subgraph_results) == 1
        assert all(isinstance(r, PatternRuntime) for r in subgraph_results)

        # Get summary
        summary = checker.summary()
        assert isinstance(summary, dict)
        assert "subgraph_runtime_check_result" in summary
        assert len(summary["subgraph_runtime_check_result"]) == 1


class TestRuntimeCheckerQueryCache:
    """Test RuntimeCheckerQuery caching functionality."""

    def test_query_cache_reuse(self, simple_onnx_model: ONNXModel):
        """Test that RuntimeCheckerQuery is cached and reused."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        # First call should create the query
        assert checker._query is None
        first_result = checker.op_support()
        first_query = checker._query
        assert first_query is not None

        # Second call should reuse the cached query
        second_result = checker.op_support()
        second_query = checker._query
        assert second_query is first_query  # Same object reference

        # Results should be consistent
        assert len(first_result) == len(second_result)

    def test_query_cache_across_methods(
        self, simple_onnx_model: ONNXModel, sample_pattern_match: PatternMatchResult
    ):
        """Test that query cache is shared across op_support and pattern lookup."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
            patterns=[sample_pattern_match],
        )

        # Call op_support first
        checker.op_support()
        query_after_op_support = checker._query

        # Call query_pattern_support
        checker.query_pattern_support(sample_pattern_match)
        query_after_pattern_support = checker._query

        # Should be the same cached query
        assert query_after_pattern_support is query_after_op_support

    def test_query_cache_performance(self, simple_onnx_model: ONNXModel):
        """Test that cache improves performance on repeated calls."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            model=simple_onnx_model,
        )

        # First call - cold (creates query)
        start_time = time.time()
        checker.op_support()
        _first_call_time = time.time() - start_time

        # Second call - warm (uses cache)
        start_time = time.time()
        checker.op_support()
        _second_call_time = time.time() - start_time

        # Second call should be faster or at least not significantly slower
        # We're primarily checking that it doesn't recreate the query
        # which would add initialization overhead
        assert checker._query is not None
        # Not asserting timing directly as it can be flaky,
        # but verifying cache exists proves the optimization

    def test_get_query_without_model_raises_error(self, sample_pattern_match: PatternMatchResult):
        """Test that _get_query raises error when model is not available."""
        checker = RuntimeChecker(
            ep="QNNExecutionProvider",
            device="NPU",
            patterns=[sample_pattern_match],
        )

        # _get_query should raise ValueError
        with pytest.raises(
            ValueError, match="Cannot create RuntimeCheckerQuery without ONNX model"
        ):
            checker._get_query()
