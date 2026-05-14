# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Unit tests for EPContextNodeChecker.

Tests verify:
- can_check() method correctly identifies EPContext nodes
- check() method validates partition_name attribute
- Runtime result generation
"""

import pytest
from onnx import helper

from tests.unit.test_helpers import stable_test_node_keys as _stable_test_node_keys
from winml.modelkit.analyze import AlternativeType, RuntimeTestResult
from winml.modelkit.analyze.core.node_checkers.ep_context_node_checker import (
    EPContextNodeChecker,  # Testing internal implementation
)
from winml.modelkit.analyze.models.runtime_checks import (
    PatternAlternative,  # Testing internal implementation
)
from winml.modelkit.onnx import ONNXDomain
from winml.modelkit.pattern import (
    OperatorPattern,
    PatternMatchResult,
    PatternType,
    SkeletonMatchResult,
)


class TestEPContextNodeChecker:
    """Test EPContextNodeChecker implementation."""

    @pytest.fixture
    def ep_context_checker(self):
        """Create EPContextNodeChecker instance."""
        return EPContextNodeChecker()

    @pytest.fixture
    def sample_pattern_match(self):
        """Create sample PatternMatchResult for testing."""
        pattern = OperatorPattern(
            pattern_id="OP/com.microsoft/EPContext",
            pattern_type=PatternType.OPERATOR,
            namespace="com.microsoft",
            op_type="EPContext",
        )

        # Create a mock node proto
        node_proto = helper.make_node("EPContext", ["input"], ["output"], name="test_node")

        # Create SkeletonMatchResult
        skeleton_result = SkeletonMatchResult(
            pattern=pattern,
            matched_nodes=[node_proto],
            matched_node_keys=_stable_test_node_keys([node_proto]),
            matcher=None,
        )

        return PatternMatchResult(
            skeleton_match_result=skeleton_result,
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={},
        )

    def test_can_check_valid_and_invalid_nodes(self, ep_context_checker):
        """Test can_check returns correct values for various node types and domains."""
        # Valid: EPContext in com.microsoft domain
        valid_node = helper.make_node("EPContext", [], [])
        assert ep_context_checker.can_check(valid_node, ONNXDomain.COM_MICROSOFT, 1) is True

        # Invalid: wrong op_type
        wrong_op = helper.make_node("Conv", [], [])
        assert ep_context_checker.can_check(wrong_op, ONNXDomain.COM_MICROSOFT, 1) is False

        # Invalid: wrong domain
        wrong_domain = helper.make_node("EPContext", [], [])
        assert ep_context_checker.can_check(wrong_domain, ONNXDomain.AI_ONNX, 1) is False

    def test_check_partition_name_scenarios(self, ep_context_checker, sample_pattern_match):
        """Test check method with various partition_name scenarios."""
        # Scenario 1: Missing partition_name
        node_missing = helper.make_node("EPContext", [], [])
        result = ep_context_checker.check(
            node=node_missing,
            op_domain=ONNXDomain.COM_MICROSOFT,
            opset_version=1,
            pattern_match=sample_pattern_match,
            alternatives=[],
            ep_name="DML",
        )
        assert result.result.no_data is True
        assert result.result.compile is False
        assert result.result.run is False
        assert "Missing 'partition_name' attribute" in result.result.reason

        # Scenario 2: Matching partition_name
        node_matching = helper.make_node("EPContext", [], [], partition_name="DML_partition_1")
        result = ep_context_checker.check(
            node=node_matching,
            op_domain=ONNXDomain.COM_MICROSOFT,
            opset_version=1,
            pattern_match=sample_pattern_match,
            alternatives=[],
            ep_name="DML",
        )
        assert result.result.compile is True
        assert result.result.run is True
        assert result.result.no_data is False
        assert result.result.reason is None

        # Scenario 3: Non-matching partition_name
        node_non_matching = helper.make_node("EPContext", [], [], partition_name="QNN_partition_1")
        result = ep_context_checker.check(
            node=node_non_matching,
            op_domain=ONNXDomain.COM_MICROSOFT,
            opset_version=1,
            pattern_match=sample_pattern_match,
            alternatives=[],
            ep_name="DML",
        )
        assert result.result.compile is False
        assert result.result.run is False
        assert result.result.no_data is False
        assert "does not match ep_name" in result.result.reason

    def test_check_with_alternatives(self, ep_context_checker, sample_pattern_match):
        """Test check preserves alternatives in result."""
        node = helper.make_node("EPContext", [], [], partition_name="DML_test")

        alternatives = [
            PatternAlternative(
                pattern_id="ALT/test",
                result=RuntimeTestResult(compile=True, run=True),
                alternative_type=AlternativeType.EQUIVALENT,
            )
        ]

        result = ep_context_checker.check(
            node=node,
            op_domain=ONNXDomain.COM_MICROSOFT,
            opset_version=1,
            pattern_match=sample_pattern_match,
            alternatives=alternatives,
            ep_name="DML",
        )

        assert result.alternatives == alternatives

    def test_get_attribute_value(self, ep_context_checker):
        """Test get_attribute_value extracts attributes correctly."""
        # Test with existing attribute
        node_with_attr = helper.make_node("EPContext", [], [], partition_name="test_partition")
        value = ep_context_checker.get_attribute_value(node_with_attr, "partition_name")
        assert value == "test_partition"

        # Test with missing attribute
        node_without_attr = helper.make_node("EPContext", [], [])
        value = ep_context_checker.get_attribute_value(node_without_attr, "nonexistent")
        assert value is None
