# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Unit tests for Pattern, OperatorPattern, SubgraphPattern, and PatternMatch validation.

Tests verify:
- Pattern base class pattern_type/pattern_id consistency validation
- OperatorPattern pattern_id format (OP/<namespace>/<op_type>)
- OperatorPattern namespace validation
- SubgraphPattern pattern_id format (SUBGRAPH/<name>)
- SubgraphPattern topology requirements
- PatternMatch node_topology validation for subgraph patterns
"""

import pytest
from pydantic import ValidationError

from tests.unit.test_helpers import stable_test_node_keys as _stable_test_node_keys
from winml.modelkit.pattern import (
    OperatorPattern,
    PatternMatchResult,
    PatternType,
    SkeletonMatchResult,
)
from winml.modelkit.pattern.models import (  # Pattern name collision with pattern.base.Pattern
    Pattern,
    SubgraphPattern,
)


def create_pattern_match_for_testing(pattern, node_protos):
    """Helper function to create PatternMatchResult for testing.

    Args:
        pattern: Pattern or SubgraphPattern instance
        node_protos: List of ONNX NodeProto objects

    Returns:
        PatternMatchResult instance
    """
    skeleton_result = SkeletonMatchResult(
        pattern=pattern,
        matched_nodes=node_protos,
        matched_node_keys=_stable_test_node_keys(node_protos),
        matcher=None,
    )

    return PatternMatchResult(
        skeleton_match_result=skeleton_result,
        schema_input_to_value={},
        schema_output_to_value={},
        type_param_to_type={},
    )


class TestPatternBaseValidation:
    """Test Pattern base class validation rules."""

    def test_pattern_type_operator_requires_op_prefix(self):
        """Test that operator pattern_type requires OP/ prefix in pattern_id."""
        # Valid: operator type with OP/ prefix
        pattern = Pattern(
            pattern_id="OP/ai.onnx/Conv",
            pattern_type=PatternType.OPERATOR,
            description="Test pattern",
        )
        assert pattern.pattern_type == PatternType.OPERATOR
        assert pattern.pattern_id.startswith("OP/")

        # Invalid: operator type without OP/ prefix
        with pytest.raises(
            ValidationError,
            match="Pattern type 'operator' requires pattern_id starting with 'OP/'",
        ):
            Pattern(
                pattern_id="INVALID/Conv",
                pattern_type=PatternType.OPERATOR,
                description="Test pattern",
            )

    def test_pattern_type_subgraph_requires_subgraph_prefix(self):
        """Test that subgraph pattern_type requires SUBGRAPH/ prefix in pattern_id."""
        # Valid: subgraph type with SUBGRAPH/ prefix
        pattern = Pattern(
            pattern_id="SUBGRAPH/GELU",
            pattern_type=PatternType.SUBGRAPH,
            description="Test subgraph",
        )
        assert pattern.pattern_type == PatternType.SUBGRAPH
        assert pattern.pattern_id.startswith("SUBGRAPH/")

        # Invalid: subgraph type without SUBGRAPH/ prefix
        with pytest.raises(
            ValidationError,
            match=("Pattern type 'subgraph' requires pattern_id starting with 'SUBGRAPH/'"),
        ):
            Pattern(
                pattern_id="OP/ai.onnx/Conv",
                pattern_type=PatternType.SUBGRAPH,
                description="Test pattern",
            )

    def test_description_optional(self):
        """Test that description field is optional."""
        # Without description
        pattern = Pattern(
            pattern_id="OP/ai.onnx/Conv",
            pattern_type=PatternType.OPERATOR,
        )
        assert pattern.description == ""

        # With description
        pattern_with_desc = Pattern(
            pattern_id="OP/ai.onnx/Conv",
            pattern_type=PatternType.OPERATOR,
            description="Convolution operator",
        )
        assert pattern_with_desc.description == "Convolution operator"


class TestOperatorPatternValidation:
    """Test OperatorPattern validation rules.

    Note: OperatorPattern now only has pattern_id, pattern_type, namespace,
    and op_type fields. Fields like node_names, attributes, dtype,
    input_shapes, and opset_version have been removed.
    """

    @pytest.mark.parametrize(
        "pattern_id",
        [
            "OP/ai.onnx/Conv",
            "OP/com.microsoft/FusedMatMul",
            "OP/ai.onnx/Relu",
        ],
    )
    def test_valid_pattern_id_format(self, pattern_id):
        """Test that valid pattern_id formats are accepted."""
        pattern = OperatorPattern(
            pattern_id=pattern_id,
            namespace="ai.onnx",
            op_type="Conv",
            description="Test pattern",
        )
        assert pattern.pattern_id == pattern_id
        assert pattern.pattern_type == PatternType.OPERATOR

    @pytest.mark.parametrize(
        "invalid_pattern_id",
        [
            "OP/Conv",  # Missing namespace
            "OP///Conv",  # Empty namespace
            "OPERATOR/ai.onnx/Conv",  # Wrong prefix
            "OP/ai.onnx/",  # Missing op_type
            "ai.onnx/Conv",  # Missing OP/ prefix
            "OP/ai.onnx/Conv/Extra",  # Too many segments
        ],
    )
    def test_invalid_pattern_id_format(self, invalid_pattern_id):
        """Test that invalid pattern_id formats are rejected."""
        with pytest.raises(ValidationError):
            OperatorPattern(
                pattern_id=invalid_pattern_id,
                namespace="ai.onnx",
                op_type="Conv",
                description="Test pattern",
            )

    @pytest.mark.parametrize("namespace", ["ai.onnx", "com.microsoft"])
    def test_valid_namespace_values(self, namespace):
        """Test that valid namespace values are accepted."""
        pattern = OperatorPattern(
            pattern_id=f"OP/{namespace}/TestOp",
            namespace=namespace,
            op_type="TestOp",
            description="Test pattern",
        )
        assert pattern.namespace == namespace

    @pytest.mark.parametrize(
        "invalid_namespace",
        [
            "custom.namespace",
            "org.pytorch",
            "ai.onnx.preview",
            "microsoft",
        ],
    )
    def test_invalid_namespace_values(self, invalid_namespace):
        """Test that invalid namespace values are rejected."""
        with pytest.raises(ValidationError, match="Namespace must be one of"):
            OperatorPattern(
                pattern_id=f"OP/{invalid_namespace}/TestOp",
                namespace=invalid_namespace,
                op_type="TestOp",
                description="Test pattern",
            )


class TestSubgraphPatternValidation:
    """Test SubgraphPattern validation rules."""

    @pytest.mark.parametrize(
        "pattern_id",
        [
            "SUBGRAPH/GELU",
            "SUBGRAPH/LayerNormalization",
            "SUBGRAPH/MultiHeadAttention",
        ],
    )
    def test_valid_subgraph_pattern_id_format(self, pattern_id):
        """Test that valid SUBGRAPH/<name> format is accepted."""
        pattern = SubgraphPattern(
            pattern_id=pattern_id,
            pattern_name="Test Subgraph",
            node_topology={"node1": "Div", "node2": "Erf"},
            edge_topology=[("node1", "node2")],
            description="Test subgraph pattern",
        )
        assert pattern.pattern_id == pattern_id
        assert pattern.pattern_type == PatternType.SUBGRAPH

    @pytest.mark.parametrize(
        "invalid_pattern_id",
        [
            "SUBGRAPH/",  # Missing name
            "SUBGRAPH",  # Missing slash
            "PATTERN/GELU",  # Wrong prefix
            "SUBGRAPH/GELU/Extra",  # Too many segments
            "subgraph/GELU",  # Lowercase prefix
        ],
    )
    def test_invalid_subgraph_pattern_id_format(self, invalid_pattern_id):
        """Test that invalid SUBGRAPH/<name> formats are rejected."""
        with pytest.raises(ValidationError):
            SubgraphPattern(
                pattern_id=invalid_pattern_id,
                pattern_name="Test Subgraph",
                node_topology={"node1": "Div"},
                edge_topology=[],
                description="Test subgraph pattern",
            )

    def test_node_topology_required(self):
        """Test that node_topology is required and must be populated."""
        # Valid: with node_topology
        pattern = SubgraphPattern(
            pattern_id="SUBGRAPH/GELU",
            pattern_name="GELU",
            node_topology={"n1": "Div", "n2": "Erf", "n3": "Add", "n4": "Mul"},
            edge_topology=[("n1", "n2"), ("n2", "n3"), ("n3", "n4")],
            description="Gaussian Error Linear Unit",
        )
        assert len(pattern.node_topology) == 4

    def test_edge_topology_required(self):
        """Test that edge_topology is required."""
        # Valid: with edge_topology
        pattern = SubgraphPattern(
            pattern_id="SUBGRAPH/GELU",
            pattern_name="GELU",
            node_topology={"n1": "Div", "n2": "Erf"},
            edge_topology=[("n1", "n2")],
            description="Test",
        )
        assert len(pattern.edge_topology) == 1
        assert pattern.edge_topology[0] == ("n1", "n2")

    def test_complex_topology_structure(self):
        """Test that complex topology structures are accepted."""
        node_topology = {
            "n1": "Div",
            "n2": "Erf",
            "n3": "Add",
            "n4": "Mul",
        }
        edge_topology = [
            ("n1", "n2"),
            ("n2", "n3"),
            ("n3", "n4"),
        ]

        pattern = SubgraphPattern(
            pattern_id="SUBGRAPH/GELU",
            pattern_name="GELU",
            node_topology=node_topology,
            edge_topology=edge_topology,
            description="Gaussian Error Linear Unit",
        )

        assert len(pattern.node_topology) == 4
        assert len(pattern.edge_topology) == 3


class TestPatternMatchValidation:
    """Test PatternMatch validation rules."""

    def test_match_id_generated_as_uuid(self):
        """Test that match_id is auto-generated as UUID."""
        from onnx import helper

        op_pattern = OperatorPattern(
            pattern_id="OP/ai.onnx/Conv",
            namespace="ai.onnx",
            op_type="Conv",
        )

        node_proto = helper.make_node("Conv", ["input"], ["output"], name="conv1")
        match = create_pattern_match_for_testing(op_pattern, [node_proto])

        # Validate that match_id is a valid UUID string
        assert isinstance(match.match_id, str)
        assert len(match.match_id) == 36  # UUID format length

    def test_operator_pattern_match(self):
        """Test PatternMatch with OperatorPattern."""
        from onnx import helper

        op_pattern = OperatorPattern(
            pattern_id="OP/ai.onnx/Conv",
            namespace="ai.onnx",
            op_type="Conv",
        )

        node_proto = helper.make_node("Conv", ["input"], ["output"], name="conv1")
        match = create_pattern_match_for_testing(op_pattern, [node_proto])

        assert match.pattern.pattern_type == PatternType.OPERATOR
        assert match.pattern.pattern_id == "OP/ai.onnx/Conv"
        assert len(match.matched_node_names) == 1

    def test_subgraph_pattern_match_requires_node_topology(self):
        """Test that subgraph PatternMatch works with subgraph patterns."""
        from onnx import helper

        subgraph_pattern = SubgraphPattern(
            pattern_id="SUBGRAPH/GELU",
            pattern_name="GELU",
            node_topology={"n1": "Div", "n2": "Erf"},
            edge_topology=[("n1", "n2")],
        )

        div_node = helper.make_node("Div", ["input"], ["div_out"], name="n1")
        erf_node = helper.make_node("Erf", ["div_out"], ["output"], name="n2")

        # Valid: subgraph match
        match = create_pattern_match_for_testing(subgraph_pattern, [div_node, erf_node])
        assert match.pattern.pattern_type == PatternType.SUBGRAPH
        assert len(match.matched_node_names) == 2

    def test_operator_pattern_match_node_topology_optional(self):
        """Test that operator PatternMatch works correctly."""
        from onnx import helper

        op_pattern = OperatorPattern(
            pattern_id="OP/ai.onnx/Relu",
            namespace="ai.onnx",
            op_type="Relu",
        )

        node_proto = helper.make_node("Relu", ["input"], ["output"], name="relu1")
        match = create_pattern_match_for_testing(op_pattern, [node_proto])
        assert match.pattern.op_type == "Relu"

    def test_optional_fields(self):
        """Test that optional fields work correctly."""
        from onnx import helper

        op_pattern = OperatorPattern(
            pattern_id="OP/ai.onnx/Conv",
            namespace="ai.onnx",
            op_type="Conv",
        )

        node_proto = helper.make_node("Conv", ["input"], ["output"], name="conv1")

        # With optional fields - use extended constructor
        skeleton_result = SkeletonMatchResult(
            pattern=op_pattern,
            matched_nodes=[node_proto],
            matched_node_keys=_stable_test_node_keys([node_proto]),
            matcher=None,
        )

        match = PatternMatchResult(
            skeleton_match_result=skeleton_result,
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={"T": "float32"},
            attributes={"kernel_shape": "[3, 3]"},
        )

        assert match.attributes == {"kernel_shape": "[3, 3]"}
        assert match.type_vars == {"T": "float32"}

        # Without optional fields
        match_minimal = create_pattern_match_for_testing(op_pattern, [node_proto])
        assert match_minimal.attributes == {}

    def test_matched_node_names_required_and_non_empty(self):
        """Test that matched_node_names is required and non-empty."""
        from onnx import helper

        op_pattern = OperatorPattern(
            pattern_id="OP/ai.onnx/Conv",
            namespace="ai.onnx",
            op_type="Conv",
        )

        node_proto = helper.make_node("Conv", ["input"], ["output"], name="conv1")

        # Valid: with matched_node_names
        match = create_pattern_match_for_testing(op_pattern, [node_proto])
        assert len(match.matched_node_names) == 1

        # Test empty nodes list
        empty_skeleton = SkeletonMatchResult(
            pattern=op_pattern,
            matched_nodes=[],
            matched_node_keys=[],
            matcher=None,
        )
        match_empty = PatternMatchResult(
            skeleton_match_result=empty_skeleton,
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={},
        )
        assert len(match_empty.matched_node_names) == 0
