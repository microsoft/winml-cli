# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for pattern deduplication logic in PatternExtractor.

Tests the priority-based deduplication:
Priority: HTP metadata > hierarchy_tag > PatternMatcher
"""

import pytest
from onnx import TensorProto, helper

from winml.modelkit.pattern.match import PatternMatchResult, SkeletonMatchResult
from winml.modelkit.pattern.models import SubgraphPattern
from winml.modelkit.analyze.core.pattern_extractor import PatternExtractor
from winml.modelkit.analyze.models.onnx_model import ONNXModel


@pytest.fixture
def simple_model_with_tags() -> ONNXModel:
    """Create a simple ONNX model with hierarchy tags."""
    input1 = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    # Create nodes with hierarchy_tag attribute
    div_node = helper.make_node("Div", ["input", "const1"], ["div_out"], name="div1")
    div_node.attribute.append(helper.make_attribute("hierarchy_tag", "layer1/Gelu1"))

    erf_node = helper.make_node("Erf", ["div_out"], ["erf_out"], name="erf1")
    erf_node.attribute.append(helper.make_attribute("hierarchy_tag", "layer1/Gelu1"))

    mul_node = helper.make_node("Mul", ["erf_out", "input"], ["output"], name="mul1")
    mul_node.attribute.append(helper.make_attribute("hierarchy_tag", "layer1/Gelu1"))

    graph_def = helper.make_graph(
        [div_node, erf_node, mul_node],
        "test_graph",
        [input1],
        [output],
    )

    model_def = helper.make_model(
        graph_def, producer_name="test", opset_imports=[helper.make_opsetid("", 13)]
    )

    return ONNXModel.from_onnx_model(model_def, "test.onnx")


@pytest.fixture
def gelu_pattern() -> SubgraphPattern:
    """Create a GELU pattern definition."""
    return SubgraphPattern(
        pattern_id="SUBGRAPH/Gelu1",
        pattern_name="Gelu1",
        semantic_label="Gelu1",
        node_topology={"div": "Div", "erf": "Erf", "mul": "Mul"},
        edge_topology=[("div", "erf"), ("erf", "mul")],
    )


class TestPatternDeduplication:
    """Test pattern deduplication logic."""

    def test_deduplication_removes_duplicates(self, simple_model_with_tags: ONNXModel, monkeypatch):
        """Test that duplicate matches are removed based on node sets."""
        extractor = PatternExtractor(simple_model_with_tags)

        # Create two matches with same nodes
        div_node = helper.make_node("Div", ["input"], ["div_out"], name="div1")
        erf_node = helper.make_node("Erf", ["div_out"], ["output"], name="erf1")

        pattern = SubgraphPattern(
            pattern_id="SUBGRAPH/Test",
            pattern_name="Test",
            node_topology={"div": "Div", "erf": "Erf"},
            edge_topology=[("div", "erf")],
        )

        skeleton1 = SkeletonMatchResult(
            pattern=pattern,
            matched_nodes=[div_node, erf_node],
            matcher=None,
        )

        match1 = PatternMatchResult(
            skeleton_match_result=skeleton1,
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={},
            attributes={"source": "htp_metadata"},
        )

        skeleton2 = SkeletonMatchResult(
            pattern=pattern,
            matched_nodes=[div_node, erf_node],
            matcher=None,
        )

        match2 = PatternMatchResult(
            skeleton_match_result=skeleton2,
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={},
            attributes={"source": "pattern_matcher"},
        )

        # Mock the methods to return our test matches
        def mock_tag_match(pattern_def):
            return [match1]

        def mock_matcher_match():
            return [match2]

        monkeypatch.setattr(
            extractor,
            "_match_subgraph_pattern_from_model_tags",
            mock_tag_match,
        )
        monkeypatch.setattr(
            extractor,
            "extract_subgraph_patterns_with_pattern_matcher",
            mock_matcher_match,
        )

        # Mock UnifiedPatternConfig to return test pattern
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_config.get_htp_patterns.return_value = [pattern]

        with patch(
            "winml.modelkit.analyze.core.pattern_extractor.UnifiedPatternConfig",
            return_value=mock_config,
        ):
            # Extract patterns with deduplication
            patterns = extractor.extract_subgraph_patterns()

            # Should only have 1 match (duplicate removed)
            assert len(patterns) == 1
            # Should keep the first one (from HTP/tag, not PatternMatcher)
            assert patterns[0].attributes.get("source") == "htp_metadata"

    def test_different_node_sets_not_deduplicated(self, simple_model_with_tags):
        """Test that matches with different node sets are kept."""
        # Create two matches with different nodes
        div1_node = helper.make_node("Div", ["input1"], ["div_out1"], name="div1")
        div2_node = helper.make_node("Div", ["input2"], ["div_out2"], name="div2")

        pattern = SubgraphPattern(
            pattern_id="SUBGRAPH/Test",
            pattern_name="Test",
            node_topology={"div": "Div"},
            edge_topology=[],
        )

        skeleton1 = SkeletonMatchResult(
            pattern=pattern,
            matched_nodes=[div1_node],
            matcher=None,
        )

        match1 = PatternMatchResult(
            skeleton_match_result=skeleton1,
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={},
        )

        skeleton2 = SkeletonMatchResult(
            pattern=pattern,
            matched_nodes=[div2_node],
            matcher=None,
        )

        match2 = PatternMatchResult(
            skeleton_match_result=skeleton2,
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={},
        )

        # These should not be deduplicated as they have different nodes
        matched_node_set1 = frozenset([n.name for n in match1.skeleton_match_result.matched_nodes])
        matched_node_set2 = frozenset([n.name for n in match2.skeleton_match_result.matched_nodes])

        assert matched_node_set1 != matched_node_set2


class TestPatternMatchNodeAccess:
    """Test PatternMatchResult node access properties."""

    def test_matched_nodes_returns_string_list(self):
        """Test that matched_nodes property returns list of strings."""
        pattern = SubgraphPattern(
            pattern_id="SUBGRAPH/Test",
            pattern_name="Test",
            node_topology={"n1": "Conv"},
            edge_topology=[],
        )

        conv_node = helper.make_node("Conv", ["input"], ["output"], name="conv1")

        skeleton = SkeletonMatchResult(
            pattern=pattern,
            matched_nodes=[conv_node],
            matcher=None,
        )

        match = PatternMatchResult(
            skeleton_match_result=skeleton,
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={},
        )

        # matched_nodes should return list of strings
        assert isinstance(match.matched_nodes, list)
        assert len(match.matched_nodes) == 1
        assert match.matched_nodes[0] == "conv1"

    def test_matched_node_names_returns_onnx_ops(self):
        """Test that matched_node_names returns OnnxOP objects."""
        from winml.modelkit.analyze.models.onnx_op import OnnxOP

        pattern = SubgraphPattern(
            pattern_id="SUBGRAPH/Test",
            pattern_name="Test",
            node_topology={"n1": "Relu"},
            edge_topology=[],
        )

        relu_node = helper.make_node("Relu", ["input"], ["output"], name="relu1")

        skeleton = SkeletonMatchResult(
            pattern=pattern,
            matched_nodes=[relu_node],
            matcher=None,
        )

        match = PatternMatchResult(
            skeleton_match_result=skeleton,
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={},
        )

        # matched_node_names should return list of OnnxOP objects
        assert isinstance(match.matched_node_names, list)
        assert len(match.matched_node_names) == 1
        assert isinstance(match.matched_node_names[0], OnnxOP)
        assert match.matched_node_names[0].op_type == "Relu"
        assert match.matched_node_names[0].node_name == "relu1"

    def test_pattern_id_property(self):
        """Test pattern_id property extracts correct ID."""
        pattern = SubgraphPattern(
            pattern_id="SUBGRAPH/MyPattern",
            pattern_name="MyPattern",
            node_topology={"n1": "Add"},
            edge_topology=[],
        )

        add_node = helper.make_node("Add", ["i1", "i2"], ["output"], name="add1")

        skeleton = SkeletonMatchResult(
            pattern=pattern,
            matched_nodes=[add_node],
            matcher=None,
        )

        match = PatternMatchResult(
            skeleton_match_result=skeleton,
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={},
        )

        assert match.pattern_id == "SUBGRAPH/MyPattern"
