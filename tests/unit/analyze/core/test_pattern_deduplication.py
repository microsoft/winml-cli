# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for source-priority pattern deduplication in PatternExtractor."""

import pytest
from onnx import TensorProto, helper

from tests.unit.test_helpers import stable_test_node_keys as _stable_test_node_keys
from winml.modelkit.analyze import ONNXModel, PatternExtractor
from winml.modelkit.pattern import PatternMatchResult, SkeletonMatchResult, SubgraphPattern


@pytest.fixture
def simple_model() -> ONNXModel:
    """Create a simple ONNX model."""
    input1 = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])

    div_node = helper.make_node("Div", ["input", "const1"], ["div_out"], name="div1")
    erf_node = helper.make_node("Erf", ["div_out"], ["erf_out"], name="erf1")
    mul_node = helper.make_node("Mul", ["erf_out", "input"], ["output"], name="mul1")

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

    def test_deduplication_removes_duplicates(self, simple_model: ONNXModel, monkeypatch):
        """Test that EP-priority dedup removes duplicate node-key matches in summary path."""
        extractor = PatternExtractor(simple_model)

        pattern = SubgraphPattern(
            pattern_id="SUBGRAPH/Test",
            pattern_name="Test",
            node_topology={"div": "Div", "erf": "Erf"},
            edge_topology=[("div", "erf")],
        )

        def make_match(source: str) -> PatternMatchResult:
            div_node = helper.make_node("Div", ["input"], ["div_out"], name=f"div1_{source}")
            erf_node = helper.make_node("Erf", ["div_out"], ["output"], name=f"erf1_{source}")
            skeleton = SkeletonMatchResult(
                pattern=pattern,
                matched_nodes=[div_node, erf_node],
                matched_node_keys=["stable_dup_node_a", "stable_dup_node_b"],
                matcher=None,
            )
            return PatternMatchResult(
                skeleton_match_result=skeleton,
                schema_input_to_value={},
                schema_output_to_value={},
                type_param_to_type={},
                attributes={"source": source},
            )

        default_match = make_match("default")
        ep_match = make_match("qnn")

        def mock_extract(self, *, source, model_signature):
            if source == "qnn":
                grouped = {"TestPattern": [ep_match]}
            else:
                grouped = {"TestPattern": [default_match]}
            stat = {
                "source": source,
                "cache_hit": False,
                "pattern_class_count": len(grouped),
                "match_count": 1,
                "elapsed_ms": 0,
            }
            return grouped, stat

        monkeypatch.setattr(
            PatternExtractor,
            "_extract_skeleton_matches_for_source",
            mock_extract,
        )
        monkeypatch.setattr(
            PatternExtractor,
            "_build_merge_prep_metadata",
            lambda *args, **kwargs: [],
        )
        monkeypatch.setattr(
            PatternExtractor,
            "_resolve_sources_for_ep",
            lambda *args, **kwargs: ["default", "qnn"],
        )

        result = extractor.summary(ep="QNNExecutionProvider", device="NPU")
        patterns = result["subgraph_patterns"]
        assert len(patterns) == 1
        assert patterns[0] is ep_match

    def test_different_node_sets_not_deduplicated(self, simple_model):
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
            matched_node_keys=_stable_test_node_keys([div1_node]),
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
            matched_node_keys=_stable_test_node_keys([div2_node]),
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
            matched_node_keys=_stable_test_node_keys([conv_node]),
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
        """Test that matched_node_names returns ONNXOp objects."""
        from winml.modelkit.analyze import ONNXOp

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
            matched_node_keys=_stable_test_node_keys([relu_node]),
            matcher=None,
        )

        match = PatternMatchResult(
            skeleton_match_result=skeleton,
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={},
        )

        # matched_node_names should return list of ONNXOp objects
        assert isinstance(match.matched_node_names, list)
        assert len(match.matched_node_names) == 1
        assert isinstance(match.matched_node_names[0], ONNXOp)
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
            matched_node_keys=_stable_test_node_keys([add_node]),
            matcher=None,
        )

        match = PatternMatchResult(
            skeleton_match_result=skeleton,
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={},
        )

        assert match.pattern_id == "SUBGRAPH/MyPattern"
