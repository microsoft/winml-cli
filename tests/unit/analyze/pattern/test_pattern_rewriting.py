# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for pattern rewriting on ONNX models."""

from pathlib import Path

import numpy as np
import onnx
import pytest

from winml.modelkit.pattern import (
    ExpandedAttentionPattern,
    Gelu2Pattern,
    MatMulAddPattern,
    PatternMatcher,
    PatternRewriter,
    ReshapeGemmReshapePattern,
    SingleGeluPattern,
    TransposeAttentionPattern,
)


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"
ERF_CONVNEXT_MODEL_PATH = FIXTURES_DIR / "erf-convnext-tiny.onnx"
BERT_TINY_OPSET23_MODEL_PATH = FIXTURES_DIR / "nsp_b0ee7fae871bae40_opt_opset23.onnx"


class TestPatternRewriting:
    """Tests for pattern rewriting functionality."""

    @pytest.fixture
    def erf_convnext_model(self):
        """Load the Erf ConvNeXt model for testing."""
        if not ERF_CONVNEXT_MODEL_PATH.exists():
            pytest.skip(f"Test model not found: {ERF_CONVNEXT_MODEL_PATH}")
        return onnx.load(str(ERF_CONVNEXT_MODEL_PATH))

    def test_rewrite_matmuladd_to_reshape_gemm_reshape(self, erf_convnext_model):
        """Test rewriting MatMulAdd patterns to ReshapeGemmReshape patterns.

        The erf-convnext-tiny model contains 36 MatMulAdd patterns.
        This test verifies that all 36 can be rewritten to ReshapeGemmReshape patterns.
        """
        # First, find all MatMulAdd patterns in the model
        matcher = PatternMatcher(erf_convnext_model)
        matcher.register_pattern(MatMulAddPattern())
        matmuladd_results = matcher.match()

        # Verify we found the expected number of MatMulAdd patterns
        assert len(matmuladd_results) == 36, (
            f"Expected 36 MatMulAdd matches, found {len(matmuladd_results)}"
        )

        # Rewrite all MatMulAdd patterns to ReshapeGemmReshape
        rewriter = PatternRewriter(erf_convnext_model)
        new_model = rewriter.rewrite([(matmuladd_results, ReshapeGemmReshapePattern)])

        # Also verify shape inference works on the rewritten model
        onnx.shape_inference.infer_shapes(new_model)
        # Verify the new model is valid - should not raise any exception
        onnx.checker.check_model(new_model)

        # Verify that MatMulAdd patterns are no longer present
        new_matcher = PatternMatcher(new_model)
        new_matcher.register_pattern(MatMulAddPattern())
        new_matmuladd_results = new_matcher.match()
        assert len(new_matmuladd_results) == 0, (
            f"Expected 0 MatMulAdd matches after rewriting, found {len(new_matmuladd_results)}"
        )

        # Verify that ReshapeGemmReshape patterns are now present
        new_matcher2 = PatternMatcher(new_model)
        new_matcher2.register_pattern(ReshapeGemmReshapePattern())
        reshape_gemm_results = new_matcher2.match()
        assert len(reshape_gemm_results) == 36, (
            f"Expected 36 ReshapeGemmReshape matches after rewriting, "
            f"found {len(reshape_gemm_results)}"
        )

    def test_rewrite_preserves_graph_structure(self, erf_convnext_model):
        """Test that rewriting preserves graph inputs and outputs."""
        # Get original input/output info
        original_inputs = [inp.name for inp in erf_convnext_model.graph.input]
        original_outputs = [out.name for out in erf_convnext_model.graph.output]

        # Find and rewrite MatMulAdd patterns
        matcher = PatternMatcher(erf_convnext_model)
        matcher.register_pattern(MatMulAddPattern())
        matmuladd_results = matcher.match()

        rewriter = PatternRewriter(erf_convnext_model)
        new_model = rewriter.rewrite([(matmuladd_results, ReshapeGemmReshapePattern)])

        # Verify inputs and outputs are preserved
        new_inputs = [inp.name for inp in new_model.graph.input]
        new_outputs = [out.name for out in new_model.graph.output]

        assert new_inputs == original_inputs, "Graph inputs should be preserved after rewriting"
        assert new_outputs == original_outputs, "Graph outputs should be preserved after rewriting"

        # Verify model validity
        onnx.checker.check_model(new_model)

    def test_rewrite_adds_new_nodes(self, erf_convnext_model):
        """Test that rewriting adds the expected new nodes."""
        # Count original nodes
        original_node_count = len(erf_convnext_model.graph.node)

        # Find and rewrite MatMulAdd patterns
        matcher = PatternMatcher(erf_convnext_model)
        matcher.register_pattern(MatMulAddPattern())
        matmuladd_results = matcher.match()

        rewriter = PatternRewriter(erf_convnext_model)
        new_model = rewriter.rewrite([(matmuladd_results, ReshapeGemmReshapePattern)])

        new_node_count = len(new_model.graph.node)

        # MatMulAdd has 2 nodes (MatMul, Add)
        # ReshapeGemmReshape has 3 nodes (Reshape, Gemm, Reshape)
        # For 36 patterns: removes 36*2 = 72 nodes, adds 36*3 = 108 nodes
        # Net change: +36 nodes
        expected_change = 36 * (3 - 2)
        expected_count = original_node_count + expected_change

        assert new_node_count == expected_count, (
            f"Expected {expected_count} nodes after rewriting, found {new_node_count}"
        )

        # Verify model validity
        onnx.checker.check_model(new_model)

    def test_rewrite_node_naming(self, erf_convnext_model):
        """Test that rewritten nodes have proper naming convention."""
        # Find and rewrite MatMulAdd patterns
        matcher = PatternMatcher(erf_convnext_model)
        matcher.register_pattern(MatMulAddPattern())
        matmuladd_results = matcher.match()

        rewriter = PatternRewriter(erf_convnext_model)
        new_model = rewriter.rewrite([(matmuladd_results, ReshapeGemmReshapePattern)])

        # Check that new nodes have the expected naming pattern
        rewrite_nodes = [
            node
            for node in new_model.graph.node
            if node.name.startswith("Rewrite_ReshapeGemmReshapePattern_")
        ]

        # Each ReshapeGemmReshape pattern has 3 nodes
        assert len(rewrite_nodes) == 36 * 3, (
            f"Expected {36 * 3} rewrite nodes, found {len(rewrite_nodes)}"
        )

        # Verify model validity
        onnx.checker.check_model(new_model)

    def test_rewrite_empty_list(self, erf_convnext_model):
        """Test rewriting with an empty list of matches."""
        rewriter = PatternRewriter(erf_convnext_model)
        new_model = rewriter.rewrite([])

        # Model should be unchanged (but a copy)
        assert len(new_model.graph.node) == len(erf_convnext_model.graph.node)
        onnx.checker.check_model(new_model)

    def test_rewrite_original_model_unchanged(self, erf_convnext_model):
        """Test that rewriting does not modify the original model."""
        original_node_count = len(erf_convnext_model.graph.node)
        original_node_names = [node.name for node in erf_convnext_model.graph.node]

        # Find and rewrite MatMulAdd patterns
        matcher = PatternMatcher(erf_convnext_model)
        matcher.register_pattern(MatMulAddPattern())
        matmuladd_results = matcher.match()

        rewriter = PatternRewriter(erf_convnext_model)
        new_model = rewriter.rewrite([(matmuladd_results, ReshapeGemmReshapePattern)])

        # Verify original model is unchanged
        assert len(erf_convnext_model.graph.node) == original_node_count
        assert [node.name for node in erf_convnext_model.graph.node] == original_node_names

        # Verify rewritten model is valid
        onnx.checker.check_model(new_model)

    def test_rewrite_adds_initializers_for_reshape(self, erf_convnext_model):
        """Test that rewriting adds required initializers for Reshape shape constants."""
        original_initializer_count = len(erf_convnext_model.graph.initializer)

        # Find and rewrite MatMulAdd patterns
        matcher = PatternMatcher(erf_convnext_model)
        matcher.register_pattern(MatMulAddPattern())
        matmuladd_results = matcher.match()

        rewriter = PatternRewriter(erf_convnext_model)
        new_model = rewriter.rewrite([(matmuladd_results, ReshapeGemmReshapePattern)])

        new_initializer_count = len(new_model.graph.initializer)

        # Each ReshapeGemmReshape pattern needs 2 shape constants (for the two Reshape nodes)
        # 36 patterns * 2 = 72 new initializers
        expected_new_initializers = 36 * 2
        assert new_initializer_count >= original_initializer_count + expected_new_initializers, (
            f"Expected at least {original_initializer_count + expected_new_initializers} "
            f"initializers, found {new_initializer_count}"
        )

        # Verify model validity
        onnx.checker.check_model(new_model)


class TestPatternRewriterWarnings:
    """Tests for warning behavior in pattern rewriting."""

    @pytest.fixture
    def erf_convnext_model(self):
        """Load the Erf ConvNeXt model for testing."""
        if not ERF_CONVNEXT_MODEL_PATH.exists():
            pytest.skip(f"Test model not found: {ERF_CONVNEXT_MODEL_PATH}")
        return onnx.load(str(ERF_CONVNEXT_MODEL_PATH))

    def test_rewrite_warns_on_non_removable(self, erf_convnext_model):
        """Test that rewriting warns when encountering non-removable patterns."""
        # Find MatMulAdd patterns
        matcher = PatternMatcher(erf_convnext_model)
        matcher.register_pattern(MatMulAddPattern())
        matmuladd_results = matcher.match()

        # Artificially make one pattern non-removable
        if matmuladd_results:
            matmuladd_results[0].skeleton_match_result.removable = False

        rewriter = PatternRewriter(erf_convnext_model)

        with pytest.warns(UserWarning, match="Skipping non-removable pattern match"):
            new_model = rewriter.rewrite([(matmuladd_results, ReshapeGemmReshapePattern)])

        # One pattern should be skipped
        new_matcher = PatternMatcher(new_model)
        new_matcher.register_pattern(ReshapeGemmReshapePattern())
        reshape_gemm_results = new_matcher.match()

        # 36 - 1 = 35 patterns should be rewritten
        assert len(reshape_gemm_results) == 35, (
            f"Expected 35 ReshapeGemmReshape matches (1 skipped), found {len(reshape_gemm_results)}"
        )


class TestComprehensivePatternRewriting:
    """Tests for rewriting multiple pattern types at once."""

    @pytest.fixture
    def erf_convnext_model(self):
        """Load the Erf ConvNeXt model for testing."""
        if not ERF_CONVNEXT_MODEL_PATH.exists():
            pytest.skip(f"Test model not found: {ERF_CONVNEXT_MODEL_PATH}")
        return onnx.load(str(ERF_CONVNEXT_MODEL_PATH))

    def test_rewrite_all_patterns_gelu_and_matmuladd(self, erf_convnext_model):
        """Test rewriting all GELU patterns to SingleGelu and MatMulAdd to ReshapeGemmReshape.

        The erf-convnext-tiny model contains:
        - 18 Gelu2Pattern matches (Erf-based GELU activations)
        - 36 MatMulAdd patterns (linear layers)

        After rewriting:
        - 0 Gelu2Pattern matches (replaced by SingleGelu)
        - 0 MatMulAdd matches (replaced by ReshapeGemmReshape)
        - 18 SingleGeluPattern matches (single Gelu nodes)
        - 36 ReshapeGemmReshape matches
        """
        # Count original nodes
        original_node_count = len(erf_convnext_model.graph.node)

        # Find all patterns before rewriting
        matcher = PatternMatcher(erf_convnext_model)
        matcher.register_pattern(Gelu2Pattern())
        matcher.register_pattern(MatMulAddPattern())
        all_results = matcher.match()

        # Categorize results
        gelu_results = [
            r for r in all_results if isinstance(r.skeleton_match_result.pattern, Gelu2Pattern)
        ]
        matmuladd_results = [
            r for r in all_results if isinstance(r.skeleton_match_result.pattern, MatMulAddPattern)
        ]

        # Verify expected counts before rewriting
        assert len(gelu_results) == 18, f"Expected 18 Gelu2 matches, found {len(gelu_results)}"
        assert len(matmuladd_results) == 36, (
            f"Expected 36 MatMulAdd matches, found {len(matmuladd_results)}"
        )

        # Rewrite all patterns
        rewriter = PatternRewriter(erf_convnext_model)
        new_model = rewriter.rewrite(
            [
                (gelu_results, SingleGeluPattern),
                (matmuladd_results, ReshapeGemmReshapePattern),
            ]
        )

        # Verify model validity
        onnx.checker.check_model(new_model)

        # Check node count change
        # Gelu2 (5 nodes) -> SingleGelu (1 node): -4 nodes per pattern, 18 patterns = -72 nodes
        # MatMulAdd (2 nodes) -> ReshapeGemmReshape (3 nodes): +1 node per pattern, 36 = +36
        # Net change from replacement: -72 + 36 = -36 nodes
        # Additional: unused constants from removed Gelu2 patterns cleaned up
        # (3 constants per pattern), 18 patterns * 3 constants = 54 removed
        base_node_change = -72 + 36  # From pattern replacement
        unused_constants_removed = 54  # 18 Gelu2 patterns * 3 constants each
        expected_node_count = original_node_count + base_node_change - unused_constants_removed
        actual_node_count = len(new_model.graph.node)
        assert actual_node_count == expected_node_count, (
            f"Expected {expected_node_count} nodes after rewriting, found {actual_node_count}"
        )

        # Verify old patterns are no longer present
        new_matcher = PatternMatcher(new_model)
        new_matcher.register_pattern(Gelu2Pattern())
        new_matcher.register_pattern(MatMulAddPattern())
        remaining_results = new_matcher.match()

        remaining_gelu = [
            r
            for r in remaining_results
            if isinstance(r.skeleton_match_result.pattern, Gelu2Pattern)
        ]
        remaining_matmuladd = [
            r
            for r in remaining_results
            if isinstance(r.skeleton_match_result.pattern, MatMulAddPattern)
        ]

        assert len(remaining_gelu) == 0, (
            f"Expected 0 Gelu2 matches after rewriting, found {len(remaining_gelu)}"
        )
        assert len(remaining_matmuladd) == 0, (
            f"Expected 0 MatMulAdd matches after rewriting, found {len(remaining_matmuladd)}"
        )

        # Verify new patterns are present
        new_matcher2 = PatternMatcher(new_model)
        new_matcher2.register_pattern(SingleGeluPattern())
        new_matcher2.register_pattern(ReshapeGemmReshapePattern())
        new_results = new_matcher2.match()

        new_msft_gelu = [
            r for r in new_results if isinstance(r.skeleton_match_result.pattern, SingleGeluPattern)
        ]
        new_reshape_gemm = [
            r
            for r in new_results
            if isinstance(r.skeleton_match_result.pattern, ReshapeGemmReshapePattern)
        ]

        assert len(new_msft_gelu) == 18, (
            f"Expected 18 SingleGelu matches after rewriting, found {len(new_msft_gelu)}"
        )
        assert len(new_reshape_gemm) == 36, (
            f"Expected 36 ReshapeGemmReshape matches after rewriting, found {len(new_reshape_gemm)}"
        )


class TestAttentionPatternRewriting:
    """Tests for rewriting ExpandedAttention patterns to TransposeAttention patterns."""

    @pytest.fixture
    def bert_tiny_model(self):
        """Load the BERT Tiny model with opset 23 for testing."""
        if not BERT_TINY_OPSET23_MODEL_PATH.exists():
            pytest.skip(f"Test model not found: {BERT_TINY_OPSET23_MODEL_PATH}")
        return onnx.load(str(BERT_TINY_OPSET23_MODEL_PATH))

    def test_rewrite_expanded_attention_to_transpose_attention(self, bert_tiny_model):
        """Test rewriting ExpandedAttention patterns to TransposeAttention patterns.

        The BERT Tiny model contains 2 ExpandedAttention patterns (one per encoder layer).
        This test verifies that both can be rewritten to TransposeAttention patterns.
        """
        # First, find all ExpandedAttention patterns in the model
        matcher = PatternMatcher(bert_tiny_model)
        matcher.register_pattern(ExpandedAttentionPattern())
        expanded_results = matcher.match()

        # Verify we found the expected number of ExpandedAttention patterns
        assert len(expanded_results) == 2, (
            f"Expected 2 ExpandedAttention matches, found {len(expanded_results)}"
        )

        # Rewrite all ExpandedAttention patterns to TransposeAttention
        rewriter = PatternRewriter(bert_tiny_model)
        new_model = rewriter.rewrite([(expanded_results, TransposeAttentionPattern)])

        # Verify the new model has Attention nodes
        attention_nodes = [n for n in new_model.graph.node if n.op_type == "Attention"]
        assert len(attention_nodes) == 2, (
            f"Expected 2 Attention nodes after rewriting, found {len(attention_nodes)}"
        )

        # Verify that ExpandedAttention patterns are no longer present
        new_matcher = PatternMatcher(new_model)
        new_matcher.register_pattern(ExpandedAttentionPattern())
        new_expanded_results = new_matcher.match()
        assert len(new_expanded_results) == 0, (
            f"Expected 0 ExpandedAttention matches after rewriting, "
            f"found {len(new_expanded_results)}"
        )

        # Verify that TransposeAttention patterns are now present
        new_matcher2 = PatternMatcher(new_model)
        new_matcher2.register_pattern(TransposeAttentionPattern())
        transpose_attn_results = new_matcher2.match()
        assert len(transpose_attn_results) == 2, (
            f"Expected 2 TransposeAttention matches after rewriting, "
            f"found {len(transpose_attn_results)}"
        )

    def test_rewrite_attention_preserves_graph_structure(self, bert_tiny_model):
        """Test that rewriting attention preserves graph inputs and outputs."""
        # Get original input/output info
        original_inputs = [inp.name for inp in bert_tiny_model.graph.input]
        original_outputs = [out.name for out in bert_tiny_model.graph.output]

        # Find and rewrite ExpandedAttention patterns
        matcher = PatternMatcher(bert_tiny_model)
        matcher.register_pattern(ExpandedAttentionPattern())
        expanded_results = matcher.match()

        rewriter = PatternRewriter(bert_tiny_model)
        new_model = rewriter.rewrite([(expanded_results, TransposeAttentionPattern)])

        # Verify inputs and outputs are preserved
        new_inputs = [inp.name for inp in new_model.graph.input]
        new_outputs = [out.name for out in new_model.graph.output]

        assert new_inputs == original_inputs, "Graph inputs should be preserved after rewriting"
        assert new_outputs == original_outputs, "Graph outputs should be preserved after rewriting"

    def test_rewrite_attention_reduces_node_count(self, bert_tiny_model):
        """Test that rewriting attention reduces the node count.

        ExpandedAttention has 10 nodes (4 Transpose, 2 Mul, 2 MatMul, 1 Add, 1 Softmax)
        TransposeAttention has 5 nodes (4 Transpose + 1 Attention)

        For 2 patterns: removes 2*10 = 20 nodes, adds 2*5 = 10 nodes
        Net change: -10 nodes (before constant cleanup)
        """
        # Count original nodes
        original_node_count = len(bert_tiny_model.graph.node)

        # Find and rewrite ExpandedAttention patterns
        matcher = PatternMatcher(bert_tiny_model)
        matcher.register_pattern(ExpandedAttentionPattern())
        expanded_results = matcher.match()

        rewriter = PatternRewriter(bert_tiny_model)
        new_model = rewriter.rewrite([(expanded_results, TransposeAttentionPattern)])

        new_node_count = len(new_model.graph.node)

        # ExpandedAttention has 10 nodes, TransposeAttention has 5 nodes
        # For 2 patterns: removes 20 nodes, adds 10 nodes = -10 nodes
        base_node_change = 2 * (5 - 10)  # -10 nodes

        # The actual node count should be reduced by at least the base change
        assert new_node_count <= original_node_count + base_node_change, (
            f"Expected at most {original_node_count + base_node_change} nodes after rewriting, "
            f"found {new_node_count}"
        )

    def test_rewrite_attention_node_naming(self, bert_tiny_model):
        """Test that rewritten attention nodes have proper naming convention."""
        # Find and rewrite ExpandedAttention patterns
        matcher = PatternMatcher(bert_tiny_model)
        matcher.register_pattern(ExpandedAttentionPattern())
        expanded_results = matcher.match()

        rewriter = PatternRewriter(bert_tiny_model)
        new_model = rewriter.rewrite([(expanded_results, TransposeAttentionPattern)])

        # Check that new nodes have the expected naming pattern
        rewrite_nodes = [
            node
            for node in new_model.graph.node
            if node.name.startswith("Rewrite_TransposeAttentionPattern_")
        ]

        # Each TransposeAttention pattern has 2 nodes (1 Transpose for K, 1 Attention)
        assert len(rewrite_nodes) == 2 * 2, (
            f"Expected {2 * 2} rewrite nodes, found {len(rewrite_nodes)}"
        )

    def test_rewrite_attention_preserves_scale(self, bert_tiny_model):
        """Test that rewriting attention preserves the scale attribute."""
        # Find ExpandedAttention patterns
        matcher = PatternMatcher(bert_tiny_model)
        matcher.register_pattern(ExpandedAttentionPattern())
        expanded_results = matcher.match()

        # Get scales from original patterns
        original_scales = [r.attributes.get("scale") for r in expanded_results]

        # Rewrite to TransposeAttention
        rewriter = PatternRewriter(bert_tiny_model)
        new_model = rewriter.rewrite([(expanded_results, TransposeAttentionPattern)])

        # Find Attention nodes and check their scale attributes
        attention_nodes = [n for n in new_model.graph.node if n.op_type == "Attention"]

        for i, attn_node in enumerate(attention_nodes):
            scale_attr = None
            for attr in attn_node.attribute:
                if attr.name == "scale":
                    scale_attr = attr.f
                    break

            if original_scales[i] is not None:
                assert scale_attr is not None, f"Attention node {i} should have scale attribute"
                np.testing.assert_allclose(
                    scale_attr,
                    original_scales[i],
                    rtol=1e-5,
                    err_msg=f"Attention node {i} scale should match original",
                )

    def test_rewrite_original_model_unchanged(self, bert_tiny_model):
        """Test that rewriting does not modify the original model."""
        original_node_count = len(bert_tiny_model.graph.node)
        original_node_names = [node.name for node in bert_tiny_model.graph.node]

        # Find and rewrite ExpandedAttention patterns
        matcher = PatternMatcher(bert_tiny_model)
        matcher.register_pattern(ExpandedAttentionPattern())
        expanded_results = matcher.match()

        rewriter = PatternRewriter(bert_tiny_model)
        _ = rewriter.rewrite([(expanded_results, TransposeAttentionPattern)])

        # Verify original model is unchanged
        assert len(bert_tiny_model.graph.node) == original_node_count
        assert [node.name for node in bert_tiny_model.graph.node] == original_node_names


class TestPatternRewriterUnnamedNodeStability:
    """Regression tests for rewriting models with unnamed nodes."""

    def test_rewrite_multiple_unnamed_matches_no_key_drift(self):
        """Rewriting multiple unnamed-node matches should not fail with key drift."""
        from onnx.defs import get_schema

        from winml.modelkit.pattern.base import make_single_op_pattern
        from winml.modelkit.pattern.match import PatternMatchResult, SkeletonMatchResult

        # Build a simple chain of unnamed nodes.
        x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])
        y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])
        n0 = onnx.helper.make_node("Relu", ["X"], ["a"])
        n1 = onnx.helper.make_node("Relu", ["a"], ["b"])
        n2 = onnx.helper.make_node("Relu", ["b"], ["c"])
        n3 = onnx.helper.make_node("Relu", ["c"], ["Y"])
        graph = onnx.helper.make_graph([n0, n1, n2, n3], "unnamed_chain", [x], [y])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])

        pattern_schema, _ = make_single_op_pattern(get_schema("Relu", 17))

        class _MatchedPattern:
            def get_schema(self):
                return pattern_schema

        class _ReplacementPattern:
            def get_schema(self):
                return pattern_schema

            def get_onnx_model(self, **kwargs):
                prefix = kwargs.get("prefix", "Rewrite_")
                input_names = kwargs["input_names"]
                output_names = kwargs["output_names"]
                identity = onnx.helper.make_node(
                    "Identity",
                    [input_names[0]],
                    [output_names[0]],
                    name=f"{prefix}Identity",
                )
                subgraph = onnx.helper.make_graph([identity], "replacement", [], [])
                return onnx.helper.make_model(
                    subgraph,
                    opset_imports=[onnx.helper.make_opsetid("", 17)],
                )

        matched_pattern = _MatchedPattern()
        match_1 = PatternMatchResult(
            skeleton_match_result=SkeletonMatchResult(
                pattern=matched_pattern,
                matched_nodes=[n0, n1],
                matched_node_keys=["node_0", "node_1"],
                matcher=None,
                inputs=["X"],
                output="b",
                removable=True,
            ),
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={"T": "float"},
            attributes={},
            input_infos={},
        )
        match_2 = PatternMatchResult(
            skeleton_match_result=SkeletonMatchResult(
                pattern=matched_pattern,
                matched_nodes=[n2, n3],
                matched_node_keys=["node_2", "node_3"],
                matcher=None,
                inputs=["b"],
                output="Y",
                removable=True,
            ),
            schema_input_to_value={},
            schema_output_to_value={},
            type_param_to_type={"T": "float"},
            attributes={},
            input_infos={},
        )

        rewriter = PatternRewriter(model)
        rewritten_model = rewriter.rewrite([([match_1, match_2], _ReplacementPattern)])

        identity_nodes = [node for node in rewritten_model.graph.node if node.op_type == "Identity"]
        assert len(identity_nodes) == 2
        onnx.checker.check_model(rewritten_model)
