# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for ReshapeTransposeReshape pattern matching and rewriting on SAM2 encoder.

These tests verify the pattern matching and rewriting functionality using the
facebook SAM2.1-hiera-small encoder model which contains 26 ReshapeTransposeReshape
patterns commonly found in attention mechanisms.
"""

from pathlib import Path

import onnx
import pytest

from winml.modelkit.pattern import (
    PatternMatcher,
    PatternRewriter,
    ReshapeTransposeReshapeLowDimPattern,
    ReshapeTransposeReshapeOverlyHighDimPattern,
)


# Path to the SAM2 encoder fixture
FIXTURE_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"
SAM2_ENCODER_PATH = FIXTURE_DIR / "facebook_sam2.1-hiera-small[sam2_encoder.py_17].onnx"


@pytest.fixture
def sam2_model():
    """Load the SAM2 encoder model."""
    if not SAM2_ENCODER_PATH.exists():
        pytest.skip(f"Test model not found: {SAM2_ENCODER_PATH}")
    return onnx.load(str(SAM2_ENCODER_PATH))


class TestSam2ReshapeTransposeReshapeOverlyHighDimMatching:
    """Tests for ReshapeTransposeReshape pattern matching on SAM2 encoder."""

    def test_original_model_has_expected_node_count(self, sam2_model):
        """Verify the original model has the expected number of nodes."""
        assert len(sam2_model.graph.node) == 654, (
            f"Expected 654 nodes, got {len(sam2_model.graph.node)}"
        )

    def test_match_reshape_transpose_reshape_patterns(self, sam2_model):
        """Test that ReshapeTransposeReshape patterns are found in SAM2 encoder.

        The SAM2 encoder contains 26 ReshapeTransposeReshape patterns from
        attention mechanisms.
        """
        matcher = PatternMatcher(sam2_model, raise_on_invalid_model=False)
        matcher.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())

        results = matcher.match()

        assert len(results) == 26, (
            f"Expected 26 ReshapeTransposeReshape matches, got {len(results)}"
        )

    def test_no_merged_patterns_in_original_model(self, sam2_model):
        """Test that ReshapeTransposeReshapeLowDim patterns are NOT found in original model.

        The original model has >=6D transposes, not <=5D merged transposes.
        """
        matcher = PatternMatcher(sam2_model, raise_on_invalid_model=False)
        matcher.register_pattern(ReshapeTransposeReshapeLowDimPattern())

        results = matcher.match()

        assert len(results) == 0, f"Expected 0 LowDim matches in original model, got {len(results)}"

    def test_matched_patterns_have_expected_structure(self, sam2_model):
        """Test that matched patterns have the expected 6D transpose structure."""
        matcher = PatternMatcher(sam2_model, raise_on_invalid_model=False)
        matcher.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())

        results = matcher.match()
        assert len(results) > 0, "No patterns matched"

        # Check first pattern's structure
        first_match = results[0]
        attrs = first_match.attributes

        # Should have 6D transpose shape
        transpose_shape = attrs["transpose_shape"]
        assert len(transpose_shape) == 6, (
            f"Expected 6D transpose_shape, got {len(transpose_shape)}D: {transpose_shape}"
        )

        # Should have 6-element perm
        perm = attrs["perm"]
        assert len(perm) == 6, f"Expected 6-element perm, got {len(perm)}: {perm}"

        # Common pattern in SAM2: perm = (0, 1, 3, 2, 4, 5)
        assert perm == (0, 1, 3, 2, 4, 5), f"Expected perm (0, 1, 3, 2, 4, 5), got {perm}"

    def test_all_matched_patterns_are_removable(self, sam2_model):
        """Test that all matched patterns are marked as removable."""
        matcher = PatternMatcher(sam2_model, raise_on_invalid_model=False)
        matcher.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())

        results = matcher.match()

        for i, result in enumerate(results):
            assert result.skeleton_match_result.removable, (
                f"Pattern match {i} is not removable: {result.skeleton_match_result.matched_nodes}"
            )


class TestSam2PatternRewriting:
    """Tests for rewriting ReshapeTransposeReshapeOverlyHighDim to ReshapeTransposeReshapeLowDim."""

    def test_rewrite_preserves_node_count(self, sam2_model):
        """Test that rewriting preserves the total node count.

        Both patterns have 3 nodes (Reshape-Transpose-Reshape), so the
        total node count should remain the same.
        """
        matcher = PatternMatcher(sam2_model, raise_on_invalid_model=False)
        matcher.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())
        results = matcher.match()

        rewriter = PatternRewriter(sam2_model)
        rewritten_model = rewriter.rewrite(
            [
                (results, ReshapeTransposeReshapeLowDimPattern),
            ]
        )

        assert len(rewritten_model.graph.node) == len(sam2_model.graph.node), (
            f"Expected {len(sam2_model.graph.node)} nodes, got {len(rewritten_model.graph.node)}"
        )

    def test_rewritten_model_matches_merged_patterns(self, sam2_model):
        """Test that rewritten model contains ReshapeTransposeReshapeLowDim patterns."""
        # Match original patterns
        matcher = PatternMatcher(sam2_model, raise_on_invalid_model=False)
        matcher.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())
        original_results = matcher.match()
        original_count = len(original_results)

        # Rewrite
        rewriter = PatternRewriter(sam2_model)
        rewritten_model = rewriter.rewrite(
            [
                (original_results, ReshapeTransposeReshapeLowDimPattern),
            ]
        )

        # Match merged patterns in rewritten model
        matcher2 = PatternMatcher(rewritten_model, raise_on_invalid_model=False)
        matcher2.register_pattern(ReshapeTransposeReshapeLowDimPattern())
        merged_results = matcher2.match()

        assert len(merged_results) == original_count, (
            f"Expected {original_count} LowDim matches, got {len(merged_results)}"
        )

    def test_rewritten_model_still_matches_original_pattern(self, sam2_model):
        """Test that rewritten model also matches ReshapeTransposeReshapeOverlyHighDim pattern.

        ReshapeTransposeReshapeLowDimPattern inherits the same skeleton topology,
        so the rewritten model still matches the OverlyHighDim pattern (just with different shapes).
        """
        # Match original patterns
        matcher = PatternMatcher(sam2_model, raise_on_invalid_model=False)
        matcher.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())
        original_results = matcher.match()
        original_count = len(original_results)

        # Rewrite
        rewriter = PatternRewriter(sam2_model)
        rewritten_model = rewriter.rewrite(
            [
                (original_results, ReshapeTransposeReshapeLowDimPattern),
            ]
        )

        # Match original pattern type in rewritten model
        matcher2 = PatternMatcher(rewritten_model, raise_on_invalid_model=False)
        matcher2.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())
        rtr_results = matcher2.match()

        assert len(rtr_results) == original_count, (
            f"Expected {original_count} OverlyHighDim matches in rewritten model, "
            f"got {len(rtr_results)}"
        )

    def test_merged_patterns_have_reduced_dimensions(self, sam2_model):
        """Test that merged patterns have reduced Transpose dimensions (4D instead of 6D)."""
        # Match and rewrite
        matcher = PatternMatcher(sam2_model, raise_on_invalid_model=False)
        matcher.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())
        original_results = matcher.match()

        rewriter = PatternRewriter(sam2_model)
        rewritten_model = rewriter.rewrite(
            [
                (original_results, ReshapeTransposeReshapeLowDimPattern),
            ]
        )

        # Match merged patterns
        matcher2 = PatternMatcher(rewritten_model, raise_on_invalid_model=False)
        matcher2.register_pattern(ReshapeTransposeReshapeLowDimPattern())
        merged_results = matcher2.match()

        assert len(merged_results) > 0, "No merged patterns found"

        # Check that merged patterns have 4D transpose (reduced from 6D)
        for i, result in enumerate(merged_results):
            attrs = result.attributes
            transpose_shape = attrs["transpose_shape"]
            perm = attrs["perm"]

            assert len(transpose_shape) == 4, (
                f"Pattern {i}: Expected 4D merged transpose_shape, "
                f"got {len(transpose_shape)}D: {transpose_shape}"
            )
            assert len(perm) == 4, (
                f"Pattern {i}: Expected 4-element merged perm, got {len(perm)}: {perm}"
            )

            # Merged perm should be (0, 2, 1, 3) for the common SAM2 pattern
            assert perm == (0, 2, 1, 3), (
                f"Pattern {i}: Expected merged perm (0, 2, 1, 3), got {perm}"
            )

    def test_merged_shapes_have_no_negative_dimensions(self, sam2_model):
        """Test that all merged shape constants have no negative dimensions.

        The merging process should resolve all -1 dimensions using the actual
        input tensor sizes.
        """
        # Match and rewrite
        matcher = PatternMatcher(sam2_model, raise_on_invalid_model=False)
        matcher.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())
        original_results = matcher.match()

        rewriter = PatternRewriter(sam2_model)
        rewritten_model = rewriter.rewrite(
            [
                (original_results, ReshapeTransposeReshapeLowDimPattern),
            ]
        )

        # Check all Reshape nodes in the rewritten model that are part of our pattern
        matcher2 = PatternMatcher(rewritten_model, raise_on_invalid_model=False)
        matcher2.register_pattern(ReshapeTransposeReshapeLowDimPattern())
        merged_results = matcher2.match()

        for i, result in enumerate(merged_results):
            matched_nodes = result.skeleton_match_result.matched_nodes

            # Get shape constants
            reshape1_node = matched_nodes[0]
            reshape2_node = matched_nodes[2]

            shape1 = matcher2.tensor_values.get(reshape1_node.input[1])
            shape2 = matcher2.tensor_values.get(reshape2_node.input[1])

            assert shape1 is not None, f"Pattern {i}: First Reshape shape not found"
            assert shape2 is not None, f"Pattern {i}: Second Reshape shape not found"

            # Check no negative dimensions
            assert all(d > 0 for d in shape1), (
                f"Pattern {i}: First Reshape has negative dims: {shape1}"
            )
            assert all(d > 0 for d in shape2), (
                f"Pattern {i}: Second Reshape has negative dims: {shape2}"
            )

    def test_rewritten_model_is_valid_onnx(self, sam2_model):
        """Test that the rewritten model passes ONNX validation."""
        # Match and rewrite
        matcher = PatternMatcher(sam2_model, raise_on_invalid_model=False)
        matcher.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())
        original_results = matcher.match()

        rewriter = PatternRewriter(sam2_model)
        rewritten_model = rewriter.rewrite(
            [
                (original_results, ReshapeTransposeReshapeLowDimPattern),
            ]
        )

        # ONNX validation
        try:
            onnx.checker.check_model(rewritten_model)
        except Exception as e:
            pytest.fail(f"Rewritten model failed ONNX validation: {e}")


class TestPatternOverlapAssertion:
    """Tests for the pattern overlap assertion in PatternRewriter."""

    def test_no_overlap_between_original_patterns(self, sam2_model):
        """Test that there's no overlap between matched ReshapeTransposeReshape patterns.

        Each Reshape-Transpose-Reshape pattern should be independent with no
        shared nodes.
        """
        matcher = PatternMatcher(sam2_model, raise_on_invalid_model=False)
        matcher.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())
        results = matcher.match()

        # Collect all matched nodes
        all_nodes = set()
        for result in results:
            matched_nodes = set(result.skeleton_match_result.matched_node_keys)
            overlap = all_nodes & matched_nodes
            assert not overlap, f"Found overlapping nodes in pattern matches: {overlap}"
            all_nodes.update(matched_nodes)

        # Total matched nodes should be 26 patterns * 3 nodes = 78 nodes
        assert len(all_nodes) == 26 * 3, (
            f"Expected {26 * 3} unique matched nodes, got {len(all_nodes)}"
        )
