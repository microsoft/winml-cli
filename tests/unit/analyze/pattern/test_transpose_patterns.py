# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Transpose pattern family cross-matching and merge-logic tests.

Verifies cross-matching between ReshapeTransposeReshapeOverlyHighDim and
ReshapeTransposeReshapeLowDim patterns, and tests the axis-merging
behavior of the LowDim variant.
"""

import numpy as np
import onnx
import onnxruntime as ort

from winml.modelkit.pattern import (
    MatMulAddPattern,
    PatternMatcher,
    ReshapeGemmReshapePattern,
    ReshapeTransposeReshapeLowDimPattern,
    ReshapeTransposeReshapeOverlyHighDimPattern,
    get_pattern_input_generator,
)
from winml.modelkit.pattern.transpose_patterns import _compute_merged_transpose

from .conftest import TEST_DOMAIN_VERSIONS


def _create_reshape_transpose_model(pattern, data_shape, transpose_shape, perm, output_shape):
    inputs = {"data": np.random.randn(*data_shape).astype(np.float32)}
    attributes = {
        "transpose_shape": transpose_shape,
        "perm": perm,
        "output_shape": output_shape,
    }
    is_constant_map = {"data": False}
    output_dtypes = ["tensor(float)"]
    return pattern.get_onnx_model(
        inputs, attributes, is_constant_map, output_dtypes, TEST_DOMAIN_VERSIONS
    )


class TestTransposePatternCrossMatching:
    """ReshapeTransposeReshape should not match unrelated patterns."""

    def test_does_not_match_other_patterns(self) -> None:
        # Use a 6-D mergeable model so ReshapeTransposeReshapeOverlyHighDimPattern actually matches
        # (check_skeleton_result filters out non-mergeable instances).
        pattern = ReshapeTransposeReshapeOverlyHighDimPattern()
        model = _create_reshape_transpose_model(
            pattern,
            data_shape=(1, 256, 256, 96),
            transpose_shape=(1, 32, 8, 32, 8, 96),
            perm=(0, 1, 3, 2, 4, 5),
            output_shape=(1024, 8, 8, 96),
        )

        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        matcher.register_pattern(MatMulAddPattern())
        matcher.register_pattern(ReshapeGemmReshapePattern())
        results = matcher.match()

        rtr_matches = [
            r
            for r in results
            if type(r.skeleton_match_result.pattern).__name__
            == "ReshapeTransposeReshapeOverlyHighDimPattern"
        ]
        assert len(rtr_matches) == 1
        assert all(
            type(r.skeleton_match_result.pattern).__name__ != "MatMulAddPattern" for r in results
        )
        assert all(
            type(r.skeleton_match_result.pattern).__name__ != "ReshapeGemmReshapePattern"
            for r in results
        )

    def test_inferred_attributes(self) -> None:
        """Matched result should contain correct transpose_shape, perm, output_shape."""
        # Use a 6-D mergeable model; check_skeleton_result filters non-mergeable instances.
        pattern = ReshapeTransposeReshapeOverlyHighDimPattern()
        model = _create_reshape_transpose_model(
            pattern,
            data_shape=(1, 256, 256, 96),
            transpose_shape=(1, 32, 8, 32, 8, 96),
            perm=(0, 1, 3, 2, 4, 5),
            output_shape=(1024, 8, 8, 96),
        )
        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        results = matcher.match()

        assert len(results) == 1
        attrs = results[0].attributes
        assert attrs["transpose_shape"] == (1, 32, 8, 32, 8, 96)
        assert attrs["perm"] == (0, 1, 3, 2, 4, 5)
        assert attrs["output_shape"] == (1024, 8, 8, 96)


class TestReshapeTransposeReshapeLowDimPattern:
    """Tests for axis-merging behaviour of ReshapeTransposeReshapeLowDimPattern."""

    def test_generates_correct_merged_shapes(self) -> None:
        """Consecutive perm axes should be merged."""
        pattern = ReshapeTransposeReshapeLowDimPattern()
        model = _create_reshape_transpose_model(
            pattern,
            data_shape=(1, 256, 256, 96),
            transpose_shape=(1, 32, 8, 32, 8, 96),
            perm=(0, 1, 3, 2, 4, 5),
            output_shape=(1024, 8, 8, 96),
        )
        onnx.checker.check_model(model)

        # First Reshape should have merged shape (32, 8, 32, 768)
        shape_name = model.graph.node[0].input[1]
        shape_tensor = next(
            onnx.numpy_helper.to_array(i) for i in model.graph.initializer if i.name == shape_name
        )
        np.testing.assert_array_equal(shape_tensor, [32, 8, 32, 768])

        # Transpose perm should be (0, 2, 1, 3)
        perm_attr = next(list(a.ints) for a in model.graph.node[1].attribute if a.name == "perm")
        assert perm_attr == [0, 2, 1, 3]

    def test_no_merging_when_not_consecutive(self) -> None:
        """Non-consecutive perm should leave shape unchanged."""
        pattern = ReshapeTransposeReshapeLowDimPattern()
        model = _create_reshape_transpose_model(
            pattern,
            data_shape=(2, 3, 4),
            transpose_shape=(2, 3, 4),
            perm=(0, 2, 1),
            output_shape=(2, 4, 3),
        )
        shape_name = model.graph.node[0].input[1]
        shape_tensor = next(
            onnx.numpy_helper.to_array(i) for i in model.graph.initializer if i.name == shape_name
        )
        np.testing.assert_array_equal(shape_tensor, [2, 3, 4])

        perm_attr = next(list(a.ints) for a in model.graph.node[1].attribute if a.name == "perm")
        assert perm_attr == [0, 2, 1]

    def test_no_merging_with_fully_reversed_perm(self) -> None:
        """Fully reversed perm has no consecutive axes, so no merging."""
        from winml.modelkit.pattern.transpose_patterns import (  # Testing internal implementation
            _compute_merged_transpose,
        )

        merged_shape, merged_perm = _compute_merged_transpose(
            (1, 32, 8, 32, 8, 96), (5, 4, 3, 2, 1, 0)
        )
        assert len(merged_shape) == 6
        assert merged_shape == (1, 32, 8, 32, 8, 96)
        assert merged_perm == (5, 4, 3, 2, 1, 0)

    def test_merged_and_original_produce_equivalent_output(self) -> None:
        """Merged and original patterns compute the same result."""
        original_pattern = ReshapeTransposeReshapeOverlyHighDimPattern()
        merged_pattern = ReshapeTransposeReshapeLowDimPattern()

        args = {
            "data_shape": (1, 256, 256, 96),
            "transpose_shape": (1, 32, 8, 32, 8, 96),
            "perm": (0, 1, 3, 2, 4, 5),
            "output_shape": (1024, 8, 8, 96),
        }
        original_model = _create_reshape_transpose_model(original_pattern, **args)
        merged_model = _create_reshape_transpose_model(merged_pattern, **args)

        onnx.checker.check_model(original_model)
        onnx.checker.check_model(merged_model)

        input_data = np.random.randn(1, 256, 256, 96).astype(np.float32)
        original_out = ort.InferenceSession(original_model.SerializeToString()).run(
            None, {"data": input_data}
        )[0]
        merged_out = ort.InferenceSession(merged_model.SerializeToString()).run(
            None, {"data": input_data}
        )[0]

        np.testing.assert_allclose(original_out, merged_out, rtol=1e-5)

    def test_already_merged_4d_rtr_is_not_matched(self) -> None:
        """Already-merged 4-D RTR instances are excluded by check_skeleton_result.

        When the transpose_shape is already fully merged (all consecutive groups are
        singletons), _compute_merged_transpose produces no change and check_skeleton_result
        returns None, so the pattern produces zero matches.
        """
        merged_model = _create_reshape_transpose_model(
            ReshapeTransposeReshapeOverlyHighDimPattern(),
            data_shape=(1, 256, 256, 96),
            transpose_shape=(32, 8, 32, 768),
            perm=(0, 2, 1, 3),
            output_shape=(1024, 8, 8, 96),
        )
        matcher = PatternMatcher(merged_model)
        matcher.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())
        results = matcher.match()
        assert len(results) == 0, (
            "Already-merged 4-D RTR must not match ReshapeTransposeReshapeOverlyHighDimPattern"
        )

    def test_unmerged_6d_rtr_is_matched(self) -> None:
        """Unmerged 6-D RTR instances are matched normally by check_skeleton_result."""
        unmerged_model = _create_reshape_transpose_model(
            ReshapeTransposeReshapeOverlyHighDimPattern(),
            data_shape=(1, 256, 256, 96),
            transpose_shape=(1, 32, 8, 32, 8, 96),
            perm=(0, 1, 3, 2, 4, 5),
            output_shape=(1024, 8, 8, 96),
        )
        matcher = PatternMatcher(unmerged_model)
        matcher.register_pattern(ReshapeTransposeReshapeOverlyHighDimPattern())
        results = matcher.match()
        assert len(results) == 1, (
            "Unmerged 6-D RTR must match ReshapeTransposeReshapeOverlyHighDimPattern"
        )


class TestRTRPatternIdAlignment:
    """Verify pattern_id values match the rule configuration expectations."""

    def test_overly_high_dim_pattern_id(self) -> None:
        """ReshapeTransposeReshapeOverlyHighDimPattern must have distinct pattern_id."""
        pattern = ReshapeTransposeReshapeOverlyHighDimPattern()
        assert pattern.pattern_id == "SUBGRAPH/ReshapeTransposeReshapeOverlyHighDimPattern"

    def test_low_dim_pattern_id(self) -> None:
        """ReshapeTransposeReshapeLowDimPattern must have distinct pattern_id."""
        pattern = ReshapeTransposeReshapeLowDimPattern()
        assert pattern.pattern_id == "SUBGRAPH/ReshapeTransposeReshapeLowDimPattern"

    def test_high_dim_pattern_id_differs_from_schema_name(self) -> None:
        """HighDim pattern_id must NOT fall back to the shared schema name."""
        pattern = ReshapeTransposeReshapeOverlyHighDimPattern()
        schema_based = f"SUBGRAPH/{pattern.get_schema().name}"
        assert pattern.pattern_id != schema_based

    def test_low_dim_pattern_id_differs_from_schema_name(self) -> None:
        """LowDim pattern_id must NOT fall back to the shared schema name."""
        pattern = ReshapeTransposeReshapeLowDimPattern()
        schema_based = f"SUBGRAPH/{pattern.get_schema().name}"
        assert pattern.pattern_id != schema_based


class TestRTRPatternInputGeneratorCoverage:
    """Regression tests for RTR pattern input-generation coverage."""

    def test_output_dim_coverage_includes_3d_and_4d(self) -> None:
        """Generator should include both 3-D and 4-D output-shape variants."""
        generator_class = get_pattern_input_generator("ReshapeTransposeReshapeOverlyHighDimPattern")
        gen = generator_class(domain_versions=TEST_DOMAIN_VERSIONS)

        combinations = gen.get_input_and_infinite_attribute_combinations()
        output_dims: set[int] = set()
        for item in combinations:
            output_shape = item.get("output_shape")
            assert isinstance(output_shape, tuple)
            output_dims.add(len(output_shape))

        assert 3 in output_dims
        assert 4 in output_dims

    def test_output_dim_remains_finite_matching_property(self) -> None:
        """output_dim should stay finite and participate in rule matching."""
        generator_class = get_pattern_input_generator("ReshapeTransposeReshapeOverlyHighDimPattern")
        gen = generator_class(domain_versions=TEST_DOMAIN_VERSIONS)

        infinite_properties = gen.get_infinite_property_names()
        assert "output_dim" not in infinite_properties

    def test_coverage_includes_merged_transpose_dim_5(self) -> None:
        """Generator should include a case producing merged_transpose_dim=5."""
        generator_class = get_pattern_input_generator("ReshapeTransposeReshapeOverlyHighDimPattern")
        gen = generator_class(domain_versions=TEST_DOMAIN_VERSIONS)

        combinations = gen.get_input_and_infinite_attribute_combinations()
        merged_dims = set()
        for item in combinations:
            transpose_shape = item.get("transpose_shape")
            perm = item.get("perm")
            assert isinstance(transpose_shape, tuple)
            assert isinstance(perm, tuple)
            merged_shape, _ = _compute_merged_transpose(
                transpose_shape,
                perm,
            )
            merged_dims.add(len(merged_shape))

        assert 5 in merged_dims
