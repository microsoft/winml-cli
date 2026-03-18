"""Transpose pattern family cross-matching and merge-logic tests.

Verifies cross-matching between ReshapeTransposeReshape and
MergedReshapeTransposeReshape patterns, and tests the axis-merging
behavior of the Merged variant.
"""

import numpy as np
import onnx
import onnxruntime as ort

from winml.modelkit.pattern import (
    MatMulAddPattern,
    MergedReshapeTransposeReshapePattern,
    PatternMatcher,
    ReshapeTransposeReshapePattern,
)
from winml.modelkit.pattern.gemm_patterns import ReshapeGemmReshapePattern

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
        pattern = ReshapeTransposeReshapePattern()
        model = _create_reshape_transpose_model(
            pattern,
            data_shape=(6, 8),
            transpose_shape=(2, 3, 8),
            perm=(0, 2, 1),
            output_shape=(16, 3),
        )

        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        matcher.register_pattern(MatMulAddPattern())
        matcher.register_pattern(ReshapeGemmReshapePattern())
        results = matcher.match()

        rtr_matches = [
            r for r in results
            if type(r.skeleton_match_result.pattern).__name__
            == "ReshapeTransposeReshapePattern"
        ]
        assert len(rtr_matches) == 1
        assert all(
            type(r.skeleton_match_result.pattern).__name__
            != "MatMulAddPattern"
            for r in results
        )
        assert all(
            type(r.skeleton_match_result.pattern).__name__
            != "ReshapeGemmReshapePattern"
            for r in results
        )

    def test_inferred_attributes(self) -> None:
        """Matched result should contain correct transpose_shape, perm, output_shape."""
        pattern = ReshapeTransposeReshapePattern()
        model = _create_reshape_transpose_model(
            pattern,
            data_shape=(2, 3, 4),
            transpose_shape=(2, 3, 4),
            perm=(0, 2, 1),
            output_shape=(2, 4, 3),
        )
        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        results = matcher.match()

        assert len(results) == 1
        attrs = results[0].attributes
        assert attrs["transpose_shape"] == (2, 3, 4)
        assert attrs["perm"] == (0, 2, 1)
        assert attrs["output_shape"] == (2, 4, 3)


class TestMergedReshapeTransposeReshapePattern:
    """Tests for axis-merging behaviour of MergedReshapeTransposeReshape."""

    def test_generates_correct_merged_shapes(self) -> None:
        """Consecutive perm axes should be merged."""
        pattern = MergedReshapeTransposeReshapePattern()
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
            onnx.numpy_helper.to_array(i)
            for i in model.graph.initializer
            if i.name == shape_name
        )
        np.testing.assert_array_equal(shape_tensor, [32, 8, 32, 768])

        # Transpose perm should be (0, 2, 1, 3)
        perm_attr = next(
            list(a.ints)
            for a in model.graph.node[1].attribute
            if a.name == "perm"
        )
        assert perm_attr == [0, 2, 1, 3]

    def test_no_merging_when_not_consecutive(self) -> None:
        """Non-consecutive perm should leave shape unchanged."""
        pattern = MergedReshapeTransposeReshapePattern()
        model = _create_reshape_transpose_model(
            pattern,
            data_shape=(2, 3, 4),
            transpose_shape=(2, 3, 4),
            perm=(0, 2, 1),
            output_shape=(2, 4, 3),
        )
        shape_name = model.graph.node[0].input[1]
        shape_tensor = next(
            onnx.numpy_helper.to_array(i)
            for i in model.graph.initializer
            if i.name == shape_name
        )
        np.testing.assert_array_equal(shape_tensor, [2, 3, 4])

        perm_attr = next(
            list(a.ints)
            for a in model.graph.node[1].attribute
            if a.name == "perm"
        )
        assert perm_attr == [0, 2, 1]

    def test_no_merging_with_fully_reversed_perm(self) -> None:
        """Fully reversed perm has no consecutive axes, so no merging."""
        from winml.modelkit.pattern.transpose_patterns import (
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
        original_pattern = ReshapeTransposeReshapePattern()
        merged_pattern = MergedReshapeTransposeReshapePattern()

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
        original_out = ort.InferenceSession(
            original_model.SerializeToString()
        ).run(None, {"data": input_data})[0]
        merged_out = ort.InferenceSession(
            merged_model.SerializeToString()
        ).run(None, {"data": input_data})[0]

        np.testing.assert_allclose(original_out, merged_out, rtol=1e-5)
