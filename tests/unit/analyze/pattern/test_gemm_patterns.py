"""Gemm pattern family cross-matching tests.

Verifies that MatMulAdd and ReshapeGemmReshape patterns do not cross-match.
"""

import numpy as np

from winml.modelkit.pattern import (
    MatMulAddPattern,
    PatternMatcher,
)
from winml.modelkit.pattern.gemm_patterns import ReshapeGemmReshapePattern

from .conftest import TEST_DOMAIN_VERSIONS


def _create_gemm_family_model(pattern, *, with_constants=False):
    inputs = {
        "A": np.random.randn(10, 2, 4).astype(np.float32),
        "B": np.random.randn(4, 8).astype(np.float32),
        "C": np.random.randn(8).astype(np.float32),
    }
    is_constant_map = {
        "A": False,
        "B": with_constants,
        "C": with_constants,
    }
    output_dtypes = ["tensor(float)"]
    return pattern.get_onnx_model(
        inputs, {}, is_constant_map, output_dtypes, TEST_DOMAIN_VERSIONS
    )


class TestGemmFamilyCrossMatching:
    """MatMulAdd and ReshapeGemmReshape should not cross-match."""

    def test_matmuladd_and_reshape_gemm_reshape_do_not_cross_match(self) -> None:
        """Each pattern's model should match only its own pattern."""
        matmuladd_pattern = MatMulAddPattern()
        reshape_gemm_pattern = ReshapeGemmReshapePattern()

        matmuladd_model = _create_gemm_family_model(matmuladd_pattern)
        reshape_gemm_model = _create_gemm_family_model(reshape_gemm_pattern)

        # MatMulAdd model: should match MatMulAdd, not ReshapeGemmReshape
        matcher1 = PatternMatcher(matmuladd_model)
        matcher1.register_pattern(matmuladd_pattern)
        matcher1.register_pattern(reshape_gemm_pattern)
        results1 = matcher1.match()

        matmuladd_matches = [
            r for r in results1
            if type(r.skeleton_match_result.pattern).__name__ == "MatMulAddPattern"
        ]
        reshape_matches = [
            r for r in results1
            if type(r.skeleton_match_result.pattern).__name__ == "ReshapeGemmReshapePattern"
        ]
        assert len(matmuladd_matches) == 1
        assert len(reshape_matches) == 0

        # ReshapeGemmReshape model: should match ReshapeGemmReshape, not MatMulAdd
        matcher2 = PatternMatcher(reshape_gemm_model)
        matcher2.register_pattern(matmuladd_pattern)
        matcher2.register_pattern(reshape_gemm_pattern)
        results2 = matcher2.match()

        matmuladd_matches2 = [
            r for r in results2
            if type(r.skeleton_match_result.pattern).__name__ == "MatMulAddPattern"
        ]
        reshape_matches2 = [
            r for r in results2
            if type(r.skeleton_match_result.pattern).__name__ == "ReshapeGemmReshapePattern"
        ]
        assert len(matmuladd_matches2) == 0
        assert len(reshape_matches2) == 1
