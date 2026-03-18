"""LayerNormalization pattern family cross-matching tests.

Verifies that Pow and Mul variants do not cross-match, and that multi-node
LayerNorm patterns do not match a transposed single-op LayerNorm model.
"""

import numpy as np
import onnx

from winml.modelkit.pattern import (
    LayerNormalizationMulPattern,
    LayerNormalizationPowPattern,
    PatternMatcher,
    TransposedSingleLayerNormalizationPattern,
)

from .conftest import TEST_DOMAIN_VERSIONS


_LAYERNORM_ATTRS = {"axis": -1, "epsilon": 1e-5}


def _create_layernorm_model(pattern, dtype=np.float32):
    inputs = {
        "X": np.random.randn(2, 128, 768).astype(dtype),
        "Scale": np.ones(768).astype(dtype),
        "B": np.zeros(768).astype(dtype),
    }
    is_constant_map = {"X": False, "Scale": True, "B": True}
    output_dtypes = ["tensor(float)" if dtype == np.float32 else "tensor(float16)"]
    return pattern.get_onnx_model(
        inputs, _LAYERNORM_ATTRS, is_constant_map, output_dtypes, TEST_DOMAIN_VERSIONS
    )


class TestLayerNormCrossMatching:
    """Pow and Mul LayerNorm variants should not cross-match."""

    def test_pow_model_matches_only_pow_pattern(self) -> None:
        """Pow model should match Pow pattern, not Mul."""
        pow_pattern = LayerNormalizationPowPattern()
        mul_pattern = LayerNormalizationMulPattern()
        model = _create_layernorm_model(pow_pattern)
        onnx.checker.check_model(model)

        matcher = PatternMatcher(model)
        matcher.register_pattern(pow_pattern)
        matcher.register_pattern(mul_pattern)
        results = matcher.match()

        pow_matches = [
            r for r in results
            if type(r.skeleton_match_result.pattern).__name__ == "LayerNormalizationPowPattern"
        ]
        mul_matches = [
            r for r in results
            if type(r.skeleton_match_result.pattern).__name__ == "LayerNormalizationMulPattern"
        ]
        assert len(pow_matches) == 1
        assert len(mul_matches) == 0
        assert pow_matches[0].skeleton_match_result.removable is True
        assert "axis" in pow_matches[0].attributes
        assert "epsilon" in pow_matches[0].attributes

    def test_mul_model_matches_only_mul_pattern(self) -> None:
        """Mul model should match Mul pattern, not Pow."""
        pow_pattern = LayerNormalizationPowPattern()
        mul_pattern = LayerNormalizationMulPattern()
        model = _create_layernorm_model(mul_pattern)
        onnx.checker.check_model(model)

        matcher = PatternMatcher(model)
        matcher.register_pattern(pow_pattern)
        matcher.register_pattern(mul_pattern)
        results = matcher.match()

        pow_matches = [
            r for r in results
            if type(r.skeleton_match_result.pattern).__name__ == "LayerNormalizationPowPattern"
        ]
        mul_matches = [
            r for r in results
            if type(r.skeleton_match_result.pattern).__name__ == "LayerNormalizationMulPattern"
        ]
        assert len(pow_matches) == 0
        assert len(mul_matches) == 1
        assert mul_matches[0].skeleton_match_result.removable is True
        assert "axis" in mul_matches[0].attributes
        assert "epsilon" in mul_matches[0].attributes

    def test_layernorm_patterns_do_not_cross_match(self) -> None:
        """Pow pattern does not match Mul model and vice versa."""
        pow_pattern = LayerNormalizationPowPattern()
        mul_pattern = LayerNormalizationMulPattern()

        pow_model = _create_layernorm_model(pow_pattern)
        mul_model = _create_layernorm_model(mul_pattern)

        matcher1 = PatternMatcher(pow_model)
        matcher1.register_pattern(mul_pattern)
        assert len(matcher1.match()) == 0, "Mul pattern should not match Pow model"

        matcher2 = PatternMatcher(mul_model)
        matcher2.register_pattern(pow_pattern)
        assert len(matcher2.match()) == 0, "Pow pattern should not match Mul model"

    def test_multi_node_patterns_do_not_match_transposed_layernorm(self) -> None:
        """Multi-node Pow/Mul patterns should not match a transposed LayerNorm model."""
        transposed_pattern = TransposedSingleLayerNormalizationPattern()
        model = _create_layernorm_model(transposed_pattern)

        matcher1 = PatternMatcher(model)
        matcher1.register_pattern(LayerNormalizationPowPattern())
        assert len(matcher1.match()) == 0, "Pow should not match transposed LayerNorm"

        matcher2 = PatternMatcher(model)
        matcher2.register_pattern(LayerNormalizationMulPattern())
        assert len(matcher2.match()) == 0, "Mul should not match transposed LayerNorm"
