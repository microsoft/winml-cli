"""GELU pattern family cross-matching tests.

Verifies that each of the 4 Erf-based GELU variants matches only its own
generated model and not the other 3 variants.
"""

import numpy as np
import onnx
import pytest

from winml.modelkit.pattern import (
    Gelu1Pattern,
    Gelu2Pattern,
    Gelu3Pattern,
    Gelu4Pattern,
    PatternMatcher,
)

from .conftest import TEST_DOMAIN_VERSIONS


_ALL_GELU_CLASSES = [Gelu1Pattern, Gelu2Pattern, Gelu3Pattern, Gelu4Pattern]


def _create_gelu_model(pattern, dtype=np.float32):
    inputs = {"X": np.random.randn(2, 4).astype(dtype)}
    is_constant_map = {"X": False}
    output_dtypes = ["tensor(float)" if dtype == np.float32 else "tensor(float16)"]
    return pattern.get_onnx_model(
        inputs, {}, is_constant_map, output_dtypes, TEST_DOMAIN_VERSIONS
    )


class TestGeluCrossMatching:
    """Each GELU variant model should match only its own pattern, not the other 3."""

    @pytest.mark.parametrize(
        "source_class",
        _ALL_GELU_CLASSES,
        ids=[cls.__name__ for cls in _ALL_GELU_CLASSES],
    )
    def test_gelu_variant_matches_only_itself(self, source_class) -> None:
        """Register all 4 GELU patterns; only the source pattern should match."""
        source_pattern = source_class()
        model = _create_gelu_model(source_pattern)
        onnx.checker.check_model(model)

        matcher = PatternMatcher(model)
        all_patterns = [cls() for cls in _ALL_GELU_CLASSES]
        for p in all_patterns:
            matcher.register_pattern(p)

        results = matcher.match()

        results_by_type = {}
        for r in results:
            name = type(r.skeleton_match_result.pattern).__name__
            results_by_type[name] = results_by_type.get(name, 0) + 1

        source_name = source_class.__name__
        assert results_by_type.get(source_name, 0) == 1, (
            f"Expected 1 {source_name} match, got {results_by_type.get(source_name, 0)}"
        )
        for cls in _ALL_GELU_CLASSES:
            if cls is not source_class:
                other_name = cls.__name__
                assert results_by_type.get(other_name, 0) == 0, (
                    f"Expected 0 {other_name} matches, got {results_by_type.get(other_name, 0)}"
                )

        for r in results:
            assert r.skeleton_match_result.removable is True
