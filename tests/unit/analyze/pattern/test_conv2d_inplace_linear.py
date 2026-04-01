# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for Conv2DInplaceLinear patterns.

Tests:
- Schema consistency: Conv2D patterns share MatMulAdd schema
- Self-matching: each pattern matches only its own generated model
- ONNX model generation: correct node structure
- PatternRewriter: MatMulAdd -> Conv2D rewriting
- Dimension mismatch: PatternMismatchedError on wrong A rank
"""

import numpy as np
import pytest

from winml.modelkit.pattern import (
    Conv2DInplaceLinear2DPattern,
    Conv2DInplaceLinear3DPattern,
    Conv2DInplaceLinear4DPattern,
    MatMulAddPattern,
    PatternMatcher,
    PatternMismatchedError,
    PatternRewriter,
)

from .conftest import TEST_DOMAIN_VERSIONS


_ALL_CONV2D_PATTERNS = [
    Conv2DInplaceLinear4DPattern,
    Conv2DInplaceLinear3DPattern,
    Conv2DInplaceLinear2DPattern,
]

_IN_F = 4
_OUT_F = 8

_PATTERN_INPUTS = {
    "Conv2DInplaceLinear4DPattern": {
        "A": np.random.randn(1, 8, 8, _IN_F).astype(np.float32),
        "B": np.random.randn(_IN_F, _OUT_F).astype(np.float32),
        "C": np.random.randn(_OUT_F).astype(np.float32),
    },
    "Conv2DInplaceLinear3DPattern": {
        "A": np.random.randn(1, 10, _IN_F).astype(np.float32),
        "B": np.random.randn(_IN_F, _OUT_F).astype(np.float32),
        "C": np.random.randn(_OUT_F).astype(np.float32),
    },
    "Conv2DInplaceLinear2DPattern": {
        "A": np.random.randn(4, _IN_F).astype(np.float32),
        "B": np.random.randn(_IN_F, _OUT_F).astype(np.float32),
        "C": np.random.randn(_OUT_F).astype(np.float32),
    },
}


def _create_model(pattern):
    """Build a standalone ONNX model from a pattern instance."""
    inputs = _PATTERN_INPUTS[type(pattern).__name__]
    return pattern.get_onnx_model(
        inputs, {}, {"A": False, "B": True, "C": True},
        ["tensor(float)"], TEST_DOMAIN_VERSIONS,
    )


def _create_matmuladd_model(a_shape):
    """Build a MatMulAdd model for rewriting tests."""
    return MatMulAddPattern().get_onnx_model(
        {
            "A": np.random.randn(*a_shape).astype(np.float32),
            "B": np.random.randn(_IN_F, _OUT_F).astype(np.float32),
            "C": np.random.randn(_OUT_F).astype(np.float32),
        },
        {}, {"A": False, "B": True, "C": True},
        ["tensor(float)"], TEST_DOMAIN_VERSIONS,
    )


class TestConv2DPatternSchemaConsistency:
    """Conv2D patterns must share the MatMulAdd schema."""

    def test_all_patterns_share_matmuladd_schema(self):
        """All Conv2D patterns return the same schema as MatMulAdd."""
        matmuladd_schema = MatMulAddPattern().get_schema()
        for cls in _ALL_CONV2D_PATTERNS:
            assert cls().get_schema() is matmuladd_schema

    def test_schema_has_a_b_c_inputs(self):
        """Schema defines (A, B, C) -> Y."""
        schema = Conv2DInplaceLinear4DPattern().get_schema()
        assert [p.name for p in schema.inputs] == ["A", "B", "C"]
        assert [p.name for p in schema.outputs] == ["Y"]


class TestConv2DPatternSelfMatching:
    """Each Conv2D pattern must match its own generated model."""

    @pytest.fixture
    def all_patterns(self):
        """Return instances of all Conv2D patterns."""
        return [cls() for cls in _ALL_CONV2D_PATTERNS]

    @pytest.mark.parametrize(
        "pattern_cls", _ALL_CONV2D_PATTERNS,
        ids=[c.__name__ for c in _ALL_CONV2D_PATTERNS],
    )
    def test_pattern_matches_itself(self, pattern_cls, all_patterns):
        """A model generated from a pattern must match that pattern."""
        model = _create_model(pattern_cls())

        matcher = PatternMatcher(model)
        for p in all_patterns:
            matcher.register_pattern(p)

        results = matcher.match()
        results_by_type = {}
        for r in results:
            name = type(r.skeleton_match_result.pattern).__name__
            results_by_type[name] = results_by_type.get(name, 0) + 1

        assert results_by_type.get(pattern_cls.__name__, 0) == 1, (
            f"Expected 1 {pattern_cls.__name__} match. All: {results_by_type}"
        )

    @pytest.mark.parametrize(
        "pattern_cls", _ALL_CONV2D_PATTERNS,
        ids=[c.__name__ for c in _ALL_CONV2D_PATTERNS],
    )
    def test_match_is_removable(self, pattern_cls, all_patterns):
        """Matches should be removable."""
        model = _create_model(pattern_cls())
        matcher = PatternMatcher(model)
        for p in all_patterns:
            matcher.register_pattern(p)

        results = matcher.match()
        assert len(results) >= 1
        for r in results:
            assert r.skeleton_match_result.removable is True


class TestConv2DPatternModelGeneration:
    """Test ONNX model generation from Conv2D patterns."""

    def test_4d_pattern_node_types(self):
        """4D: Transpose, Transpose, Reshape, Conv, Transpose."""
        model = _create_model(Conv2DInplaceLinear4DPattern())
        op_types = [n.op_type for n in model.graph.node]
        assert op_types == ["Transpose", "Transpose", "Reshape", "Conv", "Transpose"]

    def test_3d_pattern_node_types(self):
        """3D: Transpose, Unsqueeze, Transpose, Reshape, Conv, Squeeze, Transpose."""
        model = _create_model(Conv2DInplaceLinear3DPattern())
        op_types = [n.op_type for n in model.graph.node]
        assert op_types == [
            "Transpose", "Unsqueeze", "Transpose", "Reshape",
            "Conv", "Squeeze", "Transpose",
        ]

    def test_2d_pattern_node_types(self):
        """2D: Reshape, Transpose, Reshape, Conv, Reshape."""
        model = _create_model(Conv2DInplaceLinear2DPattern())
        op_types = [n.op_type for n in model.graph.node]
        assert op_types == ["Reshape", "Transpose", "Reshape", "Conv", "Reshape"]

    @pytest.mark.parametrize(
        "pattern_cls", _ALL_CONV2D_PATTERNS,
        ids=[c.__name__ for c in _ALL_CONV2D_PATTERNS],
    )
    def test_model_has_correct_io(self, pattern_cls):
        """1 dynamic input (A), 1 output, B/C as initializers."""
        model = _create_model(pattern_cls())
        initializer_names = {init.name for init in model.graph.initializer}
        dynamic_inputs = [
            inp for inp in model.graph.input if inp.name not in initializer_names
        ]
        assert len(dynamic_inputs) == 1
        assert len(model.graph.output) == 1
        assert "B" in initializer_names
        assert "C" in initializer_names


class TestConv2DPatternDimensionCheck:
    """PatternMismatchedError raised for incompatible input dimensions."""

    def test_4d_pattern_rejects_3d_input(self):
        """4D pattern raises for 3D input A."""
        pattern = Conv2DInplaceLinear4DPattern()
        with pytest.raises(PatternMismatchedError, match="4D"):
            pattern.get_internal_constants_and_attributes(
                {"A": np.zeros((1, 10, 4)), "B": np.zeros((4, 8)), "C": np.zeros(8)},
                {}, {"A": False, "B": True, "C": True}, TEST_DOMAIN_VERSIONS,
            )

    def test_3d_pattern_rejects_4d_input(self):
        """3D pattern raises for 4D input A."""
        pattern = Conv2DInplaceLinear3DPattern()
        with pytest.raises(PatternMismatchedError, match="3D"):
            pattern.get_internal_constants_and_attributes(
                {"A": np.zeros((1, 8, 8, 4)), "B": np.zeros((4, 8)), "C": np.zeros(8)},
                {}, {"A": False, "B": True, "C": True}, TEST_DOMAIN_VERSIONS,
            )

    def test_2d_pattern_rejects_3d_input(self):
        """2D pattern raises for 3D input A."""
        pattern = Conv2DInplaceLinear2DPattern()
        with pytest.raises(PatternMismatchedError, match="2D"):
            pattern.get_internal_constants_and_attributes(
                {"A": np.zeros((1, 10, 4)), "B": np.zeros((4, 8)), "C": np.zeros(8)},
                {}, {"A": False, "B": True, "C": True}, TEST_DOMAIN_VERSIONS,
            )


class TestConv2DPatternRewriting:
    """Test PatternRewriter: MatMulAdd -> Conv2D."""

    def test_rewrite_matmuladd_to_conv2d_4d(self):
        """Rewrite MatMulAdd -> Conv2D 4D."""
        model = _create_matmuladd_model(a_shape=(1, 8, 8, _IN_F))

        matcher = PatternMatcher(model)
        matcher.register_pattern(MatMulAddPattern())
        results = matcher.match()
        assert len(results) == 1

        new_model = PatternRewriter(model).rewrite(
            [(results, Conv2DInplaceLinear4DPattern)]
        )

        m = PatternMatcher(new_model)
        m.register_pattern(MatMulAddPattern())
        assert len(m.match()) == 0

        m2 = PatternMatcher(new_model)
        m2.register_pattern(Conv2DInplaceLinear4DPattern())
        assert len(m2.match()) == 1

    def test_rewrite_matmuladd_to_conv2d_3d(self):
        """Rewrite MatMulAdd -> Conv2D 3D."""
        model = _create_matmuladd_model(a_shape=(1, 10, _IN_F))

        matcher = PatternMatcher(model)
        matcher.register_pattern(MatMulAddPattern())
        results = matcher.match()
        assert len(results) == 1

        new_model = PatternRewriter(model).rewrite(
            [(results, Conv2DInplaceLinear3DPattern)]
        )

        m = PatternMatcher(new_model)
        m.register_pattern(MatMulAddPattern())
        assert len(m.match()) == 0

        m2 = PatternMatcher(new_model)
        m2.register_pattern(Conv2DInplaceLinear3DPattern())
        assert len(m2.match()) == 1

    def test_rewrite_matmuladd_to_conv2d_2d(self):
        """Rewrite MatMulAdd -> Conv2D 2D."""
        model = _create_matmuladd_model(a_shape=(4, _IN_F))

        matcher = PatternMatcher(model)
        matcher.register_pattern(MatMulAddPattern())
        results = matcher.match()
        assert len(results) == 1

        new_model = PatternRewriter(model).rewrite(
            [(results, Conv2DInplaceLinear2DPattern)]
        )

        m = PatternMatcher(new_model)
        m.register_pattern(MatMulAddPattern())
        assert len(m.match()) == 0

        m2 = PatternMatcher(new_model)
        m2.register_pattern(Conv2DInplaceLinear2DPattern())
        assert len(m2.match()) == 1

    def test_rewrite_skips_dimension_mismatch(self):
        """Rewriting 2D MatMulAdd to 4D Conv2D is skipped gracefully."""
        model = _create_matmuladd_model(a_shape=(4, _IN_F))

        matcher = PatternMatcher(model)
        matcher.register_pattern(MatMulAddPattern())
        results = matcher.match()
        assert len(results) == 1

        new_model = PatternRewriter(model).rewrite(
            [(results, Conv2DInplaceLinear4DPattern)]
        )

        m = PatternMatcher(new_model)
        m.register_pattern(MatMulAddPattern())
        assert len(m.match()) == 1

    def test_rewrite_preserves_graph_io(self):
        """Graph inputs/outputs preserved after rewriting."""
        model = _create_matmuladd_model(a_shape=(1, 8, 8, _IN_F))
        orig_in = [i.name for i in model.graph.input]
        orig_out = [o.name for o in model.graph.output]

        matcher = PatternMatcher(model)
        matcher.register_pattern(MatMulAddPattern())

        new_model = PatternRewriter(model).rewrite(
            [(matcher.match(), Conv2DInplaceLinear4DPattern)]
        )
        assert [i.name for i in new_model.graph.input] == orig_in
        assert [o.name for o in new_model.graph.output] == orig_out

    def test_rewrite_original_model_unchanged(self):
        """Original model must not be mutated."""
        model = _create_matmuladd_model(a_shape=(1, 8, 8, _IN_F))
        orig_names = [n.name for n in model.graph.node]

        matcher = PatternMatcher(model)
        matcher.register_pattern(MatMulAddPattern())
        PatternRewriter(model).rewrite(
            [(matcher.match(), Conv2DInplaceLinear4DPattern)]
        )
        assert [n.name for n in model.graph.node] == orig_names
