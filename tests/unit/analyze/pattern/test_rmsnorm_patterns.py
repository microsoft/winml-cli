# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""RMSNormalization pattern family tests.

Tests cover:
- Cross-matching: Pow and Mul variants do not cross-match
- No cross-matching with LayerNorm patterns
- Numerical equivalence with numpy reference implementation
- Scale shape validation for various axis configurations
- RMSNorm-specific schema properties (dual type constraints, no stash_type)
- Opset compatibility (axes as attribute vs input)

Universal tests (ONNX validity, self-matching, removability, node count/types,
schema basics, match structure) are covered by test_universal_pattern.py.
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from winml.modelkit.onnx import ONNXDomain
from winml.modelkit.pattern import (
    LayerNormalizationMulPattern,
    LayerNormalizationPowPattern,
    PatternMatcher,
    RMSNormalizationMulPattern,
    RMSNormalizationPowPattern,
)

from .conftest import TEST_DOMAIN_VERSIONS


def _make_rmsnorm_model(
    pattern,
    x_shape: tuple,
    scale_shape: tuple | None = None,
    axis: int = -1,
    epsilon: float = 1e-5,
    dtype=np.float32,
    domain_versions: dict | None = None,
) -> onnx.ModelProto:
    """Helper to generate an ONNX model from an RMSNorm pattern."""
    rank = len(x_shape)
    normalized_axis = axis if axis >= 0 else rank + axis
    normalized_dim = x_shape[normalized_axis]

    if scale_shape is None:
        if axis == -1 or normalized_axis == rank - 1:
            scale_shape = (normalized_dim,)
        else:
            scale_shape = tuple(normalized_dim if i == normalized_axis else 1 for i in range(rank))

    inputs = {
        "X": np.random.randn(*x_shape).astype(dtype),
        "Scale": np.ones(scale_shape, dtype=dtype),
    }
    attributes = {"axis": axis, "epsilon": epsilon}
    is_constant_map = {"X": False, "Scale": True}

    onnx_type = {
        np.float32: "tensor(float)",
        np.float16: "tensor(float16)",
    }[dtype]

    return pattern.get_onnx_model(
        inputs,
        attributes,
        is_constant_map,
        [onnx_type],
        domain_versions or TEST_DOMAIN_VERSIONS,
    )


def _numpy_rmsnorm(x: np.ndarray, scale: np.ndarray, axis: int, epsilon: float) -> np.ndarray:
    """Compute RMSNorm using numpy for reference."""
    rms = np.sqrt(np.mean(x**2, axis=axis, keepdims=True) + epsilon)
    return x / rms * scale


# ---------------------------------------------------------------------------
# Pattern-specific generation details
# ---------------------------------------------------------------------------


class TestRMSNormPatternSpecifics:
    """RMSNorm-specific generation details not covered by universal tests."""

    def test_rmsnorm_patterns_are_distinct(self) -> None:
        """Pow and Mul patterns produce different first node op types."""
        pow_model = _make_rmsnorm_model(RMSNormalizationPowPattern(), (2, 4, 6))
        mul_model = _make_rmsnorm_model(RMSNormalizationMulPattern(), (2, 4, 6))
        assert pow_model.graph.node[0].op_type == "Pow"
        assert mul_model.graph.node[0].op_type == "Mul"

    def test_rmsnorm_pow_has_correct_constants(self) -> None:
        """Pow pattern has exponent=2.0 and epsilon constant."""
        model = _make_rmsnorm_model(RMSNormalizationPowPattern(), (2, 4, 6), epsilon=1e-5)

        pow_node = model.graph.node[0]
        exponent_name = pow_node.input[1]
        exponent_init = next(i for i in model.graph.initializer if i.name == exponent_name)
        np.testing.assert_allclose(onnx.numpy_helper.to_array(exponent_init), 2.0)

        add_node = model.graph.node[2]
        eps_name = add_node.input[1]
        eps_init = next(i for i in model.graph.initializer if i.name == eps_name)
        np.testing.assert_allclose(onnx.numpy_helper.to_array(eps_init), 1e-5)

    def test_rmsnorm_mul_squaring_uses_same_input(self) -> None:
        """Mul variant squaring node has both inputs from X."""
        model = _make_rmsnorm_model(RMSNormalizationMulPattern(), (2, 4, 6))
        mul_node = model.graph.node[0]
        assert mul_node.input[0] == mul_node.input[1]

    def test_rmsnorm_schema_has_dual_type_constraints(self) -> None:
        """RMSNorm schema has both T and V type constraints."""
        schema = RMSNormalizationPowPattern().get_schema()
        type_param_strs = [tc.type_param_str for tc in schema.type_constraints]
        assert "T" in type_param_strs
        assert "V" in type_param_strs

    def test_rmsnorm_schema_attributes(self) -> None:
        """RMSNorm schema has axis and epsilon, but not stash_type."""
        schema = RMSNormalizationPowPattern().get_schema()
        assert "axis" in schema.attributes
        assert "epsilon" in schema.attributes
        assert "stash_type" not in schema.attributes


# ---------------------------------------------------------------------------
# Cross-matching
# ---------------------------------------------------------------------------


class TestRMSNormCrossMatching:
    """Pow and Mul variants, and RMSNorm vs LayerNorm, do not cross-match."""

    def test_rmsnorm_patterns_do_not_cross_match(self) -> None:
        pow_pattern = RMSNormalizationPowPattern()
        mul_pattern = RMSNormalizationMulPattern()

        pow_model = _make_rmsnorm_model(pow_pattern, (2, 4, 6))
        mul_model = _make_rmsnorm_model(mul_pattern, (2, 4, 6))

        matcher1 = PatternMatcher(pow_model)
        matcher1.register_pattern(mul_pattern)
        assert len(matcher1.match()) == 0

        matcher2 = PatternMatcher(mul_model)
        matcher2.register_pattern(pow_pattern)
        assert len(matcher2.match()) == 0

    def test_rmsnorm_subgraph_present_in_layernorm(self) -> None:
        """RMSNorm Pow subgraph structurally exists within a LayerNorm graph."""
        ln_pow = LayerNormalizationPowPattern()
        ln_model = ln_pow.get_onnx_model(
            {
                "X": np.random.randn(2, 4, 6).astype(np.float32),
                "Scale": np.ones(6, dtype=np.float32),
                "B": np.zeros(6, dtype=np.float32),
            },
            {"axis": -1, "epsilon": 1e-5},
            {"X": False, "Scale": True, "B": True},
            ["tensor(float)"],
            TEST_DOMAIN_VERSIONS,
        )

        rms_pow = RMSNormalizationPowPattern()
        matcher = PatternMatcher(ln_model)
        matcher.register_pattern(rms_pow)
        assert len(matcher.match()) == 1

    def test_layernorm_does_not_match_rmsnorm(self) -> None:
        rms_model = _make_rmsnorm_model(RMSNormalizationPowPattern(), (2, 4, 6))
        matcher = PatternMatcher(rms_model)
        matcher.register_pattern(LayerNormalizationPowPattern())
        matcher.register_pattern(LayerNormalizationMulPattern())
        assert len(matcher.match()) == 0


# ---------------------------------------------------------------------------
# Numerical equivalence
# ---------------------------------------------------------------------------


class TestRMSNormNumericalEquivalence:
    """Numerical correctness of pattern-generated models vs numpy reference."""

    @pytest.mark.parametrize(
        "pattern_class,axis,x_shape,scale_shape",
        [
            (RMSNormalizationPowPattern, -1, (2, 4, 6), (6,)),
            (RMSNormalizationPowPattern, 0, (8, 4, 6), (8, 1, 1)),
            (RMSNormalizationPowPattern, 1, (2, 8, 6), (1, 8, 1)),
            (RMSNormalizationMulPattern, -1, (2, 4, 6), (6,)),
            (RMSNormalizationMulPattern, 1, (2, 8, 6), (1, 8, 1)),
        ],
        ids=[
            "Pow-axis-last",
            "Pow-axis-0",
            "Pow-axis-1",
            "Mul-axis-last",
            "Mul-axis-1",
        ],
    )
    def test_numerical_equivalence(self, pattern_class, axis, x_shape, scale_shape) -> None:
        pattern = pattern_class()
        x_data = np.random.randn(*x_shape).astype(np.float32)
        scale_data = np.ones(scale_shape, dtype=np.float32) * 1.5

        model = pattern.get_onnx_model(
            {"X": x_data, "Scale": scale_data},
            {"axis": axis, "epsilon": 1e-5},
            {"X": False, "Scale": True},
            ["tensor(float)"],
            TEST_DOMAIN_VERSIONS,
        )

        sess = ort.InferenceSession(model.SerializeToString())
        result = sess.run(None, {"X": x_data})[0]
        expected = _numpy_rmsnorm(x_data, scale_data, axis=axis, epsilon=1e-5)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Scale validation & axis handling
# ---------------------------------------------------------------------------


class TestRMSNormScaleValidation:
    """Scale shape validation for various axis configurations."""

    def test_valid_1d_scale_last_axis(self) -> None:
        pattern = RMSNormalizationPowPattern()
        model = _make_rmsnorm_model(pattern, (2, 4, 6), scale_shape=(6,), axis=-1)
        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        assert len(matcher.match()) == 1

    def test_valid_multidim_scale_non_last_axis(self) -> None:
        pattern = RMSNormalizationPowPattern()
        model = _make_rmsnorm_model(pattern, (2, 8, 6), scale_shape=(1, 8, 1), axis=1)
        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        assert len(matcher.match()) == 1

    @pytest.mark.parametrize("axis", [-1, 0, 1])
    def test_self_matching_multiple_axes(self, axis: int) -> None:
        pattern = RMSNormalizationPowPattern()
        model = _make_rmsnorm_model(pattern, (8, 6, 4), axis=axis)
        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        results = matcher.match()
        assert len(results) == 1
        assert results[0].attributes["axis"] == axis


# ---------------------------------------------------------------------------
# RMSNorm-specific match result details
# ---------------------------------------------------------------------------


class TestRMSNormMatchResultDetails:
    """RMSNorm-specific match result properties (type params, stash_type, shapes)."""

    def test_match_result_has_dual_type_params(self) -> None:
        pattern = RMSNormalizationPowPattern()
        model = _make_rmsnorm_model(pattern, (2, 4, 6))
        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        result = matcher.match()[0]
        assert result.type_param_to_type["T"] == "tensor(float)"
        assert result.type_param_to_type["V"] == "tensor(float)"

    def test_match_result_input_infos(self) -> None:
        pattern = RMSNormalizationPowPattern()
        model = _make_rmsnorm_model(pattern, (2, 4, 6))
        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        result = matcher.match()[0]
        assert result.input_infos["X"].is_constant is False
        assert result.input_infos["Scale"].is_constant is True
        assert result.input_infos["X"].shape == (2, 4, 6)
        assert result.input_infos["Scale"].shape == (6,)

    def test_match_result_attributes_no_stash_type(self) -> None:
        pattern = RMSNormalizationPowPattern()
        model = _make_rmsnorm_model(pattern, (2, 4, 6))
        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        attrs = matcher.match()[0].attributes
        assert attrs["axis"] == -1
        assert abs(attrs["epsilon"] - 1e-5) < 1e-3
        assert "stash_type" not in attrs


# ---------------------------------------------------------------------------
# Opset compatibility
# ---------------------------------------------------------------------------


class TestRMSNormOpsetCompatibility:
    """ReduceMean axes as attribute (opset <18) vs input (opset >=18)."""

    def test_opset18_axes_as_input(self) -> None:
        domain_versions = {ONNXDomain.AI_ONNX: 18}
        pattern = RMSNormalizationPowPattern()
        model = _make_rmsnorm_model(pattern, (2, 4, 6), domain_versions=domain_versions)
        onnx.checker.check_model(model)

        reducemean_node = model.graph.node[1]
        assert len(reducemean_node.input) == 2

        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        results = matcher.match()
        assert len(results) == 1
        assert results[0].attributes["axis"] == -1

    def test_opset17_axes_as_attribute(self) -> None:
        domain_versions = {ONNXDomain.AI_ONNX: 17}
        pattern = RMSNormalizationPowPattern()
        model = _make_rmsnorm_model(pattern, (2, 4, 6), domain_versions=domain_versions)
        onnx.checker.check_model(model)

        reducemean_node = model.graph.node[1]
        axes_attr = next((a for a in reducemean_node.attribute if a.name == "axes"), None)
        assert axes_attr is not None
        assert list(axes_attr.ints) == [-1]
