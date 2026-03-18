"""Tests for transpose-wrapped RMSNorm pattern.

Tests cover:
- Transpose permutation computation for arbitrary axis values
- Pattern self-matching (generated model matches its own pattern)
- Numerical equivalence with manual RMSNorm computation
- Multi-dimensional Scale support (reshaped to 1D)
- Pattern rewriting from Pow/Mul patterns to Transposed pattern
"""

from typing import ClassVar

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from winml.modelkit.onnx.domains import ONNXDomain
from winml.modelkit.pattern import (
    PatternMatcher,
    PatternRewriter,
    RMSNormalizationMulPattern,
    RMSNormalizationPowPattern,
    TransposedSingleRMSNormalizationPattern,
)


# Access _compute_transpose_permutation via pattern instance (instance method)
_pattern_instance = TransposedSingleRMSNormalizationPattern()
_compute_transpose_permutation = _pattern_instance._compute_transpose_permutation


def _numpy_rmsnorm(
    x: np.ndarray, scale: np.ndarray, axis: int, epsilon: float
) -> np.ndarray:
    """Compute RMSNorm using numpy for reference."""
    rms = np.sqrt(np.mean(x**2, axis=axis, keepdims=True) + epsilon)
    return x / rms * scale


class TestTransposePermutation:
    """Tests for _compute_transpose_permutation utility function."""

    def test_axis_last_returns_identity(self):
        """Test that axis=-1 returns identity permutation."""
        perm_f, perm_i = _compute_transpose_permutation(axis=-1, rank=4)
        assert perm_f == [0, 1, 2, 3]
        assert perm_i == [0, 1, 2, 3]

    def test_axis_1_rank_4(self):
        """Test permutation for axis=1 in rank=4 tensor."""
        perm_f, perm_i = _compute_transpose_permutation(axis=1, rank=4)
        assert perm_f == [0, 2, 3, 1]
        for i in range(4):
            assert perm_i[perm_f[i]] == i

    def test_axis_0_rank_3(self):
        """Test permutation for axis=0 in rank=3 tensor."""
        perm_f, perm_i = _compute_transpose_permutation(axis=0, rank=3)
        assert perm_f == [1, 2, 0]
        for i in range(3):
            assert perm_i[perm_f[i]] == i

    def test_negative_axis(self):
        """Test that negative axis values are normalized correctly."""
        perm_f, perm_i = _compute_transpose_permutation(axis=-2, rank=4)
        assert perm_f == [0, 1, 3, 2]
        for i in range(4):
            assert perm_i[perm_f[i]] == i

    def test_roundtrip_transpose(self):
        """Verify that forward then inverse transpose restores original."""
        for axis in [0, 1, 2]:
            for rank in [3, 4, 5]:
                if axis < rank:
                    perm_f, perm_i = _compute_transpose_permutation(axis, rank)
                    shape = tuple(range(2, 2 + rank))
                    arr = np.arange(np.prod(shape)).reshape(shape)
                    transposed = np.transpose(arr, perm_f)
                    restored = np.transpose(transposed, perm_i)
                    np.testing.assert_array_equal(arr, restored)

    def test_axis_out_of_range(self):
        """Test that invalid axis raises ValueError."""
        with pytest.raises(ValueError, match=r"axis .* out of range"):
            _compute_transpose_permutation(axis=5, rank=4)

        with pytest.raises(ValueError, match=r"axis .* out of range"):
            _compute_transpose_permutation(axis=-5, rank=4)

    def test_inverse_is_correct(self):
        """Test that inverse permutation correctly reverses forward permutation."""
        test_cases = [
            (0, 3),
            (1, 4),
            (2, 5),
            (-2, 4),
        ]

        for axis, rank in test_cases:
            perm_f, perm_i = _compute_transpose_permutation(axis, rank)
            identity = list(range(rank))
            result = [perm_i[perm_f[i]] for i in range(rank)]
            assert result == identity, f"Failed for axis={axis}, rank={rank}"


class TestTransposedRMSNormPattern:
    """Tests for TransposedSingleRMSNormalizationPattern."""

    _DOMAIN_VERSIONS: ClassVar[dict] = {ONNXDomain.AI_ONNX: 23}

    def test_transposed_pattern_self_matching(self):
        """Test that TransposedSingleRMSNormalizationPattern matches its own output."""
        pattern = TransposedSingleRMSNormalizationPattern()

        inputs = {
            "X": np.random.randn(2, 64, 128).astype(np.float32),
            "Scale": np.ones((1, 64, 1), dtype=np.float32),
        }
        attributes = {"axis": 1, "epsilon": 1e-5}
        is_constant_map = {"X": False, "Scale": True}
        output_dtypes = ["tensor(float)"]

        model = pattern.get_onnx_model(
            inputs,
            attributes,
            is_constant_map,
            output_dtypes,
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        # Verify structure: Transpose, Reshape(Scale), RMSNormalization, Transpose
        assert len(model.graph.node) == 4
        assert model.graph.node[0].op_type == "Transpose"
        assert model.graph.node[1].op_type == "Reshape"
        assert model.graph.node[2].op_type == "RMSNormalization"
        assert model.graph.node[3].op_type == "Transpose"

        # Verify RMSNormalization has axis=-1
        rmsnorm_node = model.graph.node[2]
        axis_attr = next(
            (attr for attr in rmsnorm_node.attribute if attr.name == "axis"), None
        )
        assert axis_attr is not None
        assert axis_attr.i == -1

        # Self-match
        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        results = matcher.match()

        assert len(results) == 1
        assert results[0].attributes["axis"] == 1

    def test_transposed_pattern_numerical_equivalence(self):
        """Test numerical equivalence of TransposedSingleRMSNormalizationPattern."""
        pattern = TransposedSingleRMSNormalizationPattern()
        x_data = np.random.randn(2, 64, 128).astype(np.float32)
        scale_data = np.ones(64, dtype=np.float32)

        inputs = {"X": x_data, "Scale": scale_data}
        attributes = {"axis": 1, "epsilon": 1e-5}

        model = pattern.get_onnx_model(
            inputs,
            attributes,
            {"X": False, "Scale": True},
            ["tensor(float)"],
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        # Verify structure has Reshape node
        assert len(model.graph.node) == 4
        assert model.graph.node[1].op_type == "Reshape"

        # Run inference
        sess = ort.InferenceSession(model.SerializeToString())
        result = sess.run(None, {"X": x_data})[0]

        # Compute expected result manually (RMSNorm on axis=1)
        expected = _numpy_rmsnorm(
            x_data, scale_data[np.newaxis, :, np.newaxis], axis=1, epsilon=1e-5
        )

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_transposed_pattern_with_multidim_scale(self):
        """Test TransposedSingleRMSNormalizationPattern with multi-dimensional Scale."""
        pattern = TransposedSingleRMSNormalizationPattern()
        x_data = np.random.randn(2, 64, 128).astype(np.float32)
        scale_data = np.ones((1, 64, 1), dtype=np.float32) * 2.0

        inputs = {"X": x_data, "Scale": scale_data}
        attributes = {"axis": 1, "epsilon": 1e-5}

        model = pattern.get_onnx_model(
            inputs,
            attributes,
            {"X": False, "Scale": True},
            ["tensor(float)"],
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        sess = ort.InferenceSession(model.SerializeToString())
        result = sess.run(None, {"X": x_data})[0]

        expected = _numpy_rmsnorm(x_data, scale_data, axis=1, epsilon=1e-5)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_transposed_pattern_axis_last_identity_transpose(self):
        """Test TransposedSingleRMSNormalizationPattern with axis=-1 (identity transpose)."""
        pattern = TransposedSingleRMSNormalizationPattern()
        x_data = np.random.randn(2, 64, 128).astype(np.float32)
        scale_data = np.ones(128, dtype=np.float32) * 1.5

        inputs = {"X": x_data, "Scale": scale_data}
        attributes = {"axis": -1, "epsilon": 1e-5}

        model = pattern.get_onnx_model(
            inputs,
            attributes,
            {"X": False, "Scale": True},
            ["tensor(float)"],
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        # Verify Transpose nodes have identity permutation for axis=-1
        transpose1 = model.graph.node[0]
        transpose2 = model.graph.node[3]
        perm1 = next(
            (attr.ints for attr in transpose1.attribute if attr.name == "perm"), None
        )
        perm2 = next(
            (attr.ints for attr in transpose2.attribute if attr.name == "perm"), None
        )
        assert list(perm1) == [0, 1, 2]
        assert list(perm2) == [0, 1, 2]

        # Self-match should recover positive axis
        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        results = matcher.match()
        assert len(results) == 1
        assert results[0].attributes["axis"] == 2  # perm[-1] = 2 for identity

        # Run inference and verify numerical correctness
        sess = ort.InferenceSession(model.SerializeToString())
        result = sess.run(None, {"X": x_data})[0]

        expected = _numpy_rmsnorm(x_data, scale_data, axis=-1, epsilon=1e-5)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_transposed_pattern_axis_0(self):
        """Test TransposedSingleRMSNormalizationPattern with axis=0 and multi-dim Scale."""
        pattern = TransposedSingleRMSNormalizationPattern()
        x_data = np.random.randn(8, 64, 128).astype(np.float32)
        scale_data = np.ones((8, 1, 1), dtype=np.float32) * 2.0

        inputs = {"X": x_data, "Scale": scale_data}
        attributes = {"axis": 0, "epsilon": 1e-5}

        model = pattern.get_onnx_model(
            inputs,
            attributes,
            {"X": False, "Scale": True},
            ["tensor(float)"],
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        # Verify Transpose permutation moves axis 0 to the end
        transpose1 = model.graph.node[0]
        perm1 = list(
            next(attr.ints for attr in transpose1.attribute if attr.name == "perm")
        )
        assert perm1 == [1, 2, 0]

        # Self-match should recover axis=0
        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        results = matcher.match()
        assert len(results) == 1
        assert results[0].attributes["axis"] == 0

        # Run inference and verify numerical correctness
        sess = ort.InferenceSession(model.SerializeToString())
        result = sess.run(None, {"X": x_data})[0]

        expected = _numpy_rmsnorm(x_data, scale_data, axis=0, epsilon=1e-5)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_transposed_rmsnorm_shares_schema_with_expanded(self):
        """Test that transposed and expanded patterns share the same schema."""
        transposed = TransposedSingleRMSNormalizationPattern()
        pow_pattern = RMSNormalizationPowPattern()
        assert transposed.get_schema() == pow_pattern.get_schema()

    def test_transposed_rmsnorm_does_not_match_expanded(self):
        """Test that transposed pattern does not match expanded pattern models."""
        pattern = TransposedSingleRMSNormalizationPattern()
        pow_pattern = RMSNormalizationPowPattern()

        expanded_model = pow_pattern.get_onnx_model(
            {
                "X": np.random.randn(2, 4, 6).astype(np.float32),
                "Scale": np.ones(6, dtype=np.float32),
            },
            {"axis": -1, "epsilon": 1e-5},
            {"X": False, "Scale": True},
            ["tensor(float)"],
            {ONNXDomain.AI_ONNX: 17},
        )

        matcher = PatternMatcher(expanded_model)
        matcher.register_pattern(pattern)
        assert len(matcher.match()) == 0


class TestRMSNormalizationPatternRewriting:
    """Tests for rewriting RMSNormalization patterns to TransposedSingleRMSNormPattern."""

    _DOMAIN_VERSIONS: ClassVar[dict] = {ONNXDomain.AI_ONNX: 23}

    def test_rewrite_rmsnorm_pow_to_transposed(self):
        """Test rewriting Pow pattern to Transposed pattern (axis=-1)."""
        pow_pattern = RMSNormalizationPowPattern()
        inputs = {
            "X": np.random.randn(2, 128, 768).astype(np.float32),
            "Scale": np.ones(768).astype(np.float32),
        }
        attributes = {"axis": -1, "epsilon": 1e-5}
        is_constant_map = {"X": False, "Scale": True}
        output_dtypes = ["tensor(float)"]

        model = pow_pattern.get_onnx_model(
            inputs,
            attributes,
            is_constant_map,
            output_dtypes,
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        # Pow pattern has 6 nodes
        original_node_count = len(model.graph.node)
        assert original_node_count == 6

        matcher = PatternMatcher(model)
        matcher.register_pattern(pow_pattern)
        pow_results = matcher.match()

        assert len(pow_results) == 1

        rewriter = PatternRewriter(model)
        new_model = rewriter.rewrite(
            [(pow_results, TransposedSingleRMSNormalizationPattern)]
        )

        onnx.checker.check_model(new_model)

        # 6-node pattern -> 4-node pattern
        new_node_count = len(new_model.graph.node)
        assert new_node_count < original_node_count

        # Verify Pow pattern is no longer present
        new_matcher = PatternMatcher(new_model)
        new_matcher.register_pattern(pow_pattern)
        assert len(new_matcher.match()) == 0

        # Verify RMSNormalization node exists
        rmsnorm_nodes = [
            n for n in new_model.graph.node if n.op_type == "RMSNormalization"
        ]
        assert len(rmsnorm_nodes) == 1

    def test_rewrite_rmsnorm_mul_to_transposed(self):
        """Test rewriting Mul pattern to Transposed pattern (axis=-1)."""
        mul_pattern = RMSNormalizationMulPattern()
        inputs = {
            "X": np.random.randn(2, 128, 768).astype(np.float32),
            "Scale": np.ones(768).astype(np.float32),
        }
        attributes = {"axis": -1, "epsilon": 1e-5}
        is_constant_map = {"X": False, "Scale": True}
        output_dtypes = ["tensor(float)"]

        model = mul_pattern.get_onnx_model(
            inputs,
            attributes,
            is_constant_map,
            output_dtypes,
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        original_node_count = len(model.graph.node)
        assert original_node_count == 6

        matcher = PatternMatcher(model)
        matcher.register_pattern(mul_pattern)
        mul_results = matcher.match()

        assert len(mul_results) == 1

        rewriter = PatternRewriter(model)
        new_model = rewriter.rewrite(
            [(mul_results, TransposedSingleRMSNormalizationPattern)]
        )

        onnx.checker.check_model(new_model)

        new_node_count = len(new_model.graph.node)
        assert new_node_count < original_node_count

        new_matcher = PatternMatcher(new_model)
        new_matcher.register_pattern(mul_pattern)
        assert len(new_matcher.match()) == 0

        rmsnorm_nodes = [
            n for n in new_model.graph.node if n.op_type == "RMSNormalization"
        ]
        assert len(rmsnorm_nodes) == 1

    def test_rewrite_preserves_rmsnorm_semantics(self):
        """Test numerical equivalence after rewriting Pow pattern."""
        pow_pattern = RMSNormalizationPowPattern()
        input_data = np.random.randn(2, 128, 768).astype(np.float32)
        scale_data = np.random.randn(768).astype(np.float32)

        inputs = {"X": input_data, "Scale": scale_data}
        attributes = {"axis": -1, "epsilon": 1e-5}
        is_constant_map = {"X": False, "Scale": True}
        output_dtypes = ["tensor(float)"]

        model = pow_pattern.get_onnx_model(
            inputs,
            attributes,
            is_constant_map,
            output_dtypes,
            self._DOMAIN_VERSIONS,
        )

        # Run original model
        original_sess = ort.InferenceSession(model.SerializeToString())
        original_output = original_sess.run(None, {"X": input_data})[0]

        # Find and rewrite pattern
        matcher = PatternMatcher(model)
        matcher.register_pattern(pow_pattern)
        pow_results = matcher.match()

        rewriter = PatternRewriter(model)
        new_model = rewriter.rewrite(
            [(pow_results, TransposedSingleRMSNormalizationPattern)]
        )

        # Run rewritten model
        new_sess = ort.InferenceSession(new_model.SerializeToString())
        new_output = new_sess.run(None, {"X": input_data})[0]

        np.testing.assert_allclose(
            original_output,
            new_output,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Rewritten RMSNorm should produce equivalent output",
        )

    def test_rewrite_pow_pattern_axis_1_to_transposed(self):
        """Test rewriting Pow pattern with axis=1 and multi-dim Scale."""
        pow_pattern = RMSNormalizationPowPattern()

        input_data = np.random.randn(2, 64, 128).astype(np.float32)
        scale_data = np.ones((1, 64, 1), dtype=np.float32) * 1.5

        inputs = {"X": input_data, "Scale": scale_data}
        attributes = {"axis": 1, "epsilon": 1e-5}
        is_constant_map = {"X": False, "Scale": True}
        output_dtypes = ["tensor(float)"]

        model = pow_pattern.get_onnx_model(
            inputs,
            attributes,
            is_constant_map,
            output_dtypes,
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        # Run original model
        original_sess = ort.InferenceSession(model.SerializeToString())
        original_output = original_sess.run(None, {"X": input_data})[0]

        matcher = PatternMatcher(model)
        matcher.register_pattern(pow_pattern)
        pow_results = matcher.match()

        assert len(pow_results) == 1
        assert pow_results[0].attributes["axis"] == 1

        rewriter = PatternRewriter(model)
        new_model = rewriter.rewrite(
            [(pow_results, TransposedSingleRMSNormalizationPattern)]
        )

        onnx.checker.check_model(new_model)

        rmsnorm_nodes = [
            n for n in new_model.graph.node if n.op_type == "RMSNormalization"
        ]
        assert len(rmsnorm_nodes) == 1

        new_sess = ort.InferenceSession(new_model.SerializeToString())
        new_output = new_sess.run(None, {"X": input_data})[0]

        np.testing.assert_allclose(
            original_output,
            new_output,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Rewritten RMSNorm (axis=1) should produce equivalent output",
        )

    def test_rewrite_mul_pattern_axis_0_to_transposed(self):
        """Test rewriting Mul pattern with axis=0 and multi-dim Scale."""
        mul_pattern = RMSNormalizationMulPattern()

        input_data = np.random.randn(8, 64, 128).astype(np.float32)
        scale_data = np.ones((8, 1, 1), dtype=np.float32) * 2.0

        inputs = {"X": input_data, "Scale": scale_data}
        attributes = {"axis": 0, "epsilon": 1e-5}
        is_constant_map = {"X": False, "Scale": True}
        output_dtypes = ["tensor(float)"]

        model = mul_pattern.get_onnx_model(
            inputs,
            attributes,
            is_constant_map,
            output_dtypes,
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        original_sess = ort.InferenceSession(model.SerializeToString())
        original_output = original_sess.run(None, {"X": input_data})[0]

        matcher = PatternMatcher(model)
        matcher.register_pattern(mul_pattern)
        mul_results = matcher.match()

        assert len(mul_results) == 1
        assert mul_results[0].attributes["axis"] == 0

        rewriter = PatternRewriter(model)
        new_model = rewriter.rewrite(
            [(mul_results, TransposedSingleRMSNormalizationPattern)]
        )

        onnx.checker.check_model(new_model)

        rmsnorm_nodes = [
            n for n in new_model.graph.node if n.op_type == "RMSNormalization"
        ]
        assert len(rmsnorm_nodes) == 1

        new_sess = ort.InferenceSession(new_model.SerializeToString())
        new_output = new_sess.run(None, {"X": input_data})[0]

        np.testing.assert_allclose(
            original_output,
            new_output,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Rewritten RMSNorm (axis=0) should produce equivalent output",
        )
