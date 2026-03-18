# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for transpose-wrapped LayerNorm pattern.

Tests cover:
- Transpose permutation computation for arbitrary axis values
- Pattern self-matching (generated model matches its own pattern)
- Numerical equivalence with manual LayerNorm computation
- Multi-dimensional Scale/B support (reshaped to 1D)
- Pattern rewriting from Pow/Mul patterns to Transposed pattern
"""

from typing import ClassVar

import numpy as np
import onnx
import pytest

from winml.modelkit.onnx.domains import ONNXDomain
from winml.modelkit.pattern import (
    LayerNormalizationMulPattern,
    LayerNormalizationPowPattern,
    PatternMatcher,
    PatternRewriter,
    TransposedSingleLayerNormalizationPattern,
)


# Access _compute_transpose_permutation via pattern instance (instance method)
_pattern_instance = TransposedSingleLayerNormalizationPattern()
_compute_transpose_permutation = _pattern_instance._compute_transpose_permutation


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
        # Verify inverse: perm_i[perm_f[i]] == i
        for i in range(4):
            assert perm_i[perm_f[i]] == i

    def test_axis_0_rank_3(self):
        """Test permutation for axis=0 in rank=3 tensor."""
        perm_f, perm_i = _compute_transpose_permutation(axis=0, rank=3)
        assert perm_f == [1, 2, 0]
        # Verify inverse
        for i in range(3):
            assert perm_i[perm_f[i]] == i

    def test_negative_axis(self):
        """Test that negative axis values are normalized correctly."""
        perm_f, perm_i = _compute_transpose_permutation(axis=-2, rank=4)
        # -2 in rank 4 = axis 2, should move to end
        assert perm_f == [0, 1, 3, 2]
        # Verify inverse
        for i in range(4):
            assert perm_i[perm_f[i]] == i

    def test_roundtrip_transpose(self):
        """Verify that forward then inverse transpose restores original."""
        for axis in [0, 1, 2]:
            for rank in [3, 4, 5]:
                if axis < rank:
                    perm_f, perm_i = _compute_transpose_permutation(axis, rank)
                    # Create test array and verify roundtrip
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
            (0, 3),  # axis=0, rank=3
            (1, 4),  # axis=1, rank=4
            (2, 5),  # axis=2, rank=5
            (-2, 4),  # axis=-2, rank=4
        ]

        for axis, rank in test_cases:
            perm_f, perm_i = _compute_transpose_permutation(axis, rank)

            # Verify: applying perm_i after perm_f gives identity
            identity = list(range(rank))
            result = [perm_i[perm_f[i]] for i in range(rank)]
            assert result == identity, f"Failed for axis={axis}, rank={rank}"


class TestTransposedLayerNormPattern:
    """Tests for TransposedSingleLayerNormalizationPattern."""

    _DOMAIN_VERSIONS: ClassVar[dict] = {ONNXDomain.AI_ONNX: 17}

    def test_transposed_pattern_self_matching(self):
        """Test that TransposedSingleLayerNormalizationPattern matches its own output."""
        pattern = TransposedSingleLayerNormalizationPattern()

        inputs = {
            "X": np.random.randn(2, 64, 128).astype(np.float32),
            # axis=1, dim=64: multi-dim shape [1, 64, 1] for broadcast compatibility
            "Scale": np.ones((1, 64, 1), dtype=np.float32),
            "B": np.zeros((1, 64, 1), dtype=np.float32),
        }
        attributes = {"axis": 1, "epsilon": 1e-5}
        is_constant_map = {"X": False, "Scale": True, "B": True}
        output_dtypes = ["tensor(float)"]

        model = pattern.get_onnx_model(
            inputs,
            attributes,
            is_constant_map,
            output_dtypes,
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        # Verify structure: Transpose, Reshape(Scale), Reshape(B), LayerNorm, Transpose
        assert len(model.graph.node) == 5
        assert model.graph.node[0].op_type == "Transpose"
        assert model.graph.node[1].op_type == "Reshape"
        assert model.graph.node[2].op_type == "Reshape"
        assert model.graph.node[3].op_type == "LayerNormalization"
        assert model.graph.node[4].op_type == "Transpose"

        # Verify LayerNorm has axis=-1
        ln_node = model.graph.node[3]
        axis_attr = next((attr for attr in ln_node.attribute if attr.name == "axis"), None)
        assert axis_attr is not None
        assert axis_attr.i == -1

        # Self-match
        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        results = matcher.match()

        assert len(results) == 1
        assert results[0].attributes["axis"] == 1

    def test_transposed_pattern_numerical_equivalence(self):
        """Test numerical equivalence of TransposedSingleLayerNormalizationPattern."""
        import onnxruntime as ort

        pattern = TransposedSingleLayerNormalizationPattern()
        x_data = np.random.randn(2, 64, 128).astype(np.float32)
        scale_data = np.ones(64, dtype=np.float32)
        bias_data = np.zeros(64, dtype=np.float32)

        inputs = {"X": x_data, "Scale": scale_data, "B": bias_data}
        attributes = {"axis": 1, "epsilon": 1e-5}

        model = pattern.get_onnx_model(
            inputs,
            attributes,
            {"X": False, "Scale": True, "B": True},
            ["tensor(float)"],
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        # Verify structure has Reshape nodes
        assert len(model.graph.node) == 5
        assert model.graph.node[1].op_type == "Reshape"
        assert model.graph.node[2].op_type == "Reshape"

        # Run inference
        sess = ort.InferenceSession(model.SerializeToString())
        result = sess.run(None, {"X": x_data})[0]

        # Compute expected result manually (LayerNorm on axis=1)
        mean = np.mean(x_data, axis=1, keepdims=True)
        var = np.var(x_data, axis=1, keepdims=True)
        expected = (x_data - mean) / np.sqrt(var + 1e-5)
        expected = (
            expected * scale_data[np.newaxis, :, np.newaxis] + bias_data[np.newaxis, :, np.newaxis]
        )

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_transposed_pattern_with_multidim_scale_bias(self):
        """Test TransposedSingleLayerNormalizationPattern with multi-dimensional Scale/B."""
        import onnxruntime as ort

        pattern = TransposedSingleLayerNormalizationPattern()
        x_data = np.random.randn(2, 64, 128).astype(np.float32)
        # Scale/B are multi-dimensional but total elements = 64 (normalized_dim)
        scale_data = np.ones((1, 64, 1), dtype=np.float32) * 2.0
        bias_data = np.ones((64, 1), dtype=np.float32) * 0.5

        inputs = {"X": x_data, "Scale": scale_data, "B": bias_data}
        attributes = {"axis": 1, "epsilon": 1e-5}

        model = pattern.get_onnx_model(
            inputs,
            attributes,
            {"X": False, "Scale": True, "B": True},
            ["tensor(float)"],
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        # Run inference
        sess = ort.InferenceSession(model.SerializeToString())
        result = sess.run(None, {"X": x_data})[0]

        # Compute expected result manually (LayerNorm on axis=1)
        mean = np.mean(x_data, axis=1, keepdims=True)
        var = np.var(x_data, axis=1, keepdims=True)
        expected = (x_data - mean) / np.sqrt(var + 1e-5)
        # Scale and B are reshaped to 1D then broadcast
        expected = (
            expected * 2.0  # scale_data reshaped to (64,) then broadcast
            + 0.5  # bias_data reshaped to (64,) then broadcast
        )

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_transposed_pattern_axis_last_identity_transpose(self):
        """Test TransposedSingleLayerNormalizationPattern with axis=-1 (identity transpose)."""
        import onnxruntime as ort

        pattern = TransposedSingleLayerNormalizationPattern()
        x_data = np.random.randn(2, 64, 128).astype(np.float32)
        scale_data = np.ones(128, dtype=np.float32) * 1.5
        bias_data = np.ones(128, dtype=np.float32) * 0.3

        inputs = {"X": x_data, "Scale": scale_data, "B": bias_data}
        attributes = {"axis": -1, "epsilon": 1e-5}

        model = pattern.get_onnx_model(
            inputs,
            attributes,
            {"X": False, "Scale": True, "B": True},
            ["tensor(float)"],
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        # Verify Transpose nodes have identity permutation for axis=-1
        transpose1 = model.graph.node[0]
        transpose2 = model.graph.node[4]
        perm1 = next((attr.ints for attr in transpose1.attribute if attr.name == "perm"), None)
        perm2 = next((attr.ints for attr in transpose2.attribute if attr.name == "perm"), None)
        assert list(perm1) == [0, 1, 2], "Forward transpose should be identity for axis=-1"
        assert list(perm2) == [0, 1, 2], "Inverse transpose should be identity for axis=-1"

        # Self-match should recover axis=-1
        matcher = PatternMatcher(model)
        matcher.register_pattern(pattern)
        results = matcher.match()
        assert len(results) == 1
        # Note: axis=-1 is stored as the original axis value (last dimension index)
        assert results[0].attributes["axis"] == 2  # perm[-1] = 2 for identity

        # Run inference and verify numerical correctness
        sess = ort.InferenceSession(model.SerializeToString())
        result = sess.run(None, {"X": x_data})[0]

        # Compute expected result (LayerNorm on axis=-1)
        mean = np.mean(x_data, axis=-1, keepdims=True)
        var = np.var(x_data, axis=-1, keepdims=True)
        expected = (x_data - mean) / np.sqrt(var + 1e-5)
        expected = expected * scale_data + bias_data

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_transposed_pattern_axis_0(self):
        """Test TransposedSingleLayerNormalizationPattern with axis=0 and multi-dim Scale/B."""
        import onnxruntime as ort

        pattern = TransposedSingleLayerNormalizationPattern()
        x_data = np.random.randn(8, 64, 128).astype(np.float32)
        scale_data = np.ones((8, 1, 1), dtype=np.float32) * 2.0
        bias_data = np.ones((8, 1, 1), dtype=np.float32) * 0.5

        inputs = {"X": x_data, "Scale": scale_data, "B": bias_data}
        attributes = {"axis": 0, "epsilon": 1e-5}

        model = pattern.get_onnx_model(
            inputs,
            attributes,
            {"X": False, "Scale": True, "B": True},
            ["tensor(float)"],
            self._DOMAIN_VERSIONS,
        )

        onnx.checker.check_model(model)

        # Verify Transpose permutation moves axis 0 to the end
        transpose1 = model.graph.node[0]
        perm1 = list(next(attr.ints for attr in transpose1.attribute if attr.name == "perm"))
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

        mean = np.mean(x_data, axis=0, keepdims=True)
        var = np.var(x_data, axis=0, keepdims=True)
        expected = (x_data - mean) / np.sqrt(var + 1e-5)
        expected = expected * 2.0 + 0.5

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


class TestLayerNormalizationPatternRewriting:
    """Tests for rewriting LayerNormalization patterns to TransposedSingleLayerNormPattern."""

    _DOMAIN_VERSIONS: ClassVar[dict] = {ONNXDomain.AI_ONNX: 17}

    def test_rewrite_layernorm_pow_to_transposed(self):
        """Test rewriting Pow pattern to Transposed pattern (axis=-1)."""
        # Create model with LayerNormalizationPowPattern
        pow_pattern = LayerNormalizationPowPattern()
        inputs = {
            "X": np.random.randn(2, 128, 768).astype(np.float32),
            "Scale": np.ones(768).astype(np.float32),
            "B": np.zeros(768).astype(np.float32),
        }
        attributes = {"axis": -1, "epsilon": 1e-5}
        is_constant_map = {"X": False, "Scale": True, "B": True}
        output_dtypes = ["tensor(float)"]

        model = pow_pattern.get_onnx_model(
            inputs,
            attributes,
            is_constant_map,
            output_dtypes,
            self._DOMAIN_VERSIONS,
        )

        # Verify the original model is valid
        onnx.checker.check_model(model)

        # Count original nodes (LayerNormPow has 9 nodes)
        original_node_count = len(model.graph.node)
        assert original_node_count == 9, (
            f"Expected 9 nodes in Pow pattern, got {original_node_count}"
        )

        # Find LayerNormalizationPowPattern
        matcher = PatternMatcher(model)
        matcher.register_pattern(pow_pattern)
        pow_results = matcher.match()

        assert len(pow_results) == 1, f"Expected 1 Pow pattern match, found {len(pow_results)}"

        # Rewrite to TransposedSingleLayerNormalizationPattern
        rewriter = PatternRewriter(model)
        new_model = rewriter.rewrite([(pow_results, TransposedSingleLayerNormalizationPattern)])

        # Verify the new model is valid
        onnx.checker.check_model(new_model)

        # Check that node count decreased (9-node pattern -> 5-node pattern)
        new_node_count = len(new_model.graph.node)
        assert new_node_count < original_node_count, (
            f"Expected fewer nodes after rewriting "
            f"(was {original_node_count}, now {new_node_count})"
        )

        # Verify LayerNormalizationPowPattern is no longer present
        new_matcher = PatternMatcher(new_model)
        new_matcher.register_pattern(pow_pattern)
        remaining_pow = new_matcher.match()
        assert len(remaining_pow) == 0, (
            f"Expected 0 Pow patterns after rewriting, found {len(remaining_pow)}"
        )

        # Verify the rewritten model contains a LayerNormalization node
        ln_nodes = [node for node in new_model.graph.node if node.op_type == "LayerNormalization"]
        assert len(ln_nodes) == 1, (
            f"Expected 1 LayerNormalization node after rewriting, found {len(ln_nodes)}"
        )

    def test_rewrite_layernorm_mul_to_transposed(self):
        """Test rewriting Mul pattern to Transposed pattern (axis=-1)."""
        # Create model with LayerNormalizationMulPattern
        mul_pattern = LayerNormalizationMulPattern()
        inputs = {
            "X": np.random.randn(2, 128, 768).astype(np.float32),
            "Scale": np.ones(768).astype(np.float32),
            "B": np.zeros(768).astype(np.float32),
        }
        attributes = {"axis": -1, "epsilon": 1e-5}
        is_constant_map = {"X": False, "Scale": True, "B": True}
        output_dtypes = ["tensor(float)"]

        model = mul_pattern.get_onnx_model(
            inputs,
            attributes,
            is_constant_map,
            output_dtypes,
            self._DOMAIN_VERSIONS,
        )

        # Verify the original model is valid
        onnx.checker.check_model(model)

        # Count original nodes (LayerNormMul has 9 nodes)
        original_node_count = len(model.graph.node)
        assert original_node_count == 9, (
            f"Expected 9 nodes in Mul pattern, got {original_node_count}"
        )

        # Find LayerNormalizationMulPattern
        matcher = PatternMatcher(model)
        matcher.register_pattern(mul_pattern)
        mul_results = matcher.match()

        assert len(mul_results) == 1, f"Expected 1 Mul pattern match, found {len(mul_results)}"

        # Rewrite to TransposedSingleLayerNormalizationPattern
        rewriter = PatternRewriter(model)
        new_model = rewriter.rewrite([(mul_results, TransposedSingleLayerNormalizationPattern)])

        # Verify the new model is valid
        onnx.checker.check_model(new_model)

        # Check that node count decreased (9-node pattern -> 5-node pattern)
        new_node_count = len(new_model.graph.node)
        assert new_node_count < original_node_count, (
            f"Expected fewer nodes after rewriting "
            f"(was {original_node_count}, now {new_node_count})"
        )

        # Verify LayerNormalizationMulPattern is no longer present
        new_matcher = PatternMatcher(new_model)
        new_matcher.register_pattern(mul_pattern)
        remaining_mul = new_matcher.match()
        assert len(remaining_mul) == 0, (
            f"Expected 0 Mul patterns after rewriting, found {len(remaining_mul)}"
        )

        # Verify the rewritten model contains a LayerNormalization node
        ln_nodes = [node for node in new_model.graph.node if node.op_type == "LayerNormalization"]
        assert len(ln_nodes) == 1, (
            f"Expected 1 LayerNormalization node after rewriting, found {len(ln_nodes)}"
        )

    def test_rewrite_preserves_layernorm_semantics(self):
        """Test numerical equivalence after rewriting Pow pattern."""
        import onnxruntime as ort

        # Create model with LayerNormalizationPowPattern
        pow_pattern = LayerNormalizationPowPattern()
        input_data = np.random.randn(2, 128, 768).astype(np.float32)
        scale_data = np.random.randn(768).astype(np.float32)
        bias_data = np.random.randn(768).astype(np.float32)

        inputs = {
            "X": input_data,
            "Scale": scale_data,
            "B": bias_data,
        }
        attributes = {"axis": -1, "epsilon": 1e-5}
        is_constant_map = {"X": False, "Scale": True, "B": True}
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
        new_model = rewriter.rewrite([(pow_results, TransposedSingleLayerNormalizationPattern)])

        # Run rewritten model
        new_sess = ort.InferenceSession(new_model.SerializeToString())
        new_output = new_sess.run(None, {"X": input_data})[0]

        # Verify outputs are equivalent
        np.testing.assert_allclose(
            original_output,
            new_output,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Rewritten LayerNorm should produce equivalent output",
        )

    def test_rewrite_pow_pattern_axis_1_to_transposed(self):
        """Test rewriting Pow pattern with axis=1 and multi-dim Scale/B."""
        import onnxruntime as ort

        pow_pattern = LayerNormalizationPowPattern()

        # X shape [batch=2, normalized_dim=64, seq=128]
        # axis=1 means normalizing over dim=64
        # Scale/B must have shape [1, 64, 1] for broadcast compatibility
        input_data = np.random.randn(2, 64, 128).astype(np.float32)
        scale_data = np.ones((1, 64, 1), dtype=np.float32) * 1.5
        bias_data = np.ones((1, 64, 1), dtype=np.float32) * 0.3

        inputs = {"X": input_data, "Scale": scale_data, "B": bias_data}
        attributes = {"axis": 1, "epsilon": 1e-5}
        is_constant_map = {"X": False, "Scale": True, "B": True}
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

        # Verify Pow pattern matches axis=1 with multi-dim Scale/B
        matcher = PatternMatcher(model)
        matcher.register_pattern(pow_pattern)
        pow_results = matcher.match()

        assert len(pow_results) == 1, f"Expected 1 Pow pattern match, found {len(pow_results)}"
        assert pow_results[0].attributes["axis"] == 1, "Expected axis=1"

        # Rewrite to TransposedSingleLayerNormalizationPattern
        rewriter = PatternRewriter(model)
        new_model = rewriter.rewrite([(pow_results, TransposedSingleLayerNormalizationPattern)])

        onnx.checker.check_model(new_model)

        # Verify new model has LayerNormalization node
        ln_nodes = [n for n in new_model.graph.node if n.op_type == "LayerNormalization"]
        assert len(ln_nodes) == 1, f"Expected 1 LayerNormalization node, found {len(ln_nodes)}"

        # Run rewritten model
        new_sess = ort.InferenceSession(new_model.SerializeToString())
        new_output = new_sess.run(None, {"X": input_data})[0]

        # Verify numerical equivalence
        np.testing.assert_allclose(
            original_output,
            new_output,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Rewritten LayerNorm (axis=1) should produce equivalent output",
        )

    def test_rewrite_mul_pattern_axis_0_to_transposed(self):
        """Test rewriting Mul pattern with axis=0 and multi-dim Scale/B."""
        import onnxruntime as ort

        mul_pattern = LayerNormalizationMulPattern()

        # X shape [normalized_dim=8, batch=64, seq=128]
        # axis=0 means normalizing over dim=8
        # Scale/B must have shape [8, 1, 1] for broadcast compatibility
        input_data = np.random.randn(8, 64, 128).astype(np.float32)
        scale_data = np.ones((8, 1, 1), dtype=np.float32) * 2.0
        bias_data = np.zeros((8, 1, 1), dtype=np.float32)

        inputs = {"X": input_data, "Scale": scale_data, "B": bias_data}
        attributes = {"axis": 0, "epsilon": 1e-5}
        is_constant_map = {"X": False, "Scale": True, "B": True}
        output_dtypes = ["tensor(float)"]

        model = mul_pattern.get_onnx_model(
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

        # Verify Mul pattern matches axis=0 with multi-dim Scale/B
        matcher = PatternMatcher(model)
        matcher.register_pattern(mul_pattern)
        mul_results = matcher.match()

        assert len(mul_results) == 1, f"Expected 1 Mul pattern match, found {len(mul_results)}"
        assert mul_results[0].attributes["axis"] == 0, "Expected axis=0"

        # Rewrite to TransposedSingleLayerNormalizationPattern
        rewriter = PatternRewriter(model)
        new_model = rewriter.rewrite([(mul_results, TransposedSingleLayerNormalizationPattern)])

        onnx.checker.check_model(new_model)

        # Verify new model has LayerNormalization node
        ln_nodes = [n for n in new_model.graph.node if n.op_type == "LayerNormalization"]
        assert len(ln_nodes) == 1, f"Expected 1 LayerNormalization node, found {len(ln_nodes)}"

        # Run rewritten model
        new_sess = ort.InferenceSession(new_model.SerializeToString())
        new_output = new_sess.run(None, {"X": input_data})[0]

        # Verify numerical equivalence
        np.testing.assert_allclose(
            original_output,
            new_output,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Rewritten LayerNorm (axis=0) should produce equivalent output",
        )
