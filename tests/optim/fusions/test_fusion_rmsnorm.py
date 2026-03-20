# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for FusionRMSNorm: decomposed RMSNorm -> LpNormalization fusion.

Validates that the fusion correctly detects the backward pattern:
    Mul(weight) <- Div(root, sqrt) <- Sqrt <- Add(eps) <- ReduceMean <- Pow(2) <- root

And replaces it with:
    root -> LpNormalization(p=2, axis=-1) -> Mul(weight * sqrt(N))
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.transformers.onnx_model import OnnxModel

from winml.modelkit.optim.fusions import FusionRMSNorm


# =============================================================================
# Helpers
# =============================================================================


def _make_rmsnorm_model(
    hidden_size: int = 64,
    weight_value: np.ndarray | None = None,
    seq_len: int = 8,
    batch_size: int = 1,
) -> onnx.ModelProto:
    """Create ONNX model with a single RMSNorm pattern.

    Graph: input -> Pow(2) -> ReduceMean -> Add(eps) -> Sqrt
                                                          -> Div(input, sqrt) -> Mul(weight)
    """
    if weight_value is None:
        rng = np.random.RandomState(42)
        weight_value = rng.randn(hidden_size).astype(np.float32)

    x_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )
    y_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )

    pow_exp = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "pow_exp")
    epsilon = numpy_helper.from_array(np.array(1e-6, dtype=np.float32), "epsilon")
    weight = numpy_helper.from_array(weight_value, "weight")

    pow_node = helper.make_node("Pow", ["input", "pow_exp"], ["pow_out"])
    reduce_mean = helper.make_node(
        "ReduceMean", ["pow_out"], ["mean_out"], axes=[-1], keepdims=1
    )
    add_eps = helper.make_node("Add", ["mean_out", "epsilon"], ["add_out"])
    sqrt_node = helper.make_node("Sqrt", ["add_out"], ["sqrt_out"])
    div_node = helper.make_node("Div", ["input", "sqrt_out"], ["div_out"])
    mul_weight = helper.make_node("Mul", ["div_out", "weight"], ["output"])

    graph = helper.make_graph(
        [pow_node, reduce_mean, add_eps, sqrt_node, div_node, mul_weight],
        "rmsnorm_test",
        [x_info],
        [y_info],
        initializer=[pow_exp, epsilon, weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_multi_rmsnorm_model(
    num_layers: int = 3,
    hidden_size: int = 64,
) -> onnx.ModelProto:
    """Create ONNX model with multiple sequential RMSNorm patterns.

    Each layer: prev_output -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> next
    """
    rng = np.random.RandomState(42)

    x_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 8, hidden_size]
    )

    nodes = []
    initializers = []
    prev_output = "input"

    for i in range(num_layers):
        prefix = f"layer{i}_"
        weight_val = rng.randn(hidden_size).astype(np.float32)
        out_name = f"{prefix}output"

        pow_exp = numpy_helper.from_array(
            np.array(2.0, dtype=np.float32), f"{prefix}pow_exp"
        )
        epsilon = numpy_helper.from_array(
            np.array(1e-6, dtype=np.float32), f"{prefix}epsilon"
        )
        weight = numpy_helper.from_array(weight_val, f"{prefix}weight")
        initializers.extend([pow_exp, epsilon, weight])

        pow_node = helper.make_node(
            "Pow", [prev_output, f"{prefix}pow_exp"], [f"{prefix}pow_out"]
        )
        reduce_mean = helper.make_node(
            "ReduceMean",
            [f"{prefix}pow_out"],
            [f"{prefix}mean_out"],
            axes=[-1],
            keepdims=1,
        )
        add_eps = helper.make_node(
            "Add", [f"{prefix}mean_out", f"{prefix}epsilon"], [f"{prefix}add_out"]
        )
        sqrt_node = helper.make_node(
            "Sqrt", [f"{prefix}add_out"], [f"{prefix}sqrt_out"]
        )
        div_node = helper.make_node(
            "Div", [prev_output, f"{prefix}sqrt_out"], [f"{prefix}div_out"]
        )
        mul_weight = helper.make_node(
            "Mul", [f"{prefix}div_out", f"{prefix}weight"], [out_name]
        )
        nodes.extend([pow_node, reduce_mean, add_eps, sqrt_node, div_node, mul_weight])
        prev_output = out_name

    y_info = helper.make_tensor_value_info(
        prev_output, TensorProto.FLOAT, [1, 8, hidden_size]
    )

    graph = helper.make_graph(
        nodes, "multi_rmsnorm_test", [x_info], [y_info], initializer=initializers
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _apply_fusion(model: onnx.ModelProto) -> onnx.ModelProto:
    """Apply FusionRMSNorm and return the resulting model."""
    onnx_model = OnnxModel(model)
    FusionRMSNorm(onnx_model).apply()
    return onnx_model.model


def _get_op_types(model: onnx.ModelProto) -> set[str]:
    """Get set of all op types in model graph."""
    return {node.op_type for node in model.graph.node}


def _count_op(model: onnx.ModelProto, op_type: str) -> int:
    """Count nodes of a specific op type."""
    return sum(1 for n in model.graph.node if n.op_type == op_type)


def _run_inference(model: onnx.ModelProto, inputs: dict[str, np.ndarray]):
    """Run ONNX inference and return outputs as dict."""
    model_bytes = model.SerializeToString()
    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    output_names = [o.name for o in sess.get_outputs()]
    results = sess.run(output_names, inputs)
    return dict(zip(output_names, results, strict=False))


# =============================================================================
# Tests
# =============================================================================


class TestFusionRMSNormBasic:
    """Basic fusion matching and replacement tests."""

    def test_basic_rmsnorm_fusion(self):
        """Verify LpNormalization is created and old RMSNorm nodes removed."""
        model = _make_rmsnorm_model(hidden_size=64)
        original_ops = _get_op_types(model)
        assert "Pow" in original_ops
        assert "ReduceMean" in original_ops
        assert "Sqrt" in original_ops
        assert "Div" in original_ops

        result = _apply_fusion(model)
        result_ops = _get_op_types(result)

        # LpNormalization must exist
        assert "LpNormalization" in result_ops
        # New Mul for adjusted weight must exist
        assert "Mul" in result_ops

        # Old decomposed nodes must be gone
        assert "Pow" not in result_ops
        assert "ReduceMean" not in result_ops
        assert "Sqrt" not in result_ops
        assert "Div" not in result_ops

    def test_lpnorm_attributes(self):
        """Verify LpNormalization node has correct p=2, axis=-1 attributes."""
        model = _make_rmsnorm_model(hidden_size=32)
        result = _apply_fusion(model)

        lp_nodes = [n for n in result.graph.node if n.op_type == "LpNormalization"]
        assert len(lp_nodes) == 1

        lp_node = lp_nodes[0]
        attrs = {a.name: a for a in lp_node.attribute}
        assert attrs["p"].i == 2
        assert attrs["axis"].i == -1


class TestWeightAdjustment:
    """Tests for weight scaling by sqrt(hidden_size)."""

    def test_weight_adjustment(self):
        """Verify weight is multiplied by sqrt(hidden_size)."""
        hidden_size = 16
        original_weight = np.arange(1, hidden_size + 1, dtype=np.float32)
        model = _make_rmsnorm_model(
            hidden_size=hidden_size, weight_value=original_weight
        )
        result = _apply_fusion(model)

        # Find the adjusted weight initializer
        adjusted_inits = [
            init
            for init in result.graph.initializer
            if "l2norm_adjusted" in init.name
        ]
        assert len(adjusted_inits) == 1

        adjusted_weight = numpy_helper.to_array(adjusted_inits[0])
        expected = original_weight * np.sqrt(hidden_size).astype(np.float32)
        np.testing.assert_allclose(adjusted_weight, expected, rtol=1e-6)

    def test_all_ones_weight_collapses(self):
        """When weight is all 1.0, verify collapse to scalar [sqrt(N)]."""
        hidden_size = 32
        ones_weight = np.ones(hidden_size, dtype=np.float32)
        model = _make_rmsnorm_model(
            hidden_size=hidden_size, weight_value=ones_weight
        )
        result = _apply_fusion(model)

        adjusted_inits = [
            init
            for init in result.graph.initializer
            if "l2norm_adjusted" in init.name
        ]
        assert len(adjusted_inits) == 1

        adjusted_weight = numpy_helper.to_array(adjusted_inits[0])
        # Should be scalar array [sqrt(N)]
        assert adjusted_weight.shape == (1,)
        expected_scalar = np.sqrt(hidden_size).astype(np.float32)
        np.testing.assert_allclose(adjusted_weight, [expected_scalar], rtol=1e-6)


class TestNoMatch:
    """Tests for patterns that should NOT trigger fusion."""

    def test_no_match_without_weight_initializer(self):
        """Mul without initializer input should NOT trigger fusion."""
        hidden_size = 64
        x_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 8, hidden_size]
        )
        # A second runtime input instead of an initializer
        w_info = helper.make_tensor_value_info(
            "weight_runtime", TensorProto.FLOAT, [hidden_size]
        )
        y_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 8, hidden_size]
        )

        pow_exp = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "pow_exp")
        epsilon = numpy_helper.from_array(np.array(1e-6, dtype=np.float32), "epsilon")

        pow_node = helper.make_node("Pow", ["input", "pow_exp"], ["pow_out"])
        reduce_mean = helper.make_node(
            "ReduceMean", ["pow_out"], ["mean_out"], axes=[-1], keepdims=1
        )
        add_eps = helper.make_node("Add", ["mean_out", "epsilon"], ["add_out"])
        sqrt_node = helper.make_node("Sqrt", ["add_out"], ["sqrt_out"])
        div_node = helper.make_node("Div", ["input", "sqrt_out"], ["div_out"])
        # Mul uses runtime input, not initializer
        mul_weight = helper.make_node(
            "Mul", ["div_out", "weight_runtime"], ["output"]
        )

        graph = helper.make_graph(
            [pow_node, reduce_mean, add_eps, sqrt_node, div_node, mul_weight],
            "rmsnorm_no_init",
            [x_info, w_info],
            [y_info],
            initializer=[pow_exp, epsilon],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        original_node_count = len(model.graph.node)
        result = _apply_fusion(model)

        # Should remain unchanged
        assert len(result.graph.node) == original_node_count
        assert "LpNormalization" not in _get_op_types(result)

    def test_no_match_wrong_pow_exponent(self):
        """Pow with exponent != 2.0 should NOT match."""
        hidden_size = 64
        rng = np.random.RandomState(42)
        weight_value = rng.randn(hidden_size).astype(np.float32)

        x_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 8, hidden_size]
        )
        y_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 8, hidden_size]
        )

        # Exponent is 3.0, not 2.0
        pow_exp = numpy_helper.from_array(np.array(3.0, dtype=np.float32), "pow_exp")
        epsilon = numpy_helper.from_array(np.array(1e-6, dtype=np.float32), "epsilon")
        weight = numpy_helper.from_array(weight_value, "weight")

        pow_node = helper.make_node("Pow", ["input", "pow_exp"], ["pow_out"])
        reduce_mean = helper.make_node(
            "ReduceMean", ["pow_out"], ["mean_out"], axes=[-1], keepdims=1
        )
        add_eps = helper.make_node("Add", ["mean_out", "epsilon"], ["add_out"])
        sqrt_node = helper.make_node("Sqrt", ["add_out"], ["sqrt_out"])
        div_node = helper.make_node("Div", ["input", "sqrt_out"], ["div_out"])
        mul_weight = helper.make_node("Mul", ["div_out", "weight"], ["output"])

        graph = helper.make_graph(
            [pow_node, reduce_mean, add_eps, sqrt_node, div_node, mul_weight],
            "rmsnorm_wrong_exp",
            [x_info],
            [y_info],
            initializer=[pow_exp, epsilon, weight],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        original_node_count = len(model.graph.node)
        result = _apply_fusion(model)

        assert len(result.graph.node) == original_node_count
        assert "LpNormalization" not in _get_op_types(result)

    def test_no_match_incomplete_chain(self):
        """Partial pattern (Pow -> ReduceMean -> Add, no Sqrt/Div) should NOT match."""
        hidden_size = 64
        rng = np.random.RandomState(42)
        weight_value = rng.randn(hidden_size).astype(np.float32)

        x_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 8, hidden_size]
        )
        y_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 8, hidden_size]
        )

        pow_exp = numpy_helper.from_array(np.array(2.0, dtype=np.float32), "pow_exp")
        epsilon = numpy_helper.from_array(np.array(1e-6, dtype=np.float32), "epsilon")
        weight = numpy_helper.from_array(weight_value, "weight")

        pow_node = helper.make_node("Pow", ["input", "pow_exp"], ["pow_out"])
        reduce_mean = helper.make_node(
            "ReduceMean", ["pow_out"], ["mean_out"], axes=[-1], keepdims=1
        )
        add_eps = helper.make_node("Add", ["mean_out", "epsilon"], ["add_out"])
        # No Sqrt, no Div -- directly Mul the add output with weight
        mul_weight = helper.make_node("Mul", ["add_out", "weight"], ["output"])

        graph = helper.make_graph(
            [pow_node, reduce_mean, add_eps, mul_weight],
            "rmsnorm_incomplete",
            [x_info],
            [y_info],
            initializer=[pow_exp, epsilon, weight],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        original_node_count = len(model.graph.node)
        result = _apply_fusion(model)

        assert len(result.graph.node) == original_node_count
        assert "LpNormalization" not in _get_op_types(result)


class TestMultipleInstances:
    """Tests for models with multiple RMSNorm patterns."""

    def test_multiple_rmsnorm_instances(self):
        """Model with multiple RMSNorm patterns should all be fused."""
        num_layers = 3
        model = _make_multi_rmsnorm_model(num_layers=num_layers, hidden_size=64)

        # Before: 6 nodes per layer
        assert len(model.graph.node) == 6 * num_layers

        result = _apply_fusion(model)
        result_ops = _get_op_types(result)

        # All should be fused
        assert _count_op(result, "LpNormalization") == num_layers
        assert "Pow" not in result_ops
        assert "ReduceMean" not in result_ops
        assert "Sqrt" not in result_ops
        assert "Div" not in result_ops


class TestNodeCountReduction:
    """Tests for expected node count changes after fusion."""

    def test_node_count_reduction_single(self):
        """Each RMSNorm removes 6 nodes, adds 2 = net -4."""
        model = _make_rmsnorm_model(hidden_size=64)
        original_count = len(model.graph.node)
        assert original_count == 6

        result = _apply_fusion(model)
        result_count = len(result.graph.node)

        # 6 removed (Pow, ReduceMean, Add, Sqrt, Div, old Mul)
        # 2 added (LpNormalization, new Mul)
        assert result_count == original_count - 4
        assert result_count == 2

    def test_node_count_reduction_multiple(self):
        """Multiple RMSNorm: net -4 per instance."""
        num_layers = 4
        model = _make_multi_rmsnorm_model(num_layers=num_layers, hidden_size=64)
        original_count = len(model.graph.node)

        result = _apply_fusion(model)
        result_count = len(result.graph.node)

        expected = original_count - (4 * num_layers)
        assert result_count == expected


class TestNumericEquivalence:
    """Tests for numerical correctness of the fusion transformation."""

    def test_numeric_equivalence(self):
        """Verify outputs match within tolerance before and after fusion.

        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
        LpNorm(x, p=2) = x / L2norm(x)
        L2norm(x) = sqrt(sum(x^2))
        RMS(x) = sqrt(mean(x^2)) = sqrt(sum(x^2)/N) = L2norm(x) / sqrt(N)

        So: x / RMS(x) = x * sqrt(N) / L2norm(x) = LpNorm(x) * sqrt(N)
        And: RMSNorm(x) = LpNorm(x) * sqrt(N) * weight = LpNorm(x) * (weight * sqrt(N))
        """
        hidden_size = 32
        rng = np.random.RandomState(123)
        weight_value = rng.randn(hidden_size).astype(np.float32)

        model = _make_rmsnorm_model(
            hidden_size=hidden_size,
            weight_value=weight_value,
            seq_len=4,
            batch_size=2,
        )

        # Generate input
        test_input = rng.randn(2, 4, hidden_size).astype(np.float32)
        feeds = {"input": test_input}

        # Run original
        original_out = _run_inference(model, feeds)

        # Apply fusion and run
        fused_model = _apply_fusion(model)
        fused_out = _run_inference(fused_model, feeds)

        # Compare outputs
        for key in original_out:
            np.testing.assert_allclose(
                original_out[key],
                fused_out[key],
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Numeric mismatch for output '{key}'",
            )

    def test_numeric_equivalence_ones_weight(self):
        """Numeric equivalence when weight is all-ones (scalar collapse path)."""
        hidden_size = 16
        ones_weight = np.ones(hidden_size, dtype=np.float32)

        model = _make_rmsnorm_model(
            hidden_size=hidden_size,
            weight_value=ones_weight,
            seq_len=4,
            batch_size=1,
        )

        rng = np.random.RandomState(99)
        test_input = rng.randn(1, 4, hidden_size).astype(np.float32)
        feeds = {"input": test_input}

        original_out = _run_inference(model, feeds)
        fused_model = _apply_fusion(model)
        fused_out = _run_inference(fused_model, feeds)

        for key in original_out:
            np.testing.assert_allclose(
                original_out[key],
                fused_out[key],
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Numeric mismatch (ones weight) for output '{key}'",
            )

    def test_numeric_equivalence_large_hidden_size(self):
        """Numeric equivalence with larger hidden_size to stress sqrt(N) scaling."""
        hidden_size = 768
        rng = np.random.RandomState(7)
        weight_value = rng.randn(hidden_size).astype(np.float32)

        model = _make_rmsnorm_model(
            hidden_size=hidden_size,
            weight_value=weight_value,
            seq_len=2,
            batch_size=1,
        )

        test_input = rng.randn(1, 2, hidden_size).astype(np.float32)
        feeds = {"input": test_input}

        original_out = _run_inference(model, feeds)
        fused_model = _apply_fusion(model)
        fused_out = _run_inference(fused_model, feeds)

        for key in original_out:
            np.testing.assert_allclose(
                original_out[key],
                fused_out[key],
                rtol=1e-3,
                atol=1e-4,
                err_msg=f"Numeric mismatch (large hidden) for output '{key}'",
            )
