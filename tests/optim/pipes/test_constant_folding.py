"""Tests for ConstantFolding optimization effect.

ConstantFolding is a "basic" ORT optimizer that runs at Level 2 by default.
Unlike advanced capabilities in GRAPH_CAPABILITIES, it cannot be enabled through
GraphPipe config - it's always on unless explicitly disabled.

This test file verifies:
1. ConstantFolding actually folds constant expressions (node reduction)
2. Disabling ConstantFolding preserves constant computation nodes
3. Numerical equivalence after folding

Pattern tested: Cast -> Mul(const, scale) -> Add
- The Mul(const, scale) computes product of two constants at runtime
- ConstantFolding should pre-compute this into a single constant
- Expected: 3 nodes -> 2 nodes (Mul eliminated)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from ..assets.graphpipe.builders.core import constant_folding_builder


# =============================================================================
# MODEL FACTORY
# =============================================================================


def create_constant_folding_model() -> onnx.ModelProto:
    """Create minimal model with constant folding pattern.

    Pattern: Input -> Cast -> Add(Mul(const, scale)) -> Output

    The Mul(const, scale) multiplies two constant tensors.
    ConstantFolding should fold this into a single constant.

    Returns:
        ONNX model with foldable constant expression
    """
    initializers: list = []
    input_name = "input"
    output_name = "output"
    prefix = "cf_"

    # Build pattern using existing builder
    nodes = constant_folding_builder(input_name, output_name, prefix, initializers)

    # Create input/output
    input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 64])
    output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, None)

    # Create graph
    graph = helper.make_graph(
        nodes=nodes,
        name="constant_folding_test",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )

    # Create model
    model = helper.make_model(
        graph,
        producer_name="modelkit_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    return model


def get_op_types(model: onnx.ModelProto) -> list[str]:
    """Get list of op types in model."""
    return [node.op_type for node in model.graph.node]


def run_inference(model: onnx.ModelProto, input_data: np.ndarray) -> np.ndarray:
    """Run inference on model and return output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.onnx"
        onnx.save(model, str(model_path))
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        output = session.run(None, {"input": input_data})
        return output[0]


def optimize_model(
    model: onnx.ModelProto,
    disabled_optimizers: list[str] | None = None,
) -> onnx.ModelProto:
    """Optimize model using ORT SessionOptions and return optimized model.

    Args:
        model: Input ONNX model
        disabled_optimizers: List of optimizers to disable

    Returns:
        Optimized ONNX model
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.onnx"
        output_path = Path(tmpdir) / "optimized.onnx"

        onnx.save(model, str(input_path))

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = str(output_path)

        # Use correct ORT API key with semicolon-separated list
        if disabled_optimizers:
            disable_list = ";".join(disabled_optimizers)
            sess_options.add_session_config_entry(
                "optimization.disable_specified_optimizers",
                disable_list,
            )

        # Create session to trigger optimization
        ort.InferenceSession(str(input_path), sess_options, providers=["CPUExecutionProvider"])

        # Load optimized model
        return onnx.load(str(output_path))


# =============================================================================
# TESTS
# =============================================================================


class TestConstantFoldingEffect:
    """Test ConstantFolding actual optimization effect."""

    def test_constant_folding_node_reduction(self) -> None:
        """Verify Mul(const, scale) is folded into single constant.

        Pattern before: Cast, Mul, Add (3 nodes)
        Pattern after:  Cast, Add (2 nodes) - Mul folded away
        """
        # 1. Create model with constant folding pattern
        model = create_constant_folding_model()
        nodes_before = len(model.graph.node)
        ops_before = get_op_types(model)

        # Verify initial state
        assert nodes_before == 3, f"Expected 3 nodes before, got {nodes_before}"
        assert "Mul" in ops_before, "Mul node should exist before optimization"
        assert "Cast" in ops_before, "Cast node should exist"
        assert "Add" in ops_before, "Add node should exist"

        # 2. Optimize with ConstantFolding ENABLED (default)
        optimized = optimize_model(model, disabled_optimizers=[])
        nodes_after = len(optimized.graph.node)
        ops_after = get_op_types(optimized)

        # 3. Verify node reduction
        assert nodes_after < nodes_before, (
            f"ConstantFolding should reduce node count: {nodes_before} -> {nodes_after}"
        )
        assert "Mul" not in ops_after, f"Mul node should be folded away, but found ops: {ops_after}"

    def test_constant_folding_disabled_preserves_mul(self) -> None:
        """Verify disabling ConstantFolding preserves Mul node."""
        # 1. Create model
        model = create_constant_folding_model()
        ops_before = get_op_types(model)
        assert "Mul" in ops_before, "Mul should exist before"

        # 2. Optimize with ConstantFolding DISABLED
        optimized = optimize_model(model, disabled_optimizers=["ConstantFolding"])
        ops_after = get_op_types(optimized)

        # 3. Mul should still exist
        assert "Mul" in ops_after, (
            f"Mul node should be preserved when ConstantFolding disabled, "
            f"but found ops: {ops_after}"
        )

    def test_constant_folding_numeric_equivalence(self) -> None:
        """Verify optimized model produces same output as original."""
        # 1. Create model and optimize
        model = create_constant_folding_model()
        optimized = optimize_model(model, disabled_optimizers=[])

        # 2. Generate random input
        rng = np.random.RandomState(42)
        input_data = rng.randn(1, 64).astype(np.float32)

        # 3. Run inference on both models
        output_original = run_inference(model, input_data)
        output_optimized = run_inference(optimized, input_data)

        # 4. Compare outputs
        np.testing.assert_allclose(
            output_original,
            output_optimized,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Optimized model should produce same output as original",
        )

    def test_constant_folding_enabled_vs_disabled_comparison(self) -> None:
        """Compare enabled vs disabled ConstantFolding side by side."""
        model = create_constant_folding_model()

        # Optimize with and without ConstantFolding
        with_folding = optimize_model(model, disabled_optimizers=[])
        without_folding = optimize_model(model, disabled_optimizers=["ConstantFolding"])

        nodes_with = len(with_folding.graph.node)
        nodes_without = len(without_folding.graph.node)
        ops_with = get_op_types(with_folding)
        ops_without = get_op_types(without_folding)

        # With folding should have fewer nodes
        assert nodes_with < nodes_without, (
            f"With ConstantFolding ({nodes_with} nodes) should have fewer nodes "
            f"than without ({nodes_without} nodes)"
        )

        # Mul should only exist in non-folded version
        assert "Mul" not in ops_with, f"Mul should be folded, got: {ops_with}"
        assert "Mul" in ops_without, f"Mul should be preserved, got: {ops_without}"

        # Both should produce same output
        rng = np.random.RandomState(42)
        input_data = rng.randn(1, 64).astype(np.float32)

        output_with = run_inference(with_folding, input_data)
        output_without = run_inference(without_folding, input_data)

        np.testing.assert_allclose(
            output_with,
            output_without,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Both versions should produce same output",
        )
