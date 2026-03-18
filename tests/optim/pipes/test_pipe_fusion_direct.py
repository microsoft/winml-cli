# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for FusionPipe direct fusion class approach.

This test module validates that direct fusion classes work correctly
for the redesigned FusionPipe implementation.

Test Strategy:
    - Use 4-criteria verification for all fusion tests
    - 1. Node Count Reduction: Optimized model has fewer nodes
    - 2. Existence Check: Expected fused ops MUST exist
    - 3. Non-Existence Check: Unexpected fused ops MUST NOT exist
    - 4. Numeric Verification: Outputs must match within tolerance

Reference: docs/design/optimization/7_fusion_pipe_attention_redesign.md
"""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from ..assets.fusionpipe.builders.attention import (
    create_bert_attention_model,
    create_gpt2_attention_model,
)
from ..assets.fusionpipe.builders.layernorm import (
    create_decomposed_layernorm_model,
    create_simplified_layernorm_model,
    create_skip_layernorm_model,
)


# =============================================================================
# Test Utilities
# =============================================================================


def count_nodes(model: onnx.ModelProto) -> int:
    """Count total nodes in model graph."""
    return len(model.graph.node)


def get_op_types(model: onnx.ModelProto) -> set[str]:
    """Get set of all op types in model graph."""
    return {node.op_type for node in model.graph.node}


def has_op_type(model: onnx.ModelProto, op_type: str) -> bool:
    """Check if model contains specific op type."""
    return op_type in get_op_types(model)


def generate_random_inputs(
    model: onnx.ModelProto,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate random inputs for model inference.

    Args:
        model: ONNX model
        seed: Random seed for reproducibility

    Returns:
        Dictionary of input name to numpy array
    """
    rng = np.random.RandomState(seed)
    inputs = {}

    for inp in model.graph.input:
        # Skip initializers
        if any(init.name == inp.name for init in model.graph.initializer):
            continue

        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(1)  # Default for dynamic dims

        dtype = inp.type.tensor_type.elem_type
        if dtype == TensorProto.FLOAT:
            inputs[inp.name] = rng.randn(*shape).astype(np.float32)
        elif dtype == TensorProto.INT64:
            # Attention masks should be binary (0 or 1), use all 1s for consistency
            if "mask" in inp.name.lower():
                inputs[inp.name] = np.ones(shape, dtype=np.int64)
            else:
                inputs[inp.name] = rng.randint(0, 10, size=shape).astype(np.int64)
        elif dtype == TensorProto.INT32:
            if "mask" in inp.name.lower():
                inputs[inp.name] = np.ones(shape, dtype=np.int32)
            else:
                inputs[inp.name] = rng.randint(0, 10, size=shape).astype(np.int32)
        else:
            inputs[inp.name] = rng.randn(*shape).astype(np.float32)

    return inputs


def run_onnx_inference(
    model: onnx.ModelProto,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Run inference on ONNX model.

    Args:
        model: ONNX model
        inputs: Input dictionary

    Returns:
        Output dictionary
    """
    import onnxruntime as ort

    # Serialize model to bytes
    model_bytes = model.SerializeToString()

    # Create session
    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])

    # Run inference
    output_names = [out.name for out in sess.get_outputs()]
    outputs = sess.run(output_names, inputs)

    return dict(zip(output_names, outputs, strict=False))


def verify_numeric_equivalence(
    original_outputs: dict[str, np.ndarray],
    optimized_outputs: dict[str, np.ndarray],
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    """Verify outputs are numerically equivalent.

    Args:
        original_outputs: Original model outputs
        optimized_outputs: Optimized model outputs
        rtol: Relative tolerance
        atol: Absolute tolerance

    Raises:
        AssertionError: If outputs don't match
    """
    for name in original_outputs:
        assert name in optimized_outputs, f"Missing output: {name}"
        np.testing.assert_allclose(
            original_outputs[name],
            optimized_outputs[name],
            rtol=rtol,
            atol=atol,
            err_msg=f"Output mismatch for {name}",
        )


# =============================================================================
# Direct Fusion Application Functions
# =============================================================================


def apply_attention_fusions(model: onnx.ModelProto) -> onnx.ModelProto:
    """Apply all attention fusion classes directly.

    Args:
        model: Input ONNX model

    Returns:
        Optimized model with attention fusions applied
    """
    from onnxruntime.transformers.fusion_attention import FusionAttention
    from onnxruntime.transformers.fusion_gpt_attention import FusionGptAttention
    from onnxruntime.transformers.onnx_model import OnnxModel

    onnx_model = OnnxModel(model)

    # Apply all attention fusion classes
    # Each only modifies nodes matching its specific pattern
    FusionAttention(onnx_model, hidden_size=0, num_heads=0).apply()
    FusionGptAttention(onnx_model, num_heads=0).apply()

    onnx_model.prune_graph()
    return onnx_model.model


def apply_layernorm_fusion(model: onnx.ModelProto) -> onnx.ModelProto:
    """Apply LayerNorm fusion directly.

    Args:
        model: Input ONNX model

    Returns:
        Optimized model with LayerNorm fusion applied
    """
    from onnxruntime.transformers.fusion_layernorm import FusionLayerNormalization
    from onnxruntime.transformers.onnx_model import OnnxModel

    onnx_model = OnnxModel(model)
    FusionLayerNormalization(onnx_model).apply()
    onnx_model.prune_graph()
    return onnx_model.model


def apply_skip_layernorm_fusion(model: onnx.ModelProto) -> onnx.ModelProto:
    """Apply SkipLayerNorm fusion directly.

    Args:
        model: Input ONNX model

    Returns:
        Optimized model with SkipLayerNorm fusion applied
    """
    from onnxruntime.transformers.fusion_skiplayernorm import (
        FusionSkipLayerNormalization,
    )
    from onnxruntime.transformers.onnx_model import OnnxModel

    onnx_model = OnnxModel(model)
    FusionSkipLayerNormalization(onnx_model).apply()
    onnx_model.prune_graph()
    return onnx_model.model


def apply_simplified_layernorm_fusion(model: onnx.ModelProto) -> onnx.ModelProto:
    """Apply SimplifiedLayerNorm fusion directly.

    Args:
        model: Input ONNX model

    Returns:
        Optimized model with SimplifiedLayerNorm fusion applied
    """
    from onnxruntime.transformers.fusion_simplified_layernorm import (
        FusionSimplifiedLayerNormalization,
    )
    from onnxruntime.transformers.onnx_model import OnnxModel

    onnx_model = OnnxModel(model)
    FusionSimplifiedLayerNormalization(onnx_model).apply()
    onnx_model.prune_graph()
    return onnx_model.model


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def bert_attention_model() -> onnx.ModelProto:
    """Create BERT attention model for testing."""
    return create_bert_attention_model(hidden_size=16, num_heads=2, seq_len=3)


@pytest.fixture
def gpt2_attention_model() -> onnx.ModelProto:
    """Create GPT-2 attention model for testing."""
    return create_gpt2_attention_model(hidden_size=16, num_heads=2, seq_len=3)


@pytest.fixture
def decomposed_layernorm_model() -> onnx.ModelProto:
    """Create decomposed LayerNorm model for testing."""
    return create_decomposed_layernorm_model(hidden_size=64, seq_len=10)


@pytest.fixture
def skip_layernorm_model() -> onnx.ModelProto:
    """Create SkipLayerNorm model for testing."""
    return create_skip_layernorm_model(hidden_size=64, seq_len=10)


@pytest.fixture
def simplified_layernorm_model() -> onnx.ModelProto:
    """Create SimplifiedLayerNorm model for testing."""
    return create_simplified_layernorm_model(hidden_size=64, seq_len=10)


# =============================================================================
# Attention Fusion Tests
# =============================================================================


class TestAttentionFusion:
    """Tests for attention fusion using direct fusion classes.

    BERT attention builder matches ORT's bert_model_generator.py pattern:
        Add -> LayerNorm -> Q/K/V projections -> Attention -> Add -> LayerNorm
        with proper mask handling (Unsqueeze -> Cast -> Sub -> Mul).

    GPT-2 builder is simplified and doesn't include full pattern.
    """

    def test_bert_attention_fusion_reduces_nodes(
        self, bert_attention_model: onnx.ModelProto
    ) -> None:
        """Test that BERT attention fusion reduces node count."""
        before_count = count_nodes(bert_attention_model)

        optimized = apply_attention_fusions(bert_attention_model)
        after_count = count_nodes(optimized)

        # Should reduce significantly (from ~30 nodes to ~6)
        assert after_count < before_count, (
            f"Expected node reduction: before={before_count}, after={after_count}"
        )

    def test_bert_attention_fusion_creates_attention_op(
        self, bert_attention_model: onnx.ModelProto
    ) -> None:
        """Test that BERT attention fusion creates Attention op."""
        optimized = apply_attention_fusions(bert_attention_model)

        # Should have Attention or MultiHeadAttention op
        op_types = get_op_types(optimized)
        has_attention = "Attention" in op_types or "MultiHeadAttention" in op_types

        assert has_attention, f"Expected Attention op, got: {op_types}"

    def test_bert_attention_fusion_numeric_equivalence(
        self, bert_attention_model: onnx.ModelProto
    ) -> None:
        """Test that BERT attention fusion preserves numeric output."""
        inputs = generate_random_inputs(bert_attention_model)
        original_outputs = run_onnx_inference(bert_attention_model, inputs)

        optimized = apply_attention_fusions(bert_attention_model)
        optimized_outputs = run_onnx_inference(optimized, inputs)

        verify_numeric_equivalence(original_outputs, optimized_outputs)

    @pytest.mark.xfail(
        reason="GPT-2 builder is simplified, doesn't have full transformer block pattern"
    )
    def test_gpt2_attention_fusion_reduces_nodes(
        self, gpt2_attention_model: onnx.ModelProto
    ) -> None:
        """Test that GPT-2 attention fusion reduces node count."""
        before_count = count_nodes(gpt2_attention_model)

        optimized = apply_attention_fusions(gpt2_attention_model)
        after_count = count_nodes(optimized)

        # Should reduce (causal attention pattern)
        assert after_count < before_count, (
            f"Expected node reduction: before={before_count}, after={after_count}"
        )

    def test_gpt2_attention_fusion_numeric_equivalence(
        self, gpt2_attention_model: onnx.ModelProto
    ) -> None:
        """Test that GPT-2 attention model runs correctly (no fusion expected)."""
        inputs = generate_random_inputs(gpt2_attention_model)
        original_outputs = run_onnx_inference(gpt2_attention_model, inputs)

        # Apply fusion (may not fuse, but should still be valid)
        optimized = apply_attention_fusions(gpt2_attention_model)
        optimized_outputs = run_onnx_inference(optimized, inputs)

        verify_numeric_equivalence(original_outputs, optimized_outputs)


# =============================================================================
# LayerNorm Fusion Tests
# =============================================================================


class TestLayerNormFusion:
    """Tests for LayerNorm fusion using direct fusion classes."""

    def test_decomposed_layernorm_fusion_reduces_nodes(
        self, decomposed_layernorm_model: onnx.ModelProto
    ) -> None:
        """Test that decomposed LayerNorm fusion reduces node count."""
        before_count = count_nodes(decomposed_layernorm_model)

        optimized = apply_layernorm_fusion(decomposed_layernorm_model)
        after_count = count_nodes(optimized)

        # Should reduce from 9 nodes to 1
        assert after_count < before_count, (
            f"Expected node reduction: before={before_count}, after={after_count}"
        )

    def test_decomposed_layernorm_fusion_creates_layernorm_op(
        self, decomposed_layernorm_model: onnx.ModelProto
    ) -> None:
        """Test that decomposed LayerNorm fusion creates LayerNormalization op."""
        optimized = apply_layernorm_fusion(decomposed_layernorm_model)

        assert has_op_type(optimized, "LayerNormalization"), (
            f"Expected LayerNormalization op, got: {get_op_types(optimized)}"
        )

    def test_decomposed_layernorm_fusion_numeric_equivalence(
        self, decomposed_layernorm_model: onnx.ModelProto
    ) -> None:
        """Test that decomposed LayerNorm fusion preserves numeric output."""
        inputs = generate_random_inputs(decomposed_layernorm_model)
        original_outputs = run_onnx_inference(decomposed_layernorm_model, inputs)

        optimized = apply_layernorm_fusion(decomposed_layernorm_model)
        optimized_outputs = run_onnx_inference(optimized, inputs)

        verify_numeric_equivalence(original_outputs, optimized_outputs)

    def test_skip_layernorm_fusion_reduces_nodes(
        self, skip_layernorm_model: onnx.ModelProto
    ) -> None:
        """Test that SkipLayerNorm fusion reduces node count."""
        before_count = count_nodes(skip_layernorm_model)

        optimized = apply_skip_layernorm_fusion(skip_layernorm_model)
        after_count = count_nodes(optimized)

        # Should reduce (Add + LayerNorm -> SkipLayerNormalization)
        assert after_count < before_count, (
            f"Expected node reduction: before={before_count}, after={after_count}"
        )

    def test_skip_layernorm_fusion_creates_skip_layernorm_op(
        self, skip_layernorm_model: onnx.ModelProto
    ) -> None:
        """Test that SkipLayerNorm fusion creates SkipLayerNormalization op."""
        optimized = apply_skip_layernorm_fusion(skip_layernorm_model)

        assert has_op_type(optimized, "SkipLayerNormalization"), (
            f"Expected SkipLayerNormalization op, got: {get_op_types(optimized)}"
        )

    def test_skip_layernorm_fusion_numeric_equivalence(
        self, skip_layernorm_model: onnx.ModelProto
    ) -> None:
        """Test that SkipLayerNorm fusion preserves numeric output."""
        inputs = generate_random_inputs(skip_layernorm_model)
        original_outputs = run_onnx_inference(skip_layernorm_model, inputs)

        optimized = apply_skip_layernorm_fusion(skip_layernorm_model)
        optimized_outputs = run_onnx_inference(optimized, inputs)

        verify_numeric_equivalence(original_outputs, optimized_outputs)

    def test_simplified_layernorm_fusion_reduces_nodes(
        self, simplified_layernorm_model: onnx.ModelProto
    ) -> None:
        """Test that SimplifiedLayerNorm fusion reduces node count."""
        before_count = count_nodes(simplified_layernorm_model)

        optimized = apply_simplified_layernorm_fusion(simplified_layernorm_model)
        after_count = count_nodes(optimized)

        # Should reduce from 6 nodes to 1
        assert after_count < before_count, (
            f"Expected node reduction: before={before_count}, after={after_count}"
        )

    def test_simplified_layernorm_fusion_creates_simplified_layernorm_op(
        self, simplified_layernorm_model: onnx.ModelProto
    ) -> None:
        """Test that SimplifiedLayerNorm fusion creates SimplifiedLayerNormalization op."""
        optimized = apply_simplified_layernorm_fusion(simplified_layernorm_model)

        assert has_op_type(optimized, "SimplifiedLayerNormalization"), (
            f"Expected SimplifiedLayerNormalization op, got: {get_op_types(optimized)}"
        )

    def test_simplified_layernorm_fusion_numeric_equivalence(
        self, simplified_layernorm_model: onnx.ModelProto
    ) -> None:
        """Test that SimplifiedLayerNorm fusion preserves numeric output."""
        inputs = generate_random_inputs(simplified_layernorm_model)
        original_outputs = run_onnx_inference(simplified_layernorm_model, inputs)

        optimized = apply_simplified_layernorm_fusion(simplified_layernorm_model)
        optimized_outputs = run_onnx_inference(optimized, inputs)

        verify_numeric_equivalence(original_outputs, optimized_outputs)


# =============================================================================
# Integration Tests
# =============================================================================


class TestFusionIntegration:
    """Integration tests for combined fusion operations."""

    def test_all_fusions_combined(
        self, decomposed_layernorm_model: onnx.ModelProto
    ) -> None:
        """Test applying multiple fusions in sequence on LayerNorm model."""
        from onnxruntime.transformers.fusion_layernorm import FusionLayerNormalization
        from onnxruntime.transformers.fusion_skiplayernorm import (
            FusionSkipLayerNormalization,
        )
        from onnxruntime.transformers.onnx_model import OnnxModel

        before_count = count_nodes(decomposed_layernorm_model)

        # Apply all LayerNorm-related fusions
        onnx_model = OnnxModel(decomposed_layernorm_model)
        FusionSkipLayerNormalization(onnx_model).apply()
        FusionLayerNormalization(onnx_model).apply()
        onnx_model.prune_graph()

        optimized = onnx_model.model
        after_count = count_nodes(optimized)

        # Should reduce significantly (9 nodes -> 1)
        assert after_count < before_count, (
            f"Expected node reduction: before={before_count}, after={after_count}"
        )

    def test_no_fusion_when_pattern_not_present(self) -> None:
        """Test that fusion does nothing when pattern not present."""
        # Create a simple model without any fusion patterns
        input_tensor = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 10, 64]
        )
        output_tensor = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 10, 64]
        )

        relu_node = helper.make_node(
            "Relu",
            inputs=["input"],
            outputs=["output"],
            name="relu",
        )

        graph = helper.make_graph(
            [relu_node],
            "simple_model",
            [input_tensor],
            [output_tensor],
        )

        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17)],
        )
        model.ir_version = 8

        before_count = count_nodes(model)

        # Apply attention fusion (should do nothing)
        optimized = apply_attention_fusions(model)
        after_count = count_nodes(optimized)

        # Node count should be unchanged
        assert after_count == before_count, (
            f"Expected no change: before={before_count}, after={after_count}"
        )

    def test_fusion_isolation(
        self, bert_attention_model: onnx.ModelProto
    ) -> None:
        """Test that attention fusion doesn't create unrelated ops."""
        optimized = apply_attention_fusions(bert_attention_model)
        op_types = get_op_types(optimized)

        # Should NOT have GroupNorm or GELU ops (isolation check)
        assert "GroupNorm" not in op_types
        assert "SkipGroupNorm" not in op_types
        assert "Gelu" not in op_types
        assert "BiasGelu" not in op_types


# =============================================================================
# Model Builder Tests
# =============================================================================


class TestModelBuilders:
    """Tests for model builder functions."""

    def test_bert_attention_model_valid(self) -> None:
        """Test that BERT attention model passes ONNX checker."""
        model = create_bert_attention_model()
        onnx.checker.check_model(model)

    def test_gpt2_attention_model_valid(self) -> None:
        """Test that GPT-2 attention model passes ONNX checker."""
        model = create_gpt2_attention_model()
        onnx.checker.check_model(model)

    def test_decomposed_layernorm_model_valid(self) -> None:
        """Test that decomposed LayerNorm model passes ONNX checker."""
        model = create_decomposed_layernorm_model()
        onnx.checker.check_model(model)

    def test_skip_layernorm_model_valid(self) -> None:
        """Test that SkipLayerNorm model passes ONNX checker."""
        model = create_skip_layernorm_model()
        onnx.checker.check_model(model)

    def test_simplified_layernorm_model_valid(self) -> None:
        """Test that SimplifiedLayerNorm model passes ONNX checker."""
        model = create_simplified_layernorm_model()
        onnx.checker.check_model(model)

    def test_bert_attention_model_runnable(self) -> None:
        """Test that BERT attention model can run inference."""
        model = create_bert_attention_model()
        inputs = generate_random_inputs(model)
        outputs = run_onnx_inference(model, inputs)
        assert len(outputs) > 0

    def test_decomposed_layernorm_model_runnable(self) -> None:
        """Test that decomposed LayerNorm model can run inference."""
        model = create_decomposed_layernorm_model()
        inputs = generate_random_inputs(model)
        outputs = run_onnx_inference(model, inputs)
        assert len(outputs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
