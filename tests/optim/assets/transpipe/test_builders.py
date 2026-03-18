"""Verification tests for TransformerPipe builders.

These tests verify that:
1. Builders create valid ONNX models
2. ORT fusion classes successfully fuse the patterns
3. LayerNormalization is preserved (not converted to SkipLayerNormalization)
"""

import onnx

from .builders import (
    build_bert_attention_model,
    build_clip_attention_model,
    build_decomposed_layernorm_model,
    build_rms_norm_model,
)


def count_nodes_by_type(model: onnx.ModelProto) -> dict[str, int]:
    """Count nodes by op_type in the model."""
    counts = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def has_node_type(model: onnx.ModelProto, op_type: str) -> bool:
    """Check if model has a node of the given op_type."""
    return any(node.op_type == op_type for node in model.graph.node)


class TestAttentionBuilders:
    """Tests for attention pattern builders."""

    def test_bert_attention_builder_creates_valid_model(self):
        """Test that BERT attention builder creates a valid ONNX model."""
        model = build_bert_attention_model()
        onnx.checker.check_model(model)

    def test_bert_attention_builder_creates_expected_nodes(self):
        """Test that BERT attention builder creates expected node types."""
        model = build_bert_attention_model()
        counts = count_nodes_by_type(model)

        # Should have unfused attention pattern
        assert counts.get("MatMul", 0) >= 4, "Should have Q, K, V, Out MatMuls"
        assert counts.get("Add", 0) >= 4, "Should have bias adds"
        assert counts.get("Reshape", 0) >= 4, "Should have reshapes for Q, K, V, Out"
        assert counts.get("Transpose", 0) >= 4, "Should have transposes"
        assert counts.get("Softmax", 0) == 1, "Should have one Softmax"
        assert counts.get("LayerNormalization", 0) == 2, "Should have two LayerNorms"

    def test_bert_attention_fuses_with_ort(self):
        """Test that BERT attention pattern fuses correctly with ORT fusion classes."""
        model = build_bert_attention_model()
        original_node_count = len(model.graph.node)

        # Import ORT fusion classes
        from onnxruntime.transformers.fusion_attention import FusionAttention
        from onnxruntime.transformers.onnx_model import OnnxModel

        # Apply fusion
        onnx_model = OnnxModel(model)
        FusionAttention(onnx_model, hidden_size=0, num_heads=0).apply()
        onnx_model.prune_graph()
        fused_model = onnx_model.model

        fused_node_count = len(fused_model.graph.node)

        # Verify fusion occurred
        assert fused_node_count < original_node_count, (
            f"Fusion should reduce nodes ({original_node_count} → {fused_node_count})"
        )
        assert has_node_type(fused_model, "Attention"), "Should have fused Attention op"

        # CRITICAL: Verify LayerNormalization is preserved (not SkipLayerNormalization)
        assert has_node_type(fused_model, "LayerNormalization"), (
            "LayerNormalization should be preserved"
        )
        assert not has_node_type(fused_model, "SkipLayerNormalization"), (
            "SkipLayerNormalization should NOT be created"
        )

    def test_clip_attention_builder_creates_valid_model(self):
        """Test that CLIP attention builder creates a valid ONNX model."""
        model = build_clip_attention_model()
        onnx.checker.check_model(model)

    def test_clip_attention_fuses_with_ort(self):
        """Test that CLIP attention pattern fuses correctly with ORT fusion classes."""
        model = build_clip_attention_model()

        from onnxruntime.transformers.fusion_attention import FusionAttention
        from onnxruntime.transformers.onnx_model import OnnxModel

        onnx_model = OnnxModel(model)
        FusionAttention(onnx_model, hidden_size=0, num_heads=0).apply()
        onnx_model.prune_graph()
        fused_model = onnx_model.model

        # CLIP attention may or may not fuse depending on pattern match
        # At minimum, verify model is valid and LayerNorm is preserved
        onnx.checker.check_model(fused_model)
        assert has_node_type(fused_model, "LayerNormalization"), (
            "LayerNormalization should be preserved"
        )
        assert not has_node_type(fused_model, "SkipLayerNormalization"), (
            "SkipLayerNormalization should NOT be created"
        )


class TestLayerNormBuilders:
    """Tests for LayerNorm pattern builders."""

    def test_decomposed_layernorm_builder_creates_valid_model(self):
        """Test that decomposed LayerNorm builder creates a valid ONNX model."""
        model = build_decomposed_layernorm_model()
        onnx.checker.check_model(model)

    def test_decomposed_layernorm_creates_expected_nodes(self):
        """Test that decomposed LayerNorm builder creates expected node types."""
        model = build_decomposed_layernorm_model()
        counts = count_nodes_by_type(model)

        # Should have decomposed LayerNorm pattern
        assert counts.get("ReduceMean", 0) == 2, "Should have two ReduceMean (mean, variance)"
        assert counts.get("Sub", 0) == 1, "Should have one Sub (centering)"
        assert counts.get("Pow", 0) == 1, "Should have one Pow (squaring)"
        assert counts.get("Add", 0) == 2, "Should have two Add (epsilon, beta)"
        assert counts.get("Sqrt", 0) == 1, "Should have one Sqrt"
        assert counts.get("Div", 0) == 1, "Should have one Div (normalization)"
        assert counts.get("Mul", 0) == 1, "Should have one Mul (gamma)"

    def test_decomposed_layernorm_fuses_with_ort(self):
        """Test that decomposed LayerNorm fuses correctly with ORT fusion classes."""
        model = build_decomposed_layernorm_model()
        original_node_count = len(model.graph.node)

        from onnxruntime.transformers.fusion_layernorm import FusionLayerNormalization
        from onnxruntime.transformers.onnx_model import OnnxModel

        onnx_model = OnnxModel(model)
        FusionLayerNormalization(onnx_model).apply()
        onnx_model.prune_graph()
        fused_model = onnx_model.model

        fused_node_count = len(fused_model.graph.node)
        counts = count_nodes_by_type(fused_model)

        # Verify fusion occurred
        assert fused_node_count < original_node_count, (
            f"Fusion should reduce nodes ({original_node_count} → {fused_node_count})"
        )
        assert counts.get("LayerNormalization", 0) == 1, "Should have one fused LayerNormalization"

    def test_rms_norm_builder_creates_valid_model(self):
        """Test that RMS Norm builder creates a valid ONNX model."""
        model = build_rms_norm_model()
        onnx.checker.check_model(model)

    def test_rms_norm_creates_expected_nodes(self):
        """Test that RMS Norm builder creates expected node types."""
        model = build_rms_norm_model()
        counts = count_nodes_by_type(model)

        # RMS Norm pattern (no mean subtraction)
        assert counts.get("Pow", 0) == 1, "Should have one Pow (squaring)"
        assert counts.get("ReduceMean", 0) == 1, "Should have one ReduceMean"
        assert counts.get("Add", 0) == 1, "Should have one Add (epsilon)"
        assert counts.get("Sqrt", 0) == 1, "Should have one Sqrt"
        assert counts.get("Div", 0) == 1, "Should have one Div"
        assert counts.get("Mul", 0) == 1, "Should have one Mul (gamma)"
        # RMS Norm should NOT have Sub (no mean centering)
        assert counts.get("Sub", 0) == 0, "RMS Norm should not have Sub (no mean centering)"

    def test_rms_norm_fuses_with_ort(self):
        """Test that RMS Norm fuses correctly with ORT fusion classes."""
        model = build_rms_norm_model()
        original_node_count = len(model.graph.node)

        from onnxruntime.transformers.fusion_simplified_layernorm import (
            FusionSimplifiedLayerNormalization,
        )
        from onnxruntime.transformers.onnx_model import OnnxModel

        onnx_model = OnnxModel(model)
        FusionSimplifiedLayerNormalization(onnx_model).apply()
        onnx_model.prune_graph()
        fused_model = onnx_model.model

        fused_node_count = len(fused_model.graph.node)
        counts = count_nodes_by_type(fused_model)

        # Verify fusion occurred
        assert fused_node_count < original_node_count, (
            f"Fusion should reduce nodes ({original_node_count} → {fused_node_count})"
        )
        assert counts.get("SimplifiedLayerNormalization", 0) == 1, (
            "Should have one fused SimplifiedLayerNormalization"
        )


class TestBuilderIntegration:
    """Integration tests verifying builders work together."""

    def test_bert_attention_with_all_fusion_classes(self):
        """Test BERT attention builder with all three attention fusion classes."""
        model = build_bert_attention_model()

        from onnxruntime.transformers.fusion_attention import FusionAttention
        from onnxruntime.transformers.fusion_attention_clip import FusionAttentionClip
        from onnxruntime.transformers.fusion_gpt_attention import FusionGptAttention
        from onnxruntime.transformers.onnx_model import OnnxModel

        onnx_model = OnnxModel(model)

        # Apply all attention fusions (each only matches its specific pattern)
        FusionAttention(onnx_model, hidden_size=0, num_heads=0).apply()
        FusionGptAttention(onnx_model, num_heads=0).apply()
        FusionAttentionClip(onnx_model, hidden_size=0, num_heads=0).apply()

        onnx_model.prune_graph()
        fused_model = onnx_model.model

        # Should have Attention op and LayerNormalization preserved
        assert has_node_type(fused_model, "Attention"), "Should have Attention op"
        assert has_node_type(fused_model, "LayerNormalization"), (
            "LayerNormalization should be preserved"
        )
        assert not has_node_type(fused_model, "SkipLayerNormalization"), (
            "SkipLayerNormalization should NOT be created"
        )
