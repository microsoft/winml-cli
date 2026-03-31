# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Generate ORT Fusion Optimization Test Patterns.

Creates ONNX models for testing ORTFusionPipe control options:
1. Self-attention pattern - Tests attention-op-type and use-multi-head-attention
2. GQA pattern - Tests GroupQueryAttention (LLaMA/Phi style)
3. GroupNorm pattern - Tests group-norm-channels-last layout

NOTE: ORT's transformer optimizer expects patterns from real model exports.
These synthetic patterns may not fuse - tests verify config passing, not fusion.

Usage in pytest fixtures:
    from tests.unit.optim.assets.generate_fusion_patterns import (
        create_self_attention_model,
        create_gqa_model,
        create_groupnorm_model,
    )

    @pytest.fixture(scope="module")
    def attention_model() -> onnx.ModelProto:
        return create_self_attention_model()
"""

from __future__ import annotations

import math

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def make_compatible_model(graph: onnx.GraphProto, opset_version: int = 14) -> onnx.ModelProto:
    """Create model with IR version compatible with ORT."""
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset_version)])
    model.ir_version = 8
    return model


# =============================================================================
# ATTENTION PATTERNS
# =============================================================================


def create_self_attention_model(
    batch_size: int = 1,
    seq_len: int = 128,
    hidden_size: int = 768,
    num_heads: int = 12,
    prefix: str = "attn_",
) -> onnx.ModelProto:
    """Create self-attention pattern for Attention/MHA fusion testing.

    Pattern structure:
        X → MatMul(Wq) + bias → Q
        X → MatMul(Wk) + bias → K
        X → MatMul(Wv) + bias → V
        Q, K, V → Reshape → Transpose → Attention calc → Output

    This pattern tests:
    - attention-op-type (Attention vs MultiHeadAttention)
    - use-multi-head-attention toggle

    Note: ORT's attention fusion expects patterns from real model exports.
    This synthetic pattern may not fuse - tests verify config passing.

    Args:
        batch_size: Batch dimension (default: 1)
        seq_len: Sequence length (default: 128)
        hidden_size: Hidden dimension (default: 768)
        num_heads: Number of attention heads (default: 12)
        prefix: Node name prefix (default: "attn_")

    Returns:
        ONNX model with self-attention pattern.
    """
    head_dim = hidden_size // num_heads
    nodes = []
    initializers = []
    rng = np.random.RandomState(42)

    # Input
    input_info = helper.make_tensor_value_info(
        f"{prefix}input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )

    # Q, K, V weights and biases
    for name in ["q", "k", "v"]:
        weight = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        bias = np.zeros(hidden_size, dtype=np.float32)
        initializers.append(numpy_helper.from_array(weight, f"{prefix}w_{name}"))
        initializers.append(numpy_helper.from_array(bias, f"{prefix}b_{name}"))

    # Q projection: X @ Wq + Bq
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{prefix}input", f"{prefix}w_q"],
            [f"{prefix}q_mm"],
            name=f"{prefix}q_matmul",
        )
    )
    nodes.append(
        helper.make_node(
            "Add",
            [f"{prefix}q_mm", f"{prefix}b_q"],
            [f"{prefix}q"],
            name=f"{prefix}q_add",
        )
    )

    # K projection
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{prefix}input", f"{prefix}w_k"],
            [f"{prefix}k_mm"],
            name=f"{prefix}k_matmul",
        )
    )
    nodes.append(
        helper.make_node(
            "Add",
            [f"{prefix}k_mm", f"{prefix}b_k"],
            [f"{prefix}k"],
            name=f"{prefix}k_add",
        )
    )

    # V projection
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{prefix}input", f"{prefix}w_v"],
            [f"{prefix}v_mm"],
            name=f"{prefix}v_matmul",
        )
    )
    nodes.append(
        helper.make_node(
            "Add",
            [f"{prefix}v_mm", f"{prefix}b_v"],
            [f"{prefix}v"],
            name=f"{prefix}v_add",
        )
    )

    # Reshape Q, K, V to [B, S, num_heads, head_dim]
    shape_4d = np.array([batch_size, seq_len, num_heads, head_dim], dtype=np.int64)
    initializers.append(numpy_helper.from_array(shape_4d, f"{prefix}shape_4d"))

    for name in ["q", "k", "v"]:
        nodes.append(
            helper.make_node(
                "Reshape",
                [f"{prefix}{name}", f"{prefix}shape_4d"],
                [f"{prefix}{name}_4d"],
                name=f"{prefix}{name}_reshape",
            )
        )

    # Transpose to [B, num_heads, S, head_dim]
    for name in ["q", "k", "v"]:
        nodes.append(
            helper.make_node(
                "Transpose",
                [f"{prefix}{name}_4d"],
                [f"{prefix}{name}_t"],
                name=f"{prefix}{name}_transpose",
                perm=[0, 2, 1, 3],
            )
        )

    # Attention: Softmax(Q @ K^T / sqrt(d)) @ V

    # K transpose for Q @ K^T
    nodes.append(
        helper.make_node(
            "Transpose",
            [f"{prefix}k_t"],
            [f"{prefix}k_t_t"],
            name=f"{prefix}k_transpose2",
            perm=[0, 1, 3, 2],
        )
    )
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{prefix}q_t", f"{prefix}k_t_t"],
            [f"{prefix}qk"],
            name=f"{prefix}qk_matmul",
        )
    )

    # Scale by 1/sqrt(head_dim)
    scale = np.array([1.0 / math.sqrt(head_dim)], dtype=np.float32)
    initializers.append(numpy_helper.from_array(scale, f"{prefix}scale"))
    nodes.append(
        helper.make_node(
            "Mul",
            [f"{prefix}qk", f"{prefix}scale"],
            [f"{prefix}qk_scaled"],
            name=f"{prefix}scale_mul",
        )
    )

    # Softmax
    nodes.append(
        helper.make_node(
            "Softmax",
            [f"{prefix}qk_scaled"],
            [f"{prefix}attn_weights"],
            name=f"{prefix}softmax",
            axis=-1,
        )
    )

    # Attention @ V
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{prefix}attn_weights", f"{prefix}v_t"],
            [f"{prefix}attn_out_4d"],
            name=f"{prefix}attn_v_matmul",
        )
    )

    # Transpose back to [B, S, num_heads, head_dim]
    nodes.append(
        helper.make_node(
            "Transpose",
            [f"{prefix}attn_out_4d"],
            [f"{prefix}attn_out_t"],
            name=f"{prefix}out_transpose",
            perm=[0, 2, 1, 3],
        )
    )

    # Reshape to [B, S, hidden_size]
    shape_3d = np.array([batch_size, seq_len, hidden_size], dtype=np.int64)
    initializers.append(numpy_helper.from_array(shape_3d, f"{prefix}shape_3d"))
    nodes.append(
        helper.make_node(
            "Reshape",
            [f"{prefix}attn_out_t", f"{prefix}shape_3d"],
            [f"{prefix}output"],
            name=f"{prefix}out_reshape",
        )
    )

    # Output
    output_info = helper.make_tensor_value_info(
        f"{prefix}output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )

    graph = helper.make_graph(
        nodes,
        "self_attention_pattern",
        [input_info],
        [output_info],
        initializer=initializers,
    )
    return make_compatible_model(graph)


def create_gqa_model(
    batch_size: int = 1,
    seq_len: int = 128,
    hidden_size: int = 1024,
    num_heads: int = 32,
    kv_num_heads: int = 8,
    prefix: str = "gqa_",
) -> onnx.ModelProto:
    """Create GQA pattern for GroupQueryAttention fusion testing.

    Pattern (LLaMA-style):
        X → MatMul(Wq) → Q [B, S, num_heads * head_dim]
        X → MatMul(Wk) → K [B, S, kv_num_heads * head_dim]  # Fewer KV heads!
        X → MatMul(Wv) → V [B, S, kv_num_heads * head_dim]
        Q, K, V → GQA calculation → Output

    Key: num_heads > kv_num_heads (e.g., 32 Q heads, 8 KV heads)

    Args:
        batch_size: Batch dimension (default: 1)
        seq_len: Sequence length (default: 128)
        hidden_size: Hidden dimension (default: 1024)
        num_heads: Number of query heads (default: 32)
        kv_num_heads: Number of key/value heads (default: 8)
        prefix: Node name prefix (default: "gqa_")

    Returns:
        ONNX model with GQA pattern.
    """
    head_dim = hidden_size // num_heads
    q_size = num_heads * head_dim
    kv_size = kv_num_heads * head_dim

    nodes = []
    initializers = []
    rng = np.random.RandomState(42)

    # Input
    input_info = helper.make_tensor_value_info(
        f"{prefix}input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )

    # Q weight (full size)
    w_q = rng.randn(hidden_size, q_size).astype(np.float32) * 0.02
    initializers.append(numpy_helper.from_array(w_q, f"{prefix}w_q"))

    # K, V weights (reduced size for GQA)
    w_k = rng.randn(hidden_size, kv_size).astype(np.float32) * 0.02
    w_v = rng.randn(hidden_size, kv_size).astype(np.float32) * 0.02
    initializers.append(numpy_helper.from_array(w_k, f"{prefix}w_k"))
    initializers.append(numpy_helper.from_array(w_v, f"{prefix}w_v"))

    # Projections
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{prefix}input", f"{prefix}w_q"],
            [f"{prefix}q"],
            name=f"{prefix}q_matmul",
        )
    )
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{prefix}input", f"{prefix}w_k"],
            [f"{prefix}k"],
            name=f"{prefix}k_matmul",
        )
    )
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{prefix}input", f"{prefix}w_v"],
            [f"{prefix}v"],
            name=f"{prefix}v_matmul",
        )
    )

    # Reshape Q to [B, S, num_heads, head_dim]
    q_shape = np.array([batch_size, seq_len, num_heads, head_dim], dtype=np.int64)
    initializers.append(numpy_helper.from_array(q_shape, f"{prefix}q_shape"))
    nodes.append(
        helper.make_node(
            "Reshape",
            [f"{prefix}q", f"{prefix}q_shape"],
            [f"{prefix}q_4d"],
            name=f"{prefix}q_reshape",
        )
    )

    # Reshape K, V to [B, S, kv_num_heads, head_dim]
    kv_shape = np.array([batch_size, seq_len, kv_num_heads, head_dim], dtype=np.int64)
    initializers.append(numpy_helper.from_array(kv_shape, f"{prefix}kv_shape"))
    for name in ["k", "v"]:
        nodes.append(
            helper.make_node(
                "Reshape",
                [f"{prefix}{name}", f"{prefix}kv_shape"],
                [f"{prefix}{name}_4d"],
                name=f"{prefix}{name}_reshape",
            )
        )

    # Transpose to [B, heads, S, head_dim]
    nodes.append(
        helper.make_node(
            "Transpose",
            [f"{prefix}q_4d"],
            [f"{prefix}q_t"],
            name=f"{prefix}q_transpose",
            perm=[0, 2, 1, 3],
        )
    )
    for name in ["k", "v"]:
        nodes.append(
            helper.make_node(
                "Transpose",
                [f"{prefix}{name}_4d"],
                [f"{prefix}{name}_t"],
                name=f"{prefix}{name}_transpose",
                perm=[0, 2, 1, 3],
            )
        )

    # Tile K and V to match Q heads
    repeat_factor = num_heads // kv_num_heads
    tile_shape = np.array([1, repeat_factor, 1, 1], dtype=np.int64)
    initializers.append(numpy_helper.from_array(tile_shape, f"{prefix}tile"))

    for name in ["k", "v"]:
        nodes.append(
            helper.make_node(
                "Tile",
                [f"{prefix}{name}_t", f"{prefix}tile"],
                [f"{prefix}{name}_expanded"],
                name=f"{prefix}{name}_tile",
            )
        )

    # Attention computation
    nodes.append(
        helper.make_node(
            "Transpose",
            [f"{prefix}k_expanded"],
            [f"{prefix}k_t_t"],
            name=f"{prefix}k_transpose2",
            perm=[0, 1, 3, 2],
        )
    )
    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{prefix}q_t", f"{prefix}k_t_t"],
            [f"{prefix}qk"],
            name=f"{prefix}qk_matmul",
        )
    )

    scale = np.array([1.0 / math.sqrt(head_dim)], dtype=np.float32)
    initializers.append(numpy_helper.from_array(scale, f"{prefix}scale"))
    nodes.append(
        helper.make_node(
            "Mul",
            [f"{prefix}qk", f"{prefix}scale"],
            [f"{prefix}qk_scaled"],
            name=f"{prefix}scale_mul",
        )
    )

    nodes.append(
        helper.make_node(
            "Softmax",
            [f"{prefix}qk_scaled"],
            [f"{prefix}attn_weights"],
            name=f"{prefix}softmax",
            axis=-1,
        )
    )

    nodes.append(
        helper.make_node(
            "MatMul",
            [f"{prefix}attn_weights", f"{prefix}v_expanded"],
            [f"{prefix}attn_out_4d"],
            name=f"{prefix}attn_v_matmul",
        )
    )

    # Output reshape
    nodes.append(
        helper.make_node(
            "Transpose",
            [f"{prefix}attn_out_4d"],
            [f"{prefix}attn_out_t"],
            name=f"{prefix}out_transpose",
            perm=[0, 2, 1, 3],
        )
    )

    out_shape = np.array([batch_size, seq_len, q_size], dtype=np.int64)
    initializers.append(numpy_helper.from_array(out_shape, f"{prefix}out_shape"))
    nodes.append(
        helper.make_node(
            "Reshape",
            [f"{prefix}attn_out_t", f"{prefix}out_shape"],
            [f"{prefix}output"],
            name=f"{prefix}out_reshape",
        )
    )

    output_info = helper.make_tensor_value_info(
        f"{prefix}output", TensorProto.FLOAT, [batch_size, seq_len, q_size]
    )

    graph = helper.make_graph(
        nodes,
        "gqa_pattern",
        [input_info],
        [output_info],
        initializer=initializers,
    )
    return make_compatible_model(graph)


# =============================================================================
# GROUPNORM PATTERNS
# =============================================================================


def create_groupnorm_model(
    batch_size: int = 1,
    channels: int = 64,
    height: int = 32,
    width: int = 32,
    num_groups: int = 8,
    prefix: str = "gn_",
) -> onnx.ModelProto:
    """Create GroupNorm pattern for layout testing.

    Pattern (NCHW input):
        X [B, C, H, W] → GroupNorm calculation → Output

    Used to test group_norm_channels_last control option.

    Args:
        batch_size: Batch dimension (default: 1)
        channels: Number of channels (default: 64)
        height: Height dimension (default: 32)
        width: Width dimension (default: 32)
        num_groups: Number of groups (default: 8)
        prefix: Node name prefix (default: "gn_")

    Returns:
        ONNX model with GroupNorm pattern.
    """
    nodes = []
    initializers = []

    # Input (NCHW)
    input_info = helper.make_tensor_value_info(
        f"{prefix}input", TensorProto.FLOAT, [batch_size, channels, height, width]
    )

    # GroupNorm parameters
    gamma = np.ones(channels, dtype=np.float32)
    beta = np.zeros(channels, dtype=np.float32)
    initializers.append(numpy_helper.from_array(gamma, f"{prefix}gamma"))
    initializers.append(numpy_helper.from_array(beta, f"{prefix}beta"))

    # Reshape for GroupNorm: [B, C, H, W] → [B, G, C//G, H, W]
    channels_per_group = channels // num_groups
    gn_shape = np.array([batch_size, num_groups, channels_per_group, height, width], dtype=np.int64)
    initializers.append(numpy_helper.from_array(gn_shape, f"{prefix}gn_shape"))

    nodes.append(
        helper.make_node(
            "Reshape",
            [f"{prefix}input", f"{prefix}gn_shape"],
            [f"{prefix}reshaped"],
            name=f"{prefix}reshape1",
        )
    )

    # InstanceNorm-like computation per group
    nodes.append(
        helper.make_node(
            "ReduceMean",
            [f"{prefix}reshaped"],
            [f"{prefix}mean"],
            name=f"{prefix}mean",
            axes=[2, 3, 4],
            keepdims=1,
        )
    )

    nodes.append(
        helper.make_node(
            "Sub",
            [f"{prefix}reshaped", f"{prefix}mean"],
            [f"{prefix}centered"],
            name=f"{prefix}sub",
        )
    )

    two = np.array([2.0], dtype=np.float32)
    eps = np.array([1e-5], dtype=np.float32)
    initializers.append(numpy_helper.from_array(two, f"{prefix}two"))
    initializers.append(numpy_helper.from_array(eps, f"{prefix}eps"))

    nodes.append(
        helper.make_node(
            "Pow",
            [f"{prefix}centered", f"{prefix}two"],
            [f"{prefix}squared"],
            name=f"{prefix}pow",
        )
    )

    nodes.append(
        helper.make_node(
            "ReduceMean",
            [f"{prefix}squared"],
            [f"{prefix}var"],
            name=f"{prefix}var",
            axes=[2, 3, 4],
            keepdims=1,
        )
    )

    nodes.append(
        helper.make_node(
            "Add",
            [f"{prefix}var", f"{prefix}eps"],
            [f"{prefix}var_eps"],
            name=f"{prefix}add_eps",
        )
    )

    nodes.append(
        helper.make_node(
            "Sqrt",
            [f"{prefix}var_eps"],
            [f"{prefix}std"],
            name=f"{prefix}sqrt",
        )
    )

    nodes.append(
        helper.make_node(
            "Div",
            [f"{prefix}centered", f"{prefix}std"],
            [f"{prefix}normalized"],
            name=f"{prefix}div",
        )
    )

    # Reshape back to [B, C, H, W]
    out_shape = np.array([batch_size, channels, height, width], dtype=np.int64)
    initializers.append(numpy_helper.from_array(out_shape, f"{prefix}out_shape"))

    nodes.append(
        helper.make_node(
            "Reshape",
            [f"{prefix}normalized", f"{prefix}out_shape"],
            [f"{prefix}reshaped_back"],
            name=f"{prefix}reshape2",
        )
    )

    # Apply gamma and beta
    gamma_shape = np.array([1, channels, 1, 1], dtype=np.int64)
    initializers.append(numpy_helper.from_array(gamma_shape, f"{prefix}gamma_shape"))

    nodes.append(
        helper.make_node(
            "Reshape",
            [f"{prefix}gamma", f"{prefix}gamma_shape"],
            [f"{prefix}gamma_4d"],
            name=f"{prefix}gamma_reshape",
        )
    )

    nodes.append(
        helper.make_node(
            "Reshape",
            [f"{prefix}beta", f"{prefix}gamma_shape"],
            [f"{prefix}beta_4d"],
            name=f"{prefix}beta_reshape",
        )
    )

    nodes.append(
        helper.make_node(
            "Mul",
            [f"{prefix}reshaped_back", f"{prefix}gamma_4d"],
            [f"{prefix}scaled"],
            name=f"{prefix}mul_gamma",
        )
    )

    nodes.append(
        helper.make_node(
            "Add",
            [f"{prefix}scaled", f"{prefix}beta_4d"],
            [f"{prefix}output"],
            name=f"{prefix}add_beta",
        )
    )

    output_info = helper.make_tensor_value_info(
        f"{prefix}output", TensorProto.FLOAT, [batch_size, channels, height, width]
    )

    graph = helper.make_graph(
        nodes,
        "groupnorm_pattern",
        [input_info],
        [output_info],
        initializer=initializers,
    )
    return make_compatible_model(graph)


# =============================================================================
# PATTERN REGISTRY (for documentation and CLI)
# =============================================================================

FUSION_PATTERNS = {
    "self_attention": {
        "builder": create_self_attention_model,
        "description": "Self-attention pattern → Attention or MultiHeadAttention",
        "test_prefix": "pfc01_attention_",
    },
    "gqa": {
        "builder": create_gqa_model,
        "description": "GQA pattern → GroupQueryAttention (32 Q heads, 8 KV heads)",
        "test_prefix": "pfc03_gqa_",
    },
    "groupnorm": {
        "builder": create_groupnorm_model,
        "description": "GroupNorm pattern → layout testing (NHWC/NCHW)",
        "test_prefix": "pfc05_groupnorm_layout_",
    },
}


# =============================================================================
# CLI ENTRY POINT (optional - for manual testing)
# =============================================================================


def main() -> None:
    """Generate all fusion pattern ONNX files to temp directory."""
    from pathlib import Path

    output_dir = Path(__file__).parent.parent.parent.parent / "temp" / "fusion_test_patterns"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ORT FUSION OPTIMIZATION TEST PATTERN GENERATOR")
    print("=" * 70)

    for name, config in FUSION_PATTERNS.items():
        output_path = output_dir / f"{name}_pattern.onnx"

        print(f"\nGenerating: {name}")
        print(f"  Description: {config['description']}")

        model = config["builder"]()
        onnx.checker.check_model(model)
        onnx.save(model, str(output_path))

        print(f"  Saved: {output_path}")
        print(f"  Nodes: {len(model.graph.node)}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
