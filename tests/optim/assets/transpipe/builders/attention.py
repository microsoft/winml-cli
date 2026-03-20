# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Attention pattern builders for TransformerPipe testing.

These builders create ONNX models with unfused attention patterns that match
what ORT fusion classes expect. The patterns are derived from:
- temp/attn/test_cli_output.onnx (BERT attention)
- temp/attention_variants/unfused/ (various patterns)

Key requirements from ORT FusionAttention:
1. Q/K/V projections: MatMul → Add → Reshape → Transpose
2. Attention scores: MatMul(Q, K^T) → Div(scale) → Add(mask) → Softmax
3. Attention output: MatMul(scores, V) → Transpose → Reshape → MatMul → Add
4. Skip connection with LayerNormalization (NOT SkipLayerNormalization)
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_bert_attention_model(
    batch_size: int = 1,
    seq_len: int = 10,
    hidden_size: int = 16,
    num_heads: int = 2,
    with_attention_mask: bool = True,
    with_skip_connection: bool = True,
) -> onnx.ModelProto:
    """Build a BERT-style attention pattern that FusionAttention can fuse.

    This creates the exact pattern from test_cli_output.onnx:
    - Add → LayerNormalization (input normalization)
    - Q/K/V: MatMul → Add → Reshape → Transpose
    - Attention: MatMul(QK) → Div → Add(mask) → Softmax → MatMul(V)
    - Output: Transpose → Reshape → MatMul → Add
    - Skip: Add → LayerNormalization

    Args:
        batch_size: Batch dimension (default 1)
        seq_len: Sequence length (default 10)
        hidden_size: Hidden dimension (default 16)
        num_heads: Number of attention heads (default 2)
        with_attention_mask: Include attention mask (default True)
        with_skip_connection: Include skip connection (default True)

    Returns:
        ONNX ModelProto with unfused BERT attention pattern
    """
    head_size = hidden_size // num_heads
    assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

    # Initialize weights with random values
    np.random.seed(42)  # For reproducibility
    ln_weight = np.random.randn(hidden_size).astype(np.float32)
    ln_bias = np.random.randn(hidden_size).astype(np.float32)
    q_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    q_bias = np.random.randn(hidden_size).astype(np.float32)
    k_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    k_bias = np.random.randn(hidden_size).astype(np.float32)
    v_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    v_bias = np.random.randn(hidden_size).astype(np.float32)
    out_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    out_bias = np.random.randn(hidden_size).astype(np.float32)

    # Shape constants
    reshape_qkv = np.array([0, 0, num_heads, head_size], dtype=np.int64)
    reshape_out = np.array([0, 0, hidden_size], dtype=np.int64)
    scale = np.array([np.sqrt(head_size)], dtype=np.float32)
    sub_weight = np.array([1.0], dtype=np.float32)
    mul_weight = np.array([-10000.0], dtype=np.float32)
    unsqueeze_axes_1 = np.array([1], dtype=np.int64)
    unsqueeze_axes_2 = np.array([2], dtype=np.int64)

    # Initializers
    initializers = [
        numpy_helper.from_array(ln_weight, "attn_ln_weight"),
        numpy_helper.from_array(ln_bias, "attn_ln_bias"),
        numpy_helper.from_array(q_weight, "attn_q_weight"),
        numpy_helper.from_array(q_bias, "attn_q_bias"),
        numpy_helper.from_array(k_weight, "attn_k_weight"),
        numpy_helper.from_array(k_bias, "attn_k_bias"),
        numpy_helper.from_array(v_weight, "attn_v_weight"),
        numpy_helper.from_array(v_bias, "attn_v_bias"),
        numpy_helper.from_array(out_weight, "attn_out_weight"),
        numpy_helper.from_array(out_bias, "attn_out_bias"),
        numpy_helper.from_array(reshape_qkv, "attn_reshape_qk"),
        numpy_helper.from_array(reshape_out, "attn_reshape_out"),
        numpy_helper.from_array(scale, "attn_div_weight"),
        numpy_helper.from_array(sub_weight, "attn_sub_weight"),
        numpy_helper.from_array(mul_weight, "attn_mul_weight"),
        numpy_helper.from_array(unsqueeze_axes_1, "attn_axes_1"),
        numpy_helper.from_array(unsqueeze_axes_2, "attn_axes_2"),
    ]

    # Inputs
    inputs = [
        helper.make_tensor_value_info(
            "input_1", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        ),
        helper.make_tensor_value_info(
            "input_2", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        ),
    ]
    if with_attention_mask:
        inputs.append(
            helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT64, [batch_size, seq_len]
            )
        )

    nodes = []

    # Add inputs (for skip connection path)
    nodes.append(
        helper.make_node(
            "Add",
            ["input_1", "input_2"],
            ["attn_ln_input"],
            name="attn_add_ln",
        )
    )

    # LayerNormalization
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            ["attn_ln_input", "attn_ln_weight", "attn_ln_bias"],
            ["attn_ln_out"],
            name="attn_layernorm",
            epsilon=1e-5,
            axis=-1,
        )
    )

    # Q projection: MatMul → Add → Reshape → Transpose
    nodes.extend(
        [
            helper.make_node(
                "MatMul", ["attn_ln_out", "attn_q_weight"], ["attn_q_mm"], name="attn_matmul_q"
            ),
            helper.make_node(
                "Add", ["attn_q_mm", "attn_q_bias"], ["attn_q_add"], name="attn_add_q"
            ),
            helper.make_node(
                "Reshape",
                ["attn_q_add", "attn_reshape_qk"],
                ["attn_q_reshape"],
                name="attn_reshape_q",
            ),
            helper.make_node(
                "Transpose",
                ["attn_q_reshape"],
                ["attn_q_trans"],
                name="attn_transpose_q",
                perm=[0, 2, 1, 3],
            ),
        ]
    )

    # K projection: MatMul → Add → Reshape → Transpose (note: K has different transpose)
    nodes.extend(
        [
            helper.make_node(
                "MatMul", ["attn_ln_out", "attn_k_weight"], ["attn_k_mm"], name="attn_matmul_k"
            ),
            helper.make_node(
                "Add", ["attn_k_mm", "attn_k_bias"], ["attn_k_add"], name="attn_add_k"
            ),
            helper.make_node(
                "Reshape",
                ["attn_k_add", "attn_reshape_qk"],
                ["attn_k_reshape"],
                name="attn_reshape_k",
            ),
            helper.make_node(
                "Transpose",
                ["attn_k_reshape"],
                ["attn_k_trans"],
                name="attn_transpose_k",
                perm=[0, 2, 3, 1],
            ),
        ]
    )

    # V projection: MatMul → Add → Reshape → Transpose
    nodes.extend(
        [
            helper.make_node(
                "MatMul", ["attn_ln_out", "attn_v_weight"], ["attn_v_mm"], name="attn_matmul_v"
            ),
            helper.make_node(
                "Add", ["attn_v_mm", "attn_v_bias"], ["attn_v_add"], name="attn_add_v"
            ),
            helper.make_node(
                "Reshape",
                ["attn_v_add", "attn_reshape_qk"],
                ["attn_v_reshape"],
                name="attn_reshape_v",
            ),
            helper.make_node(
                "Transpose",
                ["attn_v_reshape"],
                ["attn_v_trans"],
                name="attn_transpose_v",
                perm=[0, 2, 1, 3],
            ),
        ]
    )

    # Attention mask processing (if enabled)
    if with_attention_mask:
        nodes.extend(
            [
                helper.make_node(
                    "Unsqueeze",
                    ["attention_mask", "attn_axes_1"],
                    ["attn_mask_unsq1"],
                    name="attn_unsqueeze1",
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["attn_mask_unsq1", "attn_axes_2"],
                    ["attn_mask_unsq2"],
                    name="attn_unsqueeze2",
                ),
                helper.make_node(
                    "Cast",
                    ["attn_mask_unsq2"],
                    ["attn_mask_cast"],
                    name="attn_cast_mask",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Sub",
                    ["attn_sub_weight", "attn_mask_cast"],
                    ["attn_mask_sub"],
                    name="attn_sub_mask",
                ),
                helper.make_node(
                    "Mul",
                    ["attn_mask_sub", "attn_mul_weight"],
                    ["attn_mask_out"],
                    name="attn_mul_mask",
                ),
            ]
        )
        mask_output = "attn_mask_out"
    else:
        mask_output = None

    # Attention scores: MatMul(Q, K^T) → Div(scale)
    nodes.extend(
        [
            helper.make_node(
                "MatMul", ["attn_q_trans", "attn_k_trans"], ["attn_qk_mm"], name="attn_matmul_qk"
            ),
            helper.make_node(
                "Div", ["attn_qk_mm", "attn_div_weight"], ["attn_qk_div"], name="attn_div_qk"
            ),
        ]
    )

    # Add mask and softmax
    if with_attention_mask:
        nodes.extend(
            [
                helper.make_node(
                    "Add", ["attn_qk_div", mask_output], ["attn_qk_add"], name="attn_add_qk"
                ),
                helper.make_node(
                    "Softmax", ["attn_qk_add"], ["attn_attn_weights"], name="attn_softmax", axis=3
                ),
            ]
        )
    else:
        nodes.append(
            helper.make_node(
                "Softmax", ["attn_qk_div"], ["attn_attn_weights"], name="attn_softmax", axis=3
            )
        )

    # Attention output: MatMul(scores, V) → Transpose → Reshape
    nodes.extend(
        [
            helper.make_node(
                "MatMul",
                ["attn_attn_weights", "attn_v_trans"],
                ["attn_attn_v"],
                name="attn_matmul_attn_v",
            ),
            helper.make_node(
                "Transpose",
                ["attn_attn_v"],
                ["attn_attn_trans"],
                name="attn_transpose_attn",
                perm=[0, 2, 1, 3],
            ),
            helper.make_node(
                "Reshape",
                ["attn_attn_trans", "attn_reshape_out"],
                ["attn_attn_reshape"],
                name="attn_reshape_attn",
            ),
        ]
    )

    # Output projection: MatMul → Add
    nodes.extend(
        [
            helper.make_node(
                "MatMul",
                ["attn_attn_reshape", "attn_out_weight"],
                ["attn_out_mm"],
                name="attn_matmul_out",
            ),
            helper.make_node(
                "Add", ["attn_out_mm", "attn_out_bias"], ["attn_out_add"], name="attn_add_out"
            ),
        ]
    )

    # Skip connection and final LayerNorm
    if with_skip_connection:
        nodes.extend(
            [
                helper.make_node(
                    "Add", ["attn_out_add", "attn_ln_out"], ["attn_skip_out"], name="attn_add_skip"
                ),
                helper.make_node(
                    "LayerNormalization",
                    ["attn_skip_out", "attn_ln_weight", "attn_ln_bias"],
                    ["output"],
                    name="attn_layernorm2",
                    epsilon=1e-5,
                    axis=-1,
                ),
            ]
        )
    else:
        # Just rename output without skip
        nodes.append(
            helper.make_node("Identity", ["attn_out_add"], ["output"], name="attn_identity_out")
        )

    # Output
    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
    ]

    # Create graph
    graph = helper.make_graph(
        nodes,
        "bert_attention_test",
        inputs,
        outputs,
        initializers,
    )

    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    return model


def build_clip_attention_model(
    batch_size: int = 1,
    seq_len: int = 17,  # 16 patches + 1 CLS token
    hidden_size: int = 64,
    num_heads: int = 2,
) -> onnx.ModelProto:
    """Build a CLIP-style attention pattern that FusionAttentionClip can fuse.

    CLIP attention differs from BERT:
    - No attention mask (vision transformer)
    - Different scale computation pattern
    - Separate Q/K/V without packed QKV

    Args:
        batch_size: Batch dimension (default 1)
        seq_len: Sequence length (default 17)
        hidden_size: Hidden dimension (default 64)
        num_heads: Number of attention heads (default 2)

    Returns:
        ONNX ModelProto with unfused CLIP attention pattern
    """
    head_size = hidden_size // num_heads
    assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

    np.random.seed(42)
    ln_weight = np.random.randn(hidden_size).astype(np.float32)
    ln_bias = np.random.randn(hidden_size).astype(np.float32)
    q_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    q_bias = np.random.randn(hidden_size).astype(np.float32)
    k_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    k_bias = np.random.randn(hidden_size).astype(np.float32)
    v_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    v_bias = np.random.randn(hidden_size).astype(np.float32)
    out_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    out_bias = np.random.randn(hidden_size).astype(np.float32)

    # Shape constants
    reshape_qkv = np.array([batch_size, seq_len, num_heads, head_size], dtype=np.int64)
    reshape_out = np.array([batch_size, seq_len, hidden_size], dtype=np.int64)
    scale = np.array([1.0 / np.sqrt(head_size)], dtype=np.float32)

    initializers = [
        numpy_helper.from_array(ln_weight, "clip_ln_weight"),
        numpy_helper.from_array(ln_bias, "clip_ln_bias"),
        numpy_helper.from_array(q_weight, "clip_q_weight"),
        numpy_helper.from_array(q_bias, "clip_q_bias"),
        numpy_helper.from_array(k_weight, "clip_k_weight"),
        numpy_helper.from_array(k_bias, "clip_k_bias"),
        numpy_helper.from_array(v_weight, "clip_v_weight"),
        numpy_helper.from_array(v_bias, "clip_v_bias"),
        numpy_helper.from_array(out_weight, "clip_out_weight"),
        numpy_helper.from_array(out_bias, "clip_out_bias"),
        numpy_helper.from_array(reshape_qkv, "clip_reshape_qkv"),
        numpy_helper.from_array(reshape_out, "clip_reshape_out"),
        numpy_helper.from_array(scale, "clip_scale"),
    ]

    inputs = [
        helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        ),
    ]

    nodes = []

    # LayerNormalization
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            ["input", "clip_ln_weight", "clip_ln_bias"],
            ["clip_ln_out"],
            name="clip_layernorm",
            epsilon=1e-5,
            axis=-1,
        )
    )

    # Q projection: MatMul → Add → Reshape → Transpose
    nodes.extend(
        [
            helper.make_node(
                "MatMul", ["clip_ln_out", "clip_q_weight"], ["clip_q_mm"], name="clip_matmul_q"
            ),
            helper.make_node(
                "Add", ["clip_q_mm", "clip_q_bias"], ["clip_q_add"], name="clip_add_q"
            ),
            helper.make_node(
                "Reshape",
                ["clip_q_add", "clip_reshape_qkv"],
                ["clip_q_reshape"],
                name="clip_reshape_q",
            ),
            helper.make_node(
                "Transpose",
                ["clip_q_reshape"],
                ["clip_q_trans"],
                name="clip_transpose_q",
                perm=[0, 2, 1, 3],
            ),
        ]
    )

    # K projection
    nodes.extend(
        [
            helper.make_node(
                "MatMul", ["clip_ln_out", "clip_k_weight"], ["clip_k_mm"], name="clip_matmul_k"
            ),
            helper.make_node(
                "Add", ["clip_k_mm", "clip_k_bias"], ["clip_k_add"], name="clip_add_k"
            ),
            helper.make_node(
                "Reshape",
                ["clip_k_add", "clip_reshape_qkv"],
                ["clip_k_reshape"],
                name="clip_reshape_k",
            ),
            helper.make_node(
                "Transpose",
                ["clip_k_reshape"],
                ["clip_k_trans"],
                name="clip_transpose_k",
                perm=[0, 2, 3, 1],
            ),
        ]
    )

    # V projection
    nodes.extend(
        [
            helper.make_node(
                "MatMul", ["clip_ln_out", "clip_v_weight"], ["clip_v_mm"], name="clip_matmul_v"
            ),
            helper.make_node(
                "Add", ["clip_v_mm", "clip_v_bias"], ["clip_v_add"], name="clip_add_v"
            ),
            helper.make_node(
                "Reshape",
                ["clip_v_add", "clip_reshape_qkv"],
                ["clip_v_reshape"],
                name="clip_reshape_v",
            ),
            helper.make_node(
                "Transpose",
                ["clip_v_reshape"],
                ["clip_v_trans"],
                name="clip_transpose_v",
                perm=[0, 2, 1, 3],
            ),
        ]
    )

    # Attention scores: MatMul(Q, K^T) → Mul(scale) → Softmax (no mask for vision)
    nodes.extend(
        [
            helper.make_node(
                "MatMul", ["clip_q_trans", "clip_k_trans"], ["clip_qk_mm"], name="clip_matmul_qk"
            ),
            helper.make_node(
                "Mul", ["clip_qk_mm", "clip_scale"], ["clip_qk_scaled"], name="clip_mul_scale"
            ),
            helper.make_node(
                "Softmax", ["clip_qk_scaled"], ["clip_attn_weights"], name="clip_softmax", axis=-1
            ),
        ]
    )

    # Attention output
    nodes.extend(
        [
            helper.make_node(
                "MatMul",
                ["clip_attn_weights", "clip_v_trans"],
                ["clip_attn_v"],
                name="clip_matmul_attn_v",
            ),
            helper.make_node(
                "Transpose",
                ["clip_attn_v"],
                ["clip_attn_trans"],
                name="clip_transpose_attn",
                perm=[0, 2, 1, 3],
            ),
            helper.make_node(
                "Reshape",
                ["clip_attn_trans", "clip_reshape_out"],
                ["clip_attn_reshape"],
                name="clip_reshape_attn",
            ),
        ]
    )

    # Output projection
    nodes.extend(
        [
            helper.make_node(
                "MatMul",
                ["clip_attn_reshape", "clip_out_weight"],
                ["clip_out_mm"],
                name="clip_matmul_out",
            ),
            helper.make_node(
                "Add", ["clip_out_mm", "clip_out_bias"], ["output"], name="clip_add_out"
            ),
        ]
    )

    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
    ]

    graph = helper.make_graph(
        nodes,
        "clip_attention_test",
        inputs,
        outputs,
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    return model
