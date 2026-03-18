# Copyright (c) 2024 BYOM Authors. All rights reserved.
# Licensed under the MIT License.
"""Attention pattern builders for ORT AttentionFusion tests.

This module builds ONNX graphs that match the BERT-style AttentionFusion pattern
as implemented in ONNX Runtime's attention_fusion.cc.

The pattern requires:
1. LayerNormalization as entry point with output edges to 3 MatMuls and 1 Add
2. Q/K/V projection paths: LayerNorm -> MatMul -> Add -> Reshape -> Transpose
3. Attention computation: Q @ K^T -> Div(scale) -> Mask -> Softmax -> @ V
4. Output path: Transpose -> Reshape -> MatMul -> Add -> Add(skip connection)

Reference: onnxruntime/core/optimizer/attention_fusion.cc
"""

import math

import numpy as np
from onnx import TensorProto, helper, numpy_helper


def attention_builder():
    """Create ONNX model that matches ORT's AttentionFusion pattern.

    This function creates a model structure compatible with ORT's AttentionFusion:

    Key requirements for ORT's AttentionFusion pattern matcher:
    1. Two inputs: hidden states [B,S,H] + int32 mask [B,S]
    2. Mask processing: Unsqueeze->Unsqueeze->Cast->Sub->Mul chain
    3. LayerNorm output feeds 3 MatMuls (Q/K/V) + 1 Add (skip connection)
    4. Div for scaling (not Mul)
    5. Softmax with explicit axis=3

    Uses opset 9 for maximum ORT compatibility (ORT accepts LayerNorm in opset 9).
    Note: ONNX checker is skipped as opset 9 doesn't officially support LayerNorm.

    Returns:
        ONNX ModelProto that triggers AttentionFusion optimization (35->6 nodes)
    """
    # Dimensions matching ORT test file
    batch_size = 1
    seq_len = 3
    hidden_size = 8
    num_heads = 2
    head_size = hidden_size // num_heads  # 4

    rng = np.random.RandomState(42)

    # =========================================================================
    # INITIALIZERS
    # =========================================================================
    initializers = []

    # LayerNorm weights
    ln_weight = np.ones(hidden_size, dtype=np.float32)
    ln_bias = np.zeros(hidden_size, dtype=np.float32)
    initializers.append(numpy_helper.from_array(ln_weight, "ln_weight"))
    initializers.append(numpy_helper.from_array(ln_bias, "ln_bias"))

    # Q/K/V projection weights [hidden_size, hidden_size]
    wq = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
    wk = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
    wv = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
    initializers.append(numpy_helper.from_array(wq, "wq"))
    initializers.append(numpy_helper.from_array(wk, "wk"))
    initializers.append(numpy_helper.from_array(wv, "wv"))

    # Q/K/V projection biases [hidden_size]
    bq = np.zeros(hidden_size, dtype=np.float32)
    bk = np.zeros(hidden_size, dtype=np.float32)
    bv = np.zeros(hidden_size, dtype=np.float32)
    initializers.append(numpy_helper.from_array(bq, "bq"))
    initializers.append(numpy_helper.from_array(bk, "bk"))
    initializers.append(numpy_helper.from_array(bv, "bv"))

    # Output projection weights
    wo = rng.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
    bo = np.zeros(hidden_size, dtype=np.float32)
    initializers.append(numpy_helper.from_array(wo, "wo"))
    initializers.append(numpy_helper.from_array(bo, "bo"))

    # =========================================================================
    # NODES
    # =========================================================================
    nodes = []

    # --- LayerNormalization (produces ln_out which feeds Q/K/V MatMuls + skip Add) ---
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            inputs=["input_1", "ln_weight", "ln_bias"],
            outputs=["ln_out"],
            name="layernorm",
            epsilon=1e-5,
            axis=-1,
        )
    )

    # --- Mask processing: Unsqueeze -> Unsqueeze -> Cast -> Sub -> Mul ---
    # input_2 is [B, S] int32, needs to become [B, 1, 1, S] float for attention mask
    # Use opset 9 style Unsqueeze with axes as attribute (required for AttentionFusion)
    nodes.append(
        helper.make_node(
            "Unsqueeze",
            inputs=["input_2"],
            outputs=["mask_unsq1"],
            name="mask_unsqueeze1",
            axes=[1],
        )
    )
    nodes.append(
        helper.make_node(
            "Unsqueeze",
            inputs=["mask_unsq1"],
            outputs=["mask_unsq2"],
            name="mask_unsqueeze2",
            axes=[2],
        )
    )
    nodes.append(
        helper.make_node(
            "Cast",
            inputs=["mask_unsq2"],
            outputs=["mask_float"],
            name="mask_cast",
            to=TensorProto.FLOAT,
        )
    )
    # Constant for Sub: 1.0
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_one"],
            name="const_one",
            value=helper.make_tensor("one", TensorProto.FLOAT, [], [1.0]),
        )
    )
    nodes.append(
        helper.make_node(
            "Sub",
            inputs=["const_one", "mask_float"],
            outputs=["mask_sub"],
            name="mask_sub",
        )
    )
    # Constant for Mul: -10000.0 (large negative for masked positions)
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["const_neg"],
            name="const_neg",
            value=helper.make_tensor("neg", TensorProto.FLOAT, [], [-10000.0]),
        )
    )
    nodes.append(
        helper.make_node(
            "Mul",
            inputs=["mask_sub", "const_neg"],
            outputs=["attention_mask"],
            name="mask_mul",
        )
    )

    # --- Q projection: MatMul -> Add -> Reshape -> Transpose ---
    nodes.append(
        helper.make_node("MatMul", ["ln_out", "wq"], ["q_matmul"], name="q_matmul")
    )
    nodes.append(helper.make_node("Add", ["q_matmul", "bq"], ["q_add"], name="q_add"))
    # Reshape shape via SEPARATE Constant node (ORT pattern matcher requires separate shapes)
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["q_shape"],
            name="q_shape_const",
            value=helper.make_tensor(
                "q_shape", TensorProto.INT64, [4], [0, 0, num_heads, head_size]
            ),
        )
    )
    nodes.append(
        helper.make_node("Reshape", ["q_add", "q_shape"], ["q_reshape"], name="q_reshape")
    )
    nodes.append(
        helper.make_node(
            "Transpose",
            ["q_reshape"],
            ["q_trans"],
            name="q_transpose",
            perm=[0, 2, 1, 3],
        )
    )

    # --- K projection: MatMul -> Add -> Reshape -> Transpose (special perm for K^T) ---
    nodes.append(
        helper.make_node("MatMul", ["ln_out", "wk"], ["k_matmul"], name="k_matmul")
    )
    nodes.append(helper.make_node("Add", ["k_matmul", "bk"], ["k_add"], name="k_add"))
    # Separate Constant for K reshape
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["k_shape"],
            name="k_shape_const",
            value=helper.make_tensor(
                "k_shape", TensorProto.INT64, [4], [0, 0, num_heads, head_size]
            ),
        )
    )
    nodes.append(
        helper.make_node("Reshape", ["k_add", "k_shape"], ["k_reshape"], name="k_reshape")
    )
    nodes.append(
        helper.make_node(
            "Transpose",
            ["k_reshape"],
            ["k_trans"],
            name="k_transpose",
            perm=[0, 2, 3, 1],  # K^T for Q @ K^T
        )
    )

    # --- V projection: MatMul -> Add -> Reshape -> Transpose ---
    nodes.append(
        helper.make_node("MatMul", ["ln_out", "wv"], ["v_matmul"], name="v_matmul")
    )
    nodes.append(helper.make_node("Add", ["v_matmul", "bv"], ["v_add"], name="v_add"))
    # Separate Constant for V reshape
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["v_shape"],
            name="v_shape_const",
            value=helper.make_tensor(
                "v_shape", TensorProto.INT64, [4], [0, 0, num_heads, head_size]
            ),
        )
    )
    nodes.append(
        helper.make_node("Reshape", ["v_add", "v_shape"], ["v_reshape"], name="v_reshape")
    )
    nodes.append(
        helper.make_node(
            "Transpose",
            ["v_reshape"],
            ["v_trans"],
            name="v_transpose",
            perm=[0, 2, 1, 3],
        )
    )

    # --- Attention: Q @ K^T -> Div(scale) -> Add(mask) -> Softmax -> @ V ---
    nodes.append(
        helper.make_node("MatMul", ["q_trans", "k_trans"], ["qk"], name="qk_matmul")
    )
    # Scale factor: sqrt(head_size) for Div
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["scale"],
            name="scale_const",
            value=helper.make_tensor(
                "scale", TensorProto.FLOAT, [], [math.sqrt(head_size)]
            ),
        )
    )
    nodes.append(helper.make_node("Div", ["qk", "scale"], ["qk_scaled"], name="qk_div"))
    nodes.append(
        helper.make_node(
            "Add", ["qk_scaled", "attention_mask"], ["qk_masked"], name="mask_add"
        )
    )
    nodes.append(
        helper.make_node(
            "Softmax", ["qk_masked"], ["attn_probs"], name="softmax", axis=3
        )
    )
    nodes.append(
        helper.make_node("MatMul", ["attn_probs", "v_trans"], ["attn_out"], name="attn_v")
    )

    # --- Output: Transpose -> Reshape -> MatMul -> Add -> Add(skip) ---
    nodes.append(
        helper.make_node(
            "Transpose",
            ["attn_out"],
            ["out_trans"],
            name="out_transpose",
            perm=[0, 2, 1, 3],
        )
    )
    # Output reshape shape via Constant [0, 0, hidden_size]
    nodes.append(
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["out_shape"],
            name="out_shape",
            value=helper.make_tensor(
                "out_shape", TensorProto.INT64, [3], [0, 0, hidden_size]
            ),
        )
    )
    nodes.append(
        helper.make_node(
            "Reshape", ["out_trans", "out_shape"], ["out_reshape"], name="out_reshape"
        )
    )
    nodes.append(
        helper.make_node("MatMul", ["out_reshape", "wo"], ["out_matmul"], name="out_matmul")
    )
    nodes.append(
        helper.make_node("Add", ["out_matmul", "bo"], ["out_add"], name="out_add")
    )
    # Skip connection: add back ln_out (this is the 4th edge from LayerNorm output)
    nodes.append(
        helper.make_node("Add", ["out_add", "ln_out"], ["output"], name="skip_add")
    )

    # =========================================================================
    # GRAPH AND MODEL
    # =========================================================================
    # Two inputs: hidden states + int32 mask
    input_1 = helper.make_tensor_value_info(
        "input_1", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )
    input_2 = helper.make_tensor_value_info(
        "input_2", TensorProto.INT32, [batch_size, seq_len]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )

    graph = helper.make_graph(
        nodes=nodes,
        name="attention_fusion_graph",
        inputs=[input_1, input_2],
        outputs=[output],
        initializer=initializers,
    )

    # Opset 9 for ORT AttentionFusion compatibility
    # Note: LayerNorm is not in opset 9 spec but ORT accepts it
    model = helper.make_model(
        graph,
        producer_name="modelkit_test",
        opset_imports=[helper.make_opsetid("", 9)],
    )
    model.ir_version = 6

    # Skip onnx.checker - opset 9 doesn't officially support LayerNorm
    # but ORT handles it correctly

    return model


def multi_head_attention_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Multi-Head Attention pattern with separate Q/K/V heads.

    P4-02: Tests multi-head-attention pattern for fusion (for FusionPipe).
    Simplified multi-head: separate Q/K/V with reshape for multi-head structure.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Q/K/V projection weights (64 -> 64)
    initializers.append(
        numpy_helper.from_array(rng.randn(64, 64).astype(np.float32) * 0.1, f"{prefix}wq")
    )
    initializers.append(
        numpy_helper.from_array(rng.randn(64, 64).astype(np.float32) * 0.1, f"{prefix}wk")
    )
    initializers.append(
        numpy_helper.from_array(rng.randn(64, 64).astype(np.float32) * 0.1, f"{prefix}wv")
    )
    # Output projection
    initializers.append(
        numpy_helper.from_array(rng.randn(64, 64).astype(np.float32) * 0.1, f"{prefix}wo")
    )
    # Multi-head reshape: [1, 64] -> [1, 4, 16] (4 heads, 16 dims each)
    initializers.append(
        numpy_helper.from_array(np.array([1, 4, 16], dtype=np.int64), f"{prefix}head_shape")
    )
    # Restore shape: [1, 4, 16] -> [1, 64]
    initializers.append(
        numpy_helper.from_array(np.array([1, 64], dtype=np.int64), f"{prefix}restore_shape")
    )
    # Scale factor
    initializers.append(
        numpy_helper.from_array(
            np.array([1.0 / math.sqrt(16)], dtype=np.float32), f"{prefix}scale"
        )
    )

    return [
        # Q = MatMul(input, Wq) -> Reshape to multi-head
        helper.make_node(
            "MatMul", [input_name, f"{prefix}wq"], [f"{prefix}q_flat"], name=f"{prefix}matmul_q"
        ),
        helper.make_node(
            "Reshape",
            [f"{prefix}q_flat", f"{prefix}head_shape"],
            [f"{prefix}q"],
            name=f"{prefix}reshape_q",
        ),
        # K = MatMul(input, Wk) -> Reshape to multi-head
        helper.make_node(
            "MatMul", [input_name, f"{prefix}wk"], [f"{prefix}k_flat"], name=f"{prefix}matmul_k"
        ),
        helper.make_node(
            "Reshape",
            [f"{prefix}k_flat", f"{prefix}head_shape"],
            [f"{prefix}k"],
            name=f"{prefix}reshape_k",
        ),
        # V = MatMul(input, Wv) -> Reshape to multi-head
        helper.make_node(
            "MatMul", [input_name, f"{prefix}wv"], [f"{prefix}v_flat"], name=f"{prefix}matmul_v"
        ),
        helper.make_node(
            "Reshape",
            [f"{prefix}v_flat", f"{prefix}head_shape"],
            [f"{prefix}v"],
            name=f"{prefix}reshape_v",
        ),
        # K^T = Transpose(K)
        helper.make_node(
            "Transpose",
            [f"{prefix}k"],
            [f"{prefix}k_t"],
            name=f"{prefix}transpose_k",
            perm=[0, 2, 1],
        ),
        # QK = MatMul(Q, K^T) * scale
        helper.make_node(
            "MatMul",
            [f"{prefix}q", f"{prefix}k_t"],
            [f"{prefix}qk"],
            name=f"{prefix}matmul_qk",
        ),
        helper.make_node(
            "Mul",
            [f"{prefix}qk", f"{prefix}scale"],
            [f"{prefix}scaled_qk"],
            name=f"{prefix}mul_scale",
        ),
        # Attention = Softmax(Scaled_QK)
        helper.make_node(
            "Softmax",
            [f"{prefix}scaled_qk"],
            [f"{prefix}attn"],
            name=f"{prefix}softmax",
            axis=-1,
        ),
        # Context = MatMul(Attention, V)
        helper.make_node(
            "MatMul",
            [f"{prefix}attn", f"{prefix}v"],
            [f"{prefix}context"],
            name=f"{prefix}matmul_av",
        ),
        # Reshape back and project
        helper.make_node(
            "Reshape",
            [f"{prefix}context", f"{prefix}restore_shape"],
            [f"{prefix}context_flat"],
            name=f"{prefix}reshape_out",
        ),
        helper.make_node(
            "MatMul",
            [f"{prefix}context_flat", f"{prefix}wo"],
            [output_name],
            name=f"{prefix}matmul_out",
        ),
    ]


def rotary_embeddings_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Rotary Embeddings (RoPE) pattern.

    P4-03: Tests rotary-embeddings pattern (ORT name: RotaryEmbedding).
    Simplified RoPE: cos/sin position encoding with element-wise rotation.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Position encoding: cos and sin values (simplified)
    initializers.append(
        numpy_helper.from_array(
            rng.uniform(0.8, 1.0, 64).astype(np.float32), f"{prefix}cos_pos"
        )
    )
    initializers.append(
        numpy_helper.from_array(
            rng.uniform(-0.2, 0.2, 64).astype(np.float32), f"{prefix}sin_pos"
        )
    )
    # Reshape for rotation: [1, 64] -> [1, 32, 2]
    initializers.append(
        numpy_helper.from_array(np.array([1, 32, 2], dtype=np.int64), f"{prefix}rot_shape")
    )
    # Restore shape
    initializers.append(
        numpy_helper.from_array(np.array([1, 64], dtype=np.int64), f"{prefix}restore_shape")
    )

    return [
        # Reshape input for rotation: [1, 64] -> [1, 32, 2]
        helper.make_node(
            "Reshape",
            [input_name, f"{prefix}rot_shape"],
            [f"{prefix}reshaped"],
            name=f"{prefix}reshape_in",
        ),
        # Split into two halves along last dimension
        helper.make_node(
            "Split",
            [f"{prefix}reshaped"],
            [f"{prefix}x0", f"{prefix}x1"],
            name=f"{prefix}split",
            axis=2,
        ),
        # Reshape cos/sin to broadcast shape [1, 32, 2]
        helper.make_node(
            "Reshape",
            [f"{prefix}cos_pos", f"{prefix}rot_shape"],
            [f"{prefix}cos"],
            name=f"{prefix}reshape_cos",
        ),
        helper.make_node(
            "Reshape",
            [f"{prefix}sin_pos", f"{prefix}rot_shape"],
            [f"{prefix}sin"],
            name=f"{prefix}reshape_sin",
        ),
        # Split cos/sin for rotation
        helper.make_node(
            "Split",
            [f"{prefix}cos"],
            [f"{prefix}cos0", f"{prefix}cos1"],
            name=f"{prefix}split_cos",
            axis=2,
        ),
        helper.make_node(
            "Split",
            [f"{prefix}sin"],
            [f"{prefix}sin0", f"{prefix}sin1"],
            name=f"{prefix}split_sin",
            axis=2,
        ),
        # Rotation: y0 = x0*cos - x1*sin, y1 = x0*sin + x1*cos
        helper.make_node(
            "Mul",
            [f"{prefix}x0", f"{prefix}cos0"],
            [f"{prefix}x0_cos"],
            name=f"{prefix}mul_x0_cos",
        ),
        helper.make_node(
            "Mul",
            [f"{prefix}x1", f"{prefix}sin0"],
            [f"{prefix}x1_sin"],
            name=f"{prefix}mul_x1_sin",
        ),
        helper.make_node(
            "Sub",
            [f"{prefix}x0_cos", f"{prefix}x1_sin"],
            [f"{prefix}y0"],
            name=f"{prefix}sub_y0",
        ),
        helper.make_node(
            "Mul",
            [f"{prefix}x0", f"{prefix}sin1"],
            [f"{prefix}x0_sin"],
            name=f"{prefix}mul_x0_sin",
        ),
        helper.make_node(
            "Mul",
            [f"{prefix}x1", f"{prefix}cos1"],
            [f"{prefix}x1_cos"],
            name=f"{prefix}mul_x1_cos",
        ),
        helper.make_node(
            "Add",
            [f"{prefix}x0_sin", f"{prefix}x1_cos"],
            [f"{prefix}y1"],
            name=f"{prefix}add_y1",
        ),
        # Concat back
        helper.make_node(
            "Concat",
            [f"{prefix}y0", f"{prefix}y1"],
            [f"{prefix}rotated"],
            name=f"{prefix}concat",
            axis=2,
        ),
        # Reshape back to original shape
        helper.make_node(
            "Reshape",
            [f"{prefix}rotated", f"{prefix}restore_shape"],
            [output_name],
            name=f"{prefix}reshape_out",
        ),
    ]
