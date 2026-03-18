"""Attention pattern builders for FusionPipe testing.

Creates ONNX graphs that match ORT's attention fusion patterns.
Based on: D:/BYOM/ort/onnxruntime/test/python/transformers/bert_model_generator.py

Reference: D:/BYOM/ort/onnxruntime/python/tools/transformers/fusion_attention.py
"""

from __future__ import annotations

import math

import numpy as np
from onnx import ModelProto, TensorProto, helper


def bert_attention_builder(
    input1_name: str,
    input2_name: str,
    mask_name: str,
    output_name: str,
    prefix: str,
    initializers: list,
    hidden_size: int = 16,
    num_heads: int = 2,
) -> list:
    """Create BERT-style attention pattern matching ORT's expected structure.

    Pattern (from ORT bert_model_generator.py):
        input1 + input2 -> Add -> LayerNorm -> Q/K/V projections
        Mask: Unsqueeze -> Unsqueeze -> Cast -> Sub -> Mul
        QK: MatMul(Q, K^T) -> Div -> Add(mask) -> Softmax
        Output: MatMul(attn, V) -> Transpose -> Reshape -> MatMul -> Add
        Residual: output + layernorm_out -> Add -> LayerNorm

    This pattern is recognized by FusionAttention class.

    Args:
        input1_name: Name of first input tensor [batch, seq, hidden]
        input2_name: Name of second input tensor (for skip connection)
        mask_name: Name of attention mask tensor [batch, seq]
        output_name: Name of output tensor [batch, seq, hidden]
        prefix: Unique prefix for node names
        initializers: List to append weight tensors
        hidden_size: Hidden dimension (default: 16)
        num_heads: Number of attention heads (default: 2)

    Returns:
        List of ONNX nodes forming the attention pattern
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))
    head_size = hidden_size // num_heads
    nodes = []

    # Weights
    ln_weight = helper.make_tensor(
        f"{prefix}ln_weight", TensorProto.FLOAT, [hidden_size],
        rng.randn(hidden_size).astype(np.float32),
    )
    ln_bias = helper.make_tensor(
        f"{prefix}ln_bias", TensorProto.FLOAT, [hidden_size],
        rng.randn(hidden_size).astype(np.float32),
    )
    initializers.extend([ln_weight, ln_bias])

    # Q, K, V projection weights
    for proj in ["q", "k", "v"]:
        weight = helper.make_tensor(
            f"{prefix}{proj}_weight", TensorProto.FLOAT, [hidden_size, hidden_size],
            rng.randn(hidden_size, hidden_size).astype(np.float32),
        )
        bias = helper.make_tensor(
            f"{prefix}{proj}_bias", TensorProto.FLOAT, [hidden_size],
            rng.randn(hidden_size).astype(np.float32),
        )
        initializers.extend([weight, bias])

    # Output projection weights
    out_weight = helper.make_tensor(
        f"{prefix}out_weight", TensorProto.FLOAT, [hidden_size, hidden_size],
        rng.randn(hidden_size, hidden_size).astype(np.float32),
    )
    out_bias = helper.make_tensor(
        f"{prefix}out_bias", TensorProto.FLOAT, [hidden_size],
        rng.randn(hidden_size).astype(np.float32),
    )
    initializers.extend([out_weight, out_bias])

    # Reshape constants
    reshape_qk = helper.make_tensor(
        f"{prefix}reshape_qk", TensorProto.INT64, [4],
        np.array([0, 0, num_heads, head_size], dtype=np.int64),
    )
    reshape_out = helper.make_tensor(
        f"{prefix}reshape_out", TensorProto.INT64, [3],
        np.array([0, 0, hidden_size], dtype=np.int64),
    )
    initializers.extend([reshape_qk, reshape_out])

    # Div weight (sqrt(head_size))
    div_weight = helper.make_tensor(
        f"{prefix}div_weight", TensorProto.FLOAT, [1],
        np.array([math.sqrt(head_size)], dtype=np.float32),
    )
    # Mask constants
    sub_weight = helper.make_tensor(
        f"{prefix}sub_weight", TensorProto.FLOAT, [1],
        np.array([1.0], dtype=np.float32),
    )
    mul_weight = helper.make_tensor(
        f"{prefix}mul_weight", TensorProto.FLOAT, [1],
        np.array([-10000.0], dtype=np.float32),
    )
    # Unsqueeze axes
    axes_1 = helper.make_tensor(f"{prefix}axes_1", TensorProto.INT64, [1], [1])
    axes_2 = helper.make_tensor(f"{prefix}axes_2", TensorProto.INT64, [1], [2])
    initializers.extend([div_weight, sub_weight, mul_weight, axes_1, axes_2])

    # === NODES ===

    # 1. Add + LayerNorm (entry point)
    nodes.append(helper.make_node(
        "Add", [input1_name, input2_name], [f"{prefix}ln_input"],
        name=f"{prefix}add_ln",
    ))
    nodes.append(helper.make_node(
        "LayerNormalization",
        [f"{prefix}ln_input", f"{prefix}ln_weight", f"{prefix}ln_bias"],
        [f"{prefix}ln_out"],
        name=f"{prefix}layernorm",
        axis=-1, epsilon=1e-5,
    ))

    # 2. Q projection: MatMul -> Add -> Reshape -> Transpose
    nodes.append(helper.make_node(
        "MatMul", [f"{prefix}ln_out", f"{prefix}q_weight"], [f"{prefix}q_mm"],
        name=f"{prefix}matmul_q",
    ))
    nodes.append(helper.make_node(
        "Add", [f"{prefix}q_mm", f"{prefix}q_bias"], [f"{prefix}q_add"],
        name=f"{prefix}add_q",
    ))
    nodes.append(helper.make_node(
        "Reshape", [f"{prefix}q_add", f"{prefix}reshape_qk"], [f"{prefix}q_reshape"],
        name=f"{prefix}reshape_q",
    ))
    nodes.append(helper.make_node(
        "Transpose", [f"{prefix}q_reshape"], [f"{prefix}q_trans"],
        name=f"{prefix}transpose_q", perm=[0, 2, 1, 3],
    ))

    # 3. K projection: MatMul -> Add -> Reshape -> Transpose (different perm for K^T)
    nodes.append(helper.make_node(
        "MatMul", [f"{prefix}ln_out", f"{prefix}k_weight"], [f"{prefix}k_mm"],
        name=f"{prefix}matmul_k",
    ))
    nodes.append(helper.make_node(
        "Add", [f"{prefix}k_mm", f"{prefix}k_bias"], [f"{prefix}k_add"],
        name=f"{prefix}add_k",
    ))
    nodes.append(helper.make_node(
        "Reshape", [f"{prefix}k_add", f"{prefix}reshape_qk"], [f"{prefix}k_reshape"],
        name=f"{prefix}reshape_k",
    ))
    nodes.append(helper.make_node(
        "Transpose", [f"{prefix}k_reshape"], [f"{prefix}k_trans"],
        name=f"{prefix}transpose_k", perm=[0, 2, 3, 1],  # K^T
    ))

    # 4. V projection: MatMul -> Add -> Reshape -> Transpose
    nodes.append(helper.make_node(
        "MatMul", [f"{prefix}ln_out", f"{prefix}v_weight"], [f"{prefix}v_mm"],
        name=f"{prefix}matmul_v",
    ))
    nodes.append(helper.make_node(
        "Add", [f"{prefix}v_mm", f"{prefix}v_bias"], [f"{prefix}v_add"],
        name=f"{prefix}add_v",
    ))
    nodes.append(helper.make_node(
        "Reshape", [f"{prefix}v_add", f"{prefix}reshape_qk"], [f"{prefix}v_reshape"],
        name=f"{prefix}reshape_v",
    ))
    nodes.append(helper.make_node(
        "Transpose", [f"{prefix}v_reshape"], [f"{prefix}v_trans"],
        name=f"{prefix}transpose_v", perm=[0, 2, 1, 3],
    ))

    # 5. Mask processing: Unsqueeze -> Unsqueeze -> Cast -> Sub -> Mul
    nodes.append(helper.make_node(
        "Unsqueeze", [mask_name, f"{prefix}axes_1"], [f"{prefix}mask_unsq1"],
        name=f"{prefix}unsqueeze1",
    ))
    nodes.append(helper.make_node(
        "Unsqueeze", [f"{prefix}mask_unsq1", f"{prefix}axes_2"], [f"{prefix}mask_unsq2"],
        name=f"{prefix}unsqueeze2",
    ))
    nodes.append(helper.make_node(
        "Cast", [f"{prefix}mask_unsq2"], [f"{prefix}mask_cast"],
        name=f"{prefix}cast_mask", to=TensorProto.FLOAT,
    ))
    nodes.append(helper.make_node(
        "Sub", [f"{prefix}sub_weight", f"{prefix}mask_cast"], [f"{prefix}mask_sub"],
        name=f"{prefix}sub_mask",
    ))
    nodes.append(helper.make_node(
        "Mul", [f"{prefix}mask_sub", f"{prefix}mul_weight"], [f"{prefix}mask_out"],
        name=f"{prefix}mul_mask",
    ))

    # 6. QK attention: MatMul(Q, K^T) -> Div -> Add(mask) -> Softmax
    nodes.append(helper.make_node(
        "MatMul", [f"{prefix}q_trans", f"{prefix}k_trans"], [f"{prefix}qk_mm"],
        name=f"{prefix}matmul_qk",
    ))
    nodes.append(helper.make_node(
        "Div", [f"{prefix}qk_mm", f"{prefix}div_weight"], [f"{prefix}qk_div"],
        name=f"{prefix}div_qk",
    ))
    nodes.append(helper.make_node(
        "Add", [f"{prefix}qk_div", f"{prefix}mask_out"], [f"{prefix}qk_add"],
        name=f"{prefix}add_qk",
    ))
    nodes.append(helper.make_node(
        "Softmax", [f"{prefix}qk_add"], [f"{prefix}attn_weights"],
        name=f"{prefix}softmax", axis=3,
    ))

    # 7. Attention @ V: MatMul -> Transpose -> Reshape
    nodes.append(helper.make_node(
        "MatMul", [f"{prefix}attn_weights", f"{prefix}v_trans"], [f"{prefix}attn_v"],
        name=f"{prefix}matmul_attn_v",
    ))
    nodes.append(helper.make_node(
        "Transpose", [f"{prefix}attn_v"], [f"{prefix}attn_trans"],
        name=f"{prefix}transpose_attn", perm=[0, 2, 1, 3],
    ))
    nodes.append(helper.make_node(
        "Reshape", [f"{prefix}attn_trans", f"{prefix}reshape_out"], [f"{prefix}attn_reshape"],
        name=f"{prefix}reshape_attn",
    ))

    # 8. Output projection: MatMul -> Add
    nodes.append(helper.make_node(
        "MatMul", [f"{prefix}attn_reshape", f"{prefix}out_weight"], [f"{prefix}out_mm"],
        name=f"{prefix}matmul_out",
    ))
    nodes.append(helper.make_node(
        "Add", [f"{prefix}out_mm", f"{prefix}out_bias"], [f"{prefix}out_add"],
        name=f"{prefix}add_out",
    ))

    # 9. Residual + Final LayerNorm: Add(output, ln_out) -> LayerNorm
    nodes.append(helper.make_node(
        "Add", [f"{prefix}out_add", f"{prefix}ln_out"], [f"{prefix}skip_out"],
        name=f"{prefix}add_skip",
    ))
    nodes.append(helper.make_node(
        "LayerNormalization",
        [f"{prefix}skip_out", f"{prefix}ln_weight", f"{prefix}ln_bias"],
        [output_name],
        name=f"{prefix}layernorm2",
        axis=-1, epsilon=1e-5,
    ))

    return nodes


def create_bert_attention_model(
    hidden_size: int = 16,
    num_heads: int = 2,
    seq_len: int = 10,
    batch_size: int = 1,
) -> ModelProto:
    """Create complete ONNX model with BERT attention pattern.

    This model matches ORT's bert_model_generator.py structure and should
    be fusible by FusionAttention.

    Args:
        hidden_size: Hidden dimension (default: 16)
        num_heads: Number of attention heads (default: 2)
        seq_len: Sequence length (default: 10)
        batch_size: Batch size (default: 1)

    Returns:
        Complete ONNX ModelProto ready for fusion testing
    """
    initializers: list = []
    nodes = bert_attention_builder(
        input1_name="input_1",
        input2_name="input_2",
        mask_name="attention_mask",
        output_name="output",
        prefix="attn_",
        initializers=initializers,
        hidden_size=hidden_size,
        num_heads=num_heads,
    )

    # Inputs
    input1 = helper.make_tensor_value_info(
        "input_1", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )
    input2 = helper.make_tensor_value_info(
        "input_2", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )
    mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, [batch_size, seq_len]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )

    graph = helper.make_graph(
        nodes,
        "bert_attention_test",
        [input1, input2, mask],
        [output],
        initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],  # opset 17 for LayerNormalization
    )
    model.ir_version = 8

    return model


def gpt2_attention_builder(
    input_name: str,
    output_name: str,
    prefix: str,
    initializers: list,
    hidden_size: int = 16,
    num_heads: int = 2,
    seq_len: int = 3,
) -> list:
    """Create GPT-2 style causal attention pattern.

    Note: GPT-2 attention has a different structure than BERT.
    This is a simplified version for testing.

    Args:
        input_name: Name of input tensor [batch, seq, hidden]
        output_name: Name of output tensor [batch, seq, hidden]
        prefix: Unique prefix for node names
        initializers: List to append weight tensors
        hidden_size: Hidden dimension (default: 16)
        num_heads: Number of attention heads (default: 2)
        seq_len: Sequence length (default: 3)

    Returns:
        List of ONNX nodes forming the GPT-2 attention pattern
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))
    head_size = hidden_size // num_heads
    nodes = []

    # Combined QKV projection (GPT-2 style)
    qkv_weight = helper.make_tensor(
        f"{prefix}qkv_weight", TensorProto.FLOAT, [hidden_size, 3 * hidden_size],
        rng.randn(hidden_size, 3 * hidden_size).astype(np.float32),
    )
    qkv_bias = helper.make_tensor(
        f"{prefix}qkv_bias", TensorProto.FLOAT, [3 * hidden_size],
        rng.randn(3 * hidden_size).astype(np.float32),
    )
    initializers.extend([qkv_weight, qkv_bias])

    # Split sizes for QKV
    split_sizes = helper.make_tensor(
        f"{prefix}split_sizes", TensorProto.INT64, [3],
        np.array([hidden_size, hidden_size, hidden_size], dtype=np.int64),
    )
    initializers.append(split_sizes)

    # Reshape and other constants
    reshape_shape = helper.make_tensor(
        f"{prefix}reshape_shape", TensorProto.INT64, [4],
        np.array([0, 0, num_heads, head_size], dtype=np.int64),
    )
    reshape_back = helper.make_tensor(
        f"{prefix}reshape_back", TensorProto.INT64, [3],
        np.array([0, 0, hidden_size], dtype=np.int64),
    )
    scale = helper.make_tensor(
        f"{prefix}scale", TensorProto.FLOAT, [],
        [np.float32(np.sqrt(head_size))],
    )
    initializers.extend([reshape_shape, reshape_back, scale])

    # Output projection
    out_weight = helper.make_tensor(
        f"{prefix}out_weight", TensorProto.FLOAT, [hidden_size, hidden_size],
        rng.randn(hidden_size, hidden_size).astype(np.float32),
    )
    out_bias = helper.make_tensor(
        f"{prefix}out_bias", TensorProto.FLOAT, [hidden_size],
        rng.randn(hidden_size).astype(np.float32),
    )
    initializers.extend([out_weight, out_bias])

    # QKV MatMul + Add
    nodes.append(helper.make_node(
        "MatMul", [input_name, f"{prefix}qkv_weight"], [f"{prefix}qkv_matmul"],
        name=f"{prefix}qkv_matmul_node",
    ))
    nodes.append(helper.make_node(
        "Add", [f"{prefix}qkv_matmul", f"{prefix}qkv_bias"], [f"{prefix}qkv_out"],
        name=f"{prefix}qkv_add_node",
    ))

    # Split QKV
    nodes.append(helper.make_node(
        "Split", [f"{prefix}qkv_out", f"{prefix}split_sizes"],
        [f"{prefix}q_split", f"{prefix}k_split", f"{prefix}v_split"],
        name=f"{prefix}qkv_split", axis=-1,
    ))

    # Reshape and transpose Q, K, V
    nodes.extend(
        helper.make_node(
            "Reshape", [f"{prefix}{proj}_split", f"{prefix}reshape_shape"],
            [f"{prefix}{proj}_reshaped"],
            name=f"{prefix}{proj}_reshape", allowzero=0,
        )
        for proj in ["q", "k", "v"]
    )

    # Transpose Q and V
    nodes.extend(
        helper.make_node(
            "Transpose", [f"{prefix}{proj}_reshaped"], [f"{prefix}{proj}_transposed"],
            name=f"{prefix}{proj}_transpose", perm=[0, 2, 1, 3],
        )
        for proj in ["q", "v"]
    )

    # K transpose for Q @ K^T
    nodes.append(helper.make_node(
        "Transpose", [f"{prefix}k_reshaped"], [f"{prefix}k_transposed"],
        name=f"{prefix}k_transpose", perm=[0, 2, 3, 1],
    ))

    # Q @ K^T
    nodes.append(helper.make_node(
        "MatMul", [f"{prefix}q_transposed", f"{prefix}k_transposed"], [f"{prefix}qk"],
        name=f"{prefix}qk_matmul",
    ))

    # Scale
    nodes.append(helper.make_node(
        "Div", [f"{prefix}qk", f"{prefix}scale"], [f"{prefix}qk_scaled"],
        name=f"{prefix}div_scale",
    ))

    # Softmax
    nodes.append(helper.make_node(
        "Softmax", [f"{prefix}qk_scaled"], [f"{prefix}attn_weights"],
        name=f"{prefix}softmax", axis=3,
    ))

    # Attention @ V
    nodes.append(helper.make_node(
        "MatMul", [f"{prefix}attn_weights", f"{prefix}v_transposed"], [f"{prefix}attn_out"],
        name=f"{prefix}attn_v_matmul",
    ))

    # Transpose back
    nodes.append(helper.make_node(
        "Transpose", [f"{prefix}attn_out"], [f"{prefix}attn_transposed"],
        name=f"{prefix}attn_transpose", perm=[0, 2, 1, 3],
    ))

    # Reshape back
    nodes.append(helper.make_node(
        "Reshape", [f"{prefix}attn_transposed", f"{prefix}reshape_back"],
        [f"{prefix}attn_reshaped"],
        name=f"{prefix}attn_reshape_back", allowzero=0,
    ))

    # Output projection
    nodes.append(helper.make_node(
        "MatMul", [f"{prefix}attn_reshaped", f"{prefix}out_weight"], [f"{prefix}out_matmul"],
        name=f"{prefix}out_matmul_node",
    ))
    nodes.append(helper.make_node(
        "Add", [f"{prefix}out_matmul", f"{prefix}out_bias"], [output_name],
        name=f"{prefix}out_add_node",
    ))

    return nodes


def create_gpt2_attention_model(
    hidden_size: int = 16,
    num_heads: int = 2,
    seq_len: int = 10,
    batch_size: int = 1,
) -> ModelProto:
    """Create complete ONNX model with GPT-2 attention pattern.

    Note: This is a simplified GPT-2 attention without causal masking.
    Full GPT-2 fusion requires FusionGptAttention class.

    Args:
        hidden_size: Hidden dimension (default: 16)
        num_heads: Number of attention heads (default: 2)
        seq_len: Sequence length (default: 10)
        batch_size: Batch size (default: 1)

    Returns:
        Complete ONNX ModelProto ready for testing
    """
    initializers: list = []
    nodes = gpt2_attention_builder(
        input_name="input",
        output_name="output",
        prefix="gpt2_attn_",
        initializers=initializers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        seq_len=seq_len,
    )

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )

    graph = helper.make_graph(
        nodes,
        "gpt2_attention_test",
        [input_tensor],
        [output_tensor],
        initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    return model
