"""LayerNorm pattern builders for FusionPipe testing.

Creates ONNX graphs that match ORT's LayerNorm fusion patterns:
- FusionLayerNormalization (decomposed LayerNorm)
- FusionSkipLayerNormalization (Add + LayerNorm)
- FusionSimplifiedLayerNormalization (RMS Normalization)

Reference: D:/BYOM/ort/onnxruntime/python/tools/transformers/fusion_layernorm.py
"""

from __future__ import annotations

import numpy as np
from onnx import ModelProto, TensorProto, helper


def decomposed_layernorm_builder(
    input_name: str,
    output_name: str,
    prefix: str,
    initializers: list,
    hidden_size: int = 64,
) -> list:
    """Create decomposed LayerNorm pattern (9 nodes -> 1 node).

    Pattern: X -> ReduceMean -> Sub -> Pow(2) -> ReduceMean -> Add(eps) ->
             Sqrt -> Div -> Mul(gamma) -> Add(beta)

    This pattern is recognized by FusionLayerNormalization class.

    Args:
        input_name: Name of input tensor
        output_name: Name of output tensor
        prefix: Unique prefix for node names
        initializers: List to append weight tensors
        hidden_size: Hidden dimension

    Returns:
        List of ONNX nodes forming decomposed LayerNorm
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))
    nodes = []

    # Gamma and Beta (scale and bias)
    gamma = helper.make_tensor(
        f"{prefix}gamma",
        TensorProto.FLOAT,
        [hidden_size],
        rng.randn(hidden_size).astype(np.float32),
    )
    beta = helper.make_tensor(
        f"{prefix}beta",
        TensorProto.FLOAT,
        [hidden_size],
        rng.randn(hidden_size).astype(np.float32),
    )
    epsilon = helper.make_tensor(
        f"{prefix}eps",
        TensorProto.FLOAT,
        [],
        [np.float32(1e-5)],
    )
    pow_exp = helper.make_tensor(
        f"{prefix}pow_exp",
        TensorProto.FLOAT,
        [],
        [np.float32(2.0)],
    )
    initializers.extend([gamma, beta, epsilon, pow_exp])

    # ReduceMean (compute mean)
    nodes.append(
        helper.make_node(
            "ReduceMean",
            inputs=[input_name],
            outputs=[f"{prefix}mean"],
            name=f"{prefix}reduce_mean",
            axes=[-1],
            keepdims=1,
        )
    )

    # Sub (x - mean)
    nodes.append(
        helper.make_node(
            "Sub",
            inputs=[input_name, f"{prefix}mean"],
            outputs=[f"{prefix}centered"],
            name=f"{prefix}sub",
        )
    )

    # Pow (squared)
    nodes.append(
        helper.make_node(
            "Pow",
            inputs=[f"{prefix}centered", f"{prefix}pow_exp"],
            outputs=[f"{prefix}squared"],
            name=f"{prefix}pow",
        )
    )

    # ReduceMean (variance)
    nodes.append(
        helper.make_node(
            "ReduceMean",
            inputs=[f"{prefix}squared"],
            outputs=[f"{prefix}var"],
            name=f"{prefix}reduce_mean_var",
            axes=[-1],
            keepdims=1,
        )
    )

    # Add epsilon
    nodes.append(
        helper.make_node(
            "Add",
            inputs=[f"{prefix}var", f"{prefix}eps"],
            outputs=[f"{prefix}var_eps"],
            name=f"{prefix}add_eps",
        )
    )

    # Sqrt
    nodes.append(
        helper.make_node(
            "Sqrt",
            inputs=[f"{prefix}var_eps"],
            outputs=[f"{prefix}std"],
            name=f"{prefix}sqrt",
        )
    )

    # Div (normalize) - CRITICAL: centered must connect to both Pow AND Div
    nodes.append(
        helper.make_node(
            "Div",
            inputs=[f"{prefix}centered", f"{prefix}std"],
            outputs=[f"{prefix}normalized"],
            name=f"{prefix}div",
        )
    )

    # Mul (scale by gamma)
    nodes.append(
        helper.make_node(
            "Mul",
            inputs=[f"{prefix}normalized", f"{prefix}gamma"],
            outputs=[f"{prefix}scaled"],
            name=f"{prefix}mul_gamma",
        )
    )

    # Add (shift by beta)
    nodes.append(
        helper.make_node(
            "Add",
            inputs=[f"{prefix}scaled", f"{prefix}beta"],
            outputs=[output_name],
            name=f"{prefix}add_beta",
        )
    )

    return nodes


def skip_layernorm_builder(
    input_name: str,
    skip_name: str,
    output_name: str,
    prefix: str,
    initializers: list,
    hidden_size: int = 64,
) -> list:
    """Create SkipLayerNorm pattern (Add + LayerNorm -> SkipLayerNormalization).

    Pattern: input + skip -> LayerNormalization

    CRITICAL: Both inputs MUST be dynamic tensors (not constants).
    The skip tensor must come from a dynamic computation (e.g., MatMul).

    This pattern is recognized by FusionSkipLayerNormalization class.

    Args:
        input_name: Name of main input tensor
        skip_name: Name of skip connection tensor
        output_name: Name of output tensor
        prefix: Unique prefix for node names
        initializers: List to append weight tensors
        hidden_size: Hidden dimension

    Returns:
        List of ONNX nodes forming SkipLayerNorm pattern
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))
    nodes = []

    # Gamma and Beta
    gamma = helper.make_tensor(
        f"{prefix}gamma",
        TensorProto.FLOAT,
        [hidden_size],
        rng.randn(hidden_size).astype(np.float32),
    )
    beta = helper.make_tensor(
        f"{prefix}beta",
        TensorProto.FLOAT,
        [hidden_size],
        rng.randn(hidden_size).astype(np.float32),
    )
    initializers.extend([gamma, beta])

    # Add (skip connection)
    nodes.append(
        helper.make_node(
            "Add",
            inputs=[input_name, skip_name],
            outputs=[f"{prefix}add_out"],
            name=f"{prefix}add",
        )
    )

    # LayerNormalization
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            inputs=[f"{prefix}add_out", f"{prefix}gamma", f"{prefix}beta"],
            outputs=[output_name],
            name=f"{prefix}layernorm",
            axis=-1,
            epsilon=1e-5,
        )
    )

    return nodes


def simplified_layernorm_builder(
    input_name: str,
    output_name: str,
    prefix: str,
    initializers: list,
    hidden_size: int = 64,
) -> list:
    """Create SimplifiedLayerNorm pattern (RMS Normalization, 6 nodes -> 1 node).

    Pattern: X -> Pow(2) -> ReduceMean -> Add(eps) -> Sqrt -> Div -> Mul(gamma)

    Note: No mean subtraction (variance-only normalization).

    This pattern is recognized by FusionSimplifiedLayerNormalization class.

    Args:
        input_name: Name of input tensor
        output_name: Name of output tensor
        prefix: Unique prefix for node names
        initializers: List to append weight tensors
        hidden_size: Hidden dimension

    Returns:
        List of ONNX nodes forming simplified LayerNorm (RMS Norm)
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))
    nodes = []

    # Gamma (scale only, no beta for RMS norm)
    gamma = helper.make_tensor(
        f"{prefix}gamma",
        TensorProto.FLOAT,
        [hidden_size],
        rng.randn(hidden_size).astype(np.float32),
    )
    epsilon = helper.make_tensor(
        f"{prefix}eps",
        TensorProto.FLOAT,
        [],
        [np.float32(1e-5)],
    )
    pow_exp = helper.make_tensor(
        f"{prefix}pow_exp",
        TensorProto.FLOAT,
        [],
        [np.float32(2.0)],
    )
    initializers.extend([gamma, epsilon, pow_exp])

    # Pow (squared) - directly on input, no mean subtraction
    nodes.append(
        helper.make_node(
            "Pow",
            inputs=[input_name, f"{prefix}pow_exp"],
            outputs=[f"{prefix}squared"],
            name=f"{prefix}pow",
        )
    )

    # ReduceMean (mean of squared values = variance for zero-mean)
    nodes.append(
        helper.make_node(
            "ReduceMean",
            inputs=[f"{prefix}squared"],
            outputs=[f"{prefix}var"],
            name=f"{prefix}reduce_mean",
            axes=[-1],
            keepdims=1,
        )
    )

    # Add epsilon
    nodes.append(
        helper.make_node(
            "Add",
            inputs=[f"{prefix}var", f"{prefix}eps"],
            outputs=[f"{prefix}var_eps"],
            name=f"{prefix}add_eps",
        )
    )

    # Sqrt (RMS)
    nodes.append(
        helper.make_node(
            "Sqrt",
            inputs=[f"{prefix}var_eps"],
            outputs=[f"{prefix}rms"],
            name=f"{prefix}sqrt",
        )
    )

    # Div (normalize by RMS)
    nodes.append(
        helper.make_node(
            "Div",
            inputs=[input_name, f"{prefix}rms"],
            outputs=[f"{prefix}normalized"],
            name=f"{prefix}div",
        )
    )

    # Mul (scale by gamma)
    nodes.append(
        helper.make_node(
            "Mul",
            inputs=[f"{prefix}normalized", f"{prefix}gamma"],
            outputs=[output_name],
            name=f"{prefix}mul_gamma",
        )
    )

    return nodes


def create_decomposed_layernorm_model(
    hidden_size: int = 64,
    seq_len: int = 10,
    batch_size: int = 1,
) -> ModelProto:
    """Create complete ONNX model with decomposed LayerNorm pattern.

    Args:
        hidden_size: Hidden dimension (default: 64)
        seq_len: Sequence length (default: 10)
        batch_size: Batch size (default: 1)

    Returns:
        Complete ONNX ModelProto ready for fusion testing
    """
    initializers: list = []
    nodes = decomposed_layernorm_builder(
        input_name="input",
        output_name="output",
        prefix="ln_",
        initializers=initializers,
        hidden_size=hidden_size,
    )

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )

    graph = helper.make_graph(
        nodes,
        "decomposed_layernorm_test",
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


def create_skip_layernorm_model(
    hidden_size: int = 64,
    seq_len: int = 10,
    batch_size: int = 1,
) -> ModelProto:
    """Create complete ONNX model with SkipLayerNorm pattern.

    The model has two inputs (input and skip) that are both dynamic.
    A MatMul is added before skip to ensure it's a dynamic tensor.

    Args:
        hidden_size: Hidden dimension (default: 64)
        seq_len: Sequence length (default: 10)
        batch_size: Batch size (default: 1)

    Returns:
        Complete ONNX ModelProto ready for fusion testing
    """
    rng = np.random.RandomState(42)
    initializers: list = []
    nodes = []

    # Create a MatMul to make skip connection dynamic
    # This is CRITICAL - skip must be dynamic, not a constant
    skip_weight = helper.make_tensor(
        "skip_weight",
        TensorProto.FLOAT,
        [hidden_size, hidden_size],
        rng.randn(hidden_size, hidden_size).astype(np.float32),
    )
    initializers.append(skip_weight)

    nodes.append(
        helper.make_node(
            "MatMul",
            inputs=["skip_input", "skip_weight"],
            outputs=["skip_dynamic"],
            name="skip_matmul",
        )
    )

    # Add skip layernorm nodes
    skip_ln_nodes = skip_layernorm_builder(
        input_name="input",
        skip_name="skip_dynamic",
        output_name="output",
        prefix="skip_ln_",
        initializers=initializers,
        hidden_size=hidden_size,
    )
    nodes.extend(skip_ln_nodes)

    # Two inputs: main input and skip input
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )
    skip_input_tensor = helper.make_tensor_value_info(
        "skip_input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )

    graph = helper.make_graph(
        nodes,
        "skip_layernorm_test",
        [input_tensor, skip_input_tensor],
        [output_tensor],
        initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8

    return model


def create_simplified_layernorm_model(
    hidden_size: int = 64,
    seq_len: int = 10,
    batch_size: int = 1,
) -> ModelProto:
    """Create complete ONNX model with simplified LayerNorm (RMS Norm) pattern.

    Args:
        hidden_size: Hidden dimension (default: 64)
        seq_len: Sequence length (default: 10)
        batch_size: Batch size (default: 1)

    Returns:
        Complete ONNX ModelProto ready for fusion testing
    """
    initializers: list = []
    nodes = simplified_layernorm_builder(
        input_name="input",
        output_name="output",
        prefix="rms_",
        initializers=initializers,
        hidden_size=hidden_size,
    )

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
    )

    graph = helper.make_graph(
        nodes,
        "simplified_layernorm_test",
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
