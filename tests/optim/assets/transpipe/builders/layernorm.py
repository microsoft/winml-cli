"""LayerNorm pattern builders for TransformerPipe testing.

These builders create ONNX models with decomposed normalization patterns
that ORT fusion classes can fuse:
- FusionLayerNormalization: Full LayerNorm (mean + variance)
- FusionSimplifiedLayerNormalization: RMS Norm (variance only)

Decomposed LayerNorm pattern:
  ReduceMean → Sub → Pow → ReduceMean → Add(eps) → Sqrt → Div → Mul(gamma) → Add(beta)

Decomposed RMS Norm (SimplifiedLayerNorm) pattern:
  Pow → ReduceMean → Add(eps) → Sqrt → Div → Mul(gamma)
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_decomposed_layernorm_model(
    batch_size: int = 1,
    seq_len: int = 10,
    hidden_size: int = 16,
    epsilon: float = 1e-5,
) -> onnx.ModelProto:
    """Build a decomposed LayerNorm pattern that FusionLayerNormalization can fuse.

    Pattern: input → ReduceMean → Sub → Pow(2) → ReduceMean → Add(eps) →
             Sqrt → Div → Mul(gamma) → Add(beta) → output

    This is how PyTorch's F.layer_norm gets exported to ONNX without optimization.

    Args:
        batch_size: Batch dimension
        seq_len: Sequence length
        hidden_size: Hidden dimension (normalization axis)
        epsilon: LayerNorm epsilon

    Returns:
        ONNX ModelProto with decomposed LayerNorm pattern
    """
    np.random.seed(42)
    gamma = np.random.randn(hidden_size).astype(np.float32)
    beta = np.random.randn(hidden_size).astype(np.float32)
    eps = np.array([epsilon], dtype=np.float32)
    pow_exp = np.array([2.0], dtype=np.float32)

    initializers = [
        numpy_helper.from_array(gamma, "ln_gamma"),
        numpy_helper.from_array(beta, "ln_beta"),
        numpy_helper.from_array(eps, "ln_eps"),
        numpy_helper.from_array(pow_exp, "ln_pow_exp"),
    ]

    inputs = [
        helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        ),
    ]

    nodes = [
        # Mean: E[x]
        helper.make_node(
            "ReduceMean",
            ["input"],
            ["ln_mean"],
            name="ln_reduce_mean",
            axes=[-1],
            keepdims=1,
        ),
        # Centered: x - E[x]
        helper.make_node(
            "Sub",
            ["input", "ln_mean"],
            ["ln_centered"],
            name="ln_sub_mean",
        ),
        # Squared: (x - E[x])^2
        helper.make_node(
            "Pow",
            ["ln_centered", "ln_pow_exp"],
            ["ln_squared"],
            name="ln_pow",
        ),
        # Variance: E[(x - E[x])^2]
        helper.make_node(
            "ReduceMean",
            ["ln_squared"],
            ["ln_var"],
            name="ln_reduce_var",
            axes=[-1],
            keepdims=1,
        ),
        # Variance + eps
        helper.make_node(
            "Add",
            ["ln_var", "ln_eps"],
            ["ln_var_eps"],
            name="ln_add_eps",
        ),
        # Sqrt(variance + eps)
        helper.make_node(
            "Sqrt",
            ["ln_var_eps"],
            ["ln_std"],
            name="ln_sqrt",
        ),
        # Normalized: (x - E[x]) / sqrt(var + eps)
        helper.make_node(
            "Div",
            ["ln_centered", "ln_std"],
            ["ln_normalized"],
            name="ln_div",
        ),
        # Scale: gamma * normalized
        helper.make_node(
            "Mul",
            ["ln_normalized", "ln_gamma"],
            ["ln_scaled"],
            name="ln_mul_gamma",
        ),
        # Shift: gamma * normalized + beta
        helper.make_node(
            "Add",
            ["ln_scaled", "ln_beta"],
            ["output"],
            name="ln_add_beta",
        ),
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
    ]

    graph = helper.make_graph(
        nodes,
        "decomposed_layernorm_test",
        inputs,
        outputs,
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    return model


def build_rms_norm_model(
    batch_size: int = 1,
    seq_len: int = 10,
    hidden_size: int = 16,
    epsilon: float = 1e-5,
) -> onnx.ModelProto:
    """Build a decomposed RMS Norm pattern that FusionSimplifiedLayerNormalization can fuse.

    RMS Norm (Root Mean Square Layer Normalization) differs from LayerNorm:
    - No mean subtraction
    - Only variance normalization: x / sqrt(mean(x^2) + eps) * gamma

    Pattern: input → Pow(2) → ReduceMean → Add(eps) → Sqrt → Div → Mul(gamma) → output

    Used in models like LLaMA, Gemma.

    Args:
        batch_size: Batch dimension
        seq_len: Sequence length
        hidden_size: Hidden dimension
        epsilon: Normalization epsilon

    Returns:
        ONNX ModelProto with decomposed RMS Norm pattern
    """
    np.random.seed(42)
    gamma = np.random.randn(hidden_size).astype(np.float32)
    eps = np.array([epsilon], dtype=np.float32)
    pow_exp = np.array([2.0], dtype=np.float32)

    initializers = [
        numpy_helper.from_array(gamma, "rms_gamma"),
        numpy_helper.from_array(eps, "rms_eps"),
        numpy_helper.from_array(pow_exp, "rms_pow_exp"),
    ]

    inputs = [
        helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        ),
    ]

    nodes = [
        # Squared: x^2
        helper.make_node(
            "Pow",
            ["input", "rms_pow_exp"],
            ["rms_squared"],
            name="rms_pow",
        ),
        # Mean of squared: E[x^2]
        helper.make_node(
            "ReduceMean",
            ["rms_squared"],
            ["rms_mean_sq"],
            name="rms_reduce_mean",
            axes=[-1],
            keepdims=1,
        ),
        # Mean + eps
        helper.make_node(
            "Add",
            ["rms_mean_sq", "rms_eps"],
            ["rms_mean_eps"],
            name="rms_add_eps",
        ),
        # Sqrt(E[x^2] + eps)
        helper.make_node(
            "Sqrt",
            ["rms_mean_eps"],
            ["rms_sqrt"],
            name="rms_sqrt",
        ),
        # Normalized: x / sqrt(E[x^2] + eps)
        helper.make_node(
            "Div",
            ["input", "rms_sqrt"],
            ["rms_normalized"],
            name="rms_div",
        ),
        # Scale: gamma * normalized
        helper.make_node(
            "Mul",
            ["rms_normalized", "rms_gamma"],
            ["output"],
            name="rms_mul_gamma",
        ),
    ]

    outputs = [
        helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]
        )
    ]

    graph = helper.make_graph(
        nodes,
        "rms_norm_test",
        inputs,
        outputs,
        initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    return model
