"""Activation pattern builders for ORT Graph Optimization tests.

This module contains builder functions for activation-related ONNX patterns
that test specific ONNX Runtime graph optimizations.

Pattern Status:
- BiasSoftmaxFusion: CUDA-only optimizer - skip on CPU
- BiasDropoutFusion: Training-only optimizer - skip entirely
- ReluClipFusion: Works on CPU - keep as is
"""

import numpy as np
from onnx import helper, numpy_helper


def bias_softmax_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Add+Softmax pattern for BiasSoftmaxFusion.

    NOTE: BiasSoftmaxFusion is a CUDA-only optimizer in ORT.
    The pattern requires:
    - Add node (opset 7, 13, 14) with kCudaExecutionProvider
    - Softmax node following the Add

    Tests using this pattern should be marked with:
        @pytest.mark.skip(reason="BiasSoftmaxFusion is CUDA-only optimizer")

    Pattern: Input + Bias -> Softmax -> Output
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Create bias tensor (broadcastable shape for attention mask pattern)
    # Shape [1, 1, 1, 64] for broadcasting with [batch, heads, seq, seq]
    initializers.append(
        numpy_helper.from_array(rng.randn(1, 1, 1, 64).astype(np.float32) * 0.1, f"{prefix}bias")
    )

    return [
        # Add bias (simulates attention mask addition)
        helper.make_node(
            "Add",
            [input_name, f"{prefix}bias"],
            [f"{prefix}add_out"],
            name=f"{prefix}add",
        ),
        # Softmax on the result
        helper.make_node(
            "Softmax",
            [f"{prefix}add_out"],
            [output_name],
            name=f"{prefix}softmax",
            axis=-1,
        ),
    ]


def bias_dropout_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Add+Dropout pattern for BiasDropoutFusion.

    NOTE: BiasDropoutFusion is a training-only optimizer in ORT.
    It is not available during inference because Dropout in inference mode
    is treated as identity and gets eliminated by other optimizers.

    Tests using this pattern should be marked with:
        @pytest.mark.skip(reason="BiasDropoutFusion is training-only optimizer")

    Pattern: Input + Bias -> Dropout -> Output
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    initializers.append(
        numpy_helper.from_array(rng.randn(64).astype(np.float32) * 0.1, f"{prefix}bias")
    )

    return [
        # Add bias
        helper.make_node(
            "Add",
            [input_name, f"{prefix}bias"],
            [f"{prefix}add_out"],
            name=f"{prefix}add",
        ),
        # Dropout in inference mode (training_mode not set = False by default)
        helper.make_node(
            "Dropout",
            [f"{prefix}add_out"],
            [output_name],
            name=f"{prefix}dropout",
        ),
    ]


def relu_clip_builder(input_name: str, output_name: str, prefix: str, initializers: list) -> list:
    """Build ReLU+Clip pattern for ReluClipFusion.

    Pattern: Input -> Relu -> Clip(min=0, max=6) -> Output
    ORT fuses this to Relu6 operation.

    This pattern works on CPU and is a valid test case.
    """
    # Clip min/max as inputs (opset 11+)
    initializers.append(
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), f"{prefix}clip_min")
    )
    initializers.append(
        numpy_helper.from_array(np.array(6.0, dtype=np.float32), f"{prefix}clip_max")
    )

    return [
        helper.make_node(
            "Relu",
            [input_name],
            [f"{prefix}relu_out"],
            name=f"{prefix}relu",
        ),
        helper.make_node(
            "Clip",
            [f"{prefix}relu_out", f"{prefix}clip_min", f"{prefix}clip_max"],
            [output_name],
            name=f"{prefix}clip",
        ),
    ]
