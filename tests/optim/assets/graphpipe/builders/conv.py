# Copyright (c) 2025 ModelKit Authors
# SPDX-License-Identifier: Apache-2.0
"""Conv-related pattern builders for ORT Graph Optimization tests.

This module contains corrected builders for Conv-related optimizations:

1. conv_bn_builder: Conv -> BatchNorm (existing, works)
2. conv_add_relu_builder: Conv -> Add -> Relu with extra ops (existing, works)
3. conv_activation_builder: NEW - Pure Conv -> Activation for ConvActivationFusion
4. conv_mul_builder: Conv -> Mul for ConvMulFusion (existing, works)
5. conv_add_activation_builder: FIXED - Conv -> Add -> Relu with proper 4D shapes
6. nchwc_transformer_builder: NCHWc layout (CPU AVX512 only, marked as skip)

Root Cause Analysis:
-------------------
- ConvActivationFusion: Tests were using Conv->Add->Relu (wrong pattern)
  The correct pattern is Conv->Activation DIRECTLY with no intermediate Add.
  See: onnxruntime/core/optimizer/conv_activation_fusion.cc:70-123

- ConvAddActivationFusion: Requires SPECIFIC conditions:
  1. Both Add inputs must have EQUAL 4D shapes (NCHW)
  2. Conv must be one of the Add inputs (producer)
  3. Conv must have bias slot available (< 4 inputs)
  See: onnxruntime/core/optimizer/conv_add_act_fusion.cc:124-200

- NchwcTransformer: This is NOT a pattern-matching optimization.
  It's a CPU-specific layout transformation requiring:
  1. CPU execution provider assignment
  2. AVX/SSE support with MLAS block size
  3. Static weight tensors
  See: onnxruntime/core/optimizer/nchwc_transformer.cc
"""

import numpy as np
from onnx import helper, numpy_helper


def conv_bn_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Conv -> BN pattern (shape-preserving: 16->16).

    Tests ConvBNFusion capability (ORT name: FuseConvBN).
    Conv followed by BatchNorm should fuse the BN parameters into Conv weights.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Conv 16->16 channels
    initializers.append(
        numpy_helper.from_array(
            rng.randn(16, 16, 3, 3).astype(np.float32) * 0.1, f"{prefix}conv_w"
        )
    )
    initializers.append(
        numpy_helper.from_array(np.ones(16, dtype=np.float32), f"{prefix}bn_scale")
    )
    initializers.append(
        numpy_helper.from_array(np.zeros(16, dtype=np.float32), f"{prefix}bn_bias")
    )
    initializers.append(
        numpy_helper.from_array(np.zeros(16, dtype=np.float32), f"{prefix}bn_mean")
    )
    initializers.append(
        numpy_helper.from_array(np.ones(16, dtype=np.float32), f"{prefix}bn_var")
    )

    return [
        helper.make_node(
            "Conv",
            [input_name, f"{prefix}conv_w"],
            [f"{prefix}conv_out"],
            name=f"{prefix}conv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            "BatchNormalization",
            [
                f"{prefix}conv_out",
                f"{prefix}bn_scale",
                f"{prefix}bn_bias",
                f"{prefix}bn_mean",
                f"{prefix}bn_var",
            ],
            [output_name],
            name=f"{prefix}bn",
            epsilon=1e-5,
        ),
    ]


def conv_add_relu_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Conv -> Dropout -> Add -> Relu pattern (shape-preserving: 16->16).

    P1-01: Dropout(training_mode=False) in inference mode is eliminable.
    P1-05: Expand(same_shape) after Conv is no-op, hence eliminable.
    P1-08: Reuse same bias constant twice for constant-sharing test.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    initializers.append(
        numpy_helper.from_array(
            rng.randn(16, 16, 3, 3).astype(np.float32) * 0.1, f"{prefix}conv_w"
        )
    )
    # P1-08: Create bias only once, reused below
    initializers.append(
        numpy_helper.from_array(
            rng.randn(16, 1, 1).astype(np.float32) * 0.1, f"{prefix}bias"
        )
    )
    # P1-05: Expand to same shape (identity)
    initializers.append(
        numpy_helper.from_array(
            np.array([1, 16, 32, 32], dtype=np.int64), f"{prefix}expand_shape"
        )
    )

    # Dropout in inference mode (training_mode not set = False by default in opset 12+)
    return [
        helper.make_node(
            "Conv",
            [input_name, f"{prefix}conv_w"],
            [f"{prefix}conv_out"],
            name=f"{prefix}conv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        # P1-05: Expand to same shape (no-op)
        helper.make_node(
            "Expand",
            [f"{prefix}conv_out", f"{prefix}expand_shape"],
            [f"{prefix}expand_out"],
            name=f"{prefix}expand",
        ),
        helper.make_node(
            "Dropout",
            [f"{prefix}expand_out"],
            [f"{prefix}dropout_out"],
            name=f"{prefix}dropout",
        ),
        helper.make_node(
            "Add",
            [f"{prefix}dropout_out", f"{prefix}bias"],
            [f"{prefix}add_out"],
            name=f"{prefix}add",
        ),
        # P1-08: Reuse same bias constant (constant-sharing)
        helper.make_node(
            "Add",
            [f"{prefix}add_out", f"{prefix}bias"],
            [f"{prefix}add2_out"],
            name=f"{prefix}add2",
        ),
        helper.make_node(
            "Relu", [f"{prefix}add2_out"], [output_name], name=f"{prefix}relu"
        ),
    ]


def conv_activation_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Conv -> Activation pattern for ConvActivationFusion.

    P3-10: Tests conv-activation-fusion capability (ORT name: ConvActivationFusion).

    CRITICAL: This is DIFFERENT from ConvAddActivationFusion!
    ConvActivationFusion fuses Conv DIRECTLY followed by activation (no Add).

    Pattern: Conv -> Relu (or Sigmoid, Tanh, LeakyRelu, Clip, HardSigmoid)

    From conv_activation_fusion.cc:
    - Conv must have single consumer that is the activation
    - Supported activations: Relu, Sigmoid, Tanh, LeakyRelu, Clip, HardSigmoid
    - CPU EP: Float or Float16 (if MLAS_F16VEC_INTRINSICS_SUPPORTED)
    - Creates FusedConv node with 'activation' attribute

    Input shape: [1, 16, 32, 32] (4D NCHW tensor)
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Conv 16->16 channels with bias
    initializers.append(
        numpy_helper.from_array(
            rng.randn(16, 16, 3, 3).astype(np.float32) * 0.1, f"{prefix}conv_w"
        )
    )
    # Add bias to Conv to make it more realistic
    initializers.append(
        numpy_helper.from_array(
            rng.randn(16).astype(np.float32) * 0.1, f"{prefix}conv_b"
        )
    )

    return [
        # Conv with bias (3 inputs: X, W, B)
        helper.make_node(
            "Conv",
            [input_name, f"{prefix}conv_w", f"{prefix}conv_b"],
            [f"{prefix}conv_out"],
            name=f"{prefix}conv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        # Direct activation after Conv (no Add in between!)
        helper.make_node(
            "Relu",
            [f"{prefix}conv_out"],
            [output_name],
            name=f"{prefix}relu",
        ),
    ]


def conv_mul_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Conv+Mul(scale) pattern: Input -> Conv -> Mul.

    P3-08: Tests conv-mul-fusion capability (ORT name: ConvMulFusion).
    Conv followed by Mul with constant scale should fuse into Conv weights.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Conv 16->16 channels, shape-preserving
    initializers.append(
        numpy_helper.from_array(
            rng.randn(16, 16, 3, 3).astype(np.float32) * 0.1, f"{prefix}conv_w"
        )
    )
    # Scale factor for Mul (broadcasting: [1, 16, 1, 1] or just [16])
    scale_value = rng.randn(16, 1, 1).astype(np.float32) * 0.1 + 1.0
    initializers.append(numpy_helper.from_array(scale_value, f"{prefix}scale"))

    return [
        helper.make_node(
            "Conv",
            [input_name, f"{prefix}conv_w"],
            [f"{prefix}conv_out"],
            name=f"{prefix}conv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            "Mul",
            [f"{prefix}conv_out", f"{prefix}scale"],
            [output_name],
            name=f"{prefix}mul",
        ),
    ]


def conv_add_activation_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Conv+Add+Activation pattern for ConvAddActivationFusion.

    Tests conv-add-activation-fusion capability (ORT name: ConvAddActivationFusion).

    ConvAddActivationFusion fuses Conv (no bias) + Add (1D bias) + Activation into FusedConv.
    The pattern is: Conv (X, W) -> Add (Conv_out, bias_1D) -> Relu -> FusedConv

    CRITICAL REQUIREMENTS:
    - Conv must have no bias (< 3 inputs) - bias slot available for Add folding
    - Add must have a constant 1D bias of shape [C, 1, 1] for NCHW broadcast
    - Conv must have single consumer (the Add)
    - Activation must follow Add

    Input shape: [1, 16, 32, 32] (4D NCHW tensor)
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Conv 16->16 channels WITHOUT bias (important for fusion!)
    initializers.append(
        numpy_helper.from_array(
            rng.randn(16, 16, 3, 3).astype(np.float32) * 0.1, f"{prefix}conv_w"
        )
    )
    # 1D bias for Add - shape [16, 1, 1] to broadcast with NCHW output [1, 16, 32, 32]
    initializers.append(
        numpy_helper.from_array(
            rng.randn(16, 1, 1).astype(np.float32) * 0.1, f"{prefix}bias"
        )
    )

    return [
        # Conv WITHOUT bias (2 inputs only: X, W)
        helper.make_node(
            "Conv",
            [input_name, f"{prefix}conv_w"],
            [f"{prefix}conv_out"],
            name=f"{prefix}conv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        # Add with 1D bias - this will be folded into Conv's bias slot
        helper.make_node(
            "Add",
            [f"{prefix}conv_out", f"{prefix}bias"],
            [f"{prefix}add_out"],
            name=f"{prefix}add",
        ),
        # Relu activation - will be fused with Conv+Add into FusedConv
        helper.make_node(
            "Relu",
            [f"{prefix}add_out"],
            [output_name],
            name=f"{prefix}relu",
        ),
    ]


def nchwc_transformer_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build NCHWc layout transformation pattern.

    P4-01: Tests nchwc-transformer capability (ORT name: NchwcTransformer).

    IMPORTANT: This optimization is NOT a pattern match!
    NchwcTransformer is a CPU-specific layout transformation that:
    1. Requires CPU execution provider assignment
    2. Requires AVX/SSE support with MLAS block size (typically 8 or 16)
    3. Requires static weight tensors
    4. Operates on Conv nodes assigned to CPU EP

    From nchwc_transformer.cc:
    - Only works when node.GetExecutionProviderType() == kCpuExecutionProvider
    - Transforms Conv to use NCHWc (blocked) memory layout
    - Creates nodes in kMSNchwcDomain ("com.microsoft.nchwc")

    This pattern creates a simple Conv that COULD be transformed,
    but the transformation depends on runtime EP assignment and CPU features.

    For testing purposes, this builder creates the pattern but tests should:
    1. Skip on non-AVX CPUs
    2. Verify EP assignment to CPU
    3. Check for NCHWc domain in output

    Input shape: [1, 64] -> reshaped to [1, 16, 2, 2] for Conv
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Reshape input from [1, 64] to [1, 16, 2, 2] for Conv2D
    initializers.append(
        numpy_helper.from_array(
            np.array([1, 16, 2, 2], dtype=np.int64), f"{prefix}reshape_shape"
        )
    )
    # Conv weights: 16 input channels, 16 output channels, 3x3 kernel
    # NCHWc transformation works best with channel counts divisible by block size
    initializers.append(
        numpy_helper.from_array(
            rng.randn(16, 16, 3, 3).astype(np.float32) * 0.1, f"{prefix}conv_w"
        )
    )
    initializers.append(
        numpy_helper.from_array(
            rng.randn(16).astype(np.float32) * 0.1, f"{prefix}conv_b"
        )
    )
    # Reshape back to [1, 64]
    initializers.append(
        numpy_helper.from_array(
            np.array([1, 64], dtype=np.int64), f"{prefix}out_shape"
        )
    )

    return [
        # Reshape to 4D for Conv
        helper.make_node(
            "Reshape",
            [input_name, f"{prefix}reshape_shape"],
            [f"{prefix}reshaped"],
            name=f"{prefix}reshape1",
        ),
        # Conv that can benefit from NCHWc layout
        helper.make_node(
            "Conv",
            [f"{prefix}reshaped", f"{prefix}conv_w", f"{prefix}conv_b"],
            [f"{prefix}conv_out"],
            name=f"{prefix}conv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        # Reshape back to [1, 64]
        helper.make_node(
            "Reshape",
            [f"{prefix}conv_out", f"{prefix}out_shape"],
            [output_name],
            name=f"{prefix}reshape2",
        ),
    ]


def conv_add_fusion_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Conv(no bias)+Add(1D bias) pattern for ConvAddFusion.

    Tests conv-add-fusion capability (ORT name: ConvAddFusion).

    ConvAddFusion folds an Add (with 1D bias) into Conv's bias slot.
    From conv_add_fusion.cc:
    - Conv must have no existing bias (< 3 inputs)
    - Add must have a constant 1D bias of shape [C]
    - Conv must have single consumer (the Add)

    Pattern: Conv (X, W) -> Add (Conv_out, bias_1D) -> Conv (X, W, B)

    Input shape: [1, 16, 32, 32] (4D NCHW tensor)
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Conv 16->16 channels WITHOUT bias (important for fusion!)
    initializers.append(
        numpy_helper.from_array(
            rng.randn(16, 16, 3, 3).astype(np.float32) * 0.1, f"{prefix}conv_w"
        )
    )
    # 1D bias for Add - shape [16, 1, 1] to broadcast with NCHW output [1, 16, 32, 32]
    # ONNX broadcasts from rightmost axis, so [16, 1, 1] broadcasts to [1, 16, H, W]
    initializers.append(
        numpy_helper.from_array(
            rng.randn(16, 1, 1).astype(np.float32) * 0.1, f"{prefix}bias"
        )
    )

    return [
        # Conv WITHOUT bias (2 inputs only: X, W)
        helper.make_node(
            "Conv",
            [input_name, f"{prefix}conv_w"],
            [f"{prefix}conv_out"],
            name=f"{prefix}conv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        # Add with 1D bias - this should be folded into Conv's bias
        helper.make_node(
            "Add",
            [f"{prefix}conv_out", f"{prefix}bias"],
            [output_name],
            name=f"{prefix}add",
        ),
    ]


def nhwc_transformer_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build NHWC layout transformation pattern.

    Pattern: Transpose(NCHW->NHWC) -> Conv -> Transpose(NHWC->NCHW).

    P4-02: Tests nhwc-transformer capability (ORT name: NhwcTransformer).
    NCHW to NHWC layout transformation for GPU/mobile optimization.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Reshape input from [1, 64] to [1, 16, 2, 2] for Conv2D (NCHW format)
    initializers.append(
        numpy_helper.from_array(
            np.array([1, 16, 2, 2], dtype=np.int64), f"{prefix}reshape_shape"
        )
    )
    # Conv weights: 16 input channels, 16 output channels, 3x3 kernel
    initializers.append(
        numpy_helper.from_array(
            rng.randn(16, 16, 3, 3).astype(np.float32) * 0.1, f"{prefix}conv_w"
        )
    )
    initializers.append(
        numpy_helper.from_array(
            rng.randn(16).astype(np.float32) * 0.1, f"{prefix}conv_b"
        )
    )
    # Reshape back to [1, 64]
    initializers.append(
        numpy_helper.from_array(
            np.array([1, 64], dtype=np.int64), f"{prefix}out_shape"
        )
    )

    return [
        # Reshape to 4D NCHW: [1, 16, 2, 2]
        helper.make_node(
            "Reshape",
            [input_name, f"{prefix}reshape_shape"],
            [f"{prefix}nchw"],
            name=f"{prefix}reshape1",
        ),
        # Transpose NCHW -> NHWC: [0,2,3,1] permutation
        helper.make_node(
            "Transpose",
            [f"{prefix}nchw"],
            [f"{prefix}nhwc"],
            name=f"{prefix}transpose1",
            perm=[0, 2, 3, 1],  # NCHW -> NHWC
        ),
        # Conv on NHWC data (optimizer should detect and optimize)
        helper.make_node(
            "Conv",
            [f"{prefix}nhwc", f"{prefix}conv_w", f"{prefix}conv_b"],
            [f"{prefix}conv_nhwc"],
            name=f"{prefix}conv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        # Transpose NHWC -> NCHW: [0,3,1,2] permutation
        helper.make_node(
            "Transpose",
            [f"{prefix}conv_nhwc"],
            [f"{prefix}conv_nchw"],
            name=f"{prefix}transpose2",
            perm=[0, 3, 1, 2],  # NHWC -> NCHW
        ),
        # Reshape back to [1, 64]
        helper.make_node(
            "Reshape",
            [f"{prefix}conv_nchw", f"{prefix}out_shape"],
            [output_name],
            name=f"{prefix}reshape2",
        ),
    ]


def pad_conv_builder(
    input_name: str, output_name: str, prefix: str, initializers: list
) -> list:
    """Build Pad+Conv pattern: Input -> Pad -> Conv.

    P3-03: Tests pad-fusion capability (ORT name: Pad_Fusion).
    Explicit Pad followed by Conv should fuse Pad into Conv's pads attribute.

    NOTE: This pattern uses 4D conv shape [1, 1, 8, 8] which is different
    from other patterns. The template will handle shape adaptation.
    """
    rng = np.random.RandomState(hash(prefix) % (2**32))

    # Reshape input from [1, 64] to [1, 1, 8, 8] for Conv2D
    initializers.append(
        numpy_helper.from_array(
            np.array([1, 1, 8, 8], dtype=np.int64), f"{prefix}reshape_shape"
        )
    )
    # Pad values for Conv: [batch_start, c_start, H_start, W_start, batch_end, c_end, H_end, W_end]
    initializers.append(
        numpy_helper.from_array(
            np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64), f"{prefix}pads"
        )
    )
    # Conv weights: 1 input channel, 1 output channel, 3x3 kernel
    initializers.append(
        numpy_helper.from_array(
            rng.randn(1, 1, 3, 3).astype(np.float32) * 0.1, f"{prefix}conv_w"
        )
    )
    # Reshape back to [1, 64]
    initializers.append(
        numpy_helper.from_array(
            np.array([1, 64], dtype=np.int64), f"{prefix}out_shape"
        )
    )

    return [
        # Reshape to 4D for Conv
        helper.make_node(
            "Reshape",
            [input_name, f"{prefix}reshape_shape"],
            [f"{prefix}reshaped"],
            name=f"{prefix}reshape1",
        ),
        # Explicit Pad (should be fused into Conv)
        helper.make_node(
            "Pad",
            [f"{prefix}reshaped", f"{prefix}pads"],
            [f"{prefix}padded"],
            name=f"{prefix}pad",
            mode="constant",
        ),
        # Conv with no padding (Pad already applied)
        helper.make_node(
            "Conv",
            [f"{prefix}padded", f"{prefix}conv_w"],
            [f"{prefix}conv_out"],
            name=f"{prefix}conv",
            kernel_shape=[3, 3],
            pads=[0, 0, 0, 0],  # No padding since Pad already applied
        ),
        # Reshape back to [1, 64]
        helper.make_node(
            "Reshape",
            [f"{prefix}conv_out", f"{prefix}out_shape"],
            [output_name],
            name=f"{prefix}reshape2",
        ),
    ]
