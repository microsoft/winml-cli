# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Convolution fusion capabilities.

This module defines convolution operation fusion optimizations. These optimizations
detect and fuse common convolution patterns with subsequent operations, improving
performance.

Convolution operations are fundamental in computer vision architectures. These
fusions detect common patterns like Conv+BatchNorm, Conv+Add (bias), Conv+Multiply,
and Conv+Activation, replacing them with optimized fused operations or modified
weights.
"""

from __future__ import annotations

from ..registry import BoolCapability, CapabilityCategory


# Conv + Add (bias) fusion - fuses convolution with bias addition
CONV_ADD_FUSION = BoolCapability(
    name="conv-add-fusion",
    ort_name="ConvAddFusion",
    description="Fuse Conv+Add (bias) patterns",
    category=CapabilityCategory.CONVOLUTION,
    default=False,
)

# Conv + BatchNorm fusion - fuses convolution with batch normalization
CONV_BN_FUSION = BoolCapability(
    name="conv-bn-fusion",
    ort_name="ConvBNFusion",
    description="Fuse Conv+BatchNormalization into modified Conv weights",
    category=CapabilityCategory.CONVOLUTION,
    default=False,
)

# Conv + Multiply fusion - fuses convolution with element-wise multiplication
CONV_MUL_FUSION = BoolCapability(
    name="conv-mul-fusion",
    ort_name="ConvMulFusion",
    description="Fuse Conv+Multiply patterns",
    category=CapabilityCategory.CONVOLUTION,
    default=False,
)

# Conv + Activation fusion - fuses convolution with activation functions
CONV_ACTIVATION_FUSION = BoolCapability(
    name="conv-activation-fusion",
    ort_name="ConvActivationFusion",
    description="Fuse Conv+activation (ReLU, LeakyReLU, Sigmoid, Tanh, Clip)",
    category=CapabilityCategory.CONVOLUTION,
    default=False,
)
