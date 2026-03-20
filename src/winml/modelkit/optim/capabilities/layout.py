# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Layout transformation capabilities.

This module defines layout and transpose optimization capabilities. These optimizations
transform tensor data layouts and eliminate redundant transpose operations to improve
hardware performance.

Layout optimizations include:
- Transpose optimization: Eliminate redundant transpose chains
- NHWC transformation: GPU-friendly layout (memory access patterns)
- NCHWc transformation: CPU SIMD-friendly layout (vectorization)
- Conv fusion: Combine convolution with post-processing operations
"""

from __future__ import annotations

from ..registry import BoolCapability, CapabilityCategory


# Transpose optimizer - eliminate redundant transpose operations
TRANSPOSE_OPTIMIZER = BoolCapability(
    name="transpose-optimizer",
    ort_name="TransposeOptimizer",
    description="Optimize and eliminate redundant transpose operations",
    category=CapabilityCategory.LAYOUT,
    default=False,
)

# NHWC transformer - transform NCHW to NHWC layout for GPU optimization
NHWC_TRANSFORMER = BoolCapability(
    name="nhwc-transformer",
    ort_name="NhwcTransformer",
    description="Transform NCHW to NHWC layout (GPU memory access optimized)",
    category=CapabilityCategory.LAYOUT,
    default=False,
)

# NCHWc transformer - transform NCHW to NCHWc layout for CPU SIMD
NCHWC_TRANSFORMER = BoolCapability(
    name="nchwc-transformer",
    ort_name="NchwcTransformer",
    description="Transform NCHW to NCHWc layout (CPU SIMD optimized)",
    category=CapabilityCategory.LAYOUT,
    default=False,
)

# Conv+Add+Activation fusion - fuse convolution chain
CONV_ADD_ACTIVATION_FUSION = BoolCapability(
    name="conv-add-activation-fusion",
    ort_name="ConvAddActivationFusion",
    description="Fuse Conv+Add+Activation chain into single FusedConv",
    category=CapabilityCategory.LAYOUT,
    default=False,
)
