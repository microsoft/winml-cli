# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""GEMM (General Matrix Multiplication) capabilities.

This module defines GEMM fusion optimizations. These optimizations detect
and fuse GEMM operations with subsequent operations like activations, sums,
and transposes to improve performance.

GEMM is a fundamental linear algebra operation used extensively in neural
networks for fully connected layers and other matrix operations. These
fusions reduce memory bandwidth and improve computational efficiency.
"""

from __future__ import annotations

from ..registry import BoolCapability, CapabilityCategory


# GEMM + activation fusion - combines GEMM with activation functions
GEMM_ACTIVATION_FUSION = BoolCapability(
    name="gemm-activation-fusion",
    ort_name="GemmActivationFusion",
    description="Fuse GEMM+activation functions",
    category=CapabilityCategory.GEMM,
    default=False,
)

# GEMM + Sum fusion - fuses GEMM with element-wise sum operations
GEMM_SUM_FUSION = BoolCapability(
    name="gemm-sum-fusion",
    ort_name="GemmSumFusion",
    description="Fuse GEMM+Sum patterns",
    category=CapabilityCategory.GEMM,
    default=False,
)

# GEMM + Transpose fusion - fuses GEMM with transpose operations
GEMM_TRANSPOSE_FUSION = BoolCapability(
    name="gemm-transpose-fusion",
    ort_name="GemmTransposeFusion",
    description="Fuse GEMM+Transpose patterns",
    category=CapabilityCategory.GEMM,
    default=False,
)
