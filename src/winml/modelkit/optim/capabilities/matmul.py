# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""MatMul fusion capabilities.

This module defines MatMul operation fusion optimizations. These optimizations
detect and fuse MatMul operations with subsequent operations like Add, Transpose,
Scale, BatchNormalization, and activations into efficient fused operations.

MatMul (Matrix Multiplication) is a fundamental operation in neural networks.
These fusions reduce memory bandwidth and improve computational efficiency by
combining multiple operations into single kernels where possible.
"""

from __future__ import annotations

from ..registry import BoolCapability, CapabilityCategory


# MatMul+Add fusion
MATMUL_ADD_FUSION = BoolCapability(
    name="matmul-add-fusion",
    ort_name="MatMulAddFusion",
    description="Fuse MatMul+Add operations into single kernel",
    category=CapabilityCategory.MATMUL,
    default=False,
)

# MatMul+Activation fusion
# NOTE: This requires FusedMatMul (from MatmulTransposeFusion) as input, NOT regular MatMul!
# ORT's matmul_activation_fusion.cc only fuses FusedMatMul + Softmax → FusedMatMulActivation
# EP constraint: DML-only (verified against ort_optimizer_inventory.md)
MATMUL_ACTIVATION_FUSION = BoolCapability(
    name="matmul-activation-fusion",
    ort_name="MatMulActivationFusion",
    description="Fuse MatMul+activation functions (ReLU, Sigmoid, Tanh)",
    category=CapabilityCategory.MATMUL,
    default=False,
    depends_on=("matmul-transpose-fusion",),  # Requires FusedMatMul created by transpose fusion
    ep_constraint=("DML",),  # DML-only optimizer
)

# MatMul+Transpose fusion
MATMUL_TRANSPOSE_FUSION = BoolCapability(
    name="matmul-transpose-fusion",
    ort_name="MatmulTransposeFusion",
    description="Fuse MatMul+Transpose operations",
    category=CapabilityCategory.MATMUL,
    default=False,
)

# MatMul+Scale fusion
MATMUL_SCALE_FUSION = BoolCapability(
    name="matmul-scale-fusion",
    ort_name="MatMulScaleFusion",
    description="Fuse MatMul+Scale (multiply by constant)",
    category=CapabilityCategory.MATMUL,
    default=False,
)

# MatMul+BatchNormalization fusion
MATMUL_BN_FUSION = BoolCapability(
    name="matmul-bn-fusion",
    ort_name="MatMul_BatchNormalization_Fusion",
    description="Fuse MatMul+BatchNormalization",
    category=CapabilityCategory.MATMUL,
    default=False,
)

# Dynamic quantization for MatMul
DYNAMIC_QUANTIZE_MATMUL_FUSION = BoolCapability(
    name="dynamic-quantize-matmul-fusion",
    ort_name="DynamicQuantizeMatMulFusion",
    description="Dynamic quantization for MatMul operations",
    category=CapabilityCategory.MATMUL,
    default=False,
)
