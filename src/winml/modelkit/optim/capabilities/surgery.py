# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Surgery capabilities for precise model modifications.

These capabilities perform targeted graph transformations that are not part of
ONNX Runtime's standard optimization passes. Each runs at the pipeline stage
where its required graph evidence is available.

Use cases:
- Fix quantization issues (extreme values, invalid scales)
- Prepare models for specific execution providers
- Apply vendor-specific graph transformations
"""

from __future__ import annotations

from ..registry import BoolCapability, CapabilityCategory


# Clamp extreme constant values to prevent quantization issues
CLAMP_CONSTANT_VALUES = BoolCapability(
    name="clamp-constant-values",
    ort_name=None,  # Custom implementation, not ORT optimizer
    description="Clamp extreme float constants (e.g., -inf -> -1e3) to prevent quantization issues",
    category=CapabilityCategory.SURGERY,
    default=False,
)

# Remove Softmax -> IsNaN -> Where NaN guard patterns in attention.
# These guards are dead code when clamp_constant_values replaces -inf
# with a finite value (Softmax never produces NaN).
REMOVE_ISNAN_IN_ATTENTION_MASK = BoolCapability(
    name="remove-isnan-in-attention-mask",
    ort_name=None,  # Custom implementation, not ORT optimizer
    description="Remove Softmax->IsNaN->Where NaN guard patterns in attention",
    category=CapabilityCategory.SURGERY,
    default=False,
)

# Route a constant operand of a batched (rank >= 3) MatMul through a runtime
# no-op so it is no longer a compile-time constant. OpenVINO GPU's oneDNN gemm
# cannot select an implementation for a batched MatMul with a constant operand
# (e.g. transformer disentangled-attention position terms that fold to 3D
# constants); making the operand runtime-valued lets gemm impl selection
# succeed without changing numerics or splitting the batched op.
UNTIE_CONSTANT_BATCHED_MATMUL = BoolCapability(
    name="untie-constant-batched-matmul",
    ort_name=None,  # Custom implementation, not ORT optimizer
    description=(
        "Make a batched MatMul's constant operand runtime-valued so OpenVINO "
        "GPU can select a gemm implementation"
    ),
    category=CapabilityCategory.SURGERY,
    default=False,
)

# Remove operations that are provably redundant around channel-wise L2
# normalization while preserving the original ReduceL2 and Div kernels.
SIMPLIFY_L2_NORMALIZATION = BoolCapability(
    name="simplify-l2-normalization",
    ort_name=None,
    description="Remove redundant Clip(min=0) and Expand from ReduceL2-based normalization",
    category=CapabilityCategory.SURGERY,
    default=False,
)

# Replace the exporter-generated integer grid used for 2x nearest-neighbor
# spatial upsampling with the equivalent standard ONNX Resize operation.
GATHERND_TO_RESIZE = BoolCapability(
    name="gathernd-to-resize",
    ort_name=None,
    description="Replace exact 2x nearest-neighbor GatherND upsampling grids with Resize",
    category=CapabilityCategory.SURGERY,
    default=False,
)

# Emit the CUDA-supported Microsoft contrib operation for exact SiLU topology.
SILU_TO_QUICK_GELU = BoolCapability(
    name="silu-to-quick-gelu",
    ort_name=None,
    description="Replace x*Sigmoid(x) with com.microsoft QuickGelu(alpha=1)",
    category=CapabilityCategory.SURGERY,
    default=False,
    ep_constraint=("CUDA",),
)

# Move a scalar multiplication into the CUDA-supported Microsoft contrib
# MatMul kernel. This is algebraically equivalent but may change FP rounding.
SCALED_MATMUL_TO_FUSED_MATMUL = BoolCapability(
    name="scaled-matmul-to-fused-matmul",
    ort_name=None,
    description="Replace MatMul followed by scalar Mul with com.microsoft FusedMatMul",
    category=CapabilityCategory.SURGERY,
    default=False,
    ep_constraint=("CUDA",),
)
