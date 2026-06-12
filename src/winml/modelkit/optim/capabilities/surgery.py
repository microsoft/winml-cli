# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Surgery capabilities for precise model modifications.

These capabilities perform targeted graph transformations that are not part of
ONNX Runtime's standard optimization passes. They run before ORT optimizations.

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
