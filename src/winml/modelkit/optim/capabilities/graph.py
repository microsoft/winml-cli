# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""General graph optimization capabilities.

This module defines advanced graph-level optimization capabilities that users
can opt into. Basic optimizations (ConstantFolding, ConstantSharing, CSE,
ReshapeFusion) are handled automatically by ORT at GraphOptimizationLevel 2.
"""

from __future__ import annotations

from ..registry import BoolCapability, CapabilityCategory


# Concat-slice elimination - remove concat followed by slice
CONCAT_SLICE_ELIMINATION = BoolCapability(
    name="concat-slice-elimination",
    ort_name="ConcatSliceElimination",
    description="Eliminate Concat followed by Slice that extracts original tensors",
    category=CapabilityCategory.GRAPH,
    default=False,
)

# Double QDQ pairs remover - remove consecutive quantize-dequantize pairs
DOUBLE_QDQ_PAIRS_REMOVER = BoolCapability(
    name="double-qdq-pairs-remover",
    ort_name="DoubleQDQPairsRemover",
    description="Remove consecutive QuantizeLinear→DequantizeLinear pairs",
    category=CapabilityCategory.GRAPH,
    default=False,
)

# Constant folding - pre-compute constant expressions at optimization time
# WARNING: Can significantly increase model size for models with large intermediate
# tensors (e.g., SAM2 multi-scale feature maps). Disable for size-sensitive deployments.
CONSTANT_FOLDING = BoolCapability(
    name="constant-folding",
    ort_name="ConstantFolding",
    description="Pre-compute constant expressions (may increase model size)",
    category=CapabilityCategory.GRAPH,
    default=True,  # Enabled by default, can be disabled to prevent size bloat
)
