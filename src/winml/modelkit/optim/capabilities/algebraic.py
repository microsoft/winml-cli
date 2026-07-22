# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Opt-in, exact algebraic graph-rewrite capabilities."""

from __future__ import annotations

from ..registry import BoolCapability, CapabilityCategory


STATIC_SPLIT_TO_SLICE = BoolCapability(
    name="static-split-to-slice",
    ort_name=None,
    description=(
        "Replace statically bounded Split operations with standard Slice operations "
        "while preserving output tensors"
    ),
    category=CapabilityCategory.REWRITE,
    default=False,
)

CONV_CHANNEL_AFFINE_FOLDING = BoolCapability(
    name="conv-channel-affine-folding",
    ort_name=None,
    description=(
        "Fold safe scalar or channel-wise constant Mul/Add branches into Conv "
        "weights and bias; direct and static Split/Slice routing only"
    ),
    category=CapabilityCategory.REWRITE,
    default=False,
)
