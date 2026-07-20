# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Miscellaneous fusion capabilities.

This module defines miscellaneous fusion optimizations that don't fit into
other specific categories. These include pattern transformations like
Gather-to-Slice conversions, Pad fusion with Conv/Pool operations, and
logical operation fusions.

These optimizations handle edge cases and specialized patterns that can
improve performance through operation consolidation and simplification.
"""

from __future__ import annotations

from ..registry import BoolCapability, CapabilityCategory


# Gather + Slice to Split fusion - fuses gather+slice patterns into split
GATHER_SLICE_TO_SPLIT_FUSION = BoolCapability(
    name="gather-slice-to-split-fusion",
    ort_name="GatherSliceToSplitFusion",
    description="Fuse Gather+Slice patterns to Split operation",
    category=CapabilityCategory.MISC,
    default=False,
)

# Gather to Slice fusion - converts gather to slice when index is contiguous
GATHER_TO_SLICE_FUSION = BoolCapability(
    name="gather-to-slice-fusion",
    ort_name="GatherToSliceFusion",
    description="Convert Gather to Slice where index is contiguous",
    category=CapabilityCategory.MISC,
    default=False,
)

# Pad fusion - fuses pad with subsequent conv/pool operations
PAD_FUSION = BoolCapability(
    name="pad-fusion",
    ort_name="Pad_Fusion",
    description="Fuse Pad with subsequent Conv/Pool operations",
    category=CapabilityCategory.MISC,
    default=False,
)

# Not + Where fusion - fuses not+where patterns
NOT_WHERE_FUSION = BoolCapability(
    name="not-where-fusion",
    ort_name="NotWhereFusion",
    description="Fuse Not+Where patterns",
    category=CapabilityCategory.MISC,
    default=False,
)

# BiasAdd fusion - fuses the residual bias-add pattern that appears in
# Stable Diffusion UNet (Add(bias) -> Add(residual) -> LayerNorm) into the
# fused com.microsoft.BiasAdd op. This is an SD-specific FusionOptions toggle
# that complements SkipLayerNorm by absorbing the extra bias-Add.
BIAS_ADD_FUSION = BoolCapability(
    name="bias-add-fusion",
    ort_name="BiasAddFusion",  # FusionOptions attr: enable_bias_add
    description="Fuse Add(bias)+Add(residual) into BiasAdd (SD UNet)",
    category=CapabilityCategory.MISC,
    default=False,
)
