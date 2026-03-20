# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Elimination optimization capabilities.

This module defines advanced elimination capabilities that users can opt into.
Basic eliminations (Identity, Dropout, Noop, Cast) are handled automatically
by ORT at GraphOptimizationLevel 2 and are not exposed here.
"""

from __future__ import annotations

from ..registry import BoolCapability, CapabilityCategory


# Slice elimination - removes redundant slice operations
SLICE_ELIMINATION = BoolCapability(
    name="slice-elimination",
    ort_name="EliminateSlice",
    description="Eliminate redundant Slice operations",
    category=CapabilityCategory.ELIMINATION,
    default=False,
)

# Expand elimination - removes expand operations that don't change shape
EXPAND_ELIMINATION = BoolCapability(
    name="expand-elimination",
    ort_name="ExpandElimination",
    description="Eliminate Expand when output shape equals input shape",
    category=CapabilityCategory.ELIMINATION,
    default=False,
)

# Unsqueeze elimination - folds unsqueeze operations into initializers
UNSQUEEZE_ELIMINATION = BoolCapability(
    name="unsqueeze-elimination",
    ort_name="UnsqueezeElimination",
    description="Eliminate Unsqueeze of initializers (fold into weights)",
    category=CapabilityCategory.ELIMINATION,
    default=False,
)

# NOTE: ReshapeElimination was removed - this optimizer does not exist in ORT.
# Verified against ort_optimizer_inventory.md - no "ReshapeElimination" found.
# ReshapeFusion exists (L1) but that's different - it fuses patterns, not eliminates.
