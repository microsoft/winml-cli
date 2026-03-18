# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""GELU activation fusion capabilities.

This module defines GELU activation fusion optimizations. These optimizations
fuse multiple operations into single GELU operations, improving performance.

GELU (Gaussian Error Linear Unit) is a smooth activation function commonly
used in transformer architectures. These fusions detect and combine multi-op
GELU approximation patterns into single operations.
"""

from __future__ import annotations

from ..registry import BoolCapability, CapabilityCategory


# Standard GELU fusion - fuses tanh-based GELU approximation
# Note: We use graph_optimization_level=2, so we only need L2 optimizer names
GELU_FUSION = BoolCapability(
    name="gelu-fusion",
    ort_name="GeluFusionL2",
    description="Fuse multi-operation GELU approximation patterns into single GELU op",
    category=CapabilityCategory.GELU,
    default=False,
)

# Fast GELU fusion - fuses fast GELU approximation variant (tanh-based)
# NOTE: FastGeluFusion matches tanh approximation pattern, NOT Erf-based GELU
# Pattern: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) → FastGelu
FAST_GELU_FUSION = BoolCapability(
    name="fast-gelu-fusion",
    ort_name="FastGeluFusion",
    description="Fuse fast GELU approximation patterns into optimized operation",
    category=CapabilityCategory.GELU,
    default=False,
    # No dependency - FastGeluFusion works on tanh pattern, not Erf-based decomposed GELU
)

# Bias + GELU fusion - fuses bias addition with GELU activation
BIAS_GELU_FUSION = BoolCapability(
    name="bias-gelu-fusion",
    ort_name="BiasGeluFusion",
    description="Fuse bias addition and GELU activation into single operation",
    category=CapabilityCategory.GELU,
    default=False,
    depends_on=("gelu-fusion",),
)

# Quick GELU fusion - fuses QuickGelu variant patterns (sigmoid-based)
# NOTE: QuickGeluFusion matches sigmoid approximation pattern
# Pattern: x * sigmoid(1.702 * x) → FastGelu
# QuickGelu produces FastGelu op (same as FastGeluFusion)
QUICK_GELU_FUSION = BoolCapability(
    name="quick-gelu-fusion",
    ort_name="QuickGeluFusion",
    description="Fuse QuickGelu variant patterns (x * sigmoid(1.702 * x))",
    category=CapabilityCategory.GELU,
    default=False,
    # No dependency - QuickGeluFusion works on sigmoid pattern, not Erf-based decomposed GELU
)

# GELU approximation - converts exact GELU to fast approximation
# NOTE: GeluApproximation converts native Gelu → FastGelu
# It needs native Gelu ops which come from GeluFusionL2
GELU_APPROXIMATION = BoolCapability(
    name="gelu-approximation",
    ort_name="GeluApproximation",
    description="Convert exact Gelu/BiasGelu to FastGelu for improved inference speed",
    category=CapabilityCategory.GELU,
    default=False,
    depends_on=("gelu-fusion",),
)
