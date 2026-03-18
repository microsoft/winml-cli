"""Activation function fusion capabilities.

This module defines activation-related fusion optimizations beyond GELU.
These optimizations fuse activation functions with adjacent operations like
bias addition, dropout, and clipping to reduce memory access and improve
performance.

Activation fusion capabilities combine common patterns like Bias+Softmax,
Bias+Dropout, and ReLU+Clip into single fused operations that reduce
intermediate memory allocations and improve execution efficiency.
"""

from __future__ import annotations

from ..registry import BoolCapability, CapabilityCategory


# Bias + Softmax fusion - fuses bias addition with softmax operation
BIAS_SOFTMAX_FUSION = BoolCapability(
    name="bias-softmax-fusion",
    ort_name="BiasSoftmaxFusion",
    description="Fuse Bias+Softmax into single operation",
    category=CapabilityCategory.ACTIVATION,
    default=False,
)

# Bias + Dropout fusion - fuses bias addition with dropout
BIAS_DROPOUT_FUSION = BoolCapability(
    name="bias-dropout-fusion",
    ort_name="BiasDropoutFusion",
    description="Fuse Bias+Dropout patterns",
    category=CapabilityCategory.ACTIVATION,
    default=False,
)

# NOTE: FuseReluClip was removed - this optimizer only runs at Level 1, not Level 2.
# Verified against ort_optimizer_inventory.md - FuseReluClip is in L1 RewriteRule section only.
