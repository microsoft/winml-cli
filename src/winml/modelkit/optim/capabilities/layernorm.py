# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Layer normalization fusion capabilities.

This module defines layer normalization fusion optimizations. These optimizations
detect and fuse layer normalization patterns into efficient operations, improving
performance for transformer-based and normalization-heavy architectures.

Layer normalization is a fundamental building block in modern neural networks,
especially transformers. These fusions detect common normalization computation
patterns and replace them with optimized fused operations.
"""

from __future__ import annotations

from ..registry import BoolCapability, CapabilityCategory


# LayerNorm fusion - fuses multi-step LayerNorm computation
# Note: At Level 2, the disable key is "LayerNormFusionL2" (verified against ORT source).
# Level 1 has a separate "LayerNormFusion" optimizer.
LAYER_NORM_FUSION = BoolCapability(
    name="layer-norm-fusion",
    ort_name="LayerNormFusionL2",
    description="Fuse LayerNorm computation (ReduceMean→Sub→Pow→Sqrt→Div→Mul→Add)",
    category=CapabilityCategory.LAYER_NORM,
    default=False,
)

# Skip+LayerNorm fusion - fuses residual connection with LayerNorm
SKIP_LAYER_NORM_FUSION = BoolCapability(
    name="skip-layer-norm-fusion",
    ort_name="SkipLayerNormFusion",
    description="Fuse Add(residual)+LayerNorm into SkipLayerNormalization",
    category=CapabilityCategory.LAYER_NORM,
    default=False,
    depends_on=("layer-norm-fusion",),
)

# SimplifiedLayerNorm fusion - fuses simplified LayerNorm without mean-centering
SIMPLIFIED_LAYER_NORM_FUSION = BoolCapability(
    name="simplified-layer-norm-fusion",
    ort_name="SimplifiedLayerNormFusion",
    description="Fuse simplified LayerNorm (without mean-centering)",
    category=CapabilityCategory.LAYER_NORM,
    default=False,
)

# Embedding+LayerNorm fusion - fuses embeddings with LayerNorm
EMBED_LAYER_NORM_FUSION = BoolCapability(
    name="embed-layer-norm-fusion",
    ort_name="EmbedLayerNormFusion",
    description="Fuse embedding+position+token embeddings+LayerNorm",
    category=CapabilityCategory.LAYER_NORM,
    default=False,
    depends_on=("layer-norm-fusion",),
)

# Bias+Skip+LayerNorm fusion - fuses bias addition with SkipLayerNorm
# NOTE: This is a FusionOptions attribute (enable_bias_skip_layer_norm), NOT a graph optimizer.
# It controls whether the transformer optimizer fuses bias into SkipLayerNorm pattern.
# There is NO corresponding "BiasSkipLayerNorm" in the graph optimizer list.
BIAS_SKIP_LAYER_NORM_FUSION = BoolCapability(
    name="bias-skip-layer-norm-fusion",
    ort_name="BiasSkipLayerNormFusion",  # FusionOptions attr: enable_bias_skip_layer_norm
    description="Fuse Bias+Add(residual)+LayerNorm into BiasSkipLayerNorm (FusionPipe only)",
    category=CapabilityCategory.LAYER_NORM,
    default=False,
    depends_on=("skip-layer-norm-fusion",),
)

# RMSNorm → LpNormalization(p=2) fusion
# Custom fusion (not ORT built-in). Replaces decomposed RMSNorm subgraphs
# with LpNormalization(p=2, axis=-1) + weight*√N. Produces standard ONNX ops
# compatible with QNN EP. Applied after ORT's built-in fusions.
FUSE_RMSNORM = BoolCapability(
    name="fuse-rmsnorm",
    ort_name=None,  # Custom implementation, not ORT optimizer
    description="Fuse RMSNorm (Pow→ReduceMean→Add→Sqrt→Div→Mul) into LpNormalization(p=2)",
    category=CapabilityCategory.LAYER_NORM,
    default=False,
)
