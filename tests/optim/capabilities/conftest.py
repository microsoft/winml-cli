"""Capability-specific pytest fixtures for differential testing.

This module provides specialized fixtures for testing individual optimization
capabilities by comparing their effects in isolation using differential analysis.

CRITICAL (Section 9.6): This module MUST NOT import from winml.modelkit.optim.pipes
or any Pipe classes. All ORT optimizer names are defined directly here based
on the design document Appendix A.

Key Features:
    - Import utilities from parent conftest (optimize_at_level, ALL_PATTERNS, helpers)
    - ALL_ORT_OPTIMIZER_NAMES defined directly (no Pipe dependency)
    - Differential testing support (capability enabled vs disabled)

Functions:
    get_all_ort_names: Return all ORT optimizer names (from constant list)
"""

from __future__ import annotations


# =============================================================================
# ALL ORT LEVEL 2 OPTIMIZER NAMES (from design doc Appendix A)
# =============================================================================
# These are ALL 58 Level 2 optimizers. Defined directly here to avoid
# importing from winml.modelkit.optim.pipes (per Section 9.6 requirement).

ALL_ORT_OPTIMIZER_NAMES: tuple[str, ...] = (
    # =========================================================================
    # Elimination Optimizers (8)
    # =========================================================================
    "EliminateIdentity",
    "EliminateDropout",
    "NoopElimination",
    "CastElimination",
    "EliminateSlice",
    "ExpandElimination",
    "UnsqueezeElimination",
    "ReshapeElimination",
    # =========================================================================
    # Core Graph Optimizers (6)
    # =========================================================================
    "ConstantFolding",
    "ConstantSharing",
    "CommonSubexpressionElimination",
    "ReshapeFusion",
    "ConcatSliceElimination",
    "FreeDimensionOverrideDenotation",
    # =========================================================================
    # LayerNorm Optimizers (5) - Only LayerNormFusion has L2 variant
    # Reference: onnxruntime/core/optimizer/graph_transformer_utils.cc
    # =========================================================================
    "LayerNormFusion",
    "LayerNormFusionL2",  # Only this one has L2 variant
    "SimplifiedLayerNormFusion",
    "SkipLayerNormFusion",
    "EmbedLayerNormFusion",
    # =========================================================================
    # GELU Optimizers (5) - Only GeluFusion has L2 variant
    # Reference: onnxruntime/core/optimizer/graph_transformer_utils.cc
    # =========================================================================
    "GeluFusion",
    "GeluFusionL2",  # Only this one has L2 variant
    "FastGeluFusion",
    "BiasGeluFusion",
    "GeluApproximation",
    # =========================================================================
    # QuickGelu Optimizer (1)
    # =========================================================================
    "QuickGeluFusion",
    # =========================================================================
    # Convolution Optimizers (5)
    # =========================================================================
    "ConvBNFusion",
    "ConvAddFusion",
    "ConvMulFusion",
    "ConvActivationFusion",
    "ConvAddActivationFusion",
    # =========================================================================
    # MatMul/GEMM Optimizers (6) - No L2 variants
    # Reference: onnxruntime/core/optimizer/graph_transformer_utils.cc
    # =========================================================================
    "MatMulAddFusion",
    "MatMulScaleFusion",
    "MatMulIntegerToFloatFusion",
    "MatMulActivationFusion",
    "MatmulTransposeFusion",
    "GemmActivationFusion",
    # =========================================================================
    # Attention Optimizers (1) - No L2 variant
    # Reference: onnxruntime/core/optimizer/graph_transformer_utils.cc
    # =========================================================================
    "AttentionFusion",
    # =========================================================================
    # Miscellaneous Optimizers (15)
    # =========================================================================
    "NchwcTransformer",
    "NhwcTransformer",
    "TransposeOptimizer_CpuExecutionProvider",
    "BiasDropoutFusion",
    "BiasSoftmaxFusion",
    "BiasSoftmax",
    "DivMulFusion",
    "MulReluFusion",
    "AddMulFusion",
    "QDQPropagation",
    "QDQSelectorAction",
    "ConvActivationQuantFusion",
    "DoubleQDQPairsRemover",
    "EnsureUniqueDQForNodeUnit",
    "SliceElimination",
    # =========================================================================
    # Additional Fusion Optimizers
    # =========================================================================
    "MatMul_BatchNormalization_Fusion",
    "GemmSumFusion",
    "GemmTransposeFusion",
    "GatherSliceToSplitFusion",
    "GatherToSliceFusion",
    "Pad_Fusion",
    "NotWhereFusion",
    "FuseReluClip",
    "DynamicQuantizeMatMulFusion",
    "BiasSkipLayerNorm",
)


# =============================================================================
# Registry Analysis Functions
# =============================================================================


def get_all_ort_names() -> list[str]:
    """Return all ORT optimizer names from constant list.

    Per Section 9.6, this function does NOT read from Pipe classes.
    Instead, it returns the constant ALL_ORT_OPTIMIZER_NAMES defined
    in this module based on the design document Appendix A.

    Returns:
        List of all ORT Level 2 optimizer names.

    Example:
        >>> ort_names = get_all_ort_names()
        >>> "GeluFusionL2" in ort_names
        True
        >>> "BiasGeluFusion" in ort_names
        True
    """
    return list(ALL_ORT_OPTIMIZER_NAMES)
