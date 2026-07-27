# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Shared pattern matching infrastructure for WinML CLI.

This package provides pattern matching, input generation, and graph rewriting
infrastructure used by both the static analyzer and the optimizer.
"""

from .attention_patterns import (
    ExpandedAttentionPattern,
    ExpandedAttentionPatternInputGenerator,
    TransposeAttentionPattern,
    TransposeAttentionPatternInputGenerator,
)
from .base import (
    InvalidPatternMatcherModelError,
    Pattern,
    PatternInputGenerator,
    PatternMatcher,
    PatternMismatchedError,
    PatternRewriter,
    PatternSchema,
    Skeleton,
    get_pattern_input_generator,
    get_registered_pattern_input_generators,
    make_single_op_pattern,
    opschema_to_pattern_schema,
    register_pattern_input_generator,
)
from .conv2d_inplace_linear_patterns import (
    Conv2DInplaceLinear2DPattern,
    Conv2DInplaceLinear2DPatternInputGenerator,
    Conv2DInplaceLinear3DPattern,
    Conv2DInplaceLinear3DPatternInputGenerator,
    Conv2DInplaceLinear4DPattern,
    Conv2DInplaceLinear4DPatternInputGenerator,
    Conv2DInplaceLinearInputGeneratorBase,
)
from .conv_batchnorm_patterns import (
    CONV_ADD_BATCHNORM_SCHEMA,
    AddConvBatchNormalizationPattern,
    ConvAddBatchNormalizationPattern,
    FoldedConvAddPattern,
)
from .gelu_patterns import (
    Gelu1Pattern,
    Gelu1PatternInputGenerator,
    Gelu2Pattern,
    Gelu2PatternInputGenerator,
    Gelu3Pattern,
    Gelu3PatternInputGenerator,
    Gelu4Pattern,
    Gelu4PatternInputGenerator,
    SingleGeluPattern,
)
from .gemm_patterns import (
    MATMUL_ADD_SCHEMA,
    MatMulAddPattern,
    MatMulAddPatternInputGenerator,
    ReshapeGemmReshapePattern,
    ReshapeGemmReshapePatternInputGenerator,
)
from .layernorm_patterns import (
    LayerNormalizationMulPattern,
    LayerNormalizationMulPatternInputGenerator,
    LayerNormalizationPowPattern,
    LayerNormalizationPowPatternInputGenerator,
    TransposedSingleLayerNormalizationPattern,
    TransposedSingleLayerNormalizationPatternInputGenerator,
)
from .match import InputInfo, PatternMatchResult, SkeletonMatchResult
from .models import OperatorPattern, PatternType, SubgraphPattern
from .rmsnorm_patterns import (
    RMSNormalizationMulPattern,
    RMSNormalizationMulPatternInputGenerator,
    RMSNormalizationPowPattern,
    RMSNormalizationPowPatternInputGenerator,
    TransposedSingleRMSNormalizationPattern,
    TransposedSingleRMSNormalizationPatternInputGenerator,
)
from .transpose_patterns import (
    ReshapeTransposeReshapeLowDimPattern,
    ReshapeTransposeReshapeLowDimPatternInputGenerator,
    ReshapeTransposeReshapeOverlyHighDimPattern,
    ReshapeTransposeReshapeOverlyHighDimPatternInputGenerator,
)


__all__ = [
    "CONV_ADD_BATCHNORM_SCHEMA",
    "MATMUL_ADD_SCHEMA",
    "AddConvBatchNormalizationPattern",
    "Conv2DInplaceLinear2DPattern",
    "Conv2DInplaceLinear2DPatternInputGenerator",
    "Conv2DInplaceLinear3DPattern",
    "Conv2DInplaceLinear3DPatternInputGenerator",
    "Conv2DInplaceLinear4DPattern",
    "Conv2DInplaceLinear4DPatternInputGenerator",
    "Conv2DInplaceLinearInputGeneratorBase",
    "ConvAddBatchNormalizationPattern",
    "ExpandedAttentionPattern",
    "ExpandedAttentionPatternInputGenerator",
    "FoldedConvAddPattern",
    "Gelu1Pattern",
    "Gelu1PatternInputGenerator",
    "Gelu2Pattern",
    "Gelu2PatternInputGenerator",
    "Gelu3Pattern",
    "Gelu3PatternInputGenerator",
    "Gelu4Pattern",
    "Gelu4PatternInputGenerator",
    "InputInfo",
    "InvalidPatternMatcherModelError",
    "LayerNormalizationMulPattern",
    "LayerNormalizationMulPatternInputGenerator",
    "LayerNormalizationPowPattern",
    "LayerNormalizationPowPatternInputGenerator",
    "MatMulAddPattern",
    "MatMulAddPatternInputGenerator",
    "OperatorPattern",
    "Pattern",
    "PatternInputGenerator",
    "PatternMatchResult",
    "PatternMatcher",
    "PatternMismatchedError",
    "PatternRewriter",
    "PatternSchema",
    "PatternType",
    "RMSNormalizationMulPattern",
    "RMSNormalizationMulPatternInputGenerator",
    "RMSNormalizationPowPattern",
    "RMSNormalizationPowPatternInputGenerator",
    "ReshapeGemmReshapePattern",
    "ReshapeGemmReshapePatternInputGenerator",
    "ReshapeTransposeReshapeLowDimPattern",
    "ReshapeTransposeReshapeLowDimPatternInputGenerator",
    "ReshapeTransposeReshapeOverlyHighDimPattern",
    "ReshapeTransposeReshapeOverlyHighDimPatternInputGenerator",
    "SingleGeluPattern",
    "Skeleton",
    "SkeletonMatchResult",
    "SubgraphPattern",
    "TransposeAttentionPattern",
    "TransposeAttentionPatternInputGenerator",
    "TransposedSingleLayerNormalizationPattern",
    "TransposedSingleLayerNormalizationPatternInputGenerator",
    "TransposedSingleRMSNormalizationPattern",
    "TransposedSingleRMSNormalizationPatternInputGenerator",
    "get_pattern_input_generator",
    "get_registered_pattern_input_generators",
    "make_single_op_pattern",
    "opschema_to_pattern_schema",
    "register_pattern_input_generator",
]
