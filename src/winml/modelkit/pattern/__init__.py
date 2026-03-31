# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Shared pattern matching infrastructure for ModelKit.

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
    "ExpandedAttentionPattern",
    "ExpandedAttentionPatternInputGenerator",
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
