# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Opt-in CGC compatibility patterns and model metadata rewrites."""

from .cgc_constant_folding import cgc_constant_folding, fold_constant_pad_pads
from .dft_patterns import DFTWithStaticParametersPattern, MatMulDFTPattern
from .dq_rewrites import normalize_int32_dq
from .gathernd_patterns import GatherNDWithIdentityIndicesPattern, ReshapedGatherNDPattern
from .gridsample_patterns import GatherLinearGridSamplePattern, LinearGridSamplePattern
from .identity_rewrites import eliminate_identity
from .opset_rewrites import deduplicate_opset_imports
from .prelu_patterns import ExpandedPReluPattern, PReluWithFiniteSlopePattern
from .resize_patterns import (
    ResizeWithAsymmetricCoordinatesPattern,
    ResizeWithCubicInterpolationPattern,
    ResizeWithEmptyOptionalInputsPattern,
    ResizeWithLinearInterpolationPattern,
    ResizeWithOmittedOptionalInputsPattern,
    ResizeWithTfHalfPixelForNNPattern,
)


__all__ = [
    "DFTWithStaticParametersPattern",
    "ExpandedPReluPattern",
    "GatherLinearGridSamplePattern",
    "GatherNDWithIdentityIndicesPattern",
    "LinearGridSamplePattern",
    "MatMulDFTPattern",
    "PReluWithFiniteSlopePattern",
    "ReshapedGatherNDPattern",
    "ResizeWithAsymmetricCoordinatesPattern",
    "ResizeWithCubicInterpolationPattern",
    "ResizeWithEmptyOptionalInputsPattern",
    "ResizeWithLinearInterpolationPattern",
    "ResizeWithOmittedOptionalInputsPattern",
    "ResizeWithTfHalfPixelForNNPattern",
    "cgc_constant_folding",
    "deduplicate_opset_imports",
    "eliminate_identity",
    "fold_constant_pad_pads",
    "normalize_int32_dq",
]
