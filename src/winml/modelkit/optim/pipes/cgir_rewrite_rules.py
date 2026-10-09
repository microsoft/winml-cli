# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rule declarations for CGIR-specific model rewrites."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from onnx import ModelProto

from ...pattern import Pattern
from ...pattern.cgc import (
    DFTWithStaticParametersPattern,
    ExpandedPReluPattern,
    GatherLinearGridSamplePattern,
    GatherNDWithIdentityIndicesPattern,
    LinearGridSamplePattern,
    MatMulDFTPattern,
    PReluWithFiniteSlopePattern,
    ReshapedGatherNDPattern,
    ResizeWithAsymmetricCoordinatesPattern,
    ResizeWithCubicInterpolationPattern,
    ResizeWithEmptyOptionalInputsPattern,
    ResizeWithLinearInterpolationPattern,
    ResizeWithOmittedOptionalInputsPattern,
    ResizeWithTfHalfPixelForNNPattern,
    cgc_constant_folding,
    deduplicate_opset_imports,
    normalize_int32_dq,
)
from ..registry import BoolCapability, CapabilityCategory


@dataclass(frozen=True)
class CGIRRewriteRule:
    """A CGIR rewrite and its model-level opset requirement."""

    capability: BoolCapability
    source: type[Pattern]
    target: type[Pattern]
    minimum_opset: int
    aliases: tuple[BoolCapability, ...] = ()
    warning: str | None = None


@dataclass(frozen=True)
class CGIRModelRewriteRule:
    """A model-level rewrite, independent of operator pattern matching."""

    capability: BoolCapability
    transform: Callable[[ModelProto], ModelProto]
    aliases: tuple[BoolCapability, ...] = ()
    minimum_opset: int = 0


DEDUPLICATE_OPSET_IMPORTS = BoolCapability(
    name="deduplicate-opset-imports",
    ort_name=None,
    description="Remove repeated identical model opset imports before CGIR conversion",
    category=CapabilityCategory.REWRITE,
    default=False,
)

OMIT_EMPTY_RESIZE_INPUTS = BoolCapability(
    name="omit-empty-resize-inputs",
    ort_name=None,
    description="Represent empty Resize ROI and scales as omitted inputs for CGIR",
    category=CapabilityCategory.REWRITE,
    default=False,
)

CGC_CONSTANT_FOLDING = BoolCapability(
    name="cgc-constant-folding",
    ort_name=None,
    description="Fill FoundryToolbox folding gaps for static shape subgraphs",
    category=CapabilityCategory.REWRITE,
    default=False,
)

RESIZE_TF_HALF_PIXEL_FOR_NN_TO_ASYMMETRIC = BoolCapability(
    name="resize-tf-half-pixel-for-nn-to-asymmetric",
    ort_name=None,
    description=(
        "Use asymmetric coordinates for nearest/floor Resize with static positive integer scales"
    ),
    category=CapabilityCategory.REWRITE,
    default=False,
)

APPROXIMATE_CUBIC_RESIZE_WITH_LINEAR = BoolCapability(
    name="approximate-cubic-resize-with-linear",
    ort_name=None,
    description=(
        "Lossy cubic-to-linear Resize approximation without antialiasing or outside exclusion"
    ),
    category=CapabilityCategory.REWRITE,
    default=False,
)

GATHERND_TO_RESHAPE = BoolCapability(
    name="gathernd-to-reshape",
    ort_name=None,
    description=(
        "Replace unequal-rank GatherND with Reshape when static indices preserve all data in order"
    ),
    category=CapabilityCategory.REWRITE,
    default=False,
)

PRELU_TO_RELU = BoolCapability(
    name="prelu-to-relu",
    ort_name=None,
    description="Decompose floating-point PRelu with finite constant slope into Relu/Neg/Mul/Sub",
    category=CapabilityCategory.REWRITE,
    default=False,
)

DFT_TO_MATMUL = BoolCapability(
    name="dft-to-matmul",
    ort_name=None,
    description=(
        "Decompose DFT with static axis and signal length into real-valued matrix multiplications"
    ),
    category=CapabilityCategory.REWRITE,
    default=False,
)

GRIDSAMPLE_TO_GATHER = BoolCapability(
    name="gridsample-to-gather",
    ort_name=None,
    description="Decompose 2D linear zero-padded GridSample into GatherND and FP32 interpolation",
    category=CapabilityCategory.REWRITE,
    default=False,
)

NORMALIZE_INT32_DQ = BoolCapability(
    name="normalize-int32-dq",
    ort_name=None,
    description="Omit constant INT32 DQ zero points and scalarize singleton scales for CGIR",
    category=CapabilityCategory.REWRITE,
    default=False,
)

CGIR_REWRITE_RULES = (
    CGIRModelRewriteRule(
        capability=NORMALIZE_INT32_DQ,
        transform=normalize_int32_dq,
    ),
    CGIRModelRewriteRule(
        capability=CGC_CONSTANT_FOLDING,
        transform=cgc_constant_folding,
    ),
    CGIRModelRewriteRule(
        capability=DEDUPLICATE_OPSET_IMPORTS,
        transform=deduplicate_opset_imports,
    ),
    CGIRRewriteRule(
        capability=OMIT_EMPTY_RESIZE_INPUTS,
        source=ResizeWithEmptyOptionalInputsPattern,
        target=ResizeWithOmittedOptionalInputsPattern,
        minimum_opset=13,
    ),
    CGIRRewriteRule(
        capability=RESIZE_TF_HALF_PIXEL_FOR_NN_TO_ASYMMETRIC,
        source=ResizeWithTfHalfPixelForNNPattern,
        target=ResizeWithAsymmetricCoordinatesPattern,
        minimum_opset=11,
    ),
    CGIRRewriteRule(
        capability=APPROXIMATE_CUBIC_RESIZE_WITH_LINEAR,
        source=ResizeWithCubicInterpolationPattern,
        target=ResizeWithLinearInterpolationPattern,
        minimum_opset=11,
        warning=(
            "Replacing cubic Resize with linear is a lossy approximation and may reduce accuracy."
        ),
    ),
    CGIRRewriteRule(
        capability=GATHERND_TO_RESHAPE,
        source=GatherNDWithIdentityIndicesPattern,
        target=ReshapedGatherNDPattern,
        minimum_opset=11,
    ),
    CGIRRewriteRule(
        capability=PRELU_TO_RELU,
        source=PReluWithFiniteSlopePattern,
        target=ExpandedPReluPattern,
        minimum_opset=7,
    ),
    CGIRRewriteRule(
        capability=DFT_TO_MATMUL,
        source=DFTWithStaticParametersPattern,
        target=MatMulDFTPattern,
        minimum_opset=17,
    ),
    CGIRRewriteRule(
        capability=GRIDSAMPLE_TO_GATHER,
        source=LinearGridSamplePattern,
        target=GatherLinearGridSamplePattern,
        minimum_opset=16,
    ),
)

CGIR_REWRITE_CAPABILITIES = {
    capability.name: capability
    for rule in CGIR_REWRITE_RULES
    for capability in (rule.capability, *rule.aliases)
}


__all__ = [
    "APPROXIMATE_CUBIC_RESIZE_WITH_LINEAR",
    "CGC_CONSTANT_FOLDING",
    "CGIR_REWRITE_CAPABILITIES",
    "CGIR_REWRITE_RULES",
    "DEDUPLICATE_OPSET_IMPORTS",
    "DFT_TO_MATMUL",
    "GATHERND_TO_RESHAPE",
    "GRIDSAMPLE_TO_GATHER",
    "OMIT_EMPTY_RESIZE_INPUTS",
    "PRELU_TO_RELU",
    "RESIZE_TF_HALF_PIXEL_FOR_NN_TO_ASYMMETRIC",
    "CGIRModelRewriteRule",
    "CGIRRewriteRule",
]
