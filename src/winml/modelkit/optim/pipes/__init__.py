# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Optimization pipes for ONNX models.

Shape inference is a mandatory Optimizer stage. Capability-driven pipes run
before, during, and after ORT graph optimization according to their needs.
"""

from typing import Any

from .algebraic import (
    ALGEBRAIC_CAPABILITIES,
    AlgebraicRewritePipe,
    AlgebraicRewritePipeConfig,
)
from .base import BasePipe, OptimizationError, PipeConfig, caps_dict
from .fusion import ORTFusionPipe, ORTFusionPipeConfig
from .graph import GRAPH_CAPABILITIES, ORTGraphPipe, ORTGraphPipeConfig
from .rewrite import RewritePipe, RewritePipeConfig
from .surgery import (
    PRE_SURGERY_CAPABILITIES,
    SURGERY_CAPABILITIES,
    PreSurgeryPipe,
    SurgeryPipe,
    SurgeryPipeConfig,
)


# Optimization pipes to run in sequence
# - PreSurgeryPipe: Proof-dependent graph rewrites that must inspect the exported graph.
# - ORTGraphPipe: ORT graph-level optimizations (C++ optimizer), including constant folding.
#   Runs before downstream pattern pipes so they see a constant-folded graph.
# - AlgebraicRewritePipe: Exact topology-based algebraic rewrites (after ORT folding).
# - RewritePipe: Pattern-based subgraph rewriting (runs after ORT constant folding so that
#   shape constants are visible, but before ORTFusionPipe so normalised patterns are
#   available for transformer fusions).
# - ORTFusionPipe: ORT transformer fusions (Python optimizer)
# - SurgeryPipe: Post-optimization model surgery (runs last to clamp constants after folding)
PIPES: list[type[BasePipe]] = [
    PreSurgeryPipe,
    ORTGraphPipe,
    AlgebraicRewritePipe,
    RewritePipe,
    ORTFusionPipe,
    SurgeryPipe,
]


def get_all_capabilities() -> dict[str, Any]:
    """Get all capabilities from all registered pipes.

    Returns:
        Dictionary mapping capability names to capability definitions from all pipes
    """
    all_caps = {}
    for pipe_class in PIPES:
        all_caps.update(pipe_class.capabilities)
    return all_caps


__all__ = [
    "ALGEBRAIC_CAPABILITIES",
    "GRAPH_CAPABILITIES",
    "PIPES",
    "PRE_SURGERY_CAPABILITIES",
    "SURGERY_CAPABILITIES",
    "AlgebraicRewritePipe",
    "AlgebraicRewritePipeConfig",
    "BasePipe",
    "ORTFusionPipe",
    "ORTFusionPipeConfig",
    "ORTGraphPipe",
    "ORTGraphPipeConfig",
    "OptimizationError",
    "PipeConfig",
    "PreSurgeryPipe",
    "RewritePipe",
    "RewritePipeConfig",
    "SurgeryPipe",
    "SurgeryPipeConfig",
    "caps_dict",
    "get_all_capabilities",
]
