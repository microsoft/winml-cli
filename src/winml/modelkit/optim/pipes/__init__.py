# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Optimization pipes for ONNX models.

Note: Shape inference (symbolic and standard) are now mandatory stages
in the Optimizer class, not configurable pipes. Only ORTGraphPipe and
ORTFusionPipe remain as capability-driven optimization pipes.
"""

from typing import Any

from .base import BasePipe, OptimizationError, PipeConfig, caps_dict
from .fusion import ORTFusionPipe, ORTFusionPipeConfig
from .graph import GRAPH_CAPABILITIES, ORTGraphPipe, ORTGraphPipeConfig
from .rewrite import RewritePipe, RewritePipeConfig
from .surgery import SURGERY_CAPABILITIES, SurgeryPipe, SurgeryPipeConfig


# Optimization pipes to run in sequence
# - RewritePipe: Pattern-based subgraph rewriting (runs before ORT to normalise the graph)
# - ORTGraphPipe: ORT graph-level optimizations (C++ optimizer)
# - ORTFusionPipe: ORT transformer fusions (Python optimizer)
# - SurgeryPipe: Post-optimization model surgery (runs last to clamp constants after folding)
PIPES: list[type[BasePipe]] = [RewritePipe, ORTGraphPipe, ORTFusionPipe, SurgeryPipe]


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
    "GRAPH_CAPABILITIES",
    "PIPES",
    "SURGERY_CAPABILITIES",
    "BasePipe",
    "ORTFusionPipe",
    "ORTFusionPipeConfig",
    "ORTGraphPipe",
    "ORTGraphPipeConfig",
    "OptimizationError",
    "PipeConfig",
    "RewritePipe",
    "RewritePipeConfig",
    "SurgeryPipe",
    "SurgeryPipeConfig",
    "caps_dict",
    "get_all_capabilities",
]
