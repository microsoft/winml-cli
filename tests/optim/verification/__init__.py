# Verification tests for optimization module
"""Verification utilities for optimization capability testing.

This module provides re-exported verification functions from conftest.py and
additional helper functions for analyzing ONNX models.

Core verification function uses 4-criteria capability tests:
1. Node Count: Model should have fewer nodes after optimization
2. Target Effect: Expected fused ops MUST exist in optimized model
3. Locality: Other fused ops MUST NOT exist (isolation verification)
4. Numeric Verification: Outputs must match within tolerance (optional)

Usage:
    from tests.optim.verification import verify_capability_effect

    verify_capability_effect(
        model_before=baseline,
        model_after=optimized,
        existence_list=["BiasGelu"],
        non_existence_list=["Attention", "SkipLayerNormalization"],
        min_node_reduction=1,
        verify_numeric=True,  # Optional numeric verification
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export the canonical verify_capability_effect from conftest.py
from ..conftest import verify_capability_effect


if TYPE_CHECKING:
    import onnx


def get_op_types(model: onnx.ModelProto) -> set[str]:
    """Get all unique operation types in a model.

    Args:
        model: ONNX model to analyze.

    Returns:
        Set of operation type strings.
    """
    return {node.op_type for node in model.graph.node}


def count_op_type(model: onnx.ModelProto, op_type: str) -> int:
    """Count occurrences of a specific operation type.

    Args:
        model: ONNX model to analyze.
        op_type: Operation type to count.

    Returns:
        Number of nodes with the specified op_type.
    """
    return sum(1 for node in model.graph.node if node.op_type == op_type)


def has_fused_op(model: onnx.ModelProto, op_type: str) -> bool:
    """Check if model contains a specific fused operation.

    Args:
        model: ONNX model to check.
        op_type: Fused operation type to look for.

    Returns:
        True if the op_type exists in the model.
    """
    return op_type in get_op_types(model)


# Export public API
__all__ = [
    "count_op_type",
    "get_op_types",
    "has_fused_op",
    "verify_capability_effect",
]
