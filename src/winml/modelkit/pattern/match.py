# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pattern match result data models.

This module contains pure data classes for pattern matching results,
shared between modelkit.pattern and modelkit.analyze.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from onnx import NodeProto

    from .base import Pattern, PatternMatcher
    from .models import Pattern as PatternModel


@dataclass
class InputInfo:
    """Information about a pattern input.

    Attributes:
        name: Input name matching the schema.
        shape: Shape tuple if can be inferred from model, None otherwise.
        value: Numpy array if it is constant or initializer, None otherwise.
        is_constant: Whether the input is a constant or initializer.
    """

    name: str
    shape: tuple[int | str | None, ...] | None = None
    value: np.ndarray | None = None
    is_constant: bool = False


@dataclass
class SkeletonMatchResult:
    """Result of matching a pattern skeleton in an ONNX graph.

    Attributes:
        pattern: The pattern that was matched (Pattern instance or SubgraphPattern pydantic model).
        matched_nodes: List of matched NodeProto objects (actual nodes only, no virtual inputs)
                      in the same order as the skeleton's node_op_types.
        matcher: Reference to the PatternMatcher for accessing lookup tables.
        inputs: List of input edge names corresponding to virtual nodes
                (-1, -2, -3, ...) in that order.
        output: Output edge name from the exit node.
        removable: True if the skeleton nodes can be safely removed without leaving
                  dangling tensor references. A skeleton is removable iff none of the
                  intermediate tensors (outputs of skeleton nodes, excluding the final
                  skeleton output) are consumed by nodes outside the skeleton.
        matched_node_keys: List of stable node keys aligned with matched_nodes.
    """

    pattern: "Pattern | PatternModel"  # Pattern ABC instance or Pydantic Pattern model
    matched_nodes: list["NodeProto"]
    matcher: "PatternMatcher" = field(repr=False)  # PatternMatcher reference
    inputs: list[str] = field(default_factory=list)
    output: str = ""
    removable: bool = False
    matched_node_keys: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate that stable node keys are provided for each matched node."""
        if len(self.matched_node_keys) != len(self.matched_nodes):
            raise ValueError(
                "matched_node_keys must be provided and aligned with matched_nodes "
                f"(got {len(self.matched_node_keys)} keys for {len(self.matched_nodes)} nodes)"
            )


@dataclass
class PatternMatchResult:
    """Result of successfully matching and validating a pattern in an ONNX graph.

    This class represents a detected pattern instance in a model graph, containing
    both the matching information and validation metadata.

    Attributes:
        skeleton_match_result: The underlying skeleton match result.
        schema_input_to_value: Map from schema input names to actual tensor value names.
        schema_output_to_value: Map from schema output names to actual tensor value names.
        type_param_to_type: Map from type parameter strings to actual type strings.
        attributes: Map of inferred attributes from PatternSchema.
        input_infos: Map from input names to InputInfo objects.
        match_id: Unique identifier for this match instance.
    """

    skeleton_match_result: SkeletonMatchResult
    schema_input_to_value: dict[str, str]
    schema_output_to_value: dict[str, str]
    type_param_to_type: dict[str, str]
    attributes: dict[str, Any] = field(default_factory=dict)
    input_infos: dict[str, InputInfo] = field(default_factory=dict)
    match_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def pattern(self) -> Pattern | PatternModel:
        """Get the pattern that was matched."""
        return self.skeleton_match_result.pattern

    @property
    def pattern_id(self) -> str:
        """Get the pattern identifier.

        Attempts to extract pattern_id from the pattern object. Falls back
        to schema-based naming or "SUBGRAPH/Unknown" if unavailable.

        Returns:
            Pattern identifier string (e.g., "SUBGRAPH/Gelu1").
        """
        pattern = self.skeleton_match_result.pattern
        # Handle both Pattern instances and SubgraphPattern pydantic models
        if hasattr(pattern, "pattern_id"):
            return pattern.pattern_id
        # Fallback for Pattern instances without pattern_id
        if hasattr(pattern, "get_schema"):
            return f"SUBGRAPH/{pattern.get_schema().name}"
        return "SUBGRAPH/Unknown"

    @property
    def matched_nodes(self) -> list[str]:
        """Get matched stable node keys as strings.

        Returns:
            List of stable node keys (e.g., ["node1", "node_0"]).
        """
        return self.skeleton_match_result.matched_node_keys

    @property
    def matched_node_keys(self) -> list[str]:
        """Get matched stable node keys as strings."""
        return self.skeleton_match_result.matched_node_keys

    @property
    def matched_node_names(self):
        """Get matched nodes as ONNXOp objects.

        Note: Despite the name, this returns ONNXOp objects, not strings.
        This is for backward compatibility. Use matched_nodes for string names.

        Returns:
            List of ONNXOp instances containing node metadata (when used from analyze).
            Falls back to dicts when ONNXOp is not available.
        """
        try:
            from ..analyze import ONNXOp

            node_keys = self.skeleton_match_result.matched_node_keys

            return [
                ONNXOp(
                    node_name=node_keys[idx],
                    op_type=node.op_type,
                    namespace=node.domain if node.domain else "ai.onnx",
                )
                for idx, node in enumerate(self.skeleton_match_result.matched_nodes)
            ]
        except ImportError:
            # When used outside analyze context, return node info as dicts
            node_keys = self.skeleton_match_result.matched_node_keys
            return [
                {
                    "node_name": node_keys[idx],
                    "op_type": node.op_type,
                    "namespace": node.domain if node.domain else "ai.onnx",
                }
                for idx, node in enumerate(self.skeleton_match_result.matched_nodes)
            ]

    @property
    def type_vars(self) -> dict[str, str]:
        """Alias for type_param_to_type for API compatibility."""
        return self.type_param_to_type

    @property
    def input_shapes(self) -> dict[str, tuple[int | str | None, ...] | None]:
        """Extract input shapes from input_infos.

        Returns:
            Dictionary mapping input names to shape tuples.
            Shape elements can be int (concrete), str (symbolic), or None (unknown).
        """
        return {name: info.shape for name, info in self.input_infos.items()}
