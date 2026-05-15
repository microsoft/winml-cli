# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Node-level metadata system for ONNX models.

This module provides a comprehensive metadata system for tracking ONNX node origins,
transformations, optimizations, and semantic information through the WinML CLI pipeline.

Metadata is stored as custom ONNX node attributes with the 'winml.' prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import onnx


@dataclass
class NodeMetadata:
    """Metadata for an ONNX node.

    All values are stored as strings in ONNX attributes with the 'winml.' prefix.

    Attributes:
        name: ONNX node name (required).
        origin: Module that created this node - 'export', 'optimize', 'quantize',
            'compile', or 'external' (required).
        hierarchy_tag: Full hierarchical path from PyTorch module structure.
        hierarchy_depth: Depth in module hierarchy (stored as int, serialized as string).
        semantic_type: Semantic type with optional component (e.g., 'attention/query').
        semantic_layer_id: Layer index within model (e.g., '0', '1', '11').
        optim_applied: List of optimization names applied to this node.
        optim_sources: List of original node names that were fused into this node.
    """

    # Core (required)
    name: str
    origin: str  # export, optimize, quantize, compile, external

    # Hierarchy (optional)
    hierarchy_tag: str | None = None
    hierarchy_depth: int | None = None

    # Semantic (optional)
    semantic_type: str | None = None  # Format: "type" or "type/component"
    semantic_layer_id: str | None = None

    # Optimization (optional)
    optim_applied: list[str] = field(default_factory=list)
    optim_sources: list[str] = field(default_factory=list)

    def to_attributes(self) -> list[onnx.AttributeProto]:
        """Convert metadata to ONNX attributes.

        Returns:
            List of ONNX AttributeProto objects with 'winml.' prefix.
        """
        import onnx

        attrs = [
            onnx.helper.make_attribute("winml.node.name", self.name),
            onnx.helper.make_attribute("winml.node.origin", self.origin),
        ]

        if self.hierarchy_tag:
            attrs.append(onnx.helper.make_attribute("winml.hierarchy.tag", self.hierarchy_tag))
        if self.hierarchy_depth is not None:
            attrs.append(
                onnx.helper.make_attribute("winml.hierarchy.depth", str(self.hierarchy_depth))
            )
        if self.semantic_type:
            attrs.append(onnx.helper.make_attribute("winml.semantic.type", self.semantic_type))
        if self.semantic_layer_id:
            attrs.append(
                onnx.helper.make_attribute("winml.semantic.layer_id", self.semantic_layer_id)
            )
        if self.optim_applied:
            attrs.append(
                onnx.helper.make_attribute("winml.optim.applied", ",".join(self.optim_applied))
            )
        if self.optim_sources:
            attrs.append(
                onnx.helper.make_attribute("winml.optim.sources", ",".join(self.optim_sources))
            )

        return attrs

    @classmethod
    def from_node(cls, node: onnx.NodeProto) -> NodeMetadata | None:
        """Extract metadata from an ONNX node.

        Args:
            node: ONNX node to extract metadata from.

        Returns:
            NodeMetadata instance if winml.node.name attribute is present,
            None otherwise.
        """
        import onnx

        # Extract all winml.* attributes
        attrs: dict[str, str] = {}
        for attr in node.attribute:
            if attr.name.startswith("winml."):
                # Handle string attributes
                if attr.type == onnx.AttributeProto.STRING:
                    attrs[attr.name] = attr.s.decode("utf-8")
                else:
                    # Fallback for other types (shouldn't happen, but be defensive)
                    attrs[attr.name] = str(attr.s)

        # Must have at least node name to be valid metadata
        if "winml.node.name" not in attrs:
            return None

        # Parse optim_applied list
        optim_applied_str = attrs.get("winml.optim.applied", "")
        optim_applied = (
            [opt.strip() for opt in optim_applied_str.split(",") if opt.strip()]
            if optim_applied_str
            else []
        )

        # Parse optim_sources list
        optim_sources_str = attrs.get("winml.optim.sources", "")
        optim_sources = (
            [src.strip() for src in optim_sources_str.split(",") if src.strip()]
            if optim_sources_str
            else []
        )

        # Parse hierarchy_depth as int
        hierarchy_depth = None
        if "winml.hierarchy.depth" in attrs:
            try:
                hierarchy_depth = int(attrs["winml.hierarchy.depth"])
            except ValueError:
                # Invalid depth value, treat as None
                hierarchy_depth = None

        return cls(
            name=attrs.get("winml.node.name", node.name),
            origin=attrs.get("winml.node.origin", "external"),
            hierarchy_tag=attrs.get("winml.hierarchy.tag"),
            hierarchy_depth=hierarchy_depth,
            semantic_type=attrs.get("winml.semantic.type"),
            semantic_layer_id=attrs.get("winml.semantic.layer_id"),
            optim_applied=optim_applied,
            optim_sources=optim_sources,
        )


def add_metadata_to_node(node: onnx.NodeProto, metadata: NodeMetadata) -> None:
    """Add metadata attributes to an ONNX node.

    Removes any existing winml.* attributes and replaces them with the new metadata.

    Args:
        node: ONNX node to modify.
        metadata: Metadata to add to the node.
    """
    # Remove existing winml.* attributes
    # Cannot use slice assignment on ONNX repeated fields
    filtered_attrs = [attr for attr in node.attribute if not attr.name.startswith("winml.")]
    del node.attribute[:]
    node.attribute.extend(filtered_attrs)

    # Add new metadata attributes
    node.attribute.extend(metadata.to_attributes())


def get_metadata_from_node(node: onnx.NodeProto) -> NodeMetadata | None:
    """Extract metadata from an ONNX node.

    Args:
        node: ONNX node to extract metadata from.

    Returns:
        NodeMetadata instance if metadata is present, None otherwise.
    """
    return NodeMetadata.from_node(node)


def set_origin_for_graph(
    graph: onnx.GraphProto,
    origin: str,
    overwrite: bool = False,
) -> None:
    """Set origin for all nodes in a graph.

    Args:
        graph: ONNX graph to modify.
        origin: Origin value ('export', 'optimize', 'quantize', 'compile', 'external').
        overwrite: If True, overwrite existing origin; if False, only set if missing.
    """
    for node in graph.node:
        existing = get_metadata_from_node(node)

        # Skip if metadata exists and overwrite is False
        if existing and not overwrite:
            continue

        # Create metadata, preserving existing optional fields if present
        metadata = NodeMetadata(
            name=node.name,
            origin=origin,
            hierarchy_tag=existing.hierarchy_tag if existing else None,
            hierarchy_depth=existing.hierarchy_depth if existing else None,
            semantic_type=existing.semantic_type if existing else None,
            semantic_layer_id=existing.semantic_layer_id if existing else None,
            optim_applied=existing.optim_applied if existing else [],
            optim_sources=existing.optim_sources if existing else [],
        )
        add_metadata_to_node(node, metadata)


def mark_fused_node(
    node: onnx.NodeProto,
    source_nodes: list[str],
    optimization: str,
) -> None:
    """Mark a node as result of fusion.

    Args:
        node: The fused ONNX node.
        source_nodes: Names of original nodes that were fused.
        optimization: Name of the optimization that created this fusion.
    """
    # Get existing metadata or create new
    existing = get_metadata_from_node(node)
    if existing is None:
        existing = NodeMetadata(name=node.name, origin="optimize")

    # Update fusion-related fields
    existing.origin = "optimize"
    existing.optim_sources = source_nodes

    # Accumulate optimizations (avoid duplicates)
    if optimization not in existing.optim_applied:
        existing.optim_applied.append(optimization)

    add_metadata_to_node(node, existing)


def query_nodes_by_origin(model: onnx.ModelProto, origin: str) -> list[str]:
    """Find all nodes from a specific origin.

    Args:
        model: ONNX model to query.
        origin: Origin value to search for.

    Returns:
        List of node names matching the origin.
    """
    result = []
    for node in model.graph.node:
        metadata = get_metadata_from_node(node)
        if metadata and metadata.origin == origin:
            result.append(node.name)
    return result


def query_fused_nodes(model: onnx.ModelProto) -> dict[str, list[str]]:
    """Find all fused nodes and their sources.

    Args:
        model: ONNX model to query.

    Returns:
        Dictionary mapping fused node names to their source node names.
    """
    result = {}
    for node in model.graph.node:
        metadata = get_metadata_from_node(node)
        if metadata and metadata.optim_sources:
            result[node.name] = metadata.optim_sources
    return result


def get_optimization_summary(model: onnx.ModelProto) -> dict[str, int]:
    """Get count of nodes by optimization applied.

    Args:
        model: ONNX model to analyze.

    Returns:
        Dictionary mapping optimization names to counts of nodes.
    """
    summary: dict[str, int] = {}
    for node in model.graph.node:
        metadata = get_metadata_from_node(node)
        if metadata:
            for opt in metadata.optim_applied:
                summary[opt] = summary.get(opt, 0) + 1
    return summary
