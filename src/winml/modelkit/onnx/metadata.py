# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Universal ONNX metadata capture and restore.

Graph transformations (ORT quantize, shape inference) create new ModelProto
or NodeProto objects that discard custom metadata. This module provides a
universal capture/restore mechanism that preserves:

1. **Model-level** ``metadata_props`` (e.g., ``winml.io.inputs``)
2. **Node-level** ``metadata_props`` (e.g., ``winml.hierarchy.tag``)
3. **Node-level** ``winml.*`` attributes (from the node_metadata system)

Usage::

    from winml.modelkit.onnx.metadata import capture_metadata, restore_metadata

    snapshot = capture_metadata(model)
    model = some_destructive_operation(model)
    result = restore_metadata(model, snapshot)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import onnx


logger = logging.getLogger(__name__)

# Prefix for custom winml node attributes
_WINML_ATTR_PREFIX = "winml."


@dataclass
class NodeMetadataEntry:
    """Captured metadata for a single node."""

    # metadata_props: list of (key, value) pairs
    props: list[tuple[str, str]] = field(default_factory=list)

    # winml.* attributes: list of (name, type, value) tuples
    # type is onnx.AttributeProto.AttributeType int
    attrs: list[tuple[str, int, str]] = field(default_factory=list)


@dataclass
class MetadataSnapshot:
    """Complete metadata snapshot of an ONNX model.

    Captures all metadata that may be lost during graph transformations.
    """

    # Model-level metadata_props: list of (key, value)
    model_props: list[tuple[str, str]] = field(default_factory=list)

    # Node-level metadata keyed by node name
    nodes: dict[str, NodeMetadataEntry] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        """Number of nodes with captured metadata."""
        return len(self.nodes)

    @property
    def model_prop_count(self) -> int:
        """Number of model-level metadata props captured."""
        return len(self.model_props)


@dataclass
class RestoreResult:
    """Result of a metadata restore operation."""

    nodes_restored: int = 0
    nodes_total: int = 0
    model_props_restored: int = 0


def capture_metadata(model: onnx.ModelProto) -> MetadataSnapshot:
    """Capture all metadata from an ONNX model.

    Extracts:
    - Model-level metadata_props (winml.io.inputs, etc.)
    - Node-level metadata_props (winml.hierarchy.tag, etc.)
    - Node-level winml.* attributes (from node_metadata system)

    Args:
        model: ONNX model to capture metadata from.

    Returns:
        MetadataSnapshot that can be passed to :func:`restore_metadata`.
    """
    snapshot = MetadataSnapshot()

    # Capture model-level metadata_props
    for prop in model.metadata_props:
        snapshot.model_props.append((prop.key, prop.value))

    # Capture node-level metadata
    for node in model.graph.node:
        entry = _capture_node(node)
        if entry is not None:
            snapshot.nodes[node.name] = entry

    if snapshot.nodes:
        logger.debug(
            "Captured metadata: %d model props, %d nodes",
            len(snapshot.model_props),
            len(snapshot.nodes),
        )

    return snapshot


def restore_metadata(
    model: onnx.ModelProto,
    snapshot: MetadataSnapshot,
) -> RestoreResult:
    """Restore metadata from a snapshot onto an ONNX model.

    Matches nodes by name. Model-level props are restored only if
    the target model is missing them (avoids duplicates).

    Args:
        model: ONNX model to restore metadata onto (modified in-place).
        snapshot: Metadata snapshot from :func:`capture_metadata`.

    Returns:
        RestoreResult with counts of restored items.
    """
    result = RestoreResult(nodes_total=len(model.graph.node))

    # Restore model-level metadata_props (only missing ones)
    existing_model_keys = {prop.key for prop in model.metadata_props}
    for key, value in snapshot.model_props:
        if key not in existing_model_keys:
            model.metadata_props.add(key=key, value=value)
            result.model_props_restored += 1

    # Restore node-level metadata by name match
    for node in model.graph.node:
        if node.name not in snapshot.nodes:
            continue
        entry = snapshot.nodes[node.name]
        _restore_node(node, entry)
        result.nodes_restored += 1

    if result.nodes_restored:
        logger.info(
            "Restored metadata: %d/%d nodes, %d model props",
            result.nodes_restored,
            result.nodes_total,
            result.model_props_restored,
        )

    return result


def _capture_node(node: onnx.NodeProto) -> NodeMetadataEntry | None:
    """Capture metadata from a single node.

    Returns None if the node has no custom metadata to capture.
    """
    entry = NodeMetadataEntry()

    # Capture metadata_props
    for prop in node.metadata_props:
        entry.props.append((prop.key, prop.value))

    # Capture winml.* attributes
    for attr in node.attribute:
        if attr.name.startswith(_WINML_ATTR_PREFIX):
            # Store string value (all winml attrs are strings)
            entry.attrs.append((attr.name, attr.type, attr.s.decode("utf-8")))

    if not entry.props and not entry.attrs:
        return None

    return entry


def _restore_node(node: onnx.NodeProto, entry: NodeMetadataEntry) -> None:
    """Restore metadata onto a single node."""
    # Restore metadata_props (only if not already present)
    existing_prop_keys = {p.key for p in node.metadata_props}
    for key, value in entry.props:
        if key not in existing_prop_keys:
            prop = node.metadata_props.add()
            prop.key = key
            prop.value = value

    # Restore winml.* attributes (only if not already present)
    existing_attr_names = {a.name for a in node.attribute}
    for name, _attr_type, value in entry.attrs:
        if name not in existing_attr_names:
            new_attr = onnx.helper.make_attribute(name, value)
            node.attribute.append(new_attr)
