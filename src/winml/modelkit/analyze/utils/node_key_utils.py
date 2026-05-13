# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Helpers for stable ONNX node-key resolution.

These helpers centralize the analyzer rule for stable node keys:
- prefer node.name when non-empty
- otherwise use node_<index> within the graph snapshot
"""

from __future__ import annotations

from collections.abc import Sequence

import onnx

from ...pattern.utils import make_stable_node_key


def build_node_key_by_node_id(graph_nodes: Sequence[onnx.NodeProto]) -> dict[int, str]:
    """Build id(node) -> stable-key map for a graph snapshot."""
    return {id(node): make_stable_node_key(node, index) for index, node in enumerate(graph_nodes)}


def resolve_stable_node_key(
    node: onnx.NodeProto,
    *,
    node_key_by_node_id: dict[int, str],
    graph_nodes: Sequence[onnx.NodeProto],
    unknown_unnamed_error: str,
) -> str:
    """Resolve stable key for a node against a graph snapshot.

    Resolution order:
    1) explicit id(node) sidecar mapping
    2) named node fallback (node.name)
    3) identity scan in graph snapshot for unnamed graph nodes
    4) raise KeyError for unnamed external nodes
    """
    stable_key = node_key_by_node_id.get(id(node))
    if stable_key is not None:
        return stable_key

    if node.name:
        return node.name

    for index, graph_node in enumerate(graph_nodes):
        if graph_node is node:
            stable_key = make_stable_node_key(graph_node, index)
            node_key_by_node_id[id(node)] = stable_key
            return stable_key

    raise KeyError(unknown_unnamed_error)
