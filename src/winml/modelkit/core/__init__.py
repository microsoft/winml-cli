# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Core utilities for ModelKit."""

# New API - pure torch, no external dependencies
from .node_metadata import (
    NodeMetadata,
    add_metadata_to_node,
    get_metadata_from_node,
    get_optimization_summary,
    mark_fused_node,
    query_fused_nodes,
    query_nodes_by_origin,
    set_origin_for_graph,
)
from .onnx_utils import (
    get_epcontext_info,
    get_io_config,
)


def __getattr__(name: str):
    """Lazy-load generate_dummy_inputs_from_specs to avoid importing torch at startup."""
    if name == "generate_dummy_inputs_from_specs":
        from .model_input_generator import generate_dummy_inputs_from_specs

        globals()["generate_dummy_inputs_from_specs"] = generate_dummy_inputs_from_specs
        return generate_dummy_inputs_from_specs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NodeMetadata",
    "add_metadata_to_node",
    "generate_dummy_inputs_from_specs",
    "get_epcontext_info",
    "get_io_config",
    "get_metadata_from_node",
    "get_optimization_summary",
    "mark_fused_node",
    "query_fused_nodes",
    "query_nodes_by_origin",
    "set_origin_for_graph",
]
