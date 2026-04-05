# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Core utilities for ModelKit."""

from .model_input_generator import generate_dummy_inputs_from_specs
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


def __getattr__(name: str):
    """Lazy-load onnx_utils (imports torch at module level)."""
    if name in ("get_epcontext_info", "get_io_config"):
        from .onnx_utils import get_epcontext_info, get_io_config

        globals().update(
            {
                "get_epcontext_info": get_epcontext_info,
                "get_io_config": get_io_config,
            }
        )
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
