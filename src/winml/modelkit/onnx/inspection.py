"""ONNX model inspection utilities.

Provides functions for examining tensor type info, detecting problems,
and inspecting QDQ node metadata. Used for debugging quantized models
and validating post-processing fixes.
"""

from __future__ import annotations

import logging
from typing import Any

import onnx
from onnx import TensorProto, ValueInfoProto

from ..compiler import QDQ_OP_TYPES


logger = logging.getLogger(__name__)



def get_value_info_elem_type(vi: ValueInfoProto) -> int:
    """Get the element type from a ValueInfoProto.

    Args:
        vi: An ONNX ValueInfoProto (from graph.input, graph.output, or value_info).

    Returns:
        The elem_type integer (e.g., TensorProto.FLOAT), or TensorProto.UNDEFINED
        if the type information is not set.
    """
    if vi.type.HasField("tensor_type"):
        return vi.type.tensor_type.elem_type
    return TensorProto.UNDEFINED


def get_value_info_dims(vi: ValueInfoProto) -> list[int | str] | None:
    """Get the shape dimensions from a ValueInfoProto.

    Args:
        vi: An ONNX ValueInfoProto.

    Returns:
        List of dimensions (int for static, str for symbolic), or None if
        shape info is not available.
    """
    if not vi.type.HasField("tensor_type"):
        return None
    tensor_type = vi.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims: list[int | str] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def find_undefined_types(model: onnx.ModelProto) -> list[dict[str, Any]]:
    """Find all tensors with UNDEFINED (elem_type=0) type info.

    Examines graph.input, graph.output, and value_info entries.
    Initializers are not checked since they always have data_type set.

    Args:
        model: ONNX ModelProto to inspect.

    Returns:
        List of dicts with keys: name, source, elem_type, dims.
    """
    results: list[dict[str, Any]] = []

    for source, entries in [
        ("graph_input", model.graph.input),
        ("graph_output", model.graph.output),
        ("value_info", model.graph.value_info),
    ]:
        for vi in entries:
            elem_type = get_value_info_elem_type(vi)
            if elem_type == TensorProto.UNDEFINED:
                results.append({
                    "name": vi.name,
                    "source": source,
                    "elem_type": elem_type,
                    "dims": get_value_info_dims(vi),
                })

    return results


def get_qdq_param_info(model: onnx.ModelProto) -> list[dict[str, Any]]:
    """Get detailed info about QDQ node scale/zero_point tensors.

    For each QuantizeLinear/DequantizeLinear node, reports the type
    information status of its scale and zero_point inputs across
    graph.input, value_info, and initializer entries.

    Args:
        model: ONNX ModelProto to inspect.

    Returns:
        List of dicts, one per QDQ node, with keys:
        - node_name, op_type
        - scale_name, scale_vi_type, scale_init_type
        - zp_name, zp_vi_type, zp_init_type
        - issues: list of issue descriptions
    """
    graph = model.graph

    # Build lookup tables
    vi_map: dict[str, ValueInfoProto] = {}
    for vi in list(graph.input) + list(graph.value_info):
        vi_map[vi.name] = vi

    init_map: dict[str, int] = {init.name: init.data_type for init in graph.initializer}

    results: list[dict[str, Any]] = []

    for node in graph.node:
        if node.op_type not in QDQ_OP_TYPES:
            continue

        info: dict[str, Any] = {
            "node_name": node.name,
            "op_type": node.op_type,
            "scale_name": None,
            "scale_vi_type": TensorProto.UNDEFINED,
            "scale_init_type": TensorProto.UNDEFINED,
            "zp_name": None,
            "zp_vi_type": TensorProto.UNDEFINED,
            "zp_init_type": TensorProto.UNDEFINED,
            "issues": [],
        }

        # Scale (input[1])
        if len(node.input) > 1 and node.input[1]:
            scale_name = node.input[1]
            info["scale_name"] = scale_name
            if scale_name in vi_map:
                info["scale_vi_type"] = get_value_info_elem_type(vi_map[scale_name])
            if scale_name in init_map:
                info["scale_init_type"] = init_map[scale_name]
            if info["scale_vi_type"] == TensorProto.UNDEFINED:
                info["issues"].append(f"scale '{scale_name}' has UNDEFINED type in value_info")

        # Zero point (input[2], optional)
        if len(node.input) > 2 and node.input[2]:
            zp_name = node.input[2]
            info["zp_name"] = zp_name
            if zp_name in vi_map:
                info["zp_vi_type"] = get_value_info_elem_type(vi_map[zp_name])
            if zp_name in init_map:
                info["zp_init_type"] = init_map[zp_name]
            if info["zp_vi_type"] == TensorProto.UNDEFINED:
                info["issues"].append(f"zero_point '{zp_name}' has UNDEFINED type in value_info")

        results.append(info)

    return results


def format_model_type_summary(model: onnx.ModelProto) -> str:
    """Format a human-readable summary of tensor type info in a model.

    Shows graph inputs, outputs, and highlights any UNDEFINED types
    or QDQ-specific issues.

    Args:
        model: ONNX ModelProto to inspect.

    Returns:
        Multi-line formatted string.
    """
    lines: list[str] = []
    graph = model.graph

    # Header
    node_count = len(graph.node)
    init_count = len(graph.initializer)
    vi_count = len(graph.value_info)
    lines.append(f"Model: {graph.name or '(unnamed)'}")
    lines.append(f"  Nodes: {node_count}, Initializers: {init_count}, Value infos: {vi_count}")

    # Graph inputs (non-initializer)
    init_names = {init.name for init in graph.initializer}
    lines.append("\nGraph Inputs:")
    for vi in graph.input:
        if vi.name in init_names:
            continue
        elem_type = get_value_info_elem_type(vi)
        dims = get_value_info_dims(vi)
        type_str = TensorProto.DataType.Name(elem_type)
        marker = " [UNDEFINED]" if elem_type == TensorProto.UNDEFINED else ""
        lines.append(f"  {vi.name}: {type_str} {dims}{marker}")

    # Graph outputs
    lines.append("\nGraph Outputs:")
    for vi in graph.output:
        elem_type = get_value_info_elem_type(vi)
        dims = get_value_info_dims(vi)
        type_str = TensorProto.DataType.Name(elem_type)
        marker = " [UNDEFINED]" if elem_type == TensorProto.UNDEFINED else ""
        lines.append(f"  {vi.name}: {type_str} {dims}{marker}")

    # Undefined types
    undefined = find_undefined_types(model)
    if undefined:
        lines.append(f"\nUNDEFINED Types ({len(undefined)}):")
        lines.extend(
            f"  {entry['name']} (source: {entry['source']})" for entry in undefined
        )

    # QDQ issues
    qdq_info = get_qdq_param_info(model)
    qdq_issues = [info for info in qdq_info if info["issues"]]
    if qdq_issues:
        lines.append(f"\nQDQ Issues ({len(qdq_issues)}):")
        for info in qdq_issues:
            lines.append(f"  {info['node_name']} ({info['op_type']}):")
            lines.extend(f"    - {issue}" for issue in info["issues"])
    elif qdq_info:
        lines.append(f"\nQDQ Nodes: {len(qdq_info)} (all OK)")

    return "\n".join(lines)
