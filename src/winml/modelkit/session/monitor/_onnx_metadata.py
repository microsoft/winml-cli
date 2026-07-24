# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Private ONNX metadata extraction for operator tracing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


def _load_onnx_operator_data(onnx_path: Path) -> dict[str, dict[str, Any]]:
    """Load inferred ONNX node metadata keyed by exact node name."""
    from ...onnx import infer_shapes, load_onnx

    model = infer_shapes(load_onnx(onnx_path, load_weights=False, validate=False))
    value_info = _collect_value_info(model)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    opset_versions = {opset.domain: opset.version for opset in model.opset_import}

    return {
        node.name: {
            "onnx_op_type": node.op_type,
            "onnx_attributes": {
                attribute.name: _serialize_attribute(attribute) for attribute in node.attribute
            },
            "onnx_inputs": _node_value_metadata(
                node,
                node.input,
                "input",
                opset_versions,
                value_info,
                initializers,
            ),
            "onnx_outputs": _node_value_metadata(
                node,
                node.output,
                "output",
                opset_versions,
                value_info,
                initializers,
            ),
        }
        for node in model.graph.node
        if node.name
    }


def _collect_value_info(model: Any) -> dict[str, Any]:
    """Collect graph tensor type and shape entries by tensor name."""
    return {
        value.name: value
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)
    }


def _node_value_metadata(
    node: Any,
    value_names: Any,
    value_kind: str,
    opset_versions: dict[str, int],
    value_info: dict[str, Any],
    initializers: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return tensor metadata keyed by ONNX schema formal parameter name."""
    metadata: dict[str, dict[str, Any]] = {}
    for index, value_name in enumerate(value_names):
        if not value_name:
            continue
        schema_name = _schema_value_name(node, index, value_kind, opset_versions)
        key = schema_name if schema_name not in metadata else f"{schema_name}[{index}]"
        metadata[key] = _tensor_value_metadata(value_name, value_info, initializers)
    return metadata


def _schema_value_name(
    node: Any,
    value_index: int,
    value_kind: str,
    opset_versions: dict[str, int],
) -> str:
    """Resolve an input/output index to its ONNX schema formal name."""
    from onnx import defs

    opset_version = opset_versions.get(node.domain)
    if opset_version is None:
        logger.debug(
            "Could not resolve ONNX opset version for %s domain=%r %s=%d",
            node.op_type,
            node.domain,
            value_kind,
            value_index,
        )
        return f"{value_kind}_{value_index}"

    try:
        schema = defs.get_schema(
            node.op_type,
            max_inclusive_version=opset_version,
            domain=node.domain,
        )
    except defs.SchemaError:
        logger.debug(
            "Could not resolve ONNX schema for %s domain=%r %s=%d",
            node.op_type,
            node.domain,
            value_kind,
            value_index,
            exc_info=True,
        )
        return f"{value_kind}_{value_index}"

    schema_values = schema.inputs if value_kind == "input" else schema.outputs
    if value_index < len(schema_values):
        return schema_values[value_index].name
    if schema_values:
        return schema_values[-1].name
    return f"{value_kind}_{value_index}"


def _tensor_value_metadata(
    name: str,
    value_info: dict[str, Any],
    initializers: dict[str, Any],
) -> dict[str, Any]:
    """Return JSON-safe type and shape metadata for a tensor value."""
    metadata: dict[str, Any] = {"name": name}
    if name in initializers:
        initializer = initializers[name]
        metadata["dims"] = list(initializer.dims)
        metadata["data_type"] = _tensor_data_type_name(initializer.data_type)
        return metadata

    value = value_info.get(name)
    if value is None or not value.type.HasField("tensor_type"):
        return metadata

    tensor_type = value.type.tensor_type
    metadata["data_type"] = _tensor_data_type_name(tensor_type.elem_type)
    if tensor_type.HasField("shape"):
        metadata["dims"] = [_shape_dim_to_value(dim) for dim in tensor_type.shape.dim]
    return metadata


def _shape_dim_to_value(dim: Any) -> int | str | None:
    """Convert an ONNX dimension into a JSON-safe value."""
    if dim.HasField("dim_value"):
        return cast("int", dim.dim_value)
    if dim.HasField("dim_param"):
        return cast("str", dim.dim_param)
    return None


def _tensor_data_type_name(data_type: int) -> str:
    """Return the ONNX TensorProto datatype name for an enum value."""
    from onnx import TensorProto

    return TensorProto.DataType.Name(data_type)


def _serialize_attribute(attribute: Any) -> Any:
    """Convert an ONNX AttributeProto value into compact JSON-safe metadata."""
    from onnx import AttributeProto, helper

    value = helper.get_attribute_value(attribute)
    attribute_type = attribute.type
    if attribute_type == AttributeProto.STRING:
        return value.decode("utf-8", errors="replace")
    if attribute_type == AttributeProto.STRINGS:
        return [item.decode("utf-8", errors="replace") for item in value]
    if attribute_type == AttributeProto.TENSOR:
        return _tensor_attribute_metadata(value)
    if attribute_type == AttributeProto.TENSORS:
        return [_tensor_attribute_metadata(tensor) for tensor in value]
    if attribute_type in (AttributeProto.INTS, AttributeProto.FLOATS):
        return list(value)
    if attribute_type == AttributeProto.GRAPH:
        return _graph_attribute_metadata(value)
    if attribute_type == AttributeProto.GRAPHS:
        return [_graph_attribute_metadata(graph) for graph in value]
    if attribute_type == AttributeProto.SPARSE_TENSOR:
        return _sparse_tensor_attribute_metadata(value)
    if attribute_type == AttributeProto.SPARSE_TENSORS:
        return [_sparse_tensor_attribute_metadata(tensor) for tensor in value]
    if attribute_type == AttributeProto.TYPE_PROTO:
        return _protobuf_attribute_metadata(value)
    if attribute_type == AttributeProto.TYPE_PROTOS:
        return [_protobuf_attribute_metadata(type_proto) for type_proto in value]
    return _json_safe_attribute_value(value)


def _protobuf_attribute_metadata(value: Any) -> dict[str, Any] | str:
    """Convert protobuf attribute payloads to JSON-safe metadata."""
    from google.protobuf import json_format

    try:
        return json_format.MessageToDict(value, preserving_proto_field_name=True)
    except Exception:
        logger.debug("Could not convert protobuf attribute payload to JSON", exc_info=True)
        return str(value)


def _json_safe_attribute_value(value: Any) -> Any:
    """Return a JSON-safe fallback for unsupported ONNX attribute payloads."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_json_safe_attribute_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_attribute_value(item) for key, item in value.items()}
    if hasattr(value, "DESCRIPTOR"):
        return _protobuf_attribute_metadata(value)
    return str(value)


def _tensor_attribute_metadata(tensor: Any) -> dict[str, Any]:
    """Summarize tensor attributes without embedding raw tensor data."""
    return {
        "name": tensor.name,
        "dims": list(tensor.dims),
        "data_type": _tensor_data_type_name(tensor.data_type),
    }


def _sparse_tensor_attribute_metadata(tensor: Any) -> dict[str, Any]:
    """Summarize sparse tensor attributes without embedding raw data."""
    return {
        "dims": list(tensor.dims),
        "values": _tensor_attribute_metadata(tensor.values),
    }


def _graph_attribute_metadata(graph: Any) -> dict[str, Any]:
    """Summarize graph attributes without recursively dumping subgraphs."""
    return {"name": graph.name, "node_count": len(graph.node)}
