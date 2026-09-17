# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Eliminate internal tensor Identity aliases for IX compatibility."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from onnx import AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, TypeProto, helper

from ...onnx import ONNXDomain


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)


def _subgraphs(node: NodeProto) -> Iterator[GraphProto]:
    for attribute in node.attribute:
        if attribute.type == AttributeProto.GRAPH:
            yield attribute.g
        elif attribute.type == AttributeProto.GRAPHS:
            yield from attribute.graphs


def _local_names(graph: GraphProto) -> set[str]:
    return (
        {value.name for value in graph.input}
        | {value.name for value in graph.initializer}
        | {value.values.name for value in graph.sparse_initializer}
        | {name for node in graph.node for name in node.output if name}
    )


def _types(graph: GraphProto, outer: dict[str, TypeProto]) -> dict[str, TypeProto]:
    local = _local_names(graph)
    types = {name: value for name, value in outer.items() if name not in local}
    for value in (*graph.value_info, *graph.input, *graph.output):
        types[value.name] = value.type
    for value in graph.initializer:
        types.setdefault(value.name, helper.make_tensor_type_proto(value.data_type, value.dims))
    return types


def _compatible_tensor(source: TypeProto | None, output: TypeProto | None) -> bool:
    if source is None or not source.HasField("tensor_type"):
        return False
    if output is None:
        return True
    if not output.HasField("tensor_type"):
        return False
    left, right = source.tensor_type, output.tensor_type
    if left.elem_type and right.elem_type and left.elem_type != right.elem_type:
        return False
    if left.HasField("shape") and right.HasField("shape"):
        if len(left.shape.dim) != len(right.shape.dim):
            return False
        for a, b in zip(left.shape.dim, right.shape.dim, strict=True):
            if a.HasField("dim_value") and b.HasField("dim_value") and a.dim_value != b.dim_value:
                return False
            if a.dim_param and b.dim_param and a.dim_param != b.dim_param:
                return False
    return True


def _redirect_uses(
    graph: GraphProto, old: str, new: str, *, apply: bool, shadowed: bool = False
) -> bool:
    if any(value.name == old for value in graph.output):
        return False
    # Quantization annotations can name aliases independently of node inputs.
    if any(
        annotation.tensor_name == old
        or any(parameter.value == old for parameter in annotation.quant_parameter_tensor_names)
        for annotation in graph.quantization_annotation
    ):
        return False
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                if shadowed:
                    return False
                if apply:
                    node.input[index] = new
        for child in _subgraphs(node):
            local = _local_names(child)
            if old in local:
                continue
            if not _redirect_uses(
                child, old, new, apply=apply, shadowed=shadowed or new in local
            ):
                return False
    if apply:
        retained = [value for value in graph.value_info if value.name != old]
        del graph.value_info[:]
        graph.value_info.extend(retained)
    return True


def _eliminate(graph: GraphProto, outer: dict[str, TypeProto]) -> int:
    types = _types(graph, outer)
    removed = 0
    for node in list(graph.node):
        if node.domain not in {"", ONNXDomain.AI_ONNX.value} or node.op_type != "Identity":
            continue
        if len(node.input) != 1 or len(node.output) != 1 or node.attribute:
            continue
        source, output = node.input[0], node.output[0]
        if not source or not output or source == output:
            continue
        if not _compatible_tensor(types.get(source), types.get(output)):
            logger.debug(
                "Retaining Identity %r: tensor types are unknown or incompatible", node.name,
            )
            continue
        if not _redirect_uses(graph, output, source, apply=False):
            logger.debug(
                "Retaining Identity %r: output interface or scoped alias is protected", node.name
            )
            continue
        _redirect_uses(graph, output, source, apply=True)
        graph.node.remove(node)
        types.pop(output, None)
        removed += 1
    for node in graph.node:
        for child in _subgraphs(node):
            removed += _eliminate(child, types)
    return removed


def _reshape_output_aliases(model: ModelProto) -> int:
    versions = [entry.version for entry in model.opset_import if entry.domain == ""]
    graph = model.graph
    if len(versions) != 1 or versions[0] < 5 or any(list(_subgraphs(node)) for node in graph.node):
        return 0
    types: dict[str, TypeProto] = {}
    for value in (*graph.input, *graph.output, *graph.value_info):
        if value.name in types and types[value.name] != value.type:
            return 0
        types[value.name] = value.type
    outputs = {value.name for value in graph.output}
    inputs = {value.name for value in graph.input}
    protected = set()
    for annotation in graph.quantization_annotation:
        protected.add(annotation.tensor_name)
        protected.update(parameter.value for parameter in annotation.quant_parameter_tensor_names)
    names = _local_names(graph) | set(types) | protected
    names.update(name for node in graph.node for name in node.input)
    rewritten = 0
    for node in graph.node:
        if node.domain or node.op_type != "Identity" or node.attribute:
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            continue
        source, output = node.input[0], node.output[0]
        if (not source or source == output or output not in outputs or output in inputs
                or source in protected or output in protected):
            continue
        source_type, output_type = types.get(source), types.get(output)
        if (source_type is None or source_type != output_type
            or not source_type.HasField("tensor_type")):
            continue
        tensor = source_type.tensor_type
        if tensor.elem_type != TensorProto.FLOAT or not tensor.HasField("shape"):
            continue
        dimensions = tensor.shape.dim
        if not dimensions or any(not dim.HasField("dim_value") or dim.dim_value <= 0
                                 for dim in dimensions):
            continue
        shape_name = output + "_identity_shape"
        while shape_name in names:
            shape_name += "_"
        names.add(shape_name)
        graph.initializer.append(helper.make_tensor(
            shape_name, TensorProto.INT64, [len(dimensions)],
            [dim.dim_value for dim in dimensions],
        ))
        node.op_type = "Reshape"
        node.input.append(shape_name)
        rewritten += 1
    return rewritten


def eliminate_identity(model: ModelProto) -> ModelProto:
    """Remove safe internal tensor Identities, preserving graph IO and lexical bindings.

    Top-level static positive-shape FP32 output aliases use Reshape in models
    without subgraphs. Other protected aliases and lexical bindings are retained.
    """
    result = ModelProto()
    result.CopyFrom(model)
    reshaped = _reshape_output_aliases(result)
    removed = _eliminate(result.graph, {})
    if not removed and not reshaped:
        return model
    logger.info(
        "CGIR compatibility: eliminate-identity removed %d node(s), reshaped %d output alias(es)",
        removed, reshaped,
    )
    return result
