# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""FP16 conversion utility for ONNX models.

Provides a single entry point for FP32->FP16 model conversion, used by
the quantizer's ``mode="fp16"`` path.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, cast

from google.protobuf.message import EncodeError


if TYPE_CHECKING:
    from collections.abc import Sequence

    from onnx import (
        AttributeProto,
        FunctionProto,
        GraphProto,
        ModelProto,
        NodeProto,
        TensorProto,
        TypeProto,
        ValueInfoProto,
    )
    from onnx.defs import OpSchema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _InitializerOutput:
    """A graph output supplied directly by an initializer in an ORT-traversed graph."""

    graph_index: int
    name: str
    output_index: int
    has_consumers: bool


def _tensor_data_is_loaded(initializer: TensorProto) -> bool:
    """Whether a FLOAT tensor carries resident data rather than only a sidecar ref."""
    if initializer.raw_data or initializer.float_data:
        return True
    return prod(initializer.dims) == 0


def _effective_blocked_ops(op_block_list: list[str] | None) -> set[str]:
    """Return the op types ORT skips for this wrapper's exposed block-list option."""
    from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST

    return set(DEFAULT_OP_BLOCK_LIST if op_block_list is None else op_block_list)


def _ort_inference_preflight_model(model: ModelProto) -> ModelProto:
    """Run ORT's normal shape-inference preflight on an isolated clone."""
    from onnx import ModelProto as ONNXModelProto
    from onnx import shape_inference

    candidate: object = model
    if not isinstance(candidate, ONNXModelProto):
        return model
    try:
        return shape_inference.infer_shapes(deepcopy(model))
    except EncodeError:
        return model


def _graph_tensor_names(graph: GraphProto) -> set[str]:
    """Collect tensor names in one ONNX lexical scope."""
    names = {
        value.name
        for values in (
            getattr(graph, "input", []),
            getattr(graph, "output", []),
            getattr(graph, "value_info", []),
        )
        for value in values
    }
    names.update(initializer.name for initializer in getattr(graph, "initializer", []))
    names.update(sparse.values.name for sparse in getattr(graph, "sparse_initializer", []))
    names.update(
        name for node in getattr(graph, "node", []) for name in (*node.input, *node.output) if name
    )
    return names


def _all_node_names(model: ModelProto, op_block_list: list[str] | None) -> set[str]:
    """Collect generated-name collisions whose skipped nodes need conversion."""
    return {
        node.name
        for graph in _ort_traversed_graphs(model, op_block_list)
        for node in getattr(graph, "node", [])
        if node.name and not _node_is_conversion_neutral(model, graph, node)
    }


def _node_is_conversion_neutral(model: ModelProto, graph: GraphProto, node: NodeProto) -> bool:
    """Whether ORT can skip a colliding node without changing its precision."""
    if node.attribute or any(
        input_name and not _node_input_is_proven_non_float(model, graph, node, input_index)
        for input_index, input_name in enumerate(node.input)
    ):
        return False
    for output_name in node.output:
        if not output_name:
            continue
        output_types = _graph_declared_types(graph, output_name)
        if not output_types or any(
            not _type_proto_is_concrete(value_type) or _type_proto_contains_float_tensor(value_type)
            for value_type in output_types
        ):
            return False
    return True


def _all_graphs(model: ModelProto) -> list[GraphProto]:
    """Return the top-level graph and all nested attribute graphs."""
    return [model.graph, *_iter_nested_graphs(model)]


def _iter_nested_graphs(model: ModelProto) -> list[GraphProto]:
    """Return nested attribute graphs without relying on tensor-name scope."""
    from onnx import AttributeProto

    nested: list[GraphProto] = []
    pending = [model.graph]
    while pending:
        graph = pending.pop()
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == AttributeProto.GRAPH:
                    nested.append(attribute.g)
                    pending.append(attribute.g)
                elif attribute.type == AttributeProto.GRAPHS:
                    nested.extend(attribute.graphs)
                    pending.extend(attribute.graphs)
    return nested


def _ort_traversed_graphs(model: ModelProto, op_block_list: list[str] | None) -> list[GraphProto]:
    """Return graphs ORT's FP16 converter visits for the given op block list."""
    blocked_ops = _effective_blocked_ops(op_block_list)
    traversed: list[GraphProto] = []
    pending = [model.graph]
    while pending:
        graph = pending.pop()
        traversed.append(graph)
        pending.extend(_iter_ort_child_graphs(graph, blocked_ops))
    return traversed


def _direct_initializer_outputs_in_graph(
    graph: GraphProto,
    *,
    data_types: set[int] | None = None,
) -> list[TensorProto]:
    """Return direct graph-output initializers, optionally filtered by data type."""
    if not hasattr(graph, "output") or not hasattr(graph, "initializer"):
        return []

    produced = {name for node in getattr(graph, "node", []) for name in node.output if name}
    output_names = {output.name for output in graph.output}
    return [
        initializer
        for initializer in graph.initializer
        if initializer.name in output_names
        and initializer.name not in produced
        and (data_types is None or initializer.data_type in data_types)
    ]


def _direct_initializer_outputs(
    model: ModelProto,
    *,
    data_types: set[int] | None = None,
    graphs: list[GraphProto] | None = None,
) -> list[TensorProto]:
    """Return direct initializer-backed outputs across the requested graph scopes."""
    return [
        initializer
        for graph in (graphs if graphs is not None else _all_graphs(model))
        for initializer in _direct_initializer_outputs_in_graph(graph, data_types=data_types)
    ]


def _has_nested_initializer_outputs(model: ModelProto, op_block_list: list[str] | None) -> bool:
    """Whether any traversed nested graph output is supplied directly by a FLOAT initializer."""
    from onnx import TensorProto

    return any(
        _direct_initializer_outputs_in_graph(graph, data_types={TensorProto.FLOAT})
        for graph in _ort_traversed_graphs(model, op_block_list)[1:]
    )


def _has_float_sparse_initializers(model: ModelProto, op_block_list: list[str] | None) -> bool:
    """Whether any ORT-traversed graph has sparse FLOAT initializer values."""
    from onnx import TensorProto

    return any(
        sparse.values.data_type == TensorProto.FLOAT
        for graph in _ort_traversed_graphs(model, op_block_list)
        for sparse in getattr(graph, "sparse_initializer", [])
    )


def _reject_sparse_initializer_tensor_metadata(
    model: ModelProto, op_block_list: list[str] | None
) -> None:
    """Reject sparse initializer metadata that conflicts with ONNX sparse typing."""
    for graph in _ort_traversed_graphs(model, op_block_list):
        sparse_names = {sparse.values.name for sparse in getattr(graph, "sparse_initializer", [])}
        if not sparse_names:
            continue
        for value_info in (
            *getattr(graph, "input", []),
            *getattr(graph, "output", []),
            *getattr(graph, "value_info", []),
        ):
            if value_info.name in sparse_names and value_info.type.HasField("tensor_type"):
                graph_name = graph.name or "<unnamed>"
                msg = (
                    f"Sparse initializer '{graph_name}.{value_info.name}' has "
                    "tensor_type metadata; sparse initializer metadata must use "
                    "sparse_tensor_type."
                )
                raise RuntimeError(msg)


def _reject_duplicate_float_initializer_names(
    model: ModelProto, op_block_list: list[str] | None
) -> None:
    """Reject FLOAT initializer names that ORT's global conversion map cannot scope."""
    from onnx import TensorProto

    seen: set[str] = set()
    duplicates: set[str] = set()
    for graph in _ort_traversed_graphs(model, op_block_list):
        if not hasattr(graph, "initializer"):
            continue
        for initializer in graph.initializer:
            if initializer.data_type != TensorProto.FLOAT:
                continue
            if initializer.name in seen:
                duplicates.add(initializer.name)
            else:
                seen.add(initializer.name)
    if duplicates:
        names = ", ".join(sorted(duplicates))
        msg = (
            "FP16 conversion cannot safely process duplicate FLOAT initializer "
            f"names across graph scopes: {names}."
        )
        raise RuntimeError(msg)


def _ort_graphs_with_parents(
    model: ModelProto, blocked_ops: set[str]
) -> tuple[list[GraphProto], dict[int, GraphProto | None]]:
    """Return ORT's graph BFS registration order and lexical parents."""
    graphs = [model.graph]
    parents: dict[int, GraphProto | None] = {id(model.graph): None}
    for graph in graphs:
        for child in _iter_ort_child_graphs(graph, blocked_ops):
            parents[id(child)] = graph
            graphs.append(child)
    return graphs, parents


def _input_resolves_to_initializer(
    graph: GraphProto,
    name: str,
    owner: GraphProto,
    parents: dict[int, GraphProto | None],
) -> bool:
    """Whether a graph-local input name resolves to the requested initializer."""
    current: GraphProto | None = graph
    while current is not None:
        if any(initializer.name == name for initializer in getattr(current, "initializer", [])):
            return current is owner
        if (
            any(value.name == name for value in getattr(current, "input", []))
            or any(
                sparse.values.name == name for sparse in getattr(current, "sparse_initializer", [])
            )
            or any(
                output_name == name
                for node in getattr(current, "node", [])
                for output_name in node.output
                if output_name
            )
        ):
            return False
        current = parents.get(id(current))
    return False


def _initializer_tracking_analysis(
    model: ModelProto,
    *,
    keep_io_types: bool,
    blocked_ops: set[str],
    graphs: list[GraphProto],
    parents: dict[int, GraphProto | None],
) -> tuple[dict[str, GraphProto], set[str], set[str], set[str], set[str]]:
    """Model ORT and lexical FLOAT initializer conversion decisions."""
    from onnx import TensorProto

    name_mapping, io_casts = _ort_keep_io_name_mapping(model, keep_io_types=keep_io_types)
    owners: dict[str, GraphProto] = {}
    ort_fp16_initializers: set[str] = set()
    lexical_fp16_initializers: set[str] = set()
    mismatched_names: set[str] = set()
    converted_initializer_inputs: set[str] = set()
    for graph in graphs:
        float_initializers = {
            initializer.name: initializer
            for initializer in getattr(graph, "initializer", [])
            if initializer.data_type == TensorProto.FLOAT
        }
        owners.update(dict.fromkeys(float_initializers, graph))
        converted_initializer_inputs.update(
            value.name
            for value in getattr(graph, "input", [])
            if value.name in float_initializers
            and value.name not in name_mapping
            and value.type.HasField("tensor_type")
            and value.type.tensor_type.elem_type == TensorProto.FLOAT
        )
        if not keep_io_types or graph is not model.graph:
            lexical_fp16_initializers.update(
                initializer.name
                for initializer in _direct_initializer_outputs_in_graph(
                    graph, data_types={TensorProto.FLOAT}
                )
                if not _initializer_output_has_consumers(graph, initializer.name, blocked_ops)
            )
        for node in getattr(graph, "node", []):
            if node.name in io_casts:
                continue
            for input_index, original_input_name in enumerate(node.input):
                input_name = name_mapping.get(original_input_name, original_input_name)
                owner = owners.get(input_name)
                if owner is None:
                    continue
                uses_fp16 = not _node_expects_fp32_input(node, input_index, blocked_ops)
                if uses_fp16:
                    ort_fp16_initializers.add(input_name)
                if _input_resolves_to_initializer(graph, input_name, owner, parents):
                    if uses_fp16:
                        lexical_fp16_initializers.add(input_name)
                else:
                    mismatched_names.add(input_name)
    return (
        owners,
        ort_fp16_initializers,
        lexical_fp16_initializers,
        mismatched_names,
        converted_initializer_inputs,
    )


def _reject_scope_unsafe_initializer_tracking(
    model: ModelProto,
    *,
    keep_io_types: bool,
    op_block_list: list[str] | None,
) -> None:
    """Reject lexical conflation only when it changes initializer conversion."""
    from onnx import TensorProto

    blocked_ops = _effective_blocked_ops(op_block_list)
    graphs, parents = _ort_graphs_with_parents(model, blocked_ops)
    (
        owners,
        ort_fp16_initializers,
        lexical_fp16_initializers,
        mismatched_names,
        converted_initializer_inputs,
    ) = _initializer_tracking_analysis(
        model,
        keep_io_types=keep_io_types,
        blocked_ops=blocked_ops,
        graphs=graphs,
        parents=parents,
    )

    unsafe_names = sorted(
        name
        for name in mismatched_names
        if (name in ort_fp16_initializers) != (name in lexical_fp16_initializers)
        and (
            _initializer_output_has_consumers(owners[name], name, blocked_ops)
            or any(
                value.name == name
                for value in (
                    *getattr(owners[name], "input", []),
                    *getattr(owners[name], "output", []),
                )
            )
        )
    )
    if unsafe_names:
        names = ", ".join(unsafe_names)
        msg = (
            f"FLOAT initializer names are consumed through different lexical "
            f"bindings and change ORT's conversion decision: {names}; ORT's "
            "initializer tracking is not scope-aware."
        )
        raise RuntimeError(msg)
    divergent_inputs = sorted(
        converted_initializer_inputs - ort_fp16_initializers - lexical_fp16_initializers
    )
    if divergent_inputs:
        names = ", ".join(divergent_inputs)
        msg = (
            "FLOAT initializer-backed graph input declarations convert to FP16 "
            f"while their default initializers remain FLOAT: {names}."
        )
        raise RuntimeError(msg)

    name_mapping, _ = _ort_keep_io_name_mapping(model, keep_io_types=keep_io_types)
    repaired_outputs = {
        initializer.name
        for graph in graphs
        if not keep_io_types or graph is not model.graph
        for initializer in _direct_initializer_outputs_in_graph(
            graph, data_types={TensorProto.FLOAT}
        )
        if not _initializer_output_has_consumers(graph, initializer.name, blocked_ops)
    }
    incompatible_metadata = sorted(
        name
        for name, owner in owners.items()
        if name not in ort_fp16_initializers
        and name not in repaired_outputs
        and name not in name_mapping
        and any(
            value.name == name and _value_info_enters_ort_global_list(value)
            for values in (
                getattr(owner, "input", []),
                getattr(owner, "output", []),
                getattr(owner, "value_info", []),
            )
            for value in values
        )
    )
    if incompatible_metadata:
        names = ", ".join(incompatible_metadata)
        msg = (
            "ORT converts FLOAT initializer metadata to FP16 while retaining "
            f"the initializer in FP32: {names}."
        )
        raise RuntimeError(msg)


def _reject_unloaded_external_initializer_outputs(
    model: ModelProto, op_block_list: list[str] | None
) -> None:
    """Reject direct FLOAT output initializers whose external data is not resident."""
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data

    for initializer in _direct_initializer_outputs(
        model,
        data_types={TensorProto.FLOAT},
        graphs=_ort_traversed_graphs(model, op_block_list),
    ):
        if uses_external_data(initializer) and not _tensor_data_is_loaded(initializer):
            msg = (
                f"Initializer-backed output '{initializer.name}' uses unloaded external data; "
                "load external weights before FP16 conversion."
            )
            raise RuntimeError(msg)


def _internalize_external_initializer_outputs(
    model: ModelProto, op_block_list: list[str] | None
) -> None:
    """Drop stale external metadata for resident direct output initializer data."""
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data

    for initializer in _direct_initializer_outputs(
        model,
        data_types={TensorProto.FLOAT},
        graphs=_ort_traversed_graphs(model, op_block_list),
    ):
        if uses_external_data(initializer):
            del initializer.external_data[:]
            initializer.data_location = TensorProto.DEFAULT


def _ort_converted_initializer_names(
    model: ModelProto,
    *,
    keep_io_types: bool,
    blocked_ops: set[str],
) -> set[str]:
    """Return FLOAT initializer names ORT selects for FP16 conversion."""
    graphs, parents = _ort_graphs_with_parents(model, blocked_ops)
    _, converted, _, _, _ = _initializer_tracking_analysis(
        model,
        keep_io_types=keep_io_types,
        blocked_ops=blocked_ops,
        graphs=graphs,
        parents=parents,
    )
    return converted


def _internalize_selected_external_initializers(
    model: ModelProto,
    names: set[str],
    op_block_list: list[str] | None,
) -> None:
    """Drop external metadata after selected weights are loaded in memory."""
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data

    for graph in _ort_traversed_graphs(model, op_block_list):
        for initializer in graph.initializer:
            if (
                initializer.name in names
                and initializer.data_type == TensorProto.FLOAT
                and uses_external_data(initializer)
            ):
                del initializer.external_data[:]
                initializer.data_location = TensorProto.DEFAULT


def _graph_node_consumes_name(graph: GraphProto, name: str) -> bool:
    """Whether a node in this graph directly consumes the requested name."""
    return any(
        input_name == name
        for node in getattr(graph, "node", [])
        for input_name in node.input
        if input_name
    )


def _graph_defines_name(graph: GraphProto, name: str) -> bool:
    """Whether a nested graph shadows an outer-scope name."""
    return (
        any(value.name == name for value in getattr(graph, "input", []))
        or any(initializer.name == name for initializer in getattr(graph, "initializer", []))
        or any(sparse.values.name == name for sparse in getattr(graph, "sparse_initializer", []))
        or any(
            output_name == name
            for node in getattr(graph, "node", [])
            for output_name in node.output
            if output_name
        )
    )


def _graph_node_references_name(graph: GraphProto, name: str) -> bool:
    """Whether any node input or output in this graph mentions the requested name."""
    return any(
        value_name == name
        for node in getattr(graph, "node", [])
        for value_name in (*node.input, *node.output)
    )


def _graph_keep_io_mapping_declares_name(graph: GraphProto, name: str) -> bool:
    """Whether a graph formal can be hit by ORT's global keep-I/O map."""
    return any(value.name == name for value in getattr(graph, "input", []))


def _graph_keep_io_mapping_references_name(graph: GraphProto, name: str) -> bool:
    """Whether ORT's global keep-I/O map may rewrite this graph's local name."""
    return _graph_node_references_name(graph, name) or _graph_keep_io_mapping_declares_name(
        graph, name
    )


def _graph_processed_node_references_name(
    model: ModelProto,
    graph: GraphProto,
    name: str,
    io_casts: set[str],
) -> bool:
    """Whether ORT processes a node reference subject to global mapping."""
    return any(
        value_name == name
        for node in graph.node
        if node.name not in io_casts or not _node_is_conversion_neutral(model, graph, node)
        for value_name in (*node.input, *node.output)
    )


def _ort_skips_node_attributes(node: NodeProto, blocked_ops: set[str]) -> bool:
    """Whether ORT skips a node's graph-valued attributes."""
    from onnxruntime.transformers.float16 import ALWAYS_FLOAT_INPUTS

    return node.op_type in blocked_ops or node.op_type in ALWAYS_FLOAT_INPUTS


def _iter_ort_child_graphs(graph: GraphProto, blocked_ops: set[str]) -> list[GraphProto]:
    """Return child graphs ORT traverses from this graph."""
    from onnx import AttributeProto

    children: list[GraphProto] = []
    for node in getattr(graph, "node", []):
        if _ort_skips_node_attributes(node, blocked_ops):
            continue
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                children.append(attribute.g)
            elif attribute.type == AttributeProto.GRAPHS:
                children.extend(attribute.graphs)
    return children


def _iter_all_child_graphs(graph: GraphProto) -> list[GraphProto]:
    """Return all direct child graphs, including those ORT skips under blocked nodes."""
    from onnx import AttributeProto

    children: list[GraphProto] = []
    for node in getattr(graph, "node", []):
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                children.append(attribute.g)
            elif attribute.type == AttributeProto.GRAPHS:
                children.extend(attribute.graphs)
    return children


def _descendant_has_free_consumer(graph: GraphProto, name: str, blocked_ops: set[str]) -> bool:
    """Whether a traversed descendant consumes an outer name without shadowing it."""
    if _graph_defines_name(graph, name):
        return False
    if _graph_node_consumes_name(graph, name):
        return True
    return any(
        _descendant_has_free_consumer(child, name, blocked_ops)
        for child in _iter_ort_child_graphs(graph, blocked_ops)
    )


def _descendant_has_node_reference(graph: GraphProto, name: str, blocked_ops: set[str]) -> bool:
    """Whether any ORT-traversed descendant mentions a name globally mapped by ORT."""
    if _graph_keep_io_mapping_references_name(graph, name):
        return True
    return any(
        _descendant_has_node_reference(child, name, blocked_ops)
        for child in _iter_ort_child_graphs(graph, blocked_ops)
    )


def _descendant_has_shadowed_node_reference(
    model: ModelProto,
    graph: GraphProto,
    name: str,
    blocked_ops: set[str],
    io_casts: set[str],
    *,
    shadowed: bool = False,
) -> bool:
    """Whether a traversed descendant uses a local name ORT would globally rewrite."""
    shadowed = (
        shadowed
        or _graph_defines_name(graph, name)
        or _graph_keep_io_mapping_declares_name(graph, name)
    )
    if shadowed and (
        any(
            value.name == name and _type_proto_enters_ort_global_list(value.type)
            for value in graph.input
        )
        or _graph_processed_node_references_name(model, graph, name, io_casts)
    ):
        return True
    return any(
        _descendant_has_shadowed_node_reference(
            model,
            child,
            name,
            blocked_ops,
            io_casts,
            shadowed=shadowed,
        )
        for child in _iter_ort_child_graphs(graph, blocked_ops)
    )


def _descendant_has_shadowed_mapped_alias(
    model: ModelProto,
    graph: GraphProto,
    source_name: str,
    mapped_name: str,
    blocked_ops: set[str],
    io_casts: set[str],
    *,
    mapped_name_shadowed: bool = False,
) -> bool:
    """Whether mapping a free capture would bind it to a nested target alias."""
    mapped_name_shadowed = mapped_name_shadowed or _graph_defines_name(graph, mapped_name)
    if mapped_name_shadowed and _graph_processed_node_references_name(
        model, graph, source_name, io_casts
    ):
        return True
    return any(
        _descendant_has_shadowed_mapped_alias(
            model,
            child,
            source_name,
            mapped_name,
            blocked_ops,
            io_casts,
            mapped_name_shadowed=mapped_name_shadowed,
        )
        for child in _iter_ort_child_graphs(graph, blocked_ops)
    )


def _descendant_has_free_consumer_in_any_graph(graph: GraphProto, name: str) -> bool:
    """Whether any descendant consumes an outer name without shadowing it."""
    if _graph_defines_name(graph, name):
        return False
    if _graph_node_consumes_name(graph, name):
        return True
    return any(
        _descendant_has_free_consumer_in_any_graph(child, name)
        for child in _iter_all_child_graphs(graph)
    )


def _has_blocked_free_consumer(graph: GraphProto, name: str, blocked_ops: set[str]) -> bool:
    """Whether an ORT-skipped child graph can still capture an outer initializer."""
    for node in getattr(graph, "node", []):
        children = _iter_all_child_graphs_from_node(node)
        if _ort_skips_node_attributes(node, blocked_ops):
            if any(_descendant_has_free_consumer_in_any_graph(child, name) for child in children):
                return True
            continue
        if any(
            not _graph_defines_name(child, name)
            and _has_blocked_free_consumer(child, name, blocked_ops)
            for child in children
        ):
            return True
    return False


def _node_expects_fp32_input(node: NodeProto, input_index: int, blocked_ops: set[str]) -> bool:
    """Whether ORT leaves this node input in FP32 during FP16 conversion."""
    from onnxruntime.transformers.float16 import ALWAYS_FLOAT_INPUTS

    return node.op_type in blocked_ops or input_index in ALWAYS_FLOAT_INPUTS.get(node.op_type, [])


def _schema_parameter_at_index(
    parameters: Sequence[OpSchema.FormalParameter], index: int
) -> OpSchema.FormalParameter | None:
    """Resolve a fixed or trailing variadic ONNX schema parameter."""
    from onnx import defs

    if index < len(parameters):
        return parameters[index]
    if parameters and parameters[-1].option == defs.OpSchema.FormalParameterOption.Variadic:
        return parameters[-1]
    return None


def _node_schema(model: ModelProto, node: NodeProto) -> OpSchema | None:
    """Resolve a node's schema at the model's imported opset version."""
    from onnx import defs

    version = next(
        (
            opset.version
            for opset in getattr(model, "opset_import", [])
            if opset.domain == node.domain
        ),
        None,
    )
    if version is None:
        return None
    try:
        return defs.get_schema(node.op_type, version, node.domain)
    except defs.SchemaError:
        return None


def _schema_parameters_share_concrete_type(
    first: OpSchema.FormalParameter,
    second: OpSchema.FormalParameter,
) -> bool:
    """Whether schema parameters require one concrete runtime type."""
    from onnx import defs

    if first.type_str != second.type_str:
        return False
    return all(
        parameter.option != defs.OpSchema.FormalParameterOption.Variadic or parameter.is_homogeneous
        for parameter in (first, second)
    )


def _schema_parameter_allowed_types(
    schema: OpSchema,
    parameter: OpSchema.FormalParameter,
) -> set[str] | None:
    """Resolve a schema parameter's allowed type strings."""
    constraint = next(
        (
            constraint
            for constraint in schema.type_constraints
            if constraint.type_param_str == parameter.type_str
        ),
        None,
    )
    if constraint is not None:
        return set(constraint.allowed_type_strs)
    return {parameter.type_str} if "(" in parameter.type_str else None


def _schema_payload_type(type_str: str) -> str:
    """Strip ONNX container wrappers to their payload type."""
    value = type_str
    while True:
        if value.startswith(("seq(", "optional(")) and value.endswith(")"):
            value = value[value.index("(") + 1 : -1]
            continue
        if value.startswith("map(") and value.endswith(")"):
            depth = 0
            for index, character in enumerate(value[4:-1], start=4):
                if character == "(":
                    depth += 1
                elif character == ")":
                    depth -= 1
                elif character == "," and depth == 0:
                    value = value[index + 1 : -1]
                    break
            else:
                return value
            continue
        return value


def _schema_parameters_share_payload_domain(
    schema: OpSchema,
    first: OpSchema.FormalParameter,
    second: OpSchema.FormalParameter,
) -> bool:
    """Whether two parameters allow the same normalized payload types."""
    first_types = _schema_parameter_allowed_types(schema, first)
    second_types = _schema_parameter_allowed_types(schema, second)
    if first_types is None or second_types is None:
        return False
    return {_schema_payload_type(type_str) for type_str in first_types} == {
        _schema_payload_type(type_str) for type_str in second_types
    }


def _graph_declared_types(graph: GraphProto, name: str) -> list[TypeProto]:
    """Return graph metadata types declared for a binding."""
    return [
        value.type
        for values in (
            getattr(graph, "input", []),
            getattr(graph, "output", []),
            getattr(graph, "value_info", []),
        )
        for value in values
        if value.name == name
    ]


def _child_graph_input_index(
    schema: OpSchema | None,
    node: NodeProto,
    child: GraphProto,
    input_index: int,
) -> int | None:
    """Map a node input to an aligned child formal input."""
    from onnx import defs

    if (
        schema is None
        or not schema.inputs
        or schema.inputs[-1].option != defs.OpSchema.FormalParameterOption.Variadic
    ):
        return None
    if len(child.input) == len(node.input):
        return input_index
    fixed_prefix = len(schema.inputs) - 1
    if len(child.input) == len(node.input) - fixed_prefix and input_index >= fixed_prefix:
        return input_index - fixed_prefix
    return None


def _child_graph_feedback_output_index(
    model: ModelProto,
    node: NodeProto,
    child: GraphProto,
    input_index: int,
) -> int | None:
    """Map a variadic node input to its positional child feedback output."""
    from onnx import defs

    schema = _node_schema(model, node)
    if (
        schema is None
        or not schema.inputs
        or schema.inputs[-1].option != defs.OpSchema.FormalParameterOption.Variadic
    ):
        return None
    input_prefix = len(schema.inputs) - 1
    output_prefix = len(child.output) - len(node.output)
    if output_prefix <= 0 or input_prefix != output_prefix + 1 or input_index < input_prefix:
        return None
    state_index = input_index - input_prefix
    candidate = output_prefix + state_index
    input_parameter = _schema_parameter_at_index(schema.inputs, input_index)
    output_parameter = _schema_parameter_at_index(schema.outputs, state_index)
    if (
        state_index >= len(node.output)
        or candidate >= len(child.output)
        or input_parameter is None
        or output_parameter is None
        or input_parameter.type_str != output_parameter.type_str
    ):
        return None
    return candidate


def _node_input_actual_types(
    model: ModelProto,
    graph: GraphProto,
    node: NodeProto,
    input_index: int,
) -> list[TypeProto]:
    """Return concrete type sources aligned to a node input."""
    schema = _node_schema(model, node)
    value_types = _graph_declared_types(graph, node.input[input_index])
    value_types.extend(
        child.input[child_index].type
        for child in _iter_all_child_graphs_from_node(node)
        if (child_index := _child_graph_input_index(schema, node, child, input_index)) is not None
    )
    return value_types


def _node_input_is_proven_non_float(
    model: ModelProto,
    graph: GraphProto,
    node: NodeProto,
    input_index: int,
) -> bool:
    """Whether concrete schema-aligned evidence proves a non-FLOAT input."""
    schema = _node_schema(model, node)
    evidence = _node_input_actual_types(model, graph, node, input_index)
    parameter = (
        _schema_parameter_at_index(schema.inputs, input_index) if schema is not None else None
    )
    if schema is not None and parameter is not None and parameter.type_str:
        for other_index, input_name in enumerate(node.input):
            if other_index == input_index or not input_name:
                continue
            other = _schema_parameter_at_index(schema.inputs, other_index)
            if other is not None and _schema_parameters_share_concrete_type(parameter, other):
                evidence.extend(_node_input_actual_types(model, graph, node, other_index))
        for output_index, output_name in enumerate(node.output):
            if not output_name:
                continue
            other = _schema_parameter_at_index(schema.outputs, output_index)
            if other is not None and _schema_parameters_share_concrete_type(parameter, other):
                evidence.extend(_graph_declared_types(graph, output_name))
    concrete_types = [value_type for value_type in evidence if _type_proto_is_concrete(value_type)]
    return bool(concrete_types) and all(
        not _type_proto_contains_float_tensor(value_type) for value_type in concrete_types
    )


def _node_input_is_precision_coupled(
    model: ModelProto,
    graph: GraphProto,
    node: NodeProto,
    input_index: int,
    blocked_ops: set[str],
) -> bool:
    """Whether schema relationships can propagate an input's precision."""
    if _node_input_is_proven_non_float(model, graph, node, input_index):
        return False
    if any(
        _type_proto_enters_ort_global_list(value_type)
        for value_type in _node_input_actual_types(model, graph, node, input_index)
    ):
        return True
    schema = _node_schema(model, node)
    if any(
        child.input[child_index].name
        and _binding_has_precision_coupled_consumer(
            model,
            child,
            child.input[child_index].name,
            blocked_ops,
        )
        for child in _iter_all_child_graphs_from_node(node)
        if (child_index := _child_graph_input_index(schema, node, child, input_index)) is not None
    ):
        return True
    if schema is None:
        return True
    parameter = _schema_parameter_at_index(schema.inputs, input_index)
    if parameter is None or not parameter.type_str:
        return True
    related_parameters: list[tuple[OpSchema.FormalParameter, list[TypeProto]]] = []
    for other_index, input_name in enumerate(node.input):
        if other_index == input_index or not input_name:
            continue
        other = _schema_parameter_at_index(schema.inputs, other_index)
        if other is not None:
            related_parameters.append(
                (
                    other,
                    _node_input_actual_types(model, graph, node, other_index),
                )
            )
    for output_index, output_name in enumerate(node.output):
        if not output_name:
            continue
        other = _schema_parameter_at_index(schema.outputs, output_index)
        if other is not None:
            related_parameters.append(
                (
                    other,
                    _graph_declared_types(graph, output_name),
                )
            )
    if any(
        _schema_parameters_share_concrete_type(parameter, other) for other, _ in related_parameters
    ):
        return True

    input_types = _schema_parameter_allowed_types(schema, parameter)
    if input_types is None:
        return True
    for other, actual_types in related_parameters:
        other_types = _schema_parameter_allowed_types(schema, other)
        if other_types is None:
            return True
        if any(
            input_type != other_type
            and _schema_payload_type(input_type) == _schema_payload_type(other_type)
            for input_type in input_types
            for other_type in other_types
        ) and any(_type_proto_enters_ort_global_list(value_type) for value_type in actual_types):
            return True
    return False


def _binding_has_precision_coupled_consumer(
    model: ModelProto,
    graph: GraphProto,
    name: str,
    blocked_ops: set[str],
) -> bool:
    """Whether a traversed consumer propagates the binding's precision."""
    for node in getattr(graph, "node", []):
        if any(
            input_name == name
            and not _node_expects_fp32_input(node, input_index, blocked_ops)
            and _node_input_is_precision_coupled(
                model,
                graph,
                node,
                input_index,
                blocked_ops,
            )
            for input_index, input_name in enumerate(node.input)
        ):
            return True
        children = _iter_all_child_graphs_from_node(node)
        if _ort_skips_node_attributes(node, blocked_ops):
            continue
        if any(
            not _graph_defines_name(child, name)
            and _binding_has_precision_coupled_consumer(model, child, name, blocked_ops)
            for child in children
        ):
            return True
    return False


def _binding_has_unconverted_payload_consumer(
    model: ModelProto,
    graph: GraphProto,
    name: str,
    blocked_ops: set[str],
) -> bool:
    """Whether an FP16 boundary would conflict with container metadata."""
    for node in getattr(graph, "node", []):
        schema = _node_schema(model, node)
        for input_index, input_name in enumerate(node.input):
            if (
                input_name != name
                or _node_expects_fp32_input(node, input_index, blocked_ops)
                or schema is None
            ):
                continue
            parameter = _schema_parameter_at_index(schema.inputs, input_index)
            if parameter is None:
                continue
            for output_index, output_name in enumerate(node.output):
                if not output_name:
                    continue
                output_parameter = _schema_parameter_at_index(schema.outputs, output_index)
                if output_parameter is None or not _schema_parameters_share_payload_domain(
                    schema, parameter, output_parameter
                ):
                    continue
                if any(
                    _type_proto_contains_float_tensor(value_type)
                    and not _type_proto_enters_ort_global_list(value_type)
                    for value_type in _graph_declared_types(graph, output_name)
                ):
                    return True
        if _ort_skips_node_attributes(node, blocked_ops):
            continue
        if any(
            not _graph_defines_name(child, name)
            and _binding_has_unconverted_payload_consumer(model, child, name, blocked_ops)
            for child in _iter_all_child_graphs_from_node(node)
        ):
            return True
    return False


def _binding_is_proven_non_float(model: ModelProto, graph: GraphProto, name: str) -> bool:
    """Whether a binding's consumers prove that its type is non-FLOAT."""
    declared = [
        value_type
        for value_type in _graph_declared_types(graph, name)
        if _type_proto_is_concrete(value_type)
    ]
    if declared:
        return all(not _type_proto_contains_float_tensor(value_type) for value_type in declared)
    for node in getattr(graph, "node", []):
        if any(
            input_name == name and _node_input_is_proven_non_float(model, graph, node, input_index)
            for input_index, input_name in enumerate(node.input)
        ):
            return True
    return False


def _sparse_initializer_consumer_types(
    graph: GraphProto, name: str, blocked_ops: set[str]
) -> tuple[bool, bool]:
    """Return whether a sparse initializer has FP16 and/or FP32 consumers."""
    has_fp16_consumer = False
    has_fp32_consumer = False
    for node in getattr(graph, "node", []):
        for input_index, input_name in enumerate(node.input):
            if input_name != name:
                continue
            if _node_expects_fp32_input(node, input_index, blocked_ops):
                has_fp32_consumer = True
            else:
                has_fp16_consumer = True

        children = _iter_all_child_graphs_from_node(node)
        if _ort_skips_node_attributes(node, blocked_ops):
            if any(_descendant_has_free_consumer_in_any_graph(child, name) for child in children):
                has_fp32_consumer = True
            continue
        for child in children:
            if _graph_defines_name(child, name):
                continue
            child_has_fp16, child_has_fp32 = _sparse_initializer_consumer_types(
                child, name, blocked_ops
            )
            has_fp16_consumer = has_fp16_consumer or child_has_fp16
            has_fp32_consumer = has_fp32_consumer or child_has_fp32
    return has_fp16_consumer, has_fp32_consumer


def _iter_all_child_graphs_from_node(node: NodeProto) -> list[GraphProto]:
    """Return all child graphs attached to a node."""
    return [
        child for attribute in node.attribute for child in _iter_graphs_from_attribute(attribute)
    ]


def _iter_graphs_from_attribute(attribute: AttributeProto) -> list[GraphProto]:
    """Return graphs carried by one node attribute."""
    from onnx import AttributeProto as ONNXAttributeProto

    if attribute.type == ONNXAttributeProto.GRAPH:
        return [attribute.g]
    if attribute.type == ONNXAttributeProto.GRAPHS:
        return list(attribute.graphs)
    return []


def _ort_traversed_attributes(model: ModelProto, blocked_ops: set[str]) -> list[AttributeProto]:
    """Return attributes ORT's FP16 BFS processes."""
    return [
        attribute
        for graph in _ort_traversed_graphs(model, list(blocked_ops))
        for node in graph.node
        if not _ort_skips_node_attributes(node, blocked_ops)
        for attribute in node.attribute
    ]


def _attribute_requires_type_validation(
    attribute: AttributeProto,
) -> bool:
    """Whether ORT leaves potentially tensor-defining FLOAT data unchanged."""
    from onnx import AttributeProto as ONNXAttributeProto
    from onnx import TensorProto

    if attribute.type in {
        ONNXAttributeProto.FLOAT,
        ONNXAttributeProto.FLOATS,
    }:
        return True
    if attribute.type == ONNXAttributeProto.SPARSE_TENSOR:
        return attribute.sparse_tensor.values.data_type == TensorProto.FLOAT
    if attribute.type == ONNXAttributeProto.SPARSE_TENSORS:
        return any(
            sparse.values.data_type == TensorProto.FLOAT for sparse in attribute.sparse_tensors
        )
    if attribute.type == ONNXAttributeProto.TYPE_PROTO:
        return _type_proto_contains_float_tensor(attribute.tp)
    if attribute.type == ONNXAttributeProto.TYPE_PROTOS:
        return any(
            _type_proto_contains_float_tensor(value_type) for value_type in attribute.type_protos
        )
    return False


def _external_float_attribute_tensors(
    model: ModelProto, blocked_ops: set[str]
) -> list[TensorProto]:
    """Return traversed FLOAT tensor attributes backed by external data."""
    from onnx import AttributeProto as ONNXAttributeProto
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data

    tensors = []
    for attribute in _ort_traversed_attributes(model, blocked_ops):
        if attribute.type == ONNXAttributeProto.TENSOR:
            tensors.append(attribute.t)
        elif attribute.type == ONNXAttributeProto.TENSORS:
            tensors.extend(attribute.tensors)
    return [
        tensor
        for tensor in tensors
        if tensor.data_type == TensorProto.FLOAT and uses_external_data(tensor)
    ]


def _internalize_external_float_attribute_tensors(model: ModelProto, blocked_ops: set[str]) -> None:
    """Drop stale external metadata before ORT converts tensor attributes."""
    from onnx import TensorProto

    for tensor in _external_float_attribute_tensors(model, blocked_ops):
        del tensor.external_data[:]
        tensor.data_location = TensorProto.DEFAULT


def _initializer_output_has_consumers(graph: GraphProto, name: str, blocked_ops: set[str]) -> bool:
    """Resolve direct-output initializer consumers using ONNX lexical scopes."""
    if _graph_node_consumes_name(graph, name):
        return True
    return any(
        _descendant_has_free_consumer(child, name, blocked_ops)
        for child in _iter_ort_child_graphs(graph, blocked_ops)
    )


def _ort_keep_io_name_mapping(
    model: ModelProto, *, keep_io_types: bool
) -> tuple[dict[str, str], set[str]]:
    """Return ORT's global top-level I/O tensor mapping and generated Cast names."""
    from onnx import TensorProto

    if not keep_io_types:
        return {}, set()

    name_mapping: dict[str, str] = {}
    io_casts: set[str] = set()
    for io_kind, values in (
        ("input", getattr(model.graph, "input", [])),
        ("output", getattr(model.graph, "output", [])),
    ):
        for value_index, value in enumerate(values):
            if (
                not value.type.HasField("tensor_type")
                or value.type.tensor_type.elem_type != TensorProto.FLOAT
            ):
                continue
            name_mapping[value.name] = f"graph_{io_kind}_cast_{value_index}"
            io_casts.add(f"graph_{io_kind}_cast{value_index}")
    return name_mapping, io_casts


def _reject_unpreserved_float_container_io(model: ModelProto, *, keep_io_types: bool) -> None:
    """Reject FLOAT container I/O that ORT cannot preserve."""
    if not keep_io_types:
        return
    names = sorted(
        value.name
        for values in (
            getattr(model.graph, "input", []),
            getattr(model.graph, "output", []),
        )
        for value in values
        if value.type.WhichOneof("value") != "tensor_type"
        and _type_proto_enters_ort_global_list(value.type)
    )
    if names:
        joined_names = ", ".join(names)
        msg = (
            f"keep_io_types cannot preserve FLOAT container graph I/O: {joined_names}; "
            "convert with keep_io_types=False or expose tensor I/O."
        )
        raise RuntimeError(msg)


def _reject_shared_keep_io_names(model: ModelProto, *, keep_io_types: bool) -> None:
    """Reject FLOAT input/output aliases that overwrite ORT's global I/O map."""
    from onnx import TensorProto

    if not keep_io_types:
        return
    float_inputs = {
        value.name
        for value in getattr(model.graph, "input", [])
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.elem_type == TensorProto.FLOAT
    }
    float_output_names = [
        value.name
        for value in getattr(model.graph, "output", [])
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.elem_type == TensorProto.FLOAT
    ]
    repeated_outputs = sorted(
        name for name in set(float_output_names) if float_output_names.count(name) > 1
    )
    if repeated_outputs:
        names = ", ".join(repeated_outputs)
        msg = (
            "Top-level repeated FLOAT output names cannot be safely converted with "
            f"keep_io_types=True; ORT overwrites their generated aliases: {names}."
        )
        raise RuntimeError(msg)
    float_outputs = set(float_output_names)
    shared_names = float_inputs & float_outputs
    if shared_names:
        names = ", ".join(sorted(shared_names))
        msg = (
            "Top-level FLOAT names used as both input and output cannot be "
            "safely converted with keep_io_types=True; ORT's global I/O "
            f"mapping overwrites entries: {names}."
        )
        raise RuntimeError(msg)


def _reject_scope_unsafe_keep_io_mappings(
    model: ModelProto,
    *,
    keep_io_types: bool,
    blocked_ops: set[str],
) -> None:
    """Reject nested shadowing that ORT's global keep-I/O name mapping corrupts."""
    name_mapping, io_casts = _ort_keep_io_name_mapping(model, keep_io_types=keep_io_types)
    for name, mapped_name in name_mapping.items():
        if any(
            _descendant_has_shadowed_node_reference(model, child, name, blocked_ops, io_casts)
            for child in _iter_ort_child_graphs(model.graph, blocked_ops)
        ):
            msg = (
                f"Top-level keep_io_types name '{name}' is referenced by a "
                "traversed nested graph with the same local name; ORT's "
                "I/O name mapping is not scope-aware."
            )
            raise RuntimeError(msg)
        if any(
            _descendant_has_shadowed_mapped_alias(
                model,
                child,
                name,
                mapped_name,
                blocked_ops,
                io_casts,
            )
            for child in _iter_ort_child_graphs(model.graph, blocked_ops)
        ):
            msg = (
                f"Top-level keep_io_types name '{name}' maps to generated alias "
                f"'{mapped_name}', but a traversed nested graph binds that alias "
                "locally; ORT's I/O name mapping would change the captured value."
            )
            raise RuntimeError(msg)


def _lexical_binding_owner_id(
    graph: GraphProto,
    name: str,
    parents: dict[int, GraphProto | None],
    generated_top_names: set[str],
) -> int | None:
    """Resolve a tensor name to its defining graph, including virtual ORT I/O aliases."""
    current: GraphProto | None = graph
    while current is not None:
        parent = parents.get(id(current))
        if _graph_defines_name(current, name) or (parent is None and name in generated_top_names):
            return id(current)
        current = parent
    return None


def _lexical_binding_owner(
    graph: GraphProto,
    name: str,
    parents: dict[int, GraphProto | None],
) -> GraphProto | None:
    """Resolve a name to its defining graph."""
    current: GraphProto | None = graph
    while current is not None:
        if _graph_defines_name(current, name):
            return current
        current = parents.get(id(current))
    return None


def _value_info_enters_ort_global_list(value_info: ValueInfoProto) -> bool:
    """Whether ORT records this value metadata for late blocked-node processing."""
    return _type_proto_enters_ort_global_list(value_info.type)


def _type_proto_enters_ort_global_list(value_type: TypeProto) -> bool:
    """Whether ORT converts and records this declared FLOAT type."""
    from onnx import TensorProto

    if value_type.HasField("tensor_type") and value_type.tensor_type.elem_type == TensorProto.FLOAT:
        return True
    return (
        value_type.HasField("sequence_type")
        and value_type.sequence_type.elem_type.HasField("tensor_type")
        and value_type.sequence_type.elem_type.tensor_type.elem_type == TensorProto.FLOAT
    )


def _type_proto_is_concrete(value_type: TypeProto) -> bool:
    """Whether ONNX metadata declares a concrete value type."""
    from onnx import TensorProto

    type_kind = value_type.WhichOneof("value")
    if type_kind == "tensor_type":
        return value_type.tensor_type.elem_type != TensorProto.UNDEFINED
    if type_kind == "sparse_tensor_type":
        return value_type.sparse_tensor_type.elem_type != TensorProto.UNDEFINED
    if type_kind == "sequence_type":
        return _type_proto_is_concrete(value_type.sequence_type.elem_type)
    if type_kind == "optional_type":
        return _type_proto_is_concrete(value_type.optional_type.elem_type)
    if type_kind == "map_type":
        return value_type.map_type.key_type != TensorProto.UNDEFINED and _type_proto_is_concrete(
            value_type.map_type.value_type
        )
    if type_kind == "opaque_type":
        return bool(value_type.opaque_type.domain or value_type.opaque_type.name)
    return False


def _type_proto_contains_float_tensor(value_type: TypeProto) -> bool:
    """Whether a declared type contains a FLOAT tensor payload."""
    from onnx import TensorProto

    type_kind = value_type.WhichOneof("value")
    if type_kind == "tensor_type":
        return value_type.tensor_type.elem_type == TensorProto.FLOAT
    if type_kind == "sparse_tensor_type":
        return value_type.sparse_tensor_type.elem_type == TensorProto.FLOAT
    if type_kind == "sequence_type":
        return _type_proto_contains_float_tensor(value_type.sequence_type.elem_type)
    if type_kind == "optional_type":
        return _type_proto_contains_float_tensor(value_type.optional_type.elem_type)
    if type_kind == "map_type":
        return _type_proto_contains_float_tensor(value_type.map_type.value_type)
    return False


def _type_proto_has_unconverted_float_container(
    value_type: TypeProto,
) -> bool:
    """Whether ORT leaves a FLOAT-bearing container declaration unchanged."""
    type_kind = value_type.WhichOneof("value")
    if type_kind in {"optional_type", "map_type"}:
        return _type_proto_contains_float_tensor(value_type)
    if type_kind == "sequence_type":
        return not _type_proto_enters_ort_global_list(
            value_type
        ) and _type_proto_contains_float_tensor(value_type)
    return False


def _has_unconverted_float_container_declarations(
    model: ModelProto, op_block_list: list[str] | None
) -> bool:
    """Whether a traversed graph declares a FLOAT-bearing container."""
    return any(
        _type_proto_has_unconverted_float_container(value.type)
        for graph in _ort_traversed_graphs(model, op_block_list)
        for values in (
            getattr(graph, "input", []),
            getattr(graph, "output", []),
            getattr(graph, "value_info", []),
        )
        for value in values
    )


def _ort_global_value_info_bindings(
    model: ModelProto,
    *,
    keep_io_types: bool,
    blocked_ops: set[str],
    graphs: list[GraphProto],
    parents: dict[int, GraphProto | None],
) -> tuple[dict[str, int | None], dict[str, str], set[str], dict[str, str]]:
    """Simulate the first lexical owner ORT registers for global value metadata."""
    from onnx import TensorProto

    first_owner: dict[str, int | None] = {}
    first_type_kind: dict[str, str] = {}
    generated_top_names: set[str] = set()
    name_mapping, io_casts = _ort_keep_io_name_mapping(model, keep_io_types=keep_io_types)
    if keep_io_types:
        for io_kind, values in (
            ("input", getattr(model.graph, "input", [])),
            ("output", getattr(model.graph, "output", [])),
        ):
            for value_index, value in enumerate(values):
                if (
                    value.type.HasField("tensor_type")
                    and value.type.tensor_type.elem_type == TensorProto.FLOAT
                ):
                    generated_name = f"graph_{io_kind}_cast_{value_index}"
                    generated_top_names.add(generated_name)
                    first_owner.setdefault(generated_name, id(model.graph))
                    first_type_kind.setdefault(generated_name, "tensor_type")

    graph_io_to_skip = set(name_mapping)
    for graph in graphs:
        for values in (
            getattr(graph, "input", []),
            getattr(graph, "output", []),
            getattr(graph, "value_info", []),
        ):
            for value_info in values:
                if (
                    value_info.name not in graph_io_to_skip
                    and _value_info_enters_ort_global_list(value_info)
                    and value_info.name not in first_owner
                ):
                    first_owner[value_info.name] = _lexical_binding_owner_id(
                        graph,
                        value_info.name,
                        parents,
                        generated_top_names,
                    )
                    first_type_kind[value_info.name] = value_info.type.WhichOneof("value") or ""

    initializer_owners: dict[str, int] = {}
    fp16_initializers: set[str] = set()
    for graph in graphs:
        for initializer in getattr(graph, "initializer", []):
            if initializer.data_type == TensorProto.FLOAT:
                initializer_owners[initializer.name] = id(graph)
        for node in getattr(graph, "node", []):
            if node.name in io_casts:
                continue
            for input_index, original_name in enumerate(node.input):
                input_name = name_mapping.get(original_name, original_name)
                if input_name in initializer_owners and not _node_expects_fp32_input(
                    node, input_index, blocked_ops
                ):
                    fp16_initializers.add(input_name)

    for name, owner in initializer_owners.items():
        if name in fp16_initializers:
            first_owner.setdefault(name, owner)
            first_type_kind.setdefault(name, "tensor_type")
    return first_owner, name_mapping, generated_top_names, first_type_kind


def _reject_scope_unsafe_value_info_lookups(
    model: ModelProto,
    *,
    keep_io_types: bool,
    op_block_list: list[str] | None,
) -> None:
    """Reject late Casts that ORT resolves through the wrong global metadata."""
    blocked_ops = _effective_blocked_ops(op_block_list)
    graphs, parents = _ort_graphs_with_parents(model, blocked_ops)
    first_owner, name_mapping, generated_top_names, first_type_kind = (
        _ort_global_value_info_bindings(
            model,
            keep_io_types=keep_io_types,
            blocked_ops=blocked_ops,
            graphs=graphs,
            parents=parents,
        )
    )
    _, _, fp16_initializers, _, _ = _initializer_tracking_analysis(
        model,
        keep_io_types=keep_io_types,
        blocked_ops=blocked_ops,
        graphs=graphs,
        parents=parents,
    )
    owners_by_id = {id(graph): graph for graph in graphs}

    def lookup_preserves_top_binding(graph: GraphProto, name: str) -> bool:
        intended_owner = _lexical_binding_owner_id(graph, name, parents, generated_top_names)
        top_owner = _lexical_binding_owner_id(model.graph, name, parents, generated_top_names)
        return (
            intended_owner is not None
            and intended_owner == top_owner
            and first_owner[name] == intended_owner
        )

    def binding_converts_to_fp16(graph: GraphProto, name: str) -> bool:
        owner_id = _lexical_binding_owner_id(graph, name, parents, generated_top_names)
        owner = owners_by_id.get(owner_id) if owner_id is not None else None
        if owner is None or any(
            sparse.values.name == name for sparse in getattr(owner, "sparse_initializer", [])
        ):
            return False
        return _binding_converts_to_fp16(
            model,
            owner,
            name,
            keep_io_types=keep_io_types,
            blocked_ops=blocked_ops,
            name_mapping=name_mapping,
            fp16_initializers=fp16_initializers,
            parents=parents,
        )

    reserved_late_tensors: set[str] = set()
    reserved_late_nodes: set[str] = set()
    existing_top_tensors = _graph_tensor_names(model.graph) | generated_top_names
    _, generated_io_nodes = _ort_keep_io_name_mapping(model, keep_io_types=keep_io_types)
    existing_top_nodes = {
        node.name for node in getattr(model.graph, "node", []) if node.name
    } | generated_io_nodes

    def reserve_late_tensor(
        graph: GraphProto,
        node: NodeProto,
        io_kind: str,
        value_index: int,
    ) -> None:
        tensor_name = f"{node.name}_{io_kind}_cast_{value_index}"
        node_name = f"{node.name}_{io_kind}_cast{value_index}"
        shadowed = False
        current = graph
        while current is not model.graph:
            if _graph_defines_name(current, tensor_name):
                shadowed = True
                break
            parent = parents.get(id(current))
            if parent is None:
                break
            current = parent
        if tensor_name in existing_top_tensors or tensor_name in reserved_late_tensors or shadowed:
            msg = (
                f"ORT late Cast tensor '{tensor_name}' for node '{node.name}' "
                "collides with an existing or generated lexical binding."
            )
            raise RuntimeError(msg)
        if node_name in existing_top_nodes or node_name in reserved_late_nodes:
            msg = (
                f"ORT late Cast node '{node_name}' for node '{node.name}' "
                "collides with an existing or generated top-level node."
            )
            raise RuntimeError(msg)
        reserved_late_tensors.add(tensor_name)
        reserved_late_nodes.add(node_name)

    for graph in graphs:
        for node in getattr(graph, "node", []):
            for input_index, original_name in enumerate(node.input):
                if not _node_expects_fp32_input(node, input_index, blocked_ops):
                    continue
                input_name = name_mapping.get(original_name, original_name)
                if input_name not in first_owner:
                    if binding_converts_to_fp16(graph, input_name):
                        msg = (
                            "ORT cannot add a required precision-boundary Cast "
                            f"for blocked or mixed-type node '{node.name}' input "
                            f"'{input_name}' because of missing FLOAT metadata."
                        )
                        raise RuntimeError(msg)
                    continue
                if not lookup_preserves_top_binding(graph, input_name):
                    scope = "nested " if graph is not model.graph else ""
                    msg = (
                        "ORT's global value-info lookup for blocked or mixed-type "
                        f"{scope}node '{node.name}' cannot preserve the lexical input "
                        f"binding for '{input_name}'."
                    )
                    raise RuntimeError(msg)
                if first_type_kind[input_name] != "tensor_type":
                    msg = (
                        "ORT's late Cast path only supports tensor values; "
                        f"blocked or mixed-type node '{node.name}' resolves "
                        f"non-tensor input '{input_name}'."
                    )
                    raise RuntimeError(msg)
                reserve_late_tensor(graph, node, "input", input_index)

            if node.op_type not in blocked_ops:
                continue
            for output_index, original_name in enumerate(node.output):
                output_name = name_mapping.get(original_name, original_name)
                if not output_name:
                    continue
                if output_name not in first_owner:
                    if _binding_has_precision_coupled_consumer(
                        model, graph, output_name, blocked_ops
                    ):
                        msg = (
                            "ORT cannot add a required precision-boundary Cast "
                            f"for blocked node '{node.name}' output '{output_name}' "
                            "because of missing FLOAT metadata."
                        )
                        raise RuntimeError(msg)
                    continue
                if not lookup_preserves_top_binding(graph, output_name):
                    scope = "nested " if graph is not model.graph else ""
                    msg = (
                        "ORT's global value-info lookup for blocked or mixed-type "
                        f"{scope}node '{node.name}' cannot preserve the lexical output "
                        f"binding for '{output_name}'."
                    )
                    raise RuntimeError(msg)
                if first_type_kind[output_name] != "tensor_type":
                    msg = (
                        "ORT's late Cast path only supports tensor values; "
                        f"blocked node '{node.name}' resolves non-tensor output "
                        f"'{output_name}'."
                    )
                    raise RuntimeError(msg)
                if _binding_has_unconverted_payload_consumer(
                    model, graph, output_name, blocked_ops
                ):
                    msg = (
                        f"ORT's FP16 boundary for blocked node '{node.name}' "
                        f"output '{output_name}' conflicts with an unconverted "
                        "container payload declaration."
                    )
                    raise RuntimeError(msg)
                reserve_late_tensor(graph, node, "output", output_index)


def _reject_generated_io_cast_name_collisions(
    model: ModelProto,
    *,
    keep_io_types: bool,
    op_block_list: list[str] | None,
) -> None:
    """Reject names that collide with ORT's deterministic top-level I/O Casts."""
    from onnx import TensorProto

    if not keep_io_types:
        return

    tensor_names = _graph_tensor_names(model.graph)
    node_names = _all_node_names(model, op_block_list)
    for io_kind, values in (
        ("input", getattr(model.graph, "input", [])),
        ("output", getattr(model.graph, "output", [])),
    ):
        for value_index, value in enumerate(values):
            if (
                not value.type.HasField("tensor_type")
                or value.type.tensor_type.elem_type != TensorProto.FLOAT
            ):
                continue
            generated_tensor = f"graph_{io_kind}_cast_{value_index}"
            generated_node = f"graph_{io_kind}_cast{value_index}"
            collisions = [
                name
                for name, existing_names in (
                    (generated_tensor, tensor_names),
                    (generated_node, node_names),
                )
                if name in existing_names
            ]
            if collisions:
                msg = (
                    "FP16 conversion cannot safely allocate ORT "
                    f"{io_kind} Cast names for '{value.name}'; existing names "
                    f"collide: {', '.join(collisions)}."
                )
                raise RuntimeError(msg)


def _capture_safe_initializer_outputs(
    model: ModelProto,
    *,
    keep_io_types: bool,
    op_block_list: list[str] | None,
) -> list[_InitializerOutput]:
    """Capture safe direct initializer outputs or fail before ORT mutates the model.

    Top-level shared outputs are allowed for pure-FP16 conversion when ORT
    converts their consumers consistently; keep-I/O conversion still rejects
    them because ORT rewrites consumer inputs to the generated output-cast alias.
    """
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data

    if not hasattr(model.graph, "input") or not hasattr(model.graph, "output"):
        return []

    blocked_ops = _effective_blocked_ops(op_block_list)
    traversed_graphs = _ort_traversed_graphs(model, op_block_list)

    captured: list[_InitializerOutput] = []
    for graph_index, graph in enumerate(traversed_graphs):
        produced = {name for node in getattr(graph, "node", []) for name in node.output if name}
        graph_inputs = {value.name for value in getattr(graph, "input", [])}
        initializers = {
            initializer.name: initializer for initializer in getattr(graph, "initializer", [])
        }
        for output_index, output in enumerate(getattr(graph, "output", [])):
            initializer = initializers.get(output.name)
            if (
                initializer is None
                or output.name in produced
                or initializer.data_type != TensorProto.FLOAT
            ):
                continue

            if output.name in graph_inputs:
                msg = (
                    f"Initializer-backed output '{output.name}' is also a graph input; "
                    "FP16 conversion cannot preserve overridable-initializer semantics."
                )
                raise RuntimeError(msg)
            has_consumers = _initializer_output_has_consumers(graph, output.name, blocked_ops)
            if keep_io_types and graph_index == 0 and has_consumers:
                msg = (
                    f"Initializer-backed output '{output.name}' has internal consumers; "
                    "FP16 conversion cannot safely preserve keep_io_types semantics."
                )
                raise RuntimeError(msg)
            if (
                keep_io_types
                and graph_index == 0
                and not has_consumers
                and any(
                    _descendant_has_node_reference(child, output.name, blocked_ops)
                    for child in _iter_ort_child_graphs(graph, blocked_ops)
                )
            ):
                msg = (
                    f"Initializer-backed output '{output.name}' is referenced by a "
                    "traversed nested graph with the same local name; ORT's "
                    "keep_io_types output mapping is not scope-aware."
                )
                raise RuntimeError(msg)
            if (not keep_io_types or graph_index != 0) and _has_blocked_free_consumer(
                graph, output.name, blocked_ops
            ):
                msg = (
                    f"Initializer-backed output '{output.name}' is captured by a "
                    "blocked subgraph; FP16 conversion cannot safely change its "
                    "initializer type while that subgraph remains FP32."
                )
                raise RuntimeError(msg)
            if uses_external_data(initializer) and not _tensor_data_is_loaded(initializer):
                msg = (
                    f"Initializer-backed output '{output.name}' uses unloaded external data; "
                    "load external weights before FP16 conversion."
                )
                raise RuntimeError(msg)

            captured.append(
                _InitializerOutput(graph_index, output.name, output_index, has_consumers)
            )
    return captured


def _initializer_data_type(graph: GraphProto, name: str) -> int:
    """Return the initializer data type for a captured output in its graph."""
    return next(value.data_type for value in graph.initializer if value.name == name)


def _set_direct_output_elem_type(graph: GraphProto, name: str, data_type: int) -> None:
    """Set a direct graph output's tensor element type."""
    for output in graph.output:
        if output.name == name and output.type.HasField("tensor_type"):
            output.type.tensor_type.elem_type = data_type
            return


def _convert_output_initializer_to_fp16(graph: GraphProto, name: str) -> None:
    """Convert a captured output's resident FLOAT initializer in place."""
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data
    from onnxruntime.transformers.float16 import convert_tensor_float_to_float16

    initializer = next(value for value in graph.initializer if value.name == name)
    if uses_external_data(initializer):
        del initializer.external_data[:]
        initializer.data_location = TensorProto.DEFAULT
    initializer.CopyFrom(cast("TensorProto", convert_tensor_float_to_float16(initializer)))


def _convert_sparse_initializer_to_fp16(graph: GraphProto, name: str) -> None:
    """Convert a sparse FLOAT initializer's values tensor to FLOAT16."""
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data
    from onnxruntime.transformers.float16 import convert_tensor_float_to_float16

    sparse_initializer = next(
        sparse for sparse in graph.sparse_initializer if sparse.values.name == name
    )
    if uses_external_data(sparse_initializer.values):
        del sparse_initializer.values.external_data[:]
        sparse_initializer.values.data_location = TensorProto.DEFAULT
    sparse_initializer.values.CopyFrom(
        cast("TensorProto", convert_tensor_float_to_float16(sparse_initializer.values))
    )
    for value_info in (
        *getattr(graph, "input", []),
        *getattr(graph, "output", []),
        *getattr(graph, "value_info", []),
    ):
        if (
            value_info.name == name
            and value_info.type.HasField("sparse_tensor_type")
            and value_info.type.sparse_tensor_type.elem_type == TensorProto.FLOAT
        ):
            value_info.type.sparse_tensor_type.elem_type = TensorProto.FLOAT16


def _has_sparse_graph_io(graph: GraphProto, name: str) -> bool:
    """Whether a sparse initializer name is part of graph input/output metadata."""
    return any(
        value_info.name == name and value_info.type.HasField("sparse_tensor_type")
        for value_info in (*getattr(graph, "input", []), *getattr(graph, "output", []))
    )


def _has_sparse_graph_output(graph: GraphProto, name: str) -> bool:
    """Whether a sparse initializer name is a graph output."""
    return any(
        value_info.name == name and value_info.type.HasField("sparse_tensor_type")
        for value_info in getattr(graph, "output", [])
    )


def _has_sparse_graph_input(graph: GraphProto, name: str) -> bool:
    """Whether a sparse initializer name is a graph input."""
    return any(
        value_info.name == name and value_info.type.HasField("sparse_tensor_type")
        for value_info in getattr(graph, "input", [])
    )


def _is_kept_top_level_sparse_edge(
    model: ModelProto, graph: GraphProto, name: str, *, keep_io_types: bool
) -> bool:
    """Whether a nested output feeds an unchanged public sparse output."""
    from onnx import TensorProto

    if not keep_io_types:
        return False
    current_graph = graph
    current_name = name
    visited: set[int] = set()
    while current_graph is not model.graph:
        if id(current_graph) in visited:
            return False
        visited.add(id(current_graph))
        output_index = next(
            (
                index
                for index, output in enumerate(getattr(current_graph, "output", []))
                if output.name == current_name
            ),
            None,
        )
        if output_index is None:
            return False
        parents = [
            (parent_graph, node)
            for parent_graph in _all_graphs(model)
            for node in getattr(parent_graph, "node", [])
            if any(child is current_graph for child in _iter_all_child_graphs_from_node(node))
        ]
        if len(parents) != 1:
            return False
        parent_graph, node = parents[0]
        if len(node.output) != len(current_graph.output) or output_index >= len(node.output):
            return False
        parent_name = node.output[output_index]
        if not parent_name or _graph_node_consumes_name(parent_graph, parent_name):
            return False
        current_graph = parent_graph
        current_name = parent_name
    return any(
        output.name == current_name
        and output.type.HasField("sparse_tensor_type")
        and output.type.sparse_tensor_type.elem_type == TensorProto.FLOAT
        for output in getattr(model.graph, "output", [])
    )


def _repair_sparse_float_initializers(
    model: ModelProto, op_block_list: list[str] | None, *, keep_io_types: bool
) -> None:
    """Convert sparse FLOAT initializers when ORT converted every consumer to FP16."""
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data

    blocked_ops = _effective_blocked_ops(op_block_list)
    for graph_index, graph in enumerate(_ort_traversed_graphs(model, op_block_list)):
        for sparse_initializer in getattr(graph, "sparse_initializer", []):
            values = sparse_initializer.values
            if values.data_type != TensorProto.FLOAT:
                continue
            has_fp16_consumer, has_fp32_consumer = _sparse_initializer_consumer_types(
                graph, values.name, blocked_ops
            )
            kept_sparse_edge = graph_index != 0 and _is_kept_top_level_sparse_edge(
                model,
                graph,
                values.name,
                keep_io_types=keep_io_types,
            )
            has_fp16_output = (
                (not keep_io_types or graph_index != 0)
                and _has_sparse_graph_output(graph, values.name)
                and not kept_sparse_edge
            )
            if (
                graph_index != 0
                and (has_fp16_consumer or has_fp16_output)
                and _has_sparse_graph_output(graph, values.name)
            ):
                msg = (
                    f"Sparse FLOAT initializer '{values.name}' is a nested sparse "
                    "graph output; FP16 conversion cannot safely propagate its type "
                    "through the parent graph edge."
                )
                raise RuntimeError(msg)
            if (has_fp16_consumer or has_fp16_output) and has_fp32_consumer:
                msg = (
                    f"Sparse FLOAT initializer '{values.name}' has both FP16 and "
                    "FP32 consumers after conversion; FP16 conversion cannot safely "
                    "choose one initializer type."
                )
                raise RuntimeError(msg)
            if not has_fp16_consumer and not has_fp16_output:
                continue
            if has_fp16_output and _has_sparse_graph_input(graph, values.name):
                msg = (
                    f"Sparse FLOAT initializer '{values.name}' is also sparse graph input "
                    "and output; FP16 conversion cannot preserve overridable-initializer "
                    "semantics."
                )
                raise RuntimeError(msg)
            if keep_io_types and _has_sparse_graph_io(graph, values.name):
                msg = (
                    f"Sparse FLOAT initializer '{values.name}' is also sparse graph I/O; "
                    "FP16 conversion cannot preserve keep_io_types semantics."
                )
                raise RuntimeError(msg)
            if uses_external_data(values) and not _tensor_data_is_loaded(values):
                msg = (
                    f"Sparse FLOAT initializer '{values.name}' uses unloaded external data; "
                    "load external weights before FP16 conversion."
                )
                raise RuntimeError(msg)
            _convert_sparse_initializer_to_fp16(graph, values.name)


def _internalize_output_initializer(graph: GraphProto, name: str) -> None:
    """Drop stale external metadata after resident bytes were loaded."""
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data

    initializer = next(value for value in graph.initializer if value.name == name)
    if uses_external_data(initializer):
        del initializer.external_data[:]
        initializer.data_location = TensorProto.DEFAULT


def _format_data_type(data_type: int) -> str:
    """Format ONNX tensor data type values for diagnostics."""
    from onnx import TensorProto

    try:
        return TensorProto.DataType.Name(data_type)
    except ValueError:
        return str(data_type)


def _validate_initializer_output_types(model: ModelProto) -> None:
    """Reject direct initializer-backed outputs whose declared type diverged."""
    mismatches: list[str] = []
    for graph in _all_graphs(model):
        if not hasattr(graph, "output") or not hasattr(graph, "initializer"):
            continue
        produced = {name for node in graph.node for name in node.output if name}
        initializers = {initializer.name: initializer for initializer in graph.initializer}
        for output in graph.output:
            initializer = initializers.get(output.name)
            if initializer is None or output.name in produced:
                continue
            if not output.type.HasField("tensor_type"):
                continue
            elem_type = output.type.tensor_type.elem_type
            if elem_type != initializer.data_type:
                graph_name = graph.name or "<unnamed>"
                mismatches.append(
                    f"{graph_name}.{output.name} declares {_format_data_type(elem_type)} "
                    f"but initializer is {_format_data_type(initializer.data_type)}"
                )
    if mismatches:
        msg = (
            "FP16 conversion produced initializer-backed outputs with mismatched "
            f"types: {'; '.join(mismatches)}."
        )
        raise RuntimeError(msg)


def _remove_orphan_output_casts(
    model: ModelProto,
    captured: list[_InitializerOutput],
) -> None:
    """Remove ORT output Casts whose inputs cannot have producers by construction."""
    from onnx import TensorProto

    if not captured:
        return

    remove_indices: list[int] = []
    orphan_inputs: set[str] = set()
    for item in captured:
        generated_tensor = f"graph_output_cast_{item.output_index}"
        generated_node = f"graph_output_cast{item.output_index}"
        matches = [
            (index, node)
            for index, node in enumerate(model.graph.node)
            if node.name == generated_node
            and node.op_type == "Cast"
            and list(node.input) == [generated_tensor]
            and list(node.output) == [item.name]
            and any(
                attribute.name == "to" and attribute.i == TensorProto.FLOAT
                for attribute in node.attribute
            )
        ]
        if len(matches) != 1:
            msg = f"Expected one ORT graph-output Cast for initializer-backed output '{item.name}'."
            raise RuntimeError(msg)
        remove_indices.append(matches[0][0])
        orphan_inputs.add(generated_tensor)

    for index in sorted(remove_indices, reverse=True):
        del model.graph.node[index]
    for item in captured:
        _internalize_output_initializer(model.graph, item.name)
    retained = [value for value in model.graph.value_info if value.name not in orphan_inputs]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained)


def _graph_free_references(graph: GraphProto) -> set[str]:
    """Return values referenced by a nested graph but defined in an outer scope."""
    local_names = {value.name for value in getattr(graph, "input", []) if value.name}
    local_names.update(
        initializer.name for initializer in getattr(graph, "initializer", []) if initializer.name
    )
    local_names.update(
        sparse.values.name
        for sparse in getattr(graph, "sparse_initializer", [])
        if sparse.values.name
    )
    local_names.update(
        output_name
        for node in getattr(graph, "node", [])
        for output_name in node.output
        if output_name
    )

    references = {
        input_name for node in getattr(graph, "node", []) for input_name in node.input if input_name
    }
    references.update(output.name for output in getattr(graph, "output", []) if output.name)
    for node in getattr(graph, "node", []):
        for child in _iter_all_child_graphs_from_node(node):
            references.update(_graph_free_references(child))
    return references - local_names


def _nested_graph_input_source(
    model: ModelProto, graph: GraphProto, name: str
) -> tuple[GraphProto, str, int | None] | None:
    """Resolve a nested formal input through a structurally aligned parent node."""
    input_index = next(
        (index for index, value in enumerate(getattr(graph, "input", [])) if value.name == name),
        None,
    )
    if input_index is None:
        return None
    for parent_graph in _all_graphs(model):
        for node in getattr(parent_graph, "node", []):
            if not any(child is graph for child in _iter_all_child_graphs_from_node(node)):
                continue
            if len(node.input) != len(graph.input) or input_index >= len(node.input):
                return None
            source_name = node.input[input_index]
            if not source_name:
                return None
            feedback_index = _child_graph_feedback_output_index(model, node, graph, input_index)
            return parent_graph, source_name, feedback_index
    return None


def _node_output_precision_sources(
    model: ModelProto,
    node: NodeProto,
    output_index: int,
    blocked_ops: set[str],
) -> list[str] | None:
    """Return direct inputs proven to carry an output's precision payload."""
    schema = _node_schema(model, node)
    if schema is None:
        return None
    children = _iter_all_child_graphs_from_node(node)
    if children:
        sources = []
        for input_index, input_name in enumerate(node.input):
            if not input_name:
                continue
            input_prefix = len(schema.inputs) - 1
            if input_index - input_prefix != output_index:
                continue
            aligned = [
                (child, child_input_index, feedback_index)
                for child in children
                if (child_input_index := _child_graph_input_index(schema, node, child, input_index))
                is not None
                and (
                    feedback_index := _child_graph_feedback_output_index(
                        model, node, child, input_index
                    )
                )
                is not None
            ]
            if len(aligned) == len(children) and all(
                _binding_preserves_precision_from(
                    model,
                    child,
                    child.output[feedback_index].name,
                    child.input[child_input_index].name,
                    blocked_ops,
                    {},
                )
                for child, child_input_index, feedback_index in aligned
            ):
                sources.append(input_name)
        return sources or None
    output_parameter = _schema_parameter_at_index(schema.outputs, output_index)
    if output_parameter is None or not output_parameter.type_str:
        return None
    return [
        input_name
        for input_index, input_name in enumerate(node.input)
        if input_name
        and (input_parameter := _schema_parameter_at_index(schema.inputs, input_index)) is not None
        and (
            _schema_parameters_share_concrete_type(input_parameter, output_parameter)
            or _schema_parameters_share_payload_domain(schema, input_parameter, output_parameter)
        )
    ]


def _binding_preserves_precision_from(
    model: ModelProto,
    graph: GraphProto,
    name: str,
    source_name: str,
    blocked_ops: set[str],
    name_mapping: dict[str, str],
) -> bool:
    """Whether a single-source producer chain preserves a binding's precision."""
    current_name = name
    visited: set[str] = set()
    while current_name != source_name:
        if current_name in name_mapping or any(
            _type_proto_enters_ort_global_list(value_type)
            for value_type in _graph_declared_types(graph, current_name)
        ):
            return False
        if current_name in visited:
            return False
        visited.add(current_name)
        producers = [
            (node, output_index)
            for node in getattr(graph, "node", [])
            if node.op_type not in blocked_ops
            for output_index, output_name in enumerate(node.output)
            if output_name == current_name
        ]
        if len(producers) != 1:
            return False
        node, output_index = producers[0]
        sources = _node_output_precision_sources(model, node, output_index, blocked_ops)
        if sources is None or len(sources) != 1:
            return False
        current_name = sources[0]
    return True


def _producer_chain_converts_to_fp16(
    model: ModelProto,
    graph: GraphProto,
    name: str,
    *,
    keep_io_types: bool,
    blocked_ops: set[str],
    name_mapping: dict[str, str],
    fp16_initializers: set[str],
    parents: dict[int, GraphProto | None],
) -> bool:
    """Trace single-source producer chains without consuming Python stack."""
    from onnx import TensorProto

    current_graph = graph
    current_name = name
    through_producer = False
    visited: set[tuple[int, str]] = set()
    while True:
        owner = _lexical_binding_owner(current_graph, current_name, parents)
        if owner is None:
            return True
        current_graph = owner
        binding = (id(current_graph), current_name)
        if binding in visited:
            return True
        visited.add(binding)
        if _binding_is_proven_non_float(model, current_graph, current_name):
            return False
        if any(
            initializer.name == current_name
            for initializer in getattr(current_graph, "initializer", [])
            if initializer.data_type == TensorProto.FLOAT
        ):
            return current_name in fp16_initializers
        if any(
            sparse.values.name == current_name
            for sparse in getattr(current_graph, "sparse_initializer", [])
            if sparse.values.data_type == TensorProto.FLOAT
        ):
            has_fp16_consumer, _ = _sparse_initializer_consumer_types(
                current_graph,
                current_name,
                blocked_ops,
            )
            has_fp16_output = (
                not keep_io_types or current_graph is not model.graph
            ) and _has_sparse_graph_output(current_graph, current_name)
            return has_fp16_consumer or has_fp16_output
        if current_name in name_mapping:
            return through_producer
        value_types = _graph_declared_types(current_graph, current_name)
        if any(_type_proto_enters_ort_global_list(value_type) for value_type in value_types):
            return True
        if any(_type_proto_contains_float_tensor(value_type) for value_type in value_types):
            if current_graph is not model.graph and any(
                value.name == current_name for value in getattr(current_graph, "input", [])
            ):
                return _binding_converts_to_fp16(
                    model,
                    current_graph,
                    current_name,
                    keep_io_types=keep_io_types,
                    blocked_ops=blocked_ops,
                    name_mapping=name_mapping,
                    fp16_initializers=fp16_initializers,
                    parents=parents,
                )
            producers = [
                (node, output_index)
                for node in getattr(current_graph, "node", [])
                if node.op_type not in blocked_ops
                for output_index, output_name in enumerate(node.output)
                if output_name == current_name
            ]
            if not producers:
                return False
            if len(producers) != 1:
                return True
            node, output_index = producers[0]
            sources = _node_output_precision_sources(model, node, output_index, blocked_ops)
            if sources is None:
                return True
            if not sources:
                return False
            if len(sources) != 1:
                return True
            current_name = sources[0]
            through_producer = True
            continue
        if any(_type_proto_is_concrete(value_type) for value_type in value_types):
            return False
        return any(
            current_name in node.output and node.op_type not in blocked_ops
            for node in getattr(current_graph, "node", [])
        )


def _binding_converts_to_fp16(
    model: ModelProto,
    graph: GraphProto,
    name: str,
    *,
    keep_io_types: bool,
    blocked_ops: set[str],
    name_mapping: dict[str, str],
    fp16_initializers: set[str],
    parents: dict[int, GraphProto | None],
) -> bool:
    """Whether ORT or wrapper repair changes an outer FLOAT binding to FP16."""
    from onnx import TensorProto

    owner = _lexical_binding_owner(graph, name, parents)
    if owner is None:
        return True
    graph = owner
    if _binding_is_proven_non_float(model, graph, name):
        return False
    if any(
        initializer.name == name
        for initializer in getattr(graph, "initializer", [])
        if initializer.data_type == TensorProto.FLOAT
    ):
        return name in fp16_initializers
    if any(
        sparse.values.name == name
        for sparse in getattr(graph, "sparse_initializer", [])
        if sparse.values.data_type == TensorProto.FLOAT
    ):
        has_fp16_consumer, _ = _sparse_initializer_consumer_types(graph, name, blocked_ops)
        has_fp16_output = (
            not keep_io_types or graph is not model.graph
        ) and _has_sparse_graph_output(graph, name)
        return has_fp16_consumer or has_fp16_output
    if name in name_mapping:
        return False
    metadata = [
        value_info
        for values in (
            getattr(graph, "input", []),
            getattr(graph, "output", []),
            getattr(graph, "value_info", []),
        )
        for value_info in values
        if value_info.name == name
    ]
    if any(_value_info_enters_ort_global_list(value_info) for value_info in metadata):
        return True
    if any(_type_proto_contains_float_tensor(value_info.type) for value_info in metadata):
        if graph is not model.graph and any(
            value.name == name for value in getattr(graph, "input", [])
        ):
            source = _nested_graph_input_source(model, graph, name)
            if source is None:
                return True
            parent_graph, source_name, feedback_index = source
            if _binding_converts_to_fp16(
                model,
                parent_graph,
                source_name,
                keep_io_types=keep_io_types,
                blocked_ops=blocked_ops,
                name_mapping=name_mapping,
                fp16_initializers=fp16_initializers,
                parents=parents,
            ):
                return True
            if feedback_index is None:
                return True
            feedback = graph.output[feedback_index]
            return not _type_proto_contains_float_tensor(
                feedback.type
            ) or not _binding_preserves_precision_from(
                model,
                graph,
                feedback.name,
                name,
                blocked_ops,
                name_mapping,
            )
        return _producer_chain_converts_to_fp16(
            model,
            graph,
            name,
            keep_io_types=keep_io_types,
            blocked_ops=blocked_ops,
            name_mapping=name_mapping,
            fp16_initializers=fp16_initializers,
            parents=parents,
        )
    if any(_type_proto_is_concrete(value_info.type) for value_info in metadata):
        return False
    return any(
        name in node.output and node.op_type not in blocked_ops
        for node in getattr(graph, "node", [])
    )


def _local_function_executed_attributes(
    model: ModelProto, node: NodeProto
) -> list[AttributeProto] | None:
    """Resolve supplied or default attributes referenced by a local function."""
    function = next(
        (
            candidate
            for candidate in getattr(model, "functions", [])
            if candidate.domain == node.domain
            and candidate.name == node.op_type
            and getattr(candidate, "overload", "") == getattr(node, "overload", "")
        ),
        None,
    )
    if function is None:
        return None
    referenced_attributes: set[str] = set()
    pending_nodes = list(function.node)
    while pending_nodes:
        function_node = pending_nodes.pop()
        for attribute in function_node.attribute:
            if attribute.ref_attr_name:
                referenced_attributes.add(attribute.ref_attr_name)
            for child in _iter_graphs_from_attribute(attribute):
                pending_nodes.extend(child.node)
    supplied = {attribute.name: attribute for attribute in node.attribute}
    defaults = {attribute.name: attribute for attribute in function.attribute_proto}
    return [
        attribute
        for name in referenced_attributes
        if (attribute := supplied.get(name, defaults.get(name))) is not None
    ]


def _function_contains_concrete_float(function: FunctionProto) -> bool:
    """Whether an unvisited function body stores a concrete FLOAT value."""
    from onnx import AttributeProto as ONNXAttributeProto
    from onnx import TensorProto

    if any(
        not any(node.input)
        and any(
            attribute.type
            in {
                ONNXAttributeProto.FLOAT,
                ONNXAttributeProto.FLOATS,
            }
            for attribute in node.attribute
        )
        for node in function.node
    ):
        return True
    if any(_type_proto_contains_float_tensor(value.type) for value in function.value_info):
        return True

    pending_attributes = [
        attribute
        for node in function.node
        for attribute in node.attribute
        if not attribute.ref_attr_name
    ]
    pending_graphs: list[GraphProto] = []
    while pending_attributes or pending_graphs:
        while pending_attributes:
            attribute = pending_attributes.pop()
            if (
                attribute.type == ONNXAttributeProto.TENSOR
                and attribute.t.data_type == TensorProto.FLOAT
            ):
                return True
            if attribute.type == ONNXAttributeProto.TENSORS and any(
                tensor.data_type == TensorProto.FLOAT for tensor in attribute.tensors
            ):
                return True
            if (
                attribute.type == ONNXAttributeProto.SPARSE_TENSOR
                and attribute.sparse_tensor.values.data_type == TensorProto.FLOAT
            ):
                return True
            if attribute.type == ONNXAttributeProto.SPARSE_TENSORS and any(
                sparse.values.data_type == TensorProto.FLOAT for sparse in attribute.sparse_tensors
            ):
                return True
            if (
                attribute.type == ONNXAttributeProto.TYPE_PROTO
                and _type_proto_contains_float_tensor(attribute.tp)
            ):
                return True
            if attribute.type == ONNXAttributeProto.TYPE_PROTOS and any(
                _type_proto_contains_float_tensor(value_type)
                for value_type in attribute.type_protos
            ):
                return True
            pending_graphs.extend(_iter_graphs_from_attribute(attribute))
        while pending_graphs:
            graph = pending_graphs.pop()
            if any(
                not any(node.input)
                and any(
                    attribute.type
                    in {
                        ONNXAttributeProto.FLOAT,
                        ONNXAttributeProto.FLOATS,
                    }
                    for attribute in node.attribute
                )
                for node in graph.node
            ):
                return True
            if any(
                initializer.data_type == TensorProto.FLOAT for initializer in graph.initializer
            ) or any(
                sparse.values.data_type == TensorProto.FLOAT for sparse in graph.sparse_initializer
            ):
                return True
            if any(
                _type_proto_contains_float_tensor(value.type)
                for values in (
                    graph.input,
                    graph.output,
                    graph.value_info,
                )
                for value in values
            ):
                return True
            pending_attributes.extend(
                attribute
                for node in graph.node
                for attribute in node.attribute
                if not attribute.ref_attr_name
            )
    return False


def _reject_unconverted_local_function_float_data(model: ModelProto, blocked_ops: set[str]) -> None:
    """Reject concrete FLOAT data in function bodies ORT never traverses."""
    functions = {
        (
            function.domain,
            function.name,
            getattr(function, "overload", ""),
        ): function
        for function in getattr(model, "functions", [])
    }
    pending = [
        function
        for graph in _ort_traversed_graphs(model, list(blocked_ops))
        for node in graph.node
        if node.op_type not in blocked_ops
        and (
            function := functions.get(
                (
                    node.domain,
                    node.op_type,
                    getattr(node, "overload", ""),
                )
            )
        )
        is not None
        and (
            any(
                _type_proto_contains_float_tensor(value_type)
                for input_index in range(len(node.input))
                for value_type in _node_input_actual_types(model, graph, node, input_index)
            )
            or any(
                _type_proto_contains_float_tensor(value_type)
                for output_name in node.output
                if output_name
                for value_type in _graph_declared_types(graph, output_name)
            )
        )
    ]
    visited: set[tuple[str, str, str]] = set()
    while pending:
        function = pending.pop()
        key = (
            function.domain,
            function.name,
            getattr(function, "overload", ""),
        )
        if key in visited:
            continue
        visited.add(key)
        if _function_contains_concrete_float(function):
            msg = (
                f"ORT does not convert concrete FLOAT data in local function "
                f"'{function.domain}::{function.name}'."
            )
            raise RuntimeError(msg)
        pending.extend(
            nested
            for node in function.node
            if (
                nested := functions.get(
                    (
                        node.domain,
                        node.op_type,
                        getattr(node, "overload", ""),
                    )
                )
            )
            is not None
        )


def _reject_blocked_subgraph_converted_captures(
    model: ModelProto,
    *,
    keep_io_types: bool,
    op_block_list: list[str] | None,
) -> None:
    """Reject free captures whose FLOAT binding changes while ORT skips the child."""
    blocked_ops = _effective_blocked_ops(op_block_list)
    graphs, parents = _ort_graphs_with_parents(model, blocked_ops)
    _, _, fp16_initializers, _, _ = _initializer_tracking_analysis(
        model,
        keep_io_types=keep_io_types,
        blocked_ops=blocked_ops,
        graphs=graphs,
        parents=parents,
    )
    name_mapping, _ = _ort_keep_io_name_mapping(model, keep_io_types=keep_io_types)
    owners_by_id = {id(graph): graph for graph in graphs}
    for graph in graphs:
        for node in getattr(graph, "node", []):
            if not _ort_skips_node_attributes(node, blocked_ops):
                continue
            executed_attributes = _local_function_executed_attributes(model, node)
            children = (
                child
                for attribute in (
                    node.attribute if executed_attributes is None else executed_attributes
                )
                for child in _iter_graphs_from_attribute(attribute)
            )
            for child in children:
                for name in _graph_free_references(child):
                    owner_id = _lexical_binding_owner_id(graph, name, parents, set())
                    owner = owners_by_id.get(owner_id) if owner_id is not None else None
                    if owner is None or not _binding_converts_to_fp16(
                        model,
                        owner,
                        name,
                        keep_io_types=keep_io_types,
                        blocked_ops=blocked_ops,
                        name_mapping=name_mapping,
                        fp16_initializers=fp16_initializers,
                        parents=parents,
                    ):
                        continue
                    msg = (
                        f"A blocked subgraph under node '{node.name}' captures FLOAT "
                        f"value '{name}' that converts to FP16 while ORT skips the "
                        "subgraph."
                    )
                    raise RuntimeError(msg)


def _validate_local_function_conversion(model: ModelProto) -> None:
    """Validate converted local functions through their expanded call sites."""
    if not getattr(model, "functions", []):
        return

    from onnx import checker, shape_inference
    from onnx.inliner import inline_local_functions

    inlined = inline_local_functions(model)
    try:
        checker.check_model(inlined)
        inferred = shape_inference.infer_shapes(inlined, check_type=True, strict_mode=True)
        checker.check_model(inferred)
    except (
        checker.ValidationError,
        shape_inference.InferenceError,
    ) as error:
        msg = (
            "ORT leaves local function bodies unconverted, producing "
            "incompatible FP16 call-site types."
        )
        raise RuntimeError(msg) from error


def _validate_converted_types(model: ModelProto) -> None:
    """Reject converted graphs whose concrete types no longer agree."""
    from onnx import checker, shape_inference

    try:
        checker.check_model(model)
        inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
        checker.check_model(inferred)
    except (
        checker.ValidationError,
        shape_inference.InferenceError,
    ) as error:
        msg = "FP16 conversion produced incompatible FP16 types."
        raise RuntimeError(msg) from error


def _node_dependencies(node: NodeProto) -> set[str]:
    """Return explicit inputs and outer values captured by the node's subgraphs."""
    dependencies = {input_name for input_name in node.input if input_name}
    for child in _iter_all_child_graphs_from_node(node):
        dependencies.update(_graph_free_references(child))
    return dependencies


def _graph_topological_sort(graph: GraphProto) -> None:
    """Topologically sort nodes while treating dense and sparse initializers as inputs."""
    deps = {initializer.name for initializer in getattr(graph, "initializer", [])}
    deps.update(sparse.values.name for sparse in getattr(graph, "sparse_initializer", []))
    deps.update(value.name for value in getattr(graph, "input", []))
    node_dependencies = [_node_dependencies(node) for node in graph.node]

    sorted_indices: set[int] = set()
    sorted_nodes = []
    last_blocked_node = None
    previous_count = -1
    while len(sorted_indices) != len(graph.node):
        if len(sorted_indices) == previous_count:
            break
        previous_count = len(sorted_indices)
        for node_index, node in enumerate(graph.node):
            if node_index in sorted_indices:
                continue
            if node_dependencies[node_index] <= deps:
                sorted_nodes.append(node)
                sorted_indices.add(node_index)
                deps.update(output for output in node.output if output)
            else:
                last_blocked_node = node.name

    if len(sorted_indices) != len(graph.node):
        msg = (
            "Graph is not a DAG: "
            f"len(sorted_node_set)={len(sorted_indices)}, "
            f"len(graph.node)={len(graph.node)}, "
            f"failed at node {last_blocked_node}"
        )
        raise RuntimeError(msg)

    del graph.node[:]
    graph.node.extend(sorted_nodes)


def convert_to_fp16(
    model: ModelProto,
    *,
    keep_io_types: bool = True,
    op_block_list: list[str] | None = None,
    node_block_list: list[str] | None = None,
) -> ModelProto:
    """Convert an ONNX model from FP32 to FP16 precision.

    Uses onnxruntime.transformers.float16.convert_float_to_float16 internally.
    The successful conversion mutates and returns ``model`` as before.

    ORT assumes each graph output has a node producer. For a safe top-level
    output supplied only by a dense FLOAT initializer, keep-I/O conversion adds
    a Cast with no producer; remove that exact Cast. Pure-FP16 conversion changes
    the output declaration but not its initializer, so convert that initializer
    explicitly.
    """
    from onnx import TensorProto
    from onnxruntime.transformers.float16 import convert_float_to_float16

    _reject_sparse_initializer_tensor_metadata(model, op_block_list)
    _reject_duplicate_float_initializer_names(model, op_block_list)
    io_preflight_model = _ort_inference_preflight_model(model)
    blocked_ops = _effective_blocked_ops(op_block_list)
    _reject_unpreserved_float_container_io(
        io_preflight_model,
        keep_io_types=keep_io_types,
    )
    _reject_scope_unsafe_keep_io_mappings(
        io_preflight_model,
        keep_io_types=keep_io_types,
        blocked_ops=blocked_ops,
    )
    _reject_generated_io_cast_name_collisions(
        io_preflight_model,
        keep_io_types=keep_io_types,
        op_block_list=op_block_list,
    )
    _reject_scope_unsafe_initializer_tracking(
        io_preflight_model,
        keep_io_types=keep_io_types,
        op_block_list=op_block_list,
    )
    _reject_unconverted_local_function_float_data(io_preflight_model, blocked_ops)
    _reject_blocked_subgraph_converted_captures(
        io_preflight_model,
        keep_io_types=keep_io_types,
        op_block_list=op_block_list,
    )
    _reject_scope_unsafe_value_info_lookups(
        io_preflight_model,
        keep_io_types=keep_io_types,
        op_block_list=op_block_list,
    )
    captured = _capture_safe_initializer_outputs(
        model, keep_io_types=keep_io_types, op_block_list=op_block_list
    )
    selected_external_names = _ort_converted_initializer_names(
        io_preflight_model,
        keep_io_types=keep_io_types,
        blocked_ops=blocked_ops,
    )
    selected_external_names.update(
        item.name for item in captured if not keep_io_types or item.graph_index != 0
    )
    from onnx.external_data_helper import uses_external_data

    selected_external_initializers = [
        initializer
        for graph in _ort_traversed_graphs(model, op_block_list)
        for initializer in graph.initializer
        if initializer.name in selected_external_names
        and initializer.data_type == TensorProto.FLOAT
        and uses_external_data(initializer)
    ]
    unloaded_external = sorted(
        initializer.name
        for initializer in selected_external_initializers
        if not _tensor_data_is_loaded(initializer)
    )
    if unloaded_external:
        names = ", ".join(unloaded_external)
        msg = (
            f"FLOAT initializers use unloaded external data: {names}; "
            "load external weights before FP16 conversion."
        )
        raise RuntimeError(msg)
    external_attribute_tensors = _external_float_attribute_tensors(model, blocked_ops)
    unloaded_attribute_tensors = sorted(
        tensor.name or "<unnamed>"
        for tensor in external_attribute_tensors
        if not _tensor_data_is_loaded(tensor)
    )
    if unloaded_attribute_tensors:
        names = ", ".join(unloaded_attribute_tensors)
        msg = (
            f"FLOAT tensor attributes use unloaded external data: {names}; "
            "load external weights before FP16 conversion."
        )
        raise RuntimeError(msg)
    requires_attribute_validation = any(
        _attribute_requires_type_validation(attribute)
        for attribute in _ort_traversed_attributes(io_preflight_model, blocked_ops)
    )
    requires_container_validation = _has_unconverted_float_container_declarations(
        io_preflight_model, op_block_list
    )
    _reject_shared_keep_io_names(
        io_preflight_model,
        keep_io_types=keep_io_types,
    )
    needs_safe_conversion = (
        bool(captured)
        or _has_nested_initializer_outputs(model, op_block_list)
        or _has_float_sparse_initializers(model, op_block_list)
        or bool(getattr(model, "functions", []))
        or bool(selected_external_initializers)
        or bool(external_attribute_tensors)
        or requires_attribute_validation
        or requires_container_validation
    )
    if needs_safe_conversion:
        _reject_unloaded_external_initializer_outputs(model, op_block_list)
    original_nodes = len(model.graph.node)
    conversion_model = deepcopy(model) if needs_safe_conversion else model
    if needs_safe_conversion:
        _internalize_external_initializer_outputs(conversion_model, op_block_list)
        _internalize_selected_external_initializers(
            conversion_model,
            selected_external_names,
            op_block_list,
        )
        _internalize_external_float_attribute_tensors(conversion_model, blocked_ops)

    logger.info("Converting model to FP16...")
    if keep_io_types:
        logger.info("  Keeping I/O types as FP32")
    if op_block_list:
        logger.info("  Keeping ops in FP32: %s", op_block_list)

    try:
        converted: ModelProto = convert_float_to_float16(
            conversion_model,
            keep_io_types=keep_io_types,
            op_block_list=op_block_list,
            node_block_list=node_block_list,
        )
    except EncodeError:
        logger.warning(
            "FP16 conversion shape inference could not serialize the model; "
            "retrying with shape inference disabled. This can happen for "
            "large ONNX models that use external data."
        )
        converted = convert_float_to_float16(
            conversion_model,
            keep_io_types=keep_io_types,
            disable_shape_infer=True,
            op_block_list=op_block_list,
            node_block_list=node_block_list,
        )

    converted_graphs = _ort_traversed_graphs(converted, op_block_list)
    if keep_io_types:
        _remove_orphan_output_casts(converted, [item for item in captured if item.graph_index == 0])

    for item in captured:
        if keep_io_types and item.graph_index == 0:
            continue
        graph = converted_graphs[item.graph_index]
        if item.has_consumers and _initializer_data_type(graph, item.name) == TensorProto.FLOAT16:
            _internalize_output_initializer(graph, item.name)
        elif not item.has_consumers:
            _set_direct_output_elem_type(graph, item.name, TensorProto.FLOAT16)
            _convert_output_initializer_to_fp16(graph, item.name)

    _repair_sparse_float_initializers(converted, op_block_list, keep_io_types=keep_io_types)
    _graph_topological_sort(converted.graph)
    _validate_initializer_output_types(converted)
    if requires_attribute_validation or external_attribute_tensors or requires_container_validation:
        _validate_converted_types(converted)
    _validate_local_function_conversion(converted)

    if converted is not model:
        model.CopyFrom(converted)
        converted = model

    converted_nodes = len(converted.graph.node)
    if converted_nodes != original_nodes:
        logger.info("FP16 conversion complete: %d -> %d nodes", original_nodes, converted_nodes)
    else:
        logger.info("FP16 conversion complete: %d nodes", converted_nodes)

    return converted
