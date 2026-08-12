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
    from onnx import GraphProto, ModelProto, NodeProto, TensorProto

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


def _all_tensor_names(model: ModelProto, op_block_list: list[str] | None) -> set[str]:
    """Collect tensor names across scopes relevant to ORT's global name maps."""
    names: set[str] = set()
    for graph in _ort_traversed_graphs(model, op_block_list):
        names.update(value.name for value in getattr(graph, "input", []))
        names.update(value.name for value in getattr(graph, "output", []))
        names.update(value.name for value in getattr(graph, "value_info", []))
        names.update(initializer.name for initializer in getattr(graph, "initializer", []))
        names.update(sparse.values.name for sparse in getattr(graph, "sparse_initializer", []))
        names.update(
            name
            for node in getattr(graph, "node", [])
            for name in (*node.input, *node.output)
            if name
        )
    return names


def _all_node_names(model: ModelProto, op_block_list: list[str] | None) -> set[str]:
    """Collect node names across scopes because ORT reserves generated names globally."""
    return {
        node.name
        for graph in _ort_traversed_graphs(model, op_block_list)
        for node in getattr(graph, "node", [])
        if node.name
    }


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
    from onnx import AttributeProto

    blocked_ops = _effective_blocked_ops(op_block_list)
    traversed: list[GraphProto] = []
    pending = [model.graph]
    while pending:
        graph = pending.pop()
        traversed.append(graph)
        for node in getattr(graph, "node", []):
            if node.op_type in blocked_ops:
                continue
            for attribute in node.attribute:
                if attribute.type == AttributeProto.GRAPH:
                    pending.append(attribute.g)
                elif attribute.type == AttributeProto.GRAPHS:
                    pending.extend(attribute.graphs)
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


def _iter_ort_child_graphs(graph: GraphProto, blocked_ops: set[str]) -> list[GraphProto]:
    """Return child graphs ORT traverses from this graph."""
    from onnx import AttributeProto

    children: list[GraphProto] = []
    for node in getattr(graph, "node", []):
        if node.op_type in blocked_ops:
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
    """Whether any ORT-traversed descendant node mentions a name globally mapped by ORT."""
    if _graph_node_references_name(graph, name):
        return True
    return any(
        _descendant_has_node_reference(child, name, blocked_ops)
        for child in _iter_ort_child_graphs(graph, blocked_ops)
    )


def _descendant_has_shadowed_node_reference(
    graph: GraphProto,
    name: str,
    blocked_ops: set[str],
    *,
    shadowed: bool = False,
) -> bool:
    """Whether a traversed descendant uses a local name ORT would globally rewrite."""
    shadowed = shadowed or _graph_defines_name(graph, name)
    if shadowed and _graph_node_references_name(graph, name):
        return True
    return any(
        _descendant_has_shadowed_node_reference(child, name, blocked_ops, shadowed=shadowed)
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
        if node.op_type in blocked_ops:
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


def _iter_all_child_graphs_from_node(node: NodeProto) -> list[GraphProto]:
    """Return all child graphs attached to a node."""
    from onnx import AttributeProto

    children: list[GraphProto] = []
    for attribute in node.attribute:
        if attribute.type == AttributeProto.GRAPH:
            children.append(attribute.g)
        elif attribute.type == AttributeProto.GRAPHS:
            children.extend(attribute.graphs)
    return children


def _initializer_output_has_consumers(graph: GraphProto, name: str, blocked_ops: set[str]) -> bool:
    """Resolve direct-output initializer consumers using ONNX lexical scopes."""
    if _graph_node_consumes_name(graph, name):
        return True
    return any(
        _descendant_has_free_consumer(child, name, blocked_ops)
        for child in _iter_ort_child_graphs(graph, blocked_ops)
    )


def _kept_top_level_io_names(model: ModelProto, *, keep_io_types: bool) -> set[str]:
    """Return top-level FLOAT I/O names ORT preserves for keep_io_types=True."""
    from onnx import TensorProto

    if not keep_io_types:
        return set()
    values = [*getattr(model.graph, "input", []), *getattr(model.graph, "output", [])]
    return {
        value.name
        for value in values
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.elem_type == TensorProto.FLOAT
    }


def _reject_scope_unsafe_keep_io_mappings(
    model: ModelProto,
    *,
    keep_io_types: bool,
    blocked_ops: set[str],
) -> None:
    """Reject nested shadowing that ORT's global keep-I/O name mapping corrupts."""
    for name in _kept_top_level_io_names(model, keep_io_types=keep_io_types):
        if any(
            _descendant_has_shadowed_node_reference(child, name, blocked_ops)
            for child in _iter_ort_child_graphs(model.graph, blocked_ops)
        ):
            msg = (
                f"Top-level keep_io_types name '{name}' is referenced by a "
                "traversed nested graph with the same local name; ORT's "
                "I/O name mapping is not scope-aware."
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
    _reject_scope_unsafe_keep_io_mappings(
        model, keep_io_types=keep_io_types, blocked_ops=blocked_ops
    )
    tensor_names = _all_tensor_names(model, op_block_list)
    node_names = _all_node_names(model, op_block_list)

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

            if keep_io_types and graph_index == 0:
                generated_tensor = f"graph_output_cast_{output_index}"
                generated_node = f"graph_output_cast{output_index}"
                collisions = []
                if generated_tensor in tensor_names:
                    collisions.append(generated_tensor)
                if generated_node in node_names:
                    collisions.append(generated_node)
                if collisions:
                    msg = (
                        "FP16 conversion cannot safely allocate ORT graph-output "
                        f"Cast names for '{output.name}'; existing names collide: "
                        f"{', '.join(collisions)}."
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


def _graph_topological_sort(graph: GraphProto) -> None:
    """Topologically sort nodes while treating dense and sparse initializers as inputs."""
    deps = {initializer.name for initializer in getattr(graph, "initializer", [])}
    deps.update(sparse.values.name for sparse in getattr(graph, "sparse_initializer", []))
    deps.update(value.name for value in getattr(graph, "input", []))

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
            if all(not input_name or input_name in deps for input_name in node.input):
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

    _reject_duplicate_float_initializer_names(model, op_block_list)
    captured = _capture_safe_initializer_outputs(
        model, keep_io_types=keep_io_types, op_block_list=op_block_list
    )
    needs_safe_conversion = bool(captured) or _has_nested_initializer_outputs(model, op_block_list)
    if needs_safe_conversion:
        _reject_unloaded_external_initializer_outputs(model, op_block_list)
    original_nodes = len(model.graph.node)
    conversion_model = deepcopy(model) if needs_safe_conversion else model
    if needs_safe_conversion:
        _internalize_external_initializer_outputs(conversion_model, op_block_list)

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

    _graph_topological_sort(converted.graph)
    _validate_initializer_output_types(converted)

    if needs_safe_conversion:
        model.CopyFrom(converted)
        converted = model

    converted_nodes = len(converted.graph.node)
    if converted_nodes != original_nodes:
        logger.info("FP16 conversion complete: %d -> %d nodes", original_nodes, converted_nodes)
    else:
        logger.info("FP16 conversion complete: %d nodes", converted_nodes)

    return converted
