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
    from onnx import GraphProto, ModelProto, TensorProto

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _InitializerOutput:
    """A safe top-level graph output supplied directly by an initializer."""

    name: str
    output_index: int
    has_consumers: bool


def _tensor_data_is_loaded(initializer: TensorProto) -> bool:
    """Whether a FLOAT tensor carries resident data rather than only a sidecar ref."""
    if initializer.raw_data or initializer.float_data:
        return True
    return prod(initializer.dims) == 0


def _all_tensor_names(model: ModelProto) -> set[str]:
    """Collect tensor names across scopes relevant to ORT's global name maps."""
    names: set[str] = set()
    for graph in [model.graph, *_iter_nested_graphs(model)]:
        names.update(value.name for value in graph.input)
        names.update(value.name for value in graph.output)
        names.update(value.name for value in graph.value_info)
        names.update(initializer.name for initializer in graph.initializer)
        names.update(sparse.values.name for sparse in graph.sparse_initializer)
        names.update(name for node in graph.node for name in (*node.input, *node.output) if name)
    return names


def _all_node_names(model: ModelProto) -> set[str]:
    """Collect node names across scopes because ORT reserves generated names globally."""
    return {
        node.name
        for graph in [model.graph, *_iter_nested_graphs(model)]
        for node in graph.node
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


def _direct_initializer_outputs_in_graph(
    graph: GraphProto,
    *,
    data_types: set[int] | None = None,
) -> list[TensorProto]:
    """Return direct graph-output initializers, optionally filtered by data type."""
    if not hasattr(graph, "output") or not hasattr(graph, "initializer"):
        return []

    produced = {name for node in graph.node for name in node.output if name}
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
) -> list[TensorProto]:
    """Return direct initializer-backed outputs across all graph scopes."""
    return [
        initializer
        for graph in _all_graphs(model)
        for initializer in _direct_initializer_outputs_in_graph(graph, data_types=data_types)
    ]


def _has_nested_initializer_outputs(model: ModelProto) -> bool:
    """Whether any nested graph output is supplied directly by a FLOAT initializer."""
    from onnx import TensorProto

    return any(
        _direct_initializer_outputs_in_graph(graph, data_types={TensorProto.FLOAT})
        for graph in _iter_nested_graphs(model)
    )


def _reject_duplicate_float_initializer_names(model: ModelProto) -> None:
    """Reject FLOAT initializer names that ORT's global conversion map cannot scope."""
    from onnx import TensorProto

    seen: set[str] = set()
    duplicates: set[str] = set()
    for graph in _all_graphs(model):
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


def _reject_unloaded_external_initializer_outputs(model: ModelProto) -> None:
    """Reject direct FLOAT output initializers whose external data is not resident."""
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data

    for initializer in _direct_initializer_outputs(model, data_types={TensorProto.FLOAT}):
        if uses_external_data(initializer) and not _tensor_data_is_loaded(initializer):
            msg = (
                f"Initializer-backed output '{initializer.name}' uses unloaded external data; "
                "load external weights before FP16 conversion."
            )
            raise RuntimeError(msg)


def _internalize_external_initializer_outputs(model: ModelProto) -> None:
    """Drop stale external metadata for resident direct output initializer data."""
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data

    for initializer in _direct_initializer_outputs(model, data_types={TensorProto.FLOAT}):
        if uses_external_data(initializer):
            del initializer.external_data[:]
            initializer.data_location = TensorProto.DEFAULT


def _all_node_inputs(model: ModelProto) -> set[str]:
    """Collect node input names across top-level and nested graphs."""
    inputs = {name for node in model.graph.node for name in node.input if name}
    for graph in _iter_nested_graphs(model):
        inputs.update(name for node in graph.node for name in node.input if name)
    return inputs


def _capture_safe_initializer_outputs(
    model: ModelProto,
    *,
    keep_io_types: bool,
) -> list[_InitializerOutput]:
    """Capture safe direct initializer outputs or fail before ORT mutates the model.

    This fix intentionally pre-validates only top-level, non-overridable dense
    FLOAT outputs. Shared outputs are allowed for pure-FP16 conversion when ORT
    converts their consumers consistently; keep-I/O conversion still rejects them
    because ORT rewrites consumer inputs to the generated output-cast alias.
    """
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data

    if not hasattr(model.graph, "input") or not hasattr(model.graph, "output"):
        return []

    produced = {name for node in model.graph.node for name in node.output if name}
    consumed = _all_node_inputs(model)
    graph_inputs = {value.name for value in model.graph.input}
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    tensor_names = _all_tensor_names(model)
    node_names = _all_node_names(model)

    captured: list[_InitializerOutput] = []
    for output_index, output in enumerate(model.graph.output):
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
        has_consumers = output.name in consumed
        if keep_io_types and has_consumers:
            msg = (
                f"Initializer-backed output '{output.name}' has internal consumers; "
                "FP16 conversion cannot safely preserve keep_io_types semantics."
            )
            raise RuntimeError(msg)
        if uses_external_data(initializer) and not _tensor_data_is_loaded(initializer):
            msg = (
                f"Initializer-backed output '{output.name}' uses unloaded external data; "
                "load external weights before FP16 conversion."
            )
            raise RuntimeError(msg)

        if keep_io_types:
            generated_tensor = f"graph_output_cast_{output_index}"
            generated_node = f"graph_output_cast{output_index}"
            collisions = []
            if generated_tensor in tensor_names:
                collisions.append(generated_tensor)
            if generated_node in node_names:
                collisions.append(generated_node)
            if collisions:
                msg = (
                    f"FP16 conversion cannot safely allocate ORT graph-output Cast names for "
                    f"'{output.name}'; existing names collide: {', '.join(collisions)}."
                )
                raise RuntimeError(msg)

        captured.append(_InitializerOutput(output.name, output_index, has_consumers))
    return captured


def _initializer_data_type(model: ModelProto, name: str) -> int:
    """Return the top-level initializer data type for a captured output."""
    return next(value.data_type for value in model.graph.initializer if value.name == name)


def _convert_output_initializer_to_fp16(model: ModelProto, name: str) -> None:
    """Convert a captured output's resident FLOAT initializer in place."""
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data
    from onnxruntime.transformers.float16 import convert_tensor_float_to_float16

    initializer = next(value for value in model.graph.initializer if value.name == name)
    if uses_external_data(initializer):
        del initializer.external_data[:]
        initializer.data_location = TensorProto.DEFAULT
    initializer.CopyFrom(cast("TensorProto", convert_tensor_float_to_float16(initializer)))


def _internalize_output_initializer(model: ModelProto, name: str) -> None:
    """Drop stale external metadata after resident bytes were loaded."""
    from onnx import TensorProto
    from onnx.external_data_helper import uses_external_data

    initializer = next(value for value in model.graph.initializer if value.name == name)
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
        _internalize_output_initializer(model, item.name)
    retained = [value for value in model.graph.value_info if value.name not in orphan_inputs]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained)


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

    fp32_types = {TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.BFLOAT16}
    initializers = [
        initializer for graph in _all_graphs(model) for initializer in graph.initializer
    ]
    if initializers:
        floating = [
            value for value in initializers if value.data_type in fp32_types | {TensorProto.FLOAT16}
        ]
        if floating and all(value.data_type == TensorProto.FLOAT16 for value in floating):
            logger.info("Model is already FP16 - skipping conversion.")
            return model

    _reject_duplicate_float_initializer_names(model)
    captured = _capture_safe_initializer_outputs(model, keep_io_types=keep_io_types)
    needs_safe_conversion = bool(captured) or _has_nested_initializer_outputs(model)
    if needs_safe_conversion:
        _reject_unloaded_external_initializer_outputs(model)
    original_nodes = len(model.graph.node)
    conversion_model = deepcopy(model) if needs_safe_conversion else model
    if needs_safe_conversion:
        _internalize_external_initializer_outputs(conversion_model)

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

    if keep_io_types:
        _remove_orphan_output_casts(converted, captured)
    else:
        for item in captured:
            if (
                item.has_consumers
                and _initializer_data_type(converted, item.name) == TensorProto.FLOAT16
            ):
                _internalize_output_initializer(converted, item.name)
            elif not item.has_consumers:
                _convert_output_initializer_to_fp16(converted, item.name)

    from onnxruntime.transformers.onnx_model import OnnxModel

    OnnxModel.graph_topological_sort(converted.graph)
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
