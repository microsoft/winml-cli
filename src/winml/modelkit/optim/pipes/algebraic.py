# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Conservative, opt-in algebraic ONNX graph rewrites.

The transformations in this pipe intentionally match topology, static tensor
metadata, attributes, and constant values.  They do not rely on exporter names
or on a particular model architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import onnx

from ..capabilities import algebraic
from .base import BasePipe, PipeConfig, caps_dict


logger = logging.getLogger(__name__)


ALGEBRAIC_CAPABILITIES: dict[str, Any] = caps_dict(
    algebraic.STATIC_SPLIT_TO_SLICE,
    algebraic.CONV_CHANNEL_AFFINE_FOLDING,
    algebraic.CONV_ADD_BATCH_NORMALIZATION_FOLDING,
)


@dataclass
class AlgebraicRewritePipeConfig(PipeConfig):
    """Independent switches for the exact algebraic rewrites."""

    static_split_to_slice: bool = False
    conv_channel_affine_folding: bool = False
    conv_add_batch_normalization_folding: bool = False


@dataclass
class _GraphIndex:
    """Small graph index rebuilt after each mutation."""

    producers: dict[str, onnx.NodeProto]
    consumers: dict[str, list[onnx.NodeProto]]
    initializers: dict[str, onnx.TensorProto]
    shapes: dict[str, tuple[int | None, ...]]
    graph_outputs: set[str]

    @classmethod
    def build(cls, model: onnx.ModelProto) -> _GraphIndex:
        from onnx import numpy_helper

        graph = model.graph
        producers: dict[str, onnx.NodeProto] = {}
        consumers: dict[str, list[onnx.NodeProto]] = {}
        for node in graph.node:
            for output in node.output:
                if output:
                    producers[output] = node
            consumed_names = {input_name for input_name in node.input if input_name}
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    consumed_names.update(_captured_tensor_names(attribute.g))
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested_graph in attribute.graphs:
                        consumed_names.update(_captured_tensor_names(nested_graph))
            for input_name in consumed_names:
                consumers.setdefault(input_name, []).append(node)

        initializers = {initializer.name: initializer for initializer in graph.initializer}
        shapes: dict[str, tuple[int | None, ...]] = {}
        for value_info in (*graph.input, *graph.value_info, *graph.output):
            shape = _value_info_shape(value_info)
            if shape is not None:
                shapes[value_info.name] = shape
        for name, initializer in initializers.items():
            shapes.setdefault(name, tuple(int(dim) for dim in initializer.dims))

        # Touch numpy_helper here so malformed tensor proto errors stay local to
        # the graph-index construction rather than a later rewrite.
        for initializer in initializers.values():
            numpy_helper.to_array(initializer)

        return cls(
            producers=producers,
            consumers=consumers,
            initializers=initializers,
            shapes=shapes,
            graph_outputs={output.name for output in graph.output if output.name},
        )


@dataclass
class _AffineCandidate:
    """A safe affine branch associated with a Conv output channel interval."""

    source_node: onnx.NodeProto
    source_output_index: int
    final_output: str
    nodes: list[onnx.NodeProto]
    start: int
    end: int
    scale: np.ndarray
    offset: np.ndarray


class _NameAllocator:
    """Allocate names without relying on optional or duplicated node names."""

    def __init__(self, model: onnx.ModelProto) -> None:
        graph = model.graph
        self._used = {
            name
            for name in (
                [initializer.name for initializer in graph.initializer]
                + [value.name for value in graph.input]
                + [value.name for value in graph.value_info]
                + [value.name for value in graph.output]
                + [node.name for node in graph.node]
                + [output for node in graph.node for output in node.output]
            )
            if name
        }

    def new(self, prefix: str) -> str:
        candidate = prefix
        suffix = 0
        while candidate in self._used:
            suffix += 1
            candidate = f"{prefix}_{suffix}"
        self._used.add(candidate)
        return candidate


def _value_info_shape(value_info: onnx.ValueInfoProto) -> tuple[int | None, ...] | None:
    """Return a tensor shape, preserving unknown dimensions as ``None``."""
    if not value_info.type.HasField("tensor_type"):
        return None
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dimensions: list[int | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        else:
            dimensions.append(None)
    return tuple(dimensions)


def _attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    from onnx import helper

    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def _constant_array(index: _GraphIndex, name: str) -> np.ndarray | None:
    """Read an initializer or a regular ONNX Constant value."""
    from onnx import numpy_helper

    if not name:
        return None
    initializer = index.initializers.get(name)
    if initializer is not None:
        return np.asarray(numpy_helper.to_array(initializer))

    producer = index.producers.get(name)
    if producer is None or producer.op_type != "Constant":
        return None
    value = _attribute(producer, "value")
    if value is not None:
        try:
            return np.asarray(numpy_helper.to_array(value))
        except (TypeError, ValueError):
            return None
    for attribute_name in ("value_float", "value_floats", "value_int", "value_ints"):
        attribute_value = _attribute(producer, attribute_name)
        if attribute_value is not None:
            return np.asarray(attribute_value)
    return None


def _constant_ints(index: _GraphIndex, name: str) -> list[int] | None:
    values = _constant_array(index, name)
    if values is None or not np.issubdtype(values.dtype, np.integer):
        return None
    return [int(value) for value in values.reshape(-1).tolist()]


def _single_attribute_or_input_ints(
    index: _GraphIndex,
    node: onnx.NodeProto,
    attribute_name: str,
    input_index: int | None,
) -> tuple[list[int] | None, bool]:
    """Read legacy attributes and newer constant inputs.

    The boolean reports whether both representations were present and
    inconsistent, which makes the rewrite ineligible rather than guessing.
    """
    attribute_value = _attribute(node, attribute_name)
    from_attribute = None
    if attribute_value is not None:
        try:
            values = np.asarray(attribute_value)
            if not np.issubdtype(values.dtype, np.integer):
                return None, True
            from_attribute = [int(value) for value in values.reshape(-1).tolist()]
        except (TypeError, ValueError):
            return None, True

    from_input = None
    if input_index is not None and len(node.input) > input_index and node.input[input_index]:
        from_input = _constant_ints(index, node.input[input_index])
        if from_input is None:
            return None, True

    if from_attribute is not None and from_input is not None and from_attribute != from_input:
        return None, True
    return from_input if from_input is not None else from_attribute, False


def _node_output(node: onnx.NodeProto) -> str | None:
    return node.output[0] if len(node.output) == 1 and node.output[0] else None


def _static_shape(index: _GraphIndex, name: str) -> tuple[int, ...] | None:
    shape = index.shapes.get(name)
    if shape is None or any(dimension is None for dimension in shape):
        return None
    return tuple(int(dimension) for dimension in shape)


def _new_initializer(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
    values: np.ndarray,
    prefix: str,
    introduced: set[str],
) -> str:
    from onnx import numpy_helper

    name = allocator.new(prefix)
    model.graph.initializer.append(numpy_helper.from_array(np.asarray(values), name))
    introduced.add(name)
    return name


def _remove_nodes(model: onnx.ModelProto, nodes: set[int]) -> None:
    remaining = [node for node in model.graph.node if id(node) not in nodes]
    del model.graph.node[:]
    model.graph.node.extend(remaining)


def _captured_tensor_names(graph: onnx.GraphProto) -> set[str]:
    """Return names a nested graph resolves from an enclosing scope."""
    locally_defined = {value.name for value in graph.input if value.name}
    locally_defined.update(initializer.name for initializer in graph.initializer)
    locally_defined.update(output for node in graph.node for output in node.output if output)
    referenced = {input_name for node in graph.node for input_name in node.input if input_name}
    referenced.update(output.name for output in graph.output if output.name)
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                referenced.update(_captured_tensor_names(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested_graph in attribute.graphs:
                    referenced.update(_captured_tensor_names(nested_graph))
    return referenced - locally_defined


def _referenced_tensor_names(graph: onnx.GraphProto) -> set[str]:
    """Collect tensor names referenced by a graph or its nested subgraphs."""
    referenced = {value.name for value in (*graph.input, *graph.output) if value.name}
    for node in graph.node:
        referenced.update(input_name for input_name in node.input if input_name)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                referenced.update(_referenced_tensor_names(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested_graph in attribute.graphs:
                    referenced.update(_referenced_tensor_names(nested_graph))
    return referenced


def _prune_unused_initializers(model: onnx.ModelProto) -> None:
    """Remove initializers made dead by a rewrite, including replaced parameters."""
    used = _referenced_tensor_names(model.graph)
    remaining = [initializer for initializer in model.graph.initializer if initializer.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(remaining)


def _prune_introduced_nodes(model: onnx.ModelProto, introduced: set[str]) -> None:
    """Remove only generated nodes whose outputs became entirely dead."""
    if not introduced:
        return
    while True:
        index = _GraphIndex.build(model)
        removable = {
            id(node)
            for node in model.graph.node
            if node.name in introduced
            and all(
                output and output not in index.graph_outputs and not index.consumers.get(output)
                for output in node.output
            )
        }
        if not removable:
            return
        _remove_nodes(model, removable)


def _prune_dead_constant_nodes(model: onnx.ModelProto) -> None:
    """Remove Constant nodes made dead by an algebraic rewrite."""
    while True:
        index = _GraphIndex.build(model)
        removable = {
            id(node)
            for node in model.graph.node
            if node.op_type == "Constant"
            and all(
                output and output not in index.graph_outputs and not index.consumers.get(output)
                for output in node.output
            )
        }
        if not removable:
            return
        _remove_nodes(model, removable)


def _split_boundaries(
    index: _GraphIndex,
    node: onnx.NodeProto,
    input_name: str,
) -> tuple[int, list[tuple[int, int]]] | None:
    """Return a static Split axis and output channel-independent boundaries."""
    input_shape = index.shapes.get(input_name)
    if input_shape is None or len(node.output) == 0:
        return None

    axis_input_index = 2 if len(node.input) > 2 else None
    axis_values, axis_conflict = _single_attribute_or_input_ints(
        index, node, "axis", axis_input_index
    )
    if axis_conflict or (axis_values is not None and len(axis_values) != 1):
        return None
    axis = axis_values[0] if axis_values is not None else 0
    if axis < -len(input_shape) or axis >= len(input_shape):
        return None
    axis %= len(input_shape)
    axis_size = input_shape[axis]
    if axis_size is None:
        return None

    split_values, split_conflict = _single_attribute_or_input_ints(index, node, "split", 1)
    if split_conflict:
        return None
    if split_values is None:
        if axis_size <= 0 or axis_size % len(node.output) != 0:
            return None
        part = axis_size // len(node.output)
        split_values = [part] * len(node.output)
    if len(split_values) != len(node.output) or any(value <= 0 for value in split_values):
        return None
    if sum(split_values) != axis_size:
        return None

    boundaries: list[tuple[int, int]] = []
    start = 0
    for size in split_values:
        boundaries.append((start, start + size))
        start += size
    return axis, boundaries


def _slice_channel_boundary(
    index: _GraphIndex,
    node: onnx.NodeProto,
    input_name: str,
    channel_axis: int,
) -> tuple[int, int] | None:
    """Read a Slice that selects a contiguous, full non-channel region."""
    if len(node.input) < 2:
        return None
    input_shape = _static_shape(index, input_name)
    if input_shape is None or channel_axis >= len(input_shape):
        return None
    starts = _constant_ints(index, node.input[1])
    ends = _constant_ints(index, node.input[2]) if len(node.input) > 2 else None
    axes = _constant_ints(index, node.input[3]) if len(node.input) > 3 else None
    steps = _constant_ints(index, node.input[4]) if len(node.input) > 4 else None
    if starts is None or ends is None:
        return None
    if axes is None:
        axes = list(range(len(starts)))
    if steps is None:
        steps = [1] * len(starts)
    if not (len(starts) == len(ends) == len(axes) == len(steps)):
        return None
    normalized_axes: list[int] = []
    for axis in axes:
        if axis < -len(input_shape) or axis >= len(input_shape):
            return None
        normalized_axes.append(axis % len(input_shape))
    if len(set(normalized_axes)) != len(normalized_axes) or any(step != 1 for step in steps):
        return None

    def normalize_bound(value: int, axis_size: int, *, is_end: bool) -> int:
        if value < 0:
            value += axis_size
        if is_end and value > axis_size:
            return axis_size
        return max(0, min(value, axis_size))

    channel_boundary: tuple[int, int] | None = None
    for start_value, end_value, axis in zip(starts, ends, normalized_axes, strict=True):
        axis_size = input_shape[axis]
        start = normalize_bound(start_value, axis_size, is_end=False)
        end = normalize_bound(end_value, axis_size, is_end=True)
        if axis == channel_axis:
            if end <= start:
                return None
            channel_boundary = (start, end)
        elif start != 0 or end != axis_size:
            return None
    if channel_boundary is None:
        return None
    return channel_boundary


def _channel_affine_values(
    values: np.ndarray,
    output_shape: tuple[int, ...],
    channels: int,
) -> np.ndarray | None:
    """Convert a scalar or a provably channel-only broadcast to ``[C]``."""
    if not np.issubdtype(values.dtype, np.floating):
        return None
    if values.size == 1:
        return np.full(channels, values.reshape(-1)[0], dtype=values.dtype)
    if values.ndim > len(output_shape):
        return None

    padded = (1,) * (len(output_shape) - values.ndim) + tuple(values.shape)
    for axis, dimension in enumerate(padded):
        if dimension not in (1, output_shape[axis]):
            return None
        if axis != 1 and dimension != 1:
            # Spatial or batch-varying transforms are deliberately outside this
            # channel-only capability.
            return None
    if padded[1] != channels:
        return None
    return np.asarray(values).reshape(channels)


def _affine_operand(
    index: _GraphIndex,
    node: onnx.NodeProto,
    data_name: str,
    output_shape: tuple[int, ...],
    channels: int,
) -> np.ndarray | None:
    constant_inputs = [
        name
        for name in node.input
        if name and name != data_name and _constant_array(index, name) is not None
    ]
    if len(constant_inputs) != 1 or len(node.input) != 2:
        return None
    values = _constant_array(index, constant_inputs[0])
    return None if values is None else _channel_affine_values(values, output_shape, channels)


def _channel_preserving_view_output(
    index: _GraphIndex,
    node: onnx.NodeProto,
    input_name: str,
    channels: int,
) -> str | None:
    """Return a shape-only view output that preserves N/C order."""
    output_name = _node_output(node)
    if (
        output_name is None
        or len(node.input) == 0
        or node.input[0] != input_name
        or node.op_type not in {"Reshape", "Squeeze", "Unsqueeze"}
    ):
        return None
    input_shape = _static_shape(index, input_name)
    output_shape = _static_shape(index, output_name)
    if (
        input_shape is None
        or output_shape is None
        or len(input_shape) < 2
        or len(output_shape) < 2
        or input_shape[:2] != output_shape[:2]
        or input_shape[1] != channels
        or np.prod(input_shape[2:], dtype=np.int64) != np.prod(output_shape[2:], dtype=np.int64)
    ):
        return None

    if node.op_type == "Reshape":
        if (
            len(node.input) < 2
            or _constant_ints(index, node.input[1]) is None
            or _attribute(node, "allowzero", 0) != 0
        ):
            return None
    else:
        axes, conflict = _single_attribute_or_input_ints(index, node, "axes", 1)
        if conflict or axes is None:
            return None
    return output_name


def _collect_affine_chain(
    index: _GraphIndex,
    first: onnx.NodeProto,
    source_name: str,
    output_shape: tuple[int, ...],
    channel_count: int,
    start: int,
    end: int,
) -> _AffineCandidate | None:
    """Collect a safe consecutive Mul/Add chain from one routed branch."""
    current = first
    current_input = source_name
    scale = np.ones(end - start, dtype=np.float32)
    offset = np.zeros(end - start, dtype=np.float32)
    matched: list[onnx.NodeProto] = []

    while current.op_type in {"Mul", "Add"}:
        if len(current.input) != 2 or current_input not in current.input:
            return None
        current_output = _node_output(current)
        if current_output is None:
            return None
        values = _affine_operand(index, current, current_input, output_shape, end - start)
        if values is None:
            return None
        values = values.astype(np.result_type(values.dtype, np.float32), copy=False)
        if current.op_type == "Mul":
            scale *= values
            offset *= values
        else:
            offset += values
        matched.append(current)

        consumers = index.consumers.get(current_output, [])
        if current_output in index.graph_outputs or len(consumers) != 1:
            break
        next_node = consumers[0]
        if next_node.op_type not in {"Mul", "Add"}:
            break
        current_input = current_output
        current = next_node

    final_output = _node_output(matched[-1]) if matched else None
    if final_output is None or (
        final_output not in index.graph_outputs and len(index.consumers.get(final_output, [])) == 0
    ):
        return None
    return _AffineCandidate(
        source_node=first,
        source_output_index=0,
        final_output=final_output,
        nodes=matched,
        start=start,
        end=end,
        scale=scale,
        offset=offset,
    )


def _collect_routed_affine_candidates(
    index: _GraphIndex,
    source_node: onnx.NodeProto,
    source_output_index: int,
    start: int,
    end: int,
) -> list[_AffineCandidate]:
    """Collect affine leaves below safe views and disjoint channel slices."""
    if source_output_index >= len(source_node.output):
        return []
    source_name = source_node.output[source_output_index]
    if not source_name or source_name in index.graph_outputs:
        return []

    current_node = source_node
    current_output_index = source_output_index
    current_name = source_name
    current_shape = _static_shape(index, current_name)
    if current_shape is None or len(current_shape) < 2 or current_shape[1] != end - start:
        return []

    consumers = index.consumers.get(current_name, [])
    while len(consumers) == 1:
        view = consumers[0]
        view_output = _channel_preserving_view_output(
            index,
            view,
            current_name,
            end - start,
        )
        if view_output is None or current_name in index.graph_outputs:
            break
        current_node = view
        current_output_index = 0
        current_name = view_output
        current_shape = _static_shape(index, current_name)
        if current_shape is None:
            return []
        consumers = index.consumers.get(current_name, [])

    if current_name in index.graph_outputs:
        return []
    if len(consumers) == 1 and consumers[0].op_type in {"Mul", "Add"}:
        candidate = _collect_affine_chain(
            index,
            consumers[0],
            current_name,
            current_shape,
            end - start,
            start,
            end,
        )
        if candidate is None:
            return []
        candidate.source_node = current_node
        candidate.source_output_index = current_output_index
        return [candidate]

    if not consumers or any(node.op_type != "Slice" for node in consumers):
        return []
    routed_slices: list[tuple[onnx.NodeProto, int, int]] = []
    for routed_slice in consumers:
        boundary = _slice_channel_boundary(index, routed_slice, current_name, 1)
        if boundary is None:
            return []
        routed_slices.append((routed_slice, *boundary))
    if any(
        left_start < right_end and right_start < left_end
        for position, (_, left_start, left_end) in enumerate(routed_slices)
        for _, right_start, right_end in routed_slices[position + 1 :]
    ):
        return []

    candidates: list[_AffineCandidate] = []
    for routed_slice, local_start, local_end in routed_slices:
        candidates.extend(
            _collect_routed_affine_candidates(
                index,
                routed_slice,
                0,
                start + local_start,
                start + local_end,
            )
        )
    return candidates


def _copy_conv_parameters(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
    introduced: set[str],
    conv: onnx.NodeProto,
    scale: np.ndarray,
    offset: np.ndarray,
) -> bool:
    from onnx import numpy_helper

    if len(conv.input) < 2:
        return False
    weight = next(
        (
            initializer
            for initializer in model.graph.initializer
            if initializer.name == conv.input[1]
        ),
        None,
    )
    if weight is None:
        return False
    weights = np.asarray(numpy_helper.to_array(weight))
    if weights.ndim < 1 or weights.shape[0] != len(scale):
        return False
    if not np.issubdtype(weights.dtype, np.floating):
        return False

    if len(conv.input) > 2 and conv.input[2]:
        bias = next(
            (
                initializer
                for initializer in model.graph.initializer
                if initializer.name == conv.input[2]
            ),
            None,
        )
        if bias is None:
            return False
        bias_values = np.asarray(numpy_helper.to_array(bias))
        if bias_values.ndim != 1 or len(bias_values) != len(scale):
            return False
        if not np.issubdtype(bias_values.dtype, np.floating):
            return False
    else:
        bias_values = np.zeros(len(scale), dtype=weights.dtype)

    new_weights = weights * scale.reshape((len(scale),) + (1,) * (weights.ndim - 1))
    weight_name = _new_initializer(
        model,
        allocator,
        np.asarray(new_weights, dtype=weights.dtype),
        "algebraic_conv_weight",
        introduced,
    )
    conv.input[1] = weight_name
    new_bias = bias_values * scale + offset
    bias_name = _new_initializer(
        model,
        allocator,
        np.asarray(new_bias, dtype=bias_values.dtype),
        "algebraic_conv_bias",
        introduced,
    )
    if len(conv.input) > 2:
        conv.input[2] = bias_name
    else:
        conv.input.append(bias_name)
    return True


def _fold_channel_affine(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
    introduced: set[str],
) -> None:
    """Fold direct or static channel-routed affine branches after Conv."""
    from onnx import numpy_helper

    index = _GraphIndex.build(model)
    for original_conv in list(model.graph.node):
        if (
            original_conv.op_type != "Conv"
            or len(original_conv.output) != 1
            or not original_conv.output[0]
        ):
            continue
        conv_output = original_conv.output[0]
        conv = index.producers.get(conv_output)
        if conv is None or conv.op_type != "Conv":
            continue
        conv_shape = _static_shape(index, conv_output)
        if conv_shape is None or len(conv_shape) < 2:
            continue
        channels = conv_shape[1]
        weight_initializer = index.initializers.get(conv.input[1]) if len(conv.input) > 1 else None
        if weight_initializer is None:
            continue
        weight_dtype = numpy_helper.to_array(weight_initializer).dtype
        if channels <= 0:
            continue

        route_name = conv_output
        route_shape = conv_shape
        route_source_node = conv
        route_source_output_index = 0
        direct_consumers = index.consumers.get(route_name, [])
        while len(direct_consumers) == 1:
            view = direct_consumers[0]
            view_output = _channel_preserving_view_output(
                index,
                view,
                route_name,
                channels,
            )
            if view_output is None or route_name in index.graph_outputs:
                break
            route_name = view_output
            route_shape = _static_shape(index, route_name)
            if route_shape is None:
                break
            route_source_node = view
            route_source_output_index = 0
            direct_consumers = index.consumers.get(route_name, [])

        candidates: list[_AffineCandidate] = []
        if route_name not in index.graph_outputs and len(direct_consumers) == 1:
            direct = _collect_affine_chain(
                index,
                direct_consumers[0],
                route_name,
                route_shape,
                channels,
                0,
                channels,
            )
            if direct is not None:
                direct.source_node = route_source_node
                direct.source_output_index = route_source_output_index
                candidates.append(direct)

        if not candidates and route_name not in index.graph_outputs and len(direct_consumers) == 1:
            router = direct_consumers[0]
            boundaries: list[tuple[int, int]] | None = None
            if router.op_type == "Split":
                split_info = _split_boundaries(index, router, route_name)
                if split_info is not None and split_info[0] == 1:
                    boundaries = split_info[1]
            elif router.op_type == "Slice":
                boundary = _slice_channel_boundary(index, router, route_name, 1)
                if boundary is not None:
                    boundaries = [boundary]

            if boundaries is not None and len(boundaries) == len(router.output):
                for output_index, (start, end) in enumerate(boundaries):
                    candidates.extend(
                        _collect_routed_affine_candidates(
                            index,
                            router,
                            output_index,
                            start,
                            end,
                        )
                    )

        if not candidates:
            continue

        # Overlapping routed slices are ambiguous even if the numerical affine
        # values happen to match.  Never silently choose one branch's meaning.
        if any(
            left.start < right.end and right.start < left.end
            for position, left in enumerate(candidates)
            for right in candidates[position + 1 :]
        ):
            continue
        if any(
            output in index.graph_outputs
            for candidate in candidates
            for node in candidate.nodes[:-1]
            for output in node.output
            if output
        ):
            continue

        calculation_dtype = np.result_type(weight_dtype, np.float32)
        scale = np.ones(channels, dtype=calculation_dtype)
        offset = np.zeros(channels, dtype=calculation_dtype)
        for candidate in candidates:
            scale[candidate.start : candidate.end] = candidate.scale
            offset[candidate.start : candidate.end] = candidate.offset
        if not _copy_conv_parameters(model, allocator, introduced, conv, scale, offset):
            continue

        removed = {id(node) for candidate in candidates for node in candidate.nodes}
        for candidate in candidates:
            candidate.source_node.output[candidate.source_output_index] = candidate.final_output
        _remove_nodes(model, removed)
        index = _GraphIndex.build(model)


def _scale_broadcast_static_tensor(
    values: np.ndarray,
    output_shape: tuple[int, ...],
    gamma: np.ndarray,
) -> np.ndarray | None:
    """Scale a static Add operand by channel while retaining valid broadcasting."""
    if not np.issubdtype(values.dtype, np.floating) or values.ndim > len(output_shape):
        return None
    padded_shape = (1,) * (len(output_shape) - values.ndim) + tuple(values.shape)
    for axis, dimension in enumerate(padded_shape):
        if dimension not in (1, output_shape[axis]):
            return None
    if padded_shape[1] not in (1, output_shape[1]):
        return None
    expanded_shape = list(padded_shape)
    expanded_shape[1] = output_shape[1]
    try:
        broadcast_values = np.broadcast_to(values.reshape(padded_shape), tuple(expanded_shape))
    except ValueError:
        return None
    factors = gamma.reshape((1, len(gamma)) + (1,) * (len(output_shape) - 2))
    return np.asarray(broadcast_values * factors, dtype=np.result_type(values.dtype, gamma.dtype))


def _copy_conv_parameters_for_bn(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
    introduced: set[str],
    conv: onnx.NodeProto,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
) -> bool:
    """Apply the affine part of inference BatchNormalization to Conv."""
    from onnx import numpy_helper

    if len(conv.input) < 2:
        return False
    weight = next((i for i in model.graph.initializer if i.name == conv.input[1]), None)
    if weight is None:
        return False
    weights = np.asarray(numpy_helper.to_array(weight))
    if weights.ndim < 1 or weights.shape[0] != len(gamma):
        return False
    if not np.issubdtype(weights.dtype, np.floating):
        return False

    if len(conv.input) > 2 and conv.input[2]:
        bias = next((i for i in model.graph.initializer if i.name == conv.input[2]), None)
        if bias is None:
            return False
        bias_values = np.asarray(numpy_helper.to_array(bias))
        if bias_values.ndim != 1 or len(bias_values) != len(gamma):
            return False
    else:
        bias_values = np.zeros(len(gamma), dtype=weights.dtype)
    if not np.issubdtype(bias_values.dtype, np.floating):
        return False
    new_weights = weights * gamma.reshape((len(gamma),) + (1,) * (weights.ndim - 1))
    conv.input[1] = _new_initializer(
        model,
        allocator,
        np.asarray(new_weights, dtype=weights.dtype),
        "algebraic_bn_conv_weight",
        introduced,
    )
    new_bias = bias_values * gamma + beta - gamma * mean
    bias_name = _new_initializer(
        model,
        allocator,
        np.asarray(new_bias, dtype=bias_values.dtype),
        "algebraic_bn_conv_bias",
        introduced,
    )
    if len(conv.input) > 2:
        conv.input[2] = bias_name
    else:
        conv.input.append(bias_name)
    return True


def _fold_conv_add_bn(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
    introduced: set[str],
) -> None:
    """Fold the inference-only ``Add(Conv(...), static) -> BN`` topology."""
    from onnx import numpy_helper

    index = _GraphIndex.build(model)
    for original_batch_norm in list(model.graph.node):
        if (
            original_batch_norm.op_type != "BatchNormalization"
            or len(original_batch_norm.output) != 1
            or not original_batch_norm.output[0]
        ):
            continue
        batch_norm = index.producers.get(original_batch_norm.output[0])
        if batch_norm is None or batch_norm.op_type != "BatchNormalization":
            continue
        # Newer schemas expose training_mode as an attribute.  A nonzero value
        # is training behavior and cannot be replaced by this inference fold.
        if len(batch_norm.input) != 5:
            continue
        try:
            if int(_attribute(batch_norm, "training_mode", 0)) != 0:
                continue
        except (TypeError, ValueError):
            continue
        add_name = batch_norm.input[0]
        add = index.producers.get(add_name)
        if add is None or add.op_type != "Add" or len(add.input) != 2 or not _node_output(add):
            continue
        if add.output[0] in index.graph_outputs:
            continue
        if len(index.consumers.get(add_name, [])) != 1:
            continue
        conv_names = [
            name
            for name in add.input
            if name
            and index.producers.get(name) is not None
            and index.producers[name].op_type == "Conv"
        ]
        if len(conv_names) != 1:
            continue
        conv_output = conv_names[0]
        conv = index.producers[conv_output]
        if (
            conv_output in index.graph_outputs
            or len(index.consumers.get(conv_output, [])) != 1
            or index.consumers[conv_output][0] is not add
        ):
            continue
        static_names = [name for name in add.input if name != conv_output and name]
        if len(static_names) != 1:
            continue
        static_values = _constant_array(index, static_names[0])
        if static_values is None:
            continue
        conv_shape = _static_shape(index, conv_output)
        if conv_shape is None or len(conv_shape) < 2:
            continue
        channels = conv_shape[1]
        weight_initializer = index.initializers.get(conv.input[1]) if len(conv.input) > 1 else None
        if weight_initializer is None:
            continue
        weight_dtype = numpy_helper.to_array(weight_initializer).dtype

        params = [_constant_array(index, name) for name in batch_norm.input[1:]]
        if any(value is None for value in params):
            continue
        scale, beta, mean, variance = (np.asarray(value) for value in params if value is not None)
        if any(
            value.ndim != 1 or len(value) != channels for value in (scale, beta, mean, variance)
        ):
            continue
        if not all(
            np.issubdtype(value.dtype, np.floating) for value in (scale, beta, mean, variance)
        ):
            continue
        if any(value.dtype != weight_dtype for value in (scale, beta, mean, variance)):
            continue
        if static_values.dtype != weight_dtype:
            continue
        epsilon = _attribute(batch_norm, "epsilon", 1e-5)
        try:
            epsilon = float(epsilon)
        except (TypeError, ValueError):
            continue
        denominator = variance + epsilon
        if epsilon < 0 or np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
            continue
        gamma = scale / np.sqrt(denominator)
        if not np.all(np.isfinite(gamma)):
            continue
        scaled_static = _scale_broadcast_static_tensor(static_values, conv_shape, gamma)
        if scaled_static is None:
            continue
        if not _copy_conv_parameters_for_bn(model, allocator, introduced, conv, gamma, beta, mean):
            continue

        static_name = _new_initializer(
            model,
            allocator,
            scaled_static,
            "algebraic_bn_add_tensor",
            introduced,
        )
        for input_index, input_name in enumerate(add.input):
            if input_name == static_names[0]:
                add.input[input_index] = static_name
        add.output[0] = batch_norm.output[0]
        _remove_nodes(model, {id(batch_norm)})
        index = _GraphIndex.build(model)


def _rewrite_static_splits(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
    introduced: set[str],
    introduced_nodes: set[str],
) -> None:
    """Replace statically bounded Split nodes with input-form Slice nodes."""
    from onnx import helper

    index = _GraphIndex.build(model)
    opset = next(
        (int(opset.version) for opset in model.opset_import if opset.domain in ("", "ai.onnx")),
        0,
    )
    if opset and opset < 10:
        return
    replacements: dict[int, list[onnx.NodeProto]] = {}
    for split in list(model.graph.node):
        if split.op_type != "Split" or len(split.input) < 1 or not split.input[0]:
            continue
        if any(not output for output in split.output):
            continue
        info = _split_boundaries(index, split, split.input[0])
        if info is None:
            continue
        axis, boundaries = info
        data_name = split.input[0]
        replacement: list[onnx.NodeProto] = []
        for output_index, (start, end) in enumerate(boundaries):
            starts_name = _new_initializer(
                model,
                allocator,
                np.asarray([start], dtype=np.int64),
                "algebraic_slice_starts",
                introduced,
            )
            ends_name = _new_initializer(
                model,
                allocator,
                np.asarray([end], dtype=np.int64),
                "algebraic_slice_ends",
                introduced,
            )
            axes_name = _new_initializer(
                model,
                allocator,
                np.asarray([axis], dtype=np.int64),
                "algebraic_slice_axes",
                introduced,
            )
            steps_name = _new_initializer(
                model,
                allocator,
                np.asarray([1], dtype=np.int64),
                "algebraic_slice_steps",
                introduced,
            )
            replacement_node = helper.make_node(
                "Slice",
                [data_name, starts_name, ends_name, axes_name, steps_name],
                [split.output[output_index]],
                name=allocator.new("algebraic_split_slice"),
            )
            replacement.append(replacement_node)
            introduced_nodes.add(replacement_node.name)
        replacements[id(split)] = replacement

    if not replacements:
        return
    rewritten: list[onnx.NodeProto] = []
    for node in model.graph.node:
        rewritten.extend(replacements.get(id(node), [node]))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)


class AlgebraicRewritePipe(BasePipe[AlgebraicRewritePipeConfig]):
    """Apply independently enabled exact algebraic graph rewrites."""

    name: ClassVar[str] = "algebraic_rewrite"
    capabilities: ClassVar[dict[str, Any]] = ALGEBRAIC_CAPABILITIES

    @classmethod
    def build_config(cls, **kwargs: Any) -> AlgebraicRewritePipeConfig:
        """Build the three independent algebraic capability switches."""
        return AlgebraicRewritePipeConfig(
            static_split_to_slice=kwargs.get("static_split_to_slice", False),
            conv_channel_affine_folding=kwargs.get("conv_channel_affine_folding", False),
            conv_add_batch_normalization_folding=kwargs.get(
                "conv_add_batch_normalization_folding", False
            ),
        )

    @classmethod
    def should_process(cls, config: AlgebraicRewritePipeConfig) -> bool:
        """Return whether at least one algebraic rewrite is enabled."""
        return any(
            (
                config.static_split_to_slice,
                config.conv_channel_affine_folding,
                config.conv_add_batch_normalization_folding,
            )
        )

    def process(
        self,
        model: onnx.ModelProto,
        config: AlgebraicRewritePipeConfig,
    ) -> onnx.ModelProto:
        """Apply enabled rewrites to a copied model and return the copy."""
        if not self.should_process(config):
            return model

        import onnx

        result = onnx.ModelProto()
        result.CopyFrom(model)
        allocator = _NameAllocator(result)
        introduced: set[str] = set()
        introduced_nodes: set[str] = set()

        # Affine folding consumes channel routing before Split-to-Slice expands
        # one router into several nodes. BN remains independently observable.
        if config.conv_channel_affine_folding:
            _fold_channel_affine(result, allocator, introduced)
        if config.conv_add_batch_normalization_folding:
            _fold_conv_add_bn(result, allocator, introduced)
        if config.static_split_to_slice:
            _rewrite_static_splits(result, allocator, introduced, introduced_nodes)
        _prune_introduced_nodes(result, introduced_nodes)
        _prune_dead_constant_nodes(result)
        _prune_unused_initializers(result)
        return result
