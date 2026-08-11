# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Conservative, opt-in algebraic ONNX graph rewrites."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any, ClassVar, cast

import numpy as np
import onnx

from ..capabilities import algebraic
from .base import BasePipe, PipeConfig, caps_dict


ALGEBRAIC_CAPABILITIES: dict[str, Any] = caps_dict(
    algebraic.STATIC_SPLIT_TO_SLICE,
    algebraic.CONV_CHANNEL_AFFINE_FOLDING,
    algebraic.EXP_POSITIVE_SCALE_FOLDING,
)
MAX_AFFINE_ROUTE_DEPTH = 64


@dataclass
class AlgebraicRewritePipeConfig(PipeConfig):
    """Configuration for exact algebraic rewrites."""

    static_split_to_slice: bool = False
    sibling_slice_to_split: bool = False
    conv_channel_affine_folding: bool = False
    exp_positive_scale_folding: bool = False


@dataclass
class _GraphIndex:
    """Graph metadata required to identify statically bounded Split nodes."""

    producers: dict[str, onnx.NodeProto]
    definition_collisions: set[str]
    has_cycle: bool
    consumers: dict[str, list[onnx.NodeProto]]
    initializers: dict[str, onnx.TensorProto]
    shapes: dict[str, tuple[int | None, ...]]
    graph_inputs: set[str]
    graph_outputs: set[str]

    @classmethod
    def build(cls, model: onnx.ModelProto) -> _GraphIndex:
        graph = model.graph
        producers: dict[str, onnx.NodeProto] = {}
        definition_collisions: set[str] = set()
        consumers: dict[str, list[onnx.NodeProto]] = {}
        graph_input_names = [value.name for value in graph.input if value.name]
        initializer_names = [
            initializer.name for initializer in graph.initializer if initializer.name
        ]
        for names in (graph_input_names, initializer_names):
            seen: set[str] = set()
            for name in names:
                if name in seen:
                    definition_collisions.add(name)
                seen.add(name)
        protected_definitions = set(graph_input_names) | set(initializer_names)
        for node in graph.node:
            for output in node.output:
                if output:
                    if output in protected_definitions or output in producers:
                        definition_collisions.add(output)
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

        node_indexes = {id(node): index for index, node in enumerate(graph.node)}
        successors: list[set[int]] = [set() for _ in graph.node]
        indegrees = [0] * len(graph.node)
        for consumer_index, node in enumerate(graph.node):
            predecessor_indexes = {
                node_indexes[id(producer)]
                for input_name in node.input
                if input_name and (producer := producers.get(input_name)) is not None
            }
            indegrees[consumer_index] = len(predecessor_indexes)
            for predecessor_index in predecessor_indexes:
                successors[predecessor_index].add(consumer_index)
        ready = [index for index, indegree in enumerate(indegrees) if indegree == 0]
        visited_count = 0
        while ready:
            node_index = ready.pop()
            visited_count += 1
            for successor_index in successors[node_index]:
                indegrees[successor_index] -= 1
                if indegrees[successor_index] == 0:
                    ready.append(successor_index)

        initializers = {initializer.name: initializer for initializer in graph.initializer}
        shapes: dict[str, tuple[int | None, ...]] = {}
        for value_info in (*graph.input, *graph.value_info, *graph.output):
            shape = _value_info_shape(value_info)
            if shape is not None:
                shapes[value_info.name] = shape
        for name, initializer in initializers.items():
            shapes.setdefault(name, tuple(int(dim) for dim in initializer.dims))

        return cls(
            producers=producers,
            definition_collisions=definition_collisions,
            has_cycle=visited_count != len(graph.node),
            consumers=consumers,
            initializers=initializers,
            shapes=shapes,
            graph_inputs={value.name for value in graph.input if value.name},
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


@dataclass
class _ExpScaleCandidate:
    """A positive post-Exp scale that can be merged into an existing bias."""

    add: onnx.NodeProto
    bias_input_index: int
    output_node: onnx.NodeProto
    mul: onnx.NodeProto
    combined_bias: np.ndarray


@dataclass
class _ExpScaleInsertCandidate:
    """A positive post-Exp scale that requires a new input Add."""

    exp: onnx.NodeProto
    output_node: onnx.NodeProto
    mul: onnx.NodeProto
    log_scale: np.ndarray


@dataclass
class _StaticSliceCandidate:
    """A one-axis static Slice that can participate in a sibling Split."""

    node: onnx.NodeProto
    input_name: str
    output_name: str
    axis: int
    start: int
    end: int


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
    dimensions = [
        int(dimension.dim_value) if dimension.HasField("dim_value") else None
        for dimension in tensor_type.shape.dim
    ]
    return tuple(dimensions)


def _attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


def _is_standard_onnx_node(node: onnx.NodeProto) -> bool:
    return node.domain in ("", "ai.onnx")


def _constant_array(index: _GraphIndex, name: str) -> np.ndarray | None:
    """Read an initializer or a regular ONNX Constant value."""
    if not name or name in index.graph_inputs:
        return None
    initializer = index.initializers.get(name)
    if initializer is not None:
        if initializer.data_location == onnx.TensorProto.EXTERNAL and not initializer.raw_data:
            return None
        try:
            return np.asarray(onnx.numpy_helper.to_array(initializer))
        except (TypeError, ValueError, RuntimeError, onnx.checker.ValidationError):
            return None

    producer = index.producers.get(name)
    if producer is None or not _is_standard_onnx_node(producer) or producer.op_type != "Constant":
        return None
    value = _attribute(producer, "value")
    if value is not None:
        try:
            return np.asarray(onnx.numpy_helper.to_array(value))
        except (TypeError, ValueError):
            return None
    for attribute_name, dtype in (
        ("value_float", np.float32),
        ("value_floats", np.float32),
        ("value_int", np.int64),
        ("value_ints", np.int64),
    ):
        attribute_value = _attribute(producer, attribute_name)
        if attribute_value is not None:
            return np.asarray(attribute_value, dtype=dtype)
    return None


def _initializer_array(index: _GraphIndex, name: str) -> np.ndarray | None:
    if not name or name not in index.initializers:
        return None
    return _constant_array(index, name)


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
    """Read legacy attributes and newer constant inputs."""
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
    return cast("tuple[int, ...]", shape)


def _new_initializer(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
    values: np.ndarray,
    prefix: str,
) -> str:
    name = allocator.new(prefix)
    model.graph.initializer.append(onnx.numpy_helper.from_array(np.asarray(values), name))
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
    used = _referenced_tensor_names(model.graph)
    remaining = [initializer for initializer in model.graph.initializer if initializer.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(remaining)


def _prune_stale_value_info(model: onnx.ModelProto) -> None:
    """Remove value metadata for tensors no longer defined by the graph."""
    graph = model.graph
    defined = {value.name for value in graph.input if value.name}
    defined.update(initializer.name for initializer in graph.initializer)
    defined.update(output for node in graph.node for output in node.output if output)
    remaining = [value for value in graph.value_info if value.name in defined]
    del graph.value_info[:]
    graph.value_info.extend(remaining)


def _prune_generated_slices(model: onnx.ModelProto, introduced: set[str]) -> None:
    """Remove generated Slice nodes whose outputs are entirely dead."""
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
    """Return a static Split axis and output boundaries."""
    input_shape = index.shapes.get(input_name)
    outputs = list(node.output)
    if (
        not _is_standard_onnx_node(node)
        or node.op_type != "Split"
        or input_shape is None
        or not outputs
        or any(not output for output in outputs)
        or len(set(outputs)) != len(outputs)
        or any(output in node.input for output in outputs)
    ):
        return None

    axis_value = _attribute(node, "axis", 0)
    if not isinstance(axis_value, (int, np.integer)):
        return None
    axis = int(axis_value)
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
        split_values = [axis_size // len(node.output)] * len(node.output)
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


def _normalize_slice_bound(value: int, axis_size: int, *, is_end: bool) -> int:
    if value < 0:
        value += axis_size
    if is_end and value > axis_size:
        return axis_size
    return max(0, min(value, axis_size))


def _static_slice_candidate(
    index: _GraphIndex,
    node: onnx.NodeProto,
) -> _StaticSliceCandidate | None:
    if (
        not _is_standard_onnx_node(node)
        or node.op_type != "Slice"
        or len(node.input) < 3
        or not node.input[0]
    ):
        return None
    output_name = _node_output(node)
    input_shape = _static_shape(index, node.input[0])
    if output_name is None or input_shape is None:
        return None
    starts = _constant_ints(index, node.input[1])
    ends = _constant_ints(index, node.input[2])
    if starts is None or ends is None or len(starts) != 1 or len(ends) != 1:
        return None
    if len(node.input) > 3 and node.input[3]:
        axes = _constant_ints(index, node.input[3])
        if axes is None:
            return None
    else:
        axes = [0]
    if len(node.input) > 4 and node.input[4]:
        steps = _constant_ints(index, node.input[4])
        if steps is None:
            return None
    else:
        steps = [1]
    if len(axes) != 1 or len(steps) != 1 or steps[0] != 1:
        return None
    axis = axes[0]
    if axis < -len(input_shape) or axis >= len(input_shape):
        return None
    axis %= len(input_shape)
    axis_size = input_shape[axis]
    if axis_size <= 0:
        return None
    start = _normalize_slice_bound(starts[0], axis_size, is_end=False)
    end = _normalize_slice_bound(ends[0], axis_size, is_end=True)
    if end <= start:
        return None
    output_shape = _static_shape(index, output_name)
    expected_shape = list(input_shape)
    expected_shape[axis] = end - start
    if output_shape is not None and output_shape != tuple(expected_shape):
        return None
    return _StaticSliceCandidate(
        node=node,
        input_name=node.input[0],
        output_name=output_name,
        axis=axis,
        start=start,
        end=end,
    )


def _sibling_slice_split_groups(
    model: onnx.ModelProto,
    index: _GraphIndex,
) -> list[list[_StaticSliceCandidate]]:
    grouped: dict[tuple[str, int], list[_StaticSliceCandidate]] = {}
    for node in model.graph.node:
        candidate = _static_slice_candidate(index, node)
        if candidate is not None:
            grouped.setdefault((candidate.input_name, candidate.axis), []).append(candidate)

    groups: list[list[_StaticSliceCandidate]] = []
    for (input_name, axis), candidates in grouped.items():
        input_shape = _static_shape(index, input_name)
        if input_shape is None or len(candidates) < 2:
            continue
        ordered = sorted(candidates, key=lambda candidate: candidate.start)
        if len({candidate.output_name for candidate in ordered}) != len(ordered):
            continue
        if ordered[0].start != 0 or ordered[-1].end != input_shape[axis]:
            continue
        if any(left.end != right.start for left, right in pairwise(ordered)):
            continue
        groups.append(ordered)
    return groups


def _fold_sibling_slices_to_split(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
) -> None:
    """Replace contiguous sibling Slice nodes with an equivalent Split."""
    opset = next(
        (int(opset.version) for opset in model.opset_import if opset.domain in ("", "ai.onnx")),
        0,
    )
    if opset < 13:
        return
    index = _GraphIndex.build(model)
    groups = _sibling_slice_split_groups(model, index)
    if not groups:
        return

    node_order = {id(node): position for position, node in enumerate(model.graph.node)}
    replacements: dict[int, onnx.NodeProto] = {}
    removed: set[int] = set()
    for group in groups:
        split_values = np.asarray(
            [candidate.end - candidate.start for candidate in group],
            dtype=np.int64,
        )
        split_name = _new_initializer(model, allocator, split_values, "algebraic_slice_splits")
        split = onnx.helper.make_node(
            "Split",
            [group[0].input_name, split_name],
            [candidate.output_name for candidate in group],
            name=allocator.new("algebraic_slice_split"),
            axis=group[0].axis,
        )
        first = min(group, key=lambda candidate: node_order[id(candidate.node)])
        replacements[id(first.node)] = split
        removed.update(
            id(candidate.node) for candidate in group if candidate.node is not first.node
        )

    rewritten: list[onnx.NodeProto] = []
    for node in model.graph.node:
        replacement = replacements.get(id(node))
        if replacement is not None:
            rewritten.append(replacement)
        elif id(node) not in removed:
            rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)


def _slice_channel_boundary(
    index: _GraphIndex,
    node: onnx.NodeProto,
    input_name: str,
    channel_axis: int,
) -> tuple[int, int] | None:
    """Read a Slice that selects a contiguous, full non-channel region."""
    if not _is_standard_onnx_node(node) or node.op_type != "Slice" or len(node.input) < 3:
        return None
    input_shape = _static_shape(index, input_name)
    if input_shape is None or channel_axis >= len(input_shape):
        return None
    starts = _constant_ints(index, node.input[1])
    ends = _constant_ints(index, node.input[2])
    if starts is None or ends is None:
        return None
    if len(node.input) > 3 and node.input[3]:
        axes = _constant_ints(index, node.input[3])
        if axes is None:
            return None
    else:
        axes = list(range(len(starts)))
    if len(node.input) > 4 and node.input[4]:
        steps = _constant_ints(index, node.input[4])
        if steps is None:
            return None
    else:
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
    return channel_boundary


def _channel_affine_values(
    values: np.ndarray,
    output_shape: tuple[int, ...],
    channels: int,
) -> np.ndarray | None:
    """Convert a scalar or a provably channel-only broadcast to ``[C]``."""
    if not np.issubdtype(values.dtype, np.floating):
        return None
    if not np.isfinite(values).all():
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
        or not _is_standard_onnx_node(node)
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
        target_shape = _constant_ints(index, node.input[1]) if len(node.input) >= 2 else None
        allowzero = _attribute(node, "allowzero", 0)
        if target_shape is None or allowzero not in (0, 1):
            return None
        if allowzero == 1 and 0 in target_shape:
            return None
    else:
        axes, conflict = _single_attribute_or_input_ints(index, node, "axes", 1)
        if conflict or axes is None:
            return None
    return output_name


def _order_preserving_view_output(
    index: _GraphIndex,
    node: onnx.NodeProto,
    input_name: str,
) -> str | None:
    """Return a static shape-only view output that preserves element order."""
    output_name = _node_output(node)
    if (
        output_name is None
        or not _is_standard_onnx_node(node)
        or not node.input
        or node.input[0] != input_name
        or node.op_type not in {"Reshape", "Squeeze", "Unsqueeze"}
    ):
        return None
    input_shape = _static_shape(index, input_name)
    output_shape = _static_shape(index, output_name)
    if (
        input_shape is None
        or output_shape is None
        or any(dimension <= 0 for dimension in (*input_shape, *output_shape))
        or np.prod(input_shape, dtype=np.int64) != np.prod(output_shape, dtype=np.int64)
    ):
        return None

    if node.op_type == "Reshape":
        target_shape = _constant_ints(index, node.input[1]) if len(node.input) == 2 else None
        allowzero = _attribute(node, "allowzero", 0)
        if target_shape is None or allowzero not in (0, 1):
            return None
        if allowzero == 1 and 0 in target_shape:
            return None
    else:
        axes, conflict = _single_attribute_or_input_ints(index, node, "axes", 1)
        if conflict or axes is None:
            return None
    return output_name


def _single_unobserved_consumer(
    index: _GraphIndex,
    tensor_name: str,
) -> onnx.NodeProto | None:
    if tensor_name in index.graph_outputs:
        return None
    consumers = index.consumers.get(tensor_name, [])
    return consumers[0] if len(consumers) == 1 else None


def _constant_input(
    index: _GraphIndex,
    node: onnx.NodeProto,
    data_name: str | None = None,
) -> tuple[int, np.ndarray] | None:
    if len(node.input) != 2 or any(not input_name for input_name in node.input):
        return None
    if data_name is not None and sum(name == data_name for name in node.input) != 1:
        return None
    constants = [
        (position, values)
        for position, input_name in enumerate(node.input)
        if (data_name is None or input_name != data_name)
        and (values := _constant_array(index, input_name)) is not None
    ]
    return constants[0] if len(constants) == 1 else None


def _post_exp_scale(
    index: _GraphIndex,
    exp: onnx.NodeProto,
    visited: set[str] | None = None,
) -> tuple[onnx.NodeProto, onnx.NodeProto, np.ndarray, tuple[int, ...]] | None:
    if not _is_standard_onnx_node(exp) or exp.op_type != "Exp" or len(exp.input) != 1:
        return None
    exp_output = _node_output(exp)
    if exp_output is None:
        return None
    route = set() if visited is None else set(visited)
    if exp_output in route or len(route) >= MAX_AFFINE_ROUTE_DEPTH:
        return None
    route.add(exp_output)
    current_name = exp_output
    current_node = exp
    next_node = _single_unobserved_consumer(index, current_name)
    while next_node is not None:
        view_output = _order_preserving_view_output(index, next_node, current_name)
        if view_output is None:
            break
        if view_output in route or len(route) >= MAX_AFFINE_ROUTE_DEPTH:
            return None
        route.add(view_output)
        current_name = view_output
        current_node = next_node
        next_node = _single_unobserved_consumer(index, current_name)

    if next_node is None or not _is_standard_onnx_node(next_node) or next_node.op_type != "Mul":
        return None
    scale_operand = _constant_input(index, next_node, current_name)
    mul_output = _node_output(next_node)
    output_shape = _static_shape(index, current_name)
    if scale_operand is None or mul_output is None or output_shape is None:
        return None
    scale = scale_operand[1]
    if (
        not np.issubdtype(scale.dtype, np.floating)
        or not np.isfinite(scale).all()
        or not np.all(scale > 0)
    ):
        return None
    try:
        np.broadcast_to(scale, output_shape)
    except ValueError:
        return None
    return current_node, next_node, scale, output_shape


def _exp_scale_candidate(
    index: _GraphIndex,
    add: onnx.NodeProto,
) -> _ExpScaleCandidate | None:
    if not _is_standard_onnx_node(add) or add.op_type != "Add":
        return None
    add_output = _node_output(add)
    bias_operand = _constant_input(index, add)
    if add_output is None or bias_operand is None:
        return None
    add_shape = _static_shape(index, add_output)
    if add_shape is None:
        return None

    current_name = add_output
    visited = {add_output}
    next_node = _single_unobserved_consumer(index, current_name)
    while next_node is not None:
        view_output = _order_preserving_view_output(index, next_node, current_name)
        if view_output is None:
            break
        if view_output in visited or len(visited) >= MAX_AFFINE_ROUTE_DEPTH:
            return None
        visited.add(view_output)
        current_name = view_output
        next_node = _single_unobserved_consumer(index, current_name)

    if (
        next_node is None
        or not _is_standard_onnx_node(next_node)
        or next_node.op_type != "Exp"
        or list(next_node.input) != [current_name]
    ):
        return None
    post_exp = _post_exp_scale(index, next_node, visited)
    if post_exp is None:
        return None
    output_node, mul, scale, output_shape = post_exp
    if output_shape != add_shape:
        return None

    bias = bias_operand[1]
    if (
        not np.issubdtype(bias.dtype, np.floating)
        or bias.dtype != scale.dtype
        or not np.isfinite(bias).all()
        or not np.isfinite(scale).all()
        or not np.all(scale > 0)
    ):
        return None
    try:
        np.broadcast_to(bias, add_shape)
        np.broadcast_to(scale, add_shape)
        combined_bias = np.asarray(bias + np.log(scale), dtype=bias.dtype)
        np.broadcast_to(combined_bias, add_shape)
    except ValueError:
        return None
    if not np.isfinite(combined_bias).all():
        return None
    return _ExpScaleCandidate(
        add=add,
        bias_input_index=bias_operand[0],
        output_node=output_node,
        mul=mul,
        combined_bias=combined_bias,
    )


def _exp_scale_insert_candidate(
    index: _GraphIndex,
    exp: onnx.NodeProto,
) -> _ExpScaleInsertCandidate | None:
    if not _is_standard_onnx_node(exp) or exp.op_type != "Exp" or len(exp.input) != 1:
        return None
    current_name = exp.input[0]
    visited = {current_name}
    producer = index.producers.get(current_name)
    while producer is not None and producer.op_type in {"Reshape", "Squeeze", "Unsqueeze"}:
        if (
            not producer.input
            or _order_preserving_view_output(index, producer, producer.input[0]) != current_name
        ):
            return None
        current_name = producer.input[0]
        if current_name in visited or len(visited) >= MAX_AFFINE_ROUTE_DEPTH:
            return None
        visited.add(current_name)
        producer = index.producers.get(current_name)

    post_exp = _post_exp_scale(index, exp)
    if post_exp is None:
        return None
    output_node, mul, scale, output_shape = post_exp
    input_shape = _static_shape(index, exp.input[0])
    if (
        input_shape is None
        or any(dimension <= 0 for dimension in (*input_shape, *output_shape))
        or np.prod(input_shape, dtype=np.int64) != np.prod(output_shape, dtype=np.int64)
    ):
        return None
    broadcast_scale = np.asarray(np.broadcast_to(scale, output_shape))
    log_scale = np.asarray(np.log(broadcast_scale).reshape(input_shape), dtype=scale.dtype)
    if not np.isfinite(log_scale).all():
        return None
    return _ExpScaleInsertCandidate(
        exp=exp,
        output_node=output_node,
        mul=mul,
        log_scale=log_scale,
    )


def _fold_exp_positive_scales(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
) -> None:
    """Fold eligible positive post-Exp constants into existing input biases."""
    index = _GraphIndex.build(model)
    for add in list(model.graph.node):
        candidate = _exp_scale_candidate(index, add)
        if candidate is None:
            continue
        combined_name = _new_initializer(
            model,
            allocator,
            candidate.combined_bias,
            "algebraic_exp_log_bias",
        )
        candidate.add.input[candidate.bias_input_index] = combined_name
        candidate.output_node.output[0] = candidate.mul.output[0]
        _remove_nodes(model, {id(candidate.mul)})
        index = _GraphIndex.build(model)

    for exp in list(model.graph.node):
        candidate = _exp_scale_insert_candidate(index, exp)
        if candidate is None:
            continue
        log_scale_name = _new_initializer(
            model,
            allocator,
            candidate.log_scale,
            "algebraic_exp_log_scale",
        )
        adjusted_name = allocator.new("algebraic_exp_adjusted")
        add = onnx.helper.make_node(
            "Add",
            [candidate.exp.input[0], log_scale_name],
            [adjusted_name],
            name=allocator.new("algebraic_exp_log_add"),
        )
        candidate.exp.input[0] = adjusted_name
        candidate.output_node.output[0] = candidate.mul.output[0]
        rewritten: list[onnx.NodeProto] = []
        for node in model.graph.node:
            if node is candidate.exp:
                rewritten.append(add)
            if node is not candidate.mul:
                rewritten.append(node)
        del model.graph.node[:]
        model.graph.node.extend(rewritten)
        index = _GraphIndex.build(model)


def _collect_affine_chain(
    index: _GraphIndex,
    first: onnx.NodeProto,
    source_name: str,
    output_shape: tuple[int, ...],
    start: int,
    end: int,
    calculation_dtype: np.dtype[Any],
    visited_routes: set[tuple[int, int, str]],
    depth: int,
) -> tuple[_AffineCandidate | None, bool]:
    """Collect a safe consecutive Mul/Add chain from one routed branch."""
    if not _is_standard_onnx_node(first) or first.op_type not in {"Mul", "Add"}:
        return None, True
    current = first
    current_input = source_name
    scale = np.ones(end - start, dtype=calculation_dtype)
    offset = np.zeros(end - start, dtype=calculation_dtype)
    matched: list[onnx.NodeProto] = []

    while current.op_type in {"Mul", "Add"}:
        if len(current.input) != 2 or current_input not in current.input:
            return None, True
        current_output = _node_output(current)
        if current_output is None:
            return None, False
        if not _visit_affine_route(current, 0, visited_routes, depth + 1):
            return None, False
        depth += 1
        values = _affine_operand(index, current, current_input, output_shape, end - start)
        if values is None:
            return None, True
        values = values.astype(calculation_dtype, copy=False)
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
        if not _is_standard_onnx_node(next_node) or next_node.op_type not in {"Mul", "Add"}:
            break
        current_input = current_output
        current = next_node

    final_output = _node_output(matched[-1]) if matched else None
    if final_output is None or (
        final_output not in index.graph_outputs and len(index.consumers.get(final_output, [])) == 0
    ):
        return None, True
    return (
        _AffineCandidate(
            source_node=first,
            source_output_index=0,
            final_output=final_output,
            nodes=matched,
            start=start,
            end=end,
            scale=scale,
            offset=offset,
        ),
        True,
    )


def _visit_affine_route(
    source_node: onnx.NodeProto,
    source_output_index: int,
    visited_routes: set[tuple[int, int, str]],
    depth: int,
) -> bool:
    """Record one unique, bounded source-slot and tensor route."""
    if (
        depth > MAX_AFFINE_ROUTE_DEPTH
        or source_output_index < 0
        or source_output_index >= len(source_node.output)
    ):
        return False
    source_name = source_node.output[source_output_index]
    if not source_name:
        return False
    source_slot = (id(source_node), source_output_index)
    if any(route[:2] == source_slot or route[2] == source_name for route in visited_routes):
        return False
    visited_routes.add((source_slot[0], source_slot[1], source_name))
    return True


def _collect_routed_affine_candidates(
    index: _GraphIndex,
    source_node: onnx.NodeProto,
    source_output_index: int,
    start: int,
    end: int,
    calculation_dtype: np.dtype[Any],
    visited_routes: set[tuple[int, int, str]],
    depth: int,
) -> list[_AffineCandidate] | None:
    """Collect affine leaves below safe views and disjoint channel slices."""
    if not _visit_affine_route(
        source_node,
        source_output_index,
        visited_routes,
        depth,
    ):
        return None
    source_name = source_node.output[source_output_index]
    if source_name in index.graph_outputs:
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
        if not _visit_affine_route(view, 0, visited_routes, depth + 1):
            return None
        depth += 1
        current_node = view
        current_output_index = 0
        current_name = view_output
        current_shape = _static_shape(index, current_name)
        if current_shape is None:
            return []
        consumers = index.consumers.get(current_name, [])

    if current_name in index.graph_outputs:
        return []
    if (
        len(consumers) == 1
        and _is_standard_onnx_node(consumers[0])
        and consumers[0].op_type in {"Mul", "Add"}
    ):
        candidate, route_is_valid = _collect_affine_chain(
            index,
            consumers[0],
            current_name,
            current_shape,
            start,
            end,
            calculation_dtype,
            visited_routes,
            depth,
        )
        if not route_is_valid:
            return None
        if candidate is None:
            return []
        candidate.source_node = current_node
        candidate.source_output_index = current_output_index
        return [candidate]

    if len(consumers) == 1 and consumers[0].op_type == "Split":
        nested_split = consumers[0]
        nested_info = _split_boundaries(index, nested_split, current_name)
        if nested_info is None or nested_info[0] != 1:
            return None
        boundaries = nested_info[1]
        if len(boundaries) != len(nested_split.output):
            return []
        candidates: list[_AffineCandidate] = []
        for output_index, (local_start, local_end) in enumerate(boundaries):
            nested_candidates = _collect_routed_affine_candidates(
                index,
                nested_split,
                output_index,
                start + local_start,
                start + local_end,
                calculation_dtype,
                visited_routes,
                depth + 1,
            )
            if nested_candidates is None:
                return None
            candidates.extend(nested_candidates)
        return candidates

    if not consumers or any(
        not _is_standard_onnx_node(node) or node.op_type != "Slice" for node in consumers
    ):
        return []
    routed_slices: list[tuple[onnx.NodeProto, int, int]] = []
    routed_outputs: list[str] = []
    for routed_slice in consumers:
        boundary = _slice_channel_boundary(index, routed_slice, current_name, 1)
        output_name = _node_output(routed_slice)
        if boundary is None or output_name is None or output_name == current_name:
            return None
        routed_slices.append((routed_slice, *boundary))
        routed_outputs.append(output_name)
    if len(set(routed_outputs)) != len(routed_outputs):
        return None
    if any(
        left_start < right_end and right_start < left_end
        for position, (_, left_start, left_end) in enumerate(routed_slices)
        for _, right_start, right_end in routed_slices[position + 1 :]
    ):
        return []

    candidates: list[_AffineCandidate] = []
    for routed_slice, local_start, local_end in routed_slices:
        routed_candidates = _collect_routed_affine_candidates(
            index,
            routed_slice,
            0,
            start + local_start,
            start + local_end,
            calculation_dtype,
            visited_routes,
            depth + 1,
        )
        if routed_candidates is None:
            return None
        candidates.extend(routed_candidates)
    return candidates


def _copy_conv_parameters(
    model: onnx.ModelProto,
    index: _GraphIndex,
    allocator: _NameAllocator,
    conv: onnx.NodeProto,
    scale: np.ndarray,
    offset: np.ndarray,
) -> bool:
    if len(conv.input) < 2 or conv.input[1] in index.graph_inputs:
        return False
    weights = _initializer_array(index, conv.input[1])
    if weights is None:
        return False
    if weights.ndim < 1 or weights.shape[0] != len(scale):
        return False
    if not np.issubdtype(weights.dtype, np.floating):
        return False

    if len(conv.input) > 2 and conv.input[2]:
        if conv.input[2] in index.graph_inputs:
            return False
        bias_values = _initializer_array(index, conv.input[2])
        if bias_values is None:
            return False
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
    )
    conv.input[1] = weight_name
    new_bias = bias_values * scale + offset
    bias_name = _new_initializer(
        model,
        allocator,
        np.asarray(new_bias, dtype=bias_values.dtype),
        "algebraic_conv_bias",
    )
    if len(conv.input) > 2:
        conv.input[2] = bias_name
    else:
        conv.input.append(bias_name)
    return True


def _fold_channel_affine(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
) -> None:
    """Fold direct or static channel-routed affine branches after Conv."""
    index = _GraphIndex.build(model)
    for original_conv in list(model.graph.node):
        if (
            not _is_standard_onnx_node(original_conv)
            or original_conv.op_type != "Conv"
            or len(original_conv.output) != 1
            or not original_conv.output[0]
        ):
            continue
        conv_output = original_conv.output[0]
        conv = index.producers.get(conv_output)
        if conv is None or not _is_standard_onnx_node(conv) or conv.op_type != "Conv":
            continue
        conv_shape = _static_shape(index, conv_output)
        if conv_shape is None or len(conv_shape) < 2:
            continue
        channels = conv_shape[1]
        weight_values = _initializer_array(index, conv.input[1]) if len(conv.input) > 1 else None
        if weight_values is None:
            continue
        weight_dtype = weight_values.dtype
        if channels <= 0:
            continue
        calculation_dtype = np.result_type(weight_dtype, np.float32)

        collected_candidates = _collect_routed_affine_candidates(
            index,
            conv,
            0,
            0,
            channels,
            calculation_dtype,
            set(),
            0,
        )
        if not collected_candidates:
            continue
        candidates = collected_candidates
        candidate_source_slots = [
            (id(candidate.source_node), candidate.source_output_index) for candidate in candidates
        ]
        candidate_source_tensors = [
            candidate.source_node.output[candidate.source_output_index] for candidate in candidates
        ]
        candidate_node_ids = [id(node) for candidate in candidates for node in candidate.nodes]
        if (
            len({id(candidate) for candidate in candidates}) != len(candidates)
            or len(set(candidate_source_slots)) != len(candidate_source_slots)
            or len(set(candidate_source_tensors)) != len(candidate_source_tensors)
            or len(set(candidate_node_ids)) != len(candidate_node_ids)
        ):
            continue
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

        scale = np.ones(channels, dtype=calculation_dtype)
        offset = np.zeros(channels, dtype=calculation_dtype)
        for candidate in candidates:
            scale[candidate.start : candidate.end] = candidate.scale
            offset[candidate.start : candidate.end] = candidate.offset
        if not _copy_conv_parameters(model, index, allocator, conv, scale, offset):
            continue

        removed = {id(node) for candidate in candidates for node in candidate.nodes}
        for candidate in candidates:
            candidate.source_node.output[candidate.source_output_index] = candidate.final_output
        _remove_nodes(model, removed)
        index = _GraphIndex.build(model)


def _rewrite_static_splits(
    model: onnx.ModelProto,
    allocator: _NameAllocator,
    introduced_nodes: set[str],
) -> None:
    """Replace statically bounded Split nodes with input-form Slice nodes."""
    index = _GraphIndex.build(model)
    opset = next(
        (int(opset.version) for opset in model.opset_import if opset.domain in ("", "ai.onnx")),
        0,
    )
    if opset and opset < 10:
        return

    replacements: dict[int, list[onnx.NodeProto]] = {}
    for split in list(model.graph.node):
        if (
            not _is_standard_onnx_node(split)
            or split.op_type != "Split"
            or len(split.input) < 1
            or not split.input[0]
        ):
            continue
        if any(not output for output in split.output):
            continue
        info = _split_boundaries(index, split, split.input[0])
        if info is None:
            continue
        axis, boundaries = info
        replacement: list[onnx.NodeProto] = []
        for output_index, (start, end) in enumerate(boundaries):
            starts_name = _new_initializer(
                model, allocator, np.asarray([start], dtype=np.int64), "algebraic_slice_starts"
            )
            ends_name = _new_initializer(
                model, allocator, np.asarray([end], dtype=np.int64), "algebraic_slice_ends"
            )
            axes_name = _new_initializer(
                model, allocator, np.asarray([axis], dtype=np.int64), "algebraic_slice_axes"
            )
            steps_name = _new_initializer(
                model, allocator, np.asarray([1], dtype=np.int64), "algebraic_slice_steps"
            )
            replacement_node = onnx.helper.make_node(
                "Slice",
                [split.input[0], starts_name, ends_name, axes_name, steps_name],
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
    """Replace statically bounded Split nodes with Slice nodes."""

    name: ClassVar[str] = "algebraic_rewrite"
    capabilities: ClassVar[dict[str, Any]] = ALGEBRAIC_CAPABILITIES

    @classmethod
    def build_config(cls, **kwargs: Any) -> AlgebraicRewritePipeConfig:
        """Build the enabled algebraic rewrite configuration."""
        return AlgebraicRewritePipeConfig(
            static_split_to_slice=kwargs.get("static_split_to_slice", False),
            sibling_slice_to_split=kwargs.get("gather_slice_to_split_fusion", False),
            conv_channel_affine_folding=kwargs.get("conv_channel_affine_folding", False),
            exp_positive_scale_folding=kwargs.get("exp_positive_scale_folding", False),
        )

    @classmethod
    def should_process(cls, config: AlgebraicRewritePipeConfig) -> bool:
        """Return whether any algebraic rewrite is enabled."""
        return (
            config.static_split_to_slice
            or config.sibling_slice_to_split
            or config.conv_channel_affine_folding
            or config.exp_positive_scale_folding
        )

    def process(
        self,
        model: onnx.ModelProto,
        config: AlgebraicRewritePipeConfig,
    ) -> onnx.ModelProto:
        """Rewrite eligible static Split nodes in a copy of the model."""
        if not self.should_process(config):
            return model

        result = onnx.ModelProto()
        result.CopyFrom(model)
        index = _GraphIndex.build(result)
        if index.definition_collisions or index.has_cycle:
            return result
        allocator = _NameAllocator(result)
        introduced_nodes: set[str] = set()
        if config.conv_channel_affine_folding:
            _fold_channel_affine(result, allocator)
        if config.exp_positive_scale_folding:
            _fold_exp_positive_scales(result, allocator)
        if config.sibling_slice_to_split:
            _fold_sibling_slices_to_split(result, allocator)
        if config.static_split_to_slice:
            _rewrite_static_splits(result, allocator, introduced_nodes)
        _prune_generated_slices(result, introduced_nodes)
        _prune_dead_constant_nodes(result)
        _prune_unused_initializers(result)
        _prune_stale_value_info(result)
        return result
