# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Conservative, opt-in algebraic ONNX graph rewrites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import onnx

from ..capabilities import algebraic
from .base import BasePipe, PipeConfig, caps_dict


ALGEBRAIC_CAPABILITIES: dict[str, Any] = caps_dict(algebraic.STATIC_SPLIT_TO_SLICE)


@dataclass
class AlgebraicRewritePipeConfig(PipeConfig):
    """Configuration for exact algebraic rewrites."""

    static_split_to_slice: bool = False


@dataclass
class _GraphIndex:
    """Graph metadata required to identify statically bounded Split nodes."""

    producers: dict[str, onnx.NodeProto]
    consumers: dict[str, list[onnx.NodeProto]]
    initializers: dict[str, onnx.TensorProto]
    shapes: dict[str, tuple[int | None, ...]]
    graph_outputs: set[str]

    @classmethod
    def build(cls, model: onnx.ModelProto) -> _GraphIndex:
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

        for initializer in initializers.values():
            onnx.numpy_helper.to_array(initializer)

        return cls(
            producers=producers,
            consumers=consumers,
            initializers=initializers,
            shapes=shapes,
            graph_outputs={output.name for output in graph.output if output.name},
        )


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


def _constant_array(index: _GraphIndex, name: str) -> np.ndarray | None:
    """Read an initializer or a regular ONNX Constant value."""
    if not name:
        return None
    initializer = index.initializers.get(name)
    if initializer is not None:
        return np.asarray(onnx.numpy_helper.to_array(initializer))

    producer = index.producers.get(name)
    if producer is None or producer.op_type != "Constant":
        return None
    value = _attribute(producer, "value")
    if value is not None:
        try:
            return np.asarray(onnx.numpy_helper.to_array(value))
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
        if split.op_type != "Split" or len(split.input) < 1 or not split.input[0]:
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
        """Build the static Split-to-Slice configuration."""
        return AlgebraicRewritePipeConfig(
            static_split_to_slice=kwargs.get("static_split_to_slice", False)
        )

    @classmethod
    def should_process(cls, config: AlgebraicRewritePipeConfig) -> bool:
        """Return whether static Split-to-Slice rewriting is enabled."""
        return config.static_split_to_slice

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
        introduced_nodes: set[str] = set()
        _rewrite_static_splits(result, _NameAllocator(result), introduced_nodes)
        _prune_generated_slices(result, introduced_nodes)
        _prune_dead_constant_nodes(result)
        _prune_unused_initializers(result)
        return result
