# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Surgery pipe for precise model modifications.

This pipe performs targeted graph transformations that are not part of
ONNX Runtime's standard optimization passes. Surgery operations run before or
after ORT optimization according to the graph evidence they require.

Use cases:
- Clamp extreme constant values to prevent quantization issues
- Prepare models for specific execution providers (QNN, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import onnx

from ..capabilities import surgery
from .base import BasePipe, PipeConfig, caps_dict


logger = logging.getLogger(__name__)


# =============================================================================
# MODULE-LEVEL CAPABILITIES
# =============================================================================

PRE_SURGERY_CAPABILITIES: dict[str, Any] = caps_dict(
    surgery.SIMPLIFY_L2_NORMALIZATION,
    surgery.GATHERND_TO_RESIZE,
    surgery.SCALED_MATMUL_TO_FUSED_MATMUL,
)

POST_SURGERY_CAPABILITIES: dict[str, Any] = caps_dict(
    surgery.CLAMP_CONSTANT_VALUES,
    surgery.REMOVE_ISNAN_IN_ATTENTION_MASK,
    surgery.UNTIE_CONSTANT_BATCHED_MATMUL,
    surgery.SILU_TO_QUICK_GELU,
)

SURGERY_CAPABILITIES: dict[str, Any] = {
    **PRE_SURGERY_CAPABILITIES,
    **POST_SURGERY_CAPABILITIES,
}


# =============================================================================
# SURGERYPIPECONFIG
# =============================================================================


@dataclass
class SurgeryPipeConfig(PipeConfig):
    """Configuration for surgery optimization pipe.

    Attributes:
        clamp_constant_values: Whether to clamp extreme float constants
        clamp_min: Minimum value for constant clamping (default: -1e3)
        clamp_max: Maximum value for constant clamping (default: 1e3)
        fix_nan_attention_mask: Replace -inf attention mask with finite value
            and remove Softmax->IsNaN->Where NaN guard patterns
        mask_value: Replacement value for -inf (default: -1e3)
        untie_constant_batched_matmul: Make a batched MatMul's constant operand
            runtime-valued so OpenVINO GPU can select a gemm implementation
        simplify_l2_normalization: Remove redundant Clip and Expand nodes from
            exact ReduceL2-based normalization patterns
        gathernd_to_resize: Replace exact 2x nearest-neighbor GatherND
            upsampling grids with standard ONNX Resize
        silu_to_quick_gelu: Replace x*Sigmoid(x) with the Microsoft QuickGelu op
        scaled_matmul_to_fused_matmul: Move a scalar Mul into Microsoft
            FusedMatMul's alpha attribute
        verbose: Enable verbose logging
    """

    clamp_constant_values: bool = False
    clamp_min: float = -1e3
    clamp_max: float = 1e3
    remove_isnan_in_attention_mask: bool = False
    untie_constant_batched_matmul: bool = False
    simplify_l2_normalization: bool = False
    gathernd_to_resize: bool = False
    silu_to_quick_gelu: bool = False
    scaled_matmul_to_fused_matmul: bool = False
    verbose: bool = False


@dataclass
class _GraphIndex:
    """Producer, consumer, constant, type, and output metadata for one graph."""

    producers: dict[str, onnx.NodeProto]
    consumers: dict[str, list[onnx.NodeProto]]
    initializers: dict[str, onnx.TensorProto]
    element_types: dict[str, int]
    shapes: dict[str, tuple[int | str | None, ...]]
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
            consumed_names = {name for name in node.input if name}
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    consumed_names.update(_captured_tensor_names(attribute.g))
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested_graph in attribute.graphs:
                        consumed_names.update(_captured_tensor_names(nested_graph))
            for name in consumed_names:
                consumers.setdefault(name, []).append(node)

        initializers = {initializer.name: initializer for initializer in graph.initializer}
        element_types: dict[str, int] = {
            initializer.name: int(initializer.data_type) for initializer in graph.initializer
        }
        for value in (*graph.input, *graph.value_info, *graph.output):
            if value.type.HasField("tensor_type"):
                element_types[value.name] = int(value.type.tensor_type.elem_type)
        shapes: dict[str, tuple[int | str | None, ...]] = {}
        for value in (*graph.input, *graph.value_info, *graph.output):
            if not value.type.HasField("tensor_type"):
                continue
            tensor_type = value.type.tensor_type
            if not tensor_type.HasField("shape"):
                continue
            dimensions: list[int | str | None] = []
            for dimension in tensor_type.shape.dim:
                if dimension.HasField("dim_value"):
                    dimensions.append(int(dimension.dim_value))
                elif dimension.HasField("dim_param") and dimension.dim_param:
                    dimensions.append(dimension.dim_param)
                else:
                    dimensions.append(None)
            shapes[value.name] = tuple(dimensions)
        for initializer in graph.initializer:
            shapes.setdefault(initializer.name, tuple(int(dim) for dim in initializer.dims))

        return cls(
            producers=producers,
            consumers=consumers,
            initializers=initializers,
            element_types=element_types,
            shapes=shapes,
            graph_outputs={output.name for output in graph.output if output.name},
        )


class _NameAllocator:
    """Allocate graph-wide unique tensor and node names."""

    def __init__(self, model: onnx.ModelProto) -> None:
        graph = model.graph
        self._used = {
            name
            for name in (
                [initializer.name for initializer in graph.initializer]
                + [value.name for value in (*graph.input, *graph.value_info, *graph.output)]
                + [node.name for node in graph.node]
                + [name for node in graph.node for name in (*node.input, *node.output)]
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


def _captured_tensor_names(graph: onnx.GraphProto) -> set[str]:
    """Return outer-scope tensor names referenced by a nested graph."""
    locally_defined = {value.name for value in graph.input if value.name}
    locally_defined.update(initializer.name for initializer in graph.initializer)
    locally_defined.update(output for node in graph.node for output in node.output if output)
    referenced = {name for node in graph.node for name in node.input if name}
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                referenced.update(_captured_tensor_names(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested_graph in attribute.graphs:
                    referenced.update(_captured_tensor_names(nested_graph))
    return referenced - locally_defined


def _referenced_tensor_names(graph: onnx.GraphProto) -> set[str]:
    """Collect tensor names referenced by a graph and its nested graphs."""
    referenced = {value.name for value in (*graph.input, *graph.output) if value.name}
    for node in graph.node:
        referenced.update(name for name in node.input if name)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                referenced.update(_referenced_tensor_names(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested_graph in attribute.graphs:
                    referenced.update(_referenced_tensor_names(nested_graph))
    return referenced


def _attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


def _tensor_array(tensor: onnx.TensorProto) -> np.ndarray | None:
    try:
        values = np.asarray(onnx.numpy_helper.to_array(tensor))
    except (TypeError, ValueError):
        return None
    if tensor.data_type == onnx.TensorProto.BFLOAT16 and np.issubdtype(values.dtype, np.integer):
        values = np.left_shift(values.astype(np.uint32), 16).view(np.float32)
    return values


def _constant_array(index: _GraphIndex, name: str) -> np.ndarray | None:
    """Read an initializer or tensor-valued Constant as a NumPy array."""
    initializer = index.initializers.get(name)
    if initializer is not None:
        return _tensor_array(initializer)

    producer = index.producers.get(name)
    if producer is None or producer.op_type != "Constant" or not _is_default_domain(producer):
        return None
    value = _attribute(producer, "value")
    if value is not None:
        return _tensor_array(value)
    for attribute_name in ("value_float", "value_floats", "value_int", "value_ints"):
        attribute_value = _attribute(producer, attribute_name)
        if attribute_value is not None:
            return np.asarray(attribute_value)
    return None


def _constant_scalar(index: _GraphIndex, name: str) -> float | int | None:
    values = _constant_array(index, name)
    if values is None or values.size != 1:
        return None
    return values.reshape(-1)[0].item()


def _constant_element_type(index: _GraphIndex, name: str) -> int | None:
    initializer = index.initializers.get(name)
    if initializer is not None:
        return int(initializer.data_type)
    producer = index.producers.get(name)
    if producer is None or producer.op_type != "Constant" or not _is_default_domain(producer):
        return index.element_types.get(name)
    value = _attribute(producer, "value")
    if isinstance(value, onnx.TensorProto):
        return int(value.data_type)
    if (
        _attribute(producer, "value_float") is not None
        or _attribute(producer, "value_floats") is not None
    ):
        return int(onnx.TensorProto.FLOAT)
    if (
        _attribute(producer, "value_int") is not None
        or _attribute(producer, "value_ints") is not None
    ):
        return int(onnx.TensorProto.INT64)
    return index.element_types.get(name)


def _is_default_domain(node: onnx.NodeProto) -> bool:
    return node.domain in ("", "ai.onnx")


def _node_output(node: onnx.NodeProto) -> str | None:
    return node.output[0] if len(node.output) == 1 and node.output[0] else None


def _only_consumer(
    index: _GraphIndex,
    tensor_name: str,
    op_type: str,
) -> onnx.NodeProto | None:
    consumers = index.consumers.get(tensor_name, [])
    if len(consumers) != 1:
        return None
    consumer = consumers[0]
    if consumer.op_type != op_type or not _is_default_domain(consumer):
        return None
    return consumer


def _constant_axes(index: _GraphIndex, node: onnx.NodeProto) -> list[int] | None:
    axes = _attribute(node, "axes")
    if axes is None and len(node.input) > 1 and node.input[1]:
        values = _constant_array(index, node.input[1])
        if values is None or not np.issubdtype(values.dtype, np.integer):
            return None
        axes = values.reshape(-1).tolist()
    if axes is None:
        return None
    return [int(axis) for axis in np.asarray(axes).reshape(-1).tolist()]


def _remove_nodes(model: onnx.ModelProto, remove_ids: set[int]) -> None:
    remaining = [node for node in model.graph.node if id(node) not in remove_ids]
    del model.graph.node[:]
    model.graph.node.extend(remaining)


def _prune_dead_ancestors(model: onnx.ModelProto, seed_names: set[str]) -> None:
    """Remove only dead producers and constants reached from removed-node inputs."""
    pending = {name for name in seed_names if name}
    candidate_initializers: set[str] = set()
    while pending:
        index = _GraphIndex.build(model)
        remove_ids: set[int] = set()
        next_pending: set[str] = set()
        for name in pending:
            if name in index.initializers:
                candidate_initializers.add(name)
            producer = index.producers.get(name)
            if producer is None:
                continue
            if any(
                output in index.graph_outputs or index.consumers.get(output)
                for output in producer.output
                if output
            ):
                continue
            remove_ids.add(id(producer))
            next_pending.update(input_name for input_name in producer.input if input_name)
        if not remove_ids:
            break
        _remove_nodes(model, remove_ids)
        pending = next_pending

    referenced = _referenced_tensor_names(model.graph)
    remaining_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name not in candidate_initializers or initializer.name in referenced
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(remaining_initializers)


def _prune_stale_value_info(model: onnx.ModelProto) -> None:
    graph = model.graph
    defined = {value.name for value in (*graph.input, *graph.output) if value.name}
    defined.update(initializer.name for initializer in graph.initializer)
    defined.update(output for node in graph.node for output in node.output if output)
    remaining = [value for value in graph.value_info if value.name in defined]
    del graph.value_info[:]
    graph.value_info.extend(remaining)


def _ensure_ms_opset(model: onnx.ModelProto) -> None:
    for opset in model.opset_import:
        if opset.domain == "com.microsoft":
            if opset.version < 1:
                opset.version = 1
            return
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))


def _default_opset(model: onnx.ModelProto) -> int:
    return next(
        (int(opset.version) for opset in model.opset_import if opset.domain in ("", "ai.onnx")),
        0,
    )


def _other_input_with_constant(
    index: _GraphIndex,
    node: onnx.NodeProto,
    expected: float,
) -> str | None:
    if len(node.input) != 2:
        return None
    for constant_index in (0, 1):
        value = _constant_scalar(index, node.input[constant_index])
        if value is not None and float(value) == expected:
            return node.input[1 - constant_index]
    return None


def _expand_preserves_shape(
    index: _GraphIndex,
    expand: onnx.NodeProto,
    source_name: str,
) -> bool:
    """Prove that Expand's target shape is exactly the source tensor shape."""
    if len(expand.input) != 2:
        return False
    target_name = expand.input[1]
    target_producer = _producer(index, target_name, "Shape")
    if (
        target_producer is not None
        and target_producer.input
        and target_producer.input[0] == source_name
        and int(_attribute(target_producer, "start", 0)) == 0
        and _attribute(target_producer, "end") is None
    ):
        return True

    source_shape = index.shapes.get(source_name)
    expanded_output = _node_output(expand)
    expanded_shape = index.shapes.get(expanded_output) if expanded_output else None
    if (
        source_shape is not None
        and expanded_shape is not None
        and None not in source_shape
        and source_shape == expanded_shape
    ):
        return True

    target_shape = _constant_array(index, target_name)
    return (
        source_shape is not None
        and None not in source_shape
        and target_shape is not None
        and np.issubdtype(target_shape.dtype, np.integer)
        and tuple(int(value) for value in target_shape.reshape(-1)) == source_shape
    )


def _producer(
    index: _GraphIndex,
    tensor_name: str,
    op_type: str,
) -> onnx.NodeProto | None:
    node = index.producers.get(tensor_name)
    if node is None or node.op_type != op_type or not _is_default_domain(node):
        return None
    return node


def _match_dimension_scalar(index: _GraphIndex, name: str) -> str | None:
    """Return the one-element shape vector representing a dimension scalar."""
    squeeze = _producer(index, name, "Squeeze")
    if squeeze is not None and squeeze.input:
        axes = _constant_axes(index, squeeze)
        if axes is None or axes in ([0], [-1]):
            shape_vector = squeeze.input[0]
            shape = _producer(index, shape_vector, "Shape")
            if shape is not None:
                start = int(_attribute(shape, "start", 0))
                end = _attribute(shape, "end")
                if end is not None and int(end) == start + 1:
                    return shape_vector

    reshaped_vectors = []
    for consumer in index.consumers.get(name, []):
        if (
            consumer.op_type != "Reshape"
            or not _is_default_domain(consumer)
            or len(consumer.input) < 2
            or consumer.input[0] != name
        ):
            continue
        target_shape = _constant_array(index, consumer.input[1])
        output = _node_output(consumer)
        if (
            target_shape is not None
            and target_shape.size == 1
            and int(target_shape.reshape(-1)[0]) == -1
            and output is not None
        ):
            reshaped_vectors.append(output)
    return reshaped_vectors[0] if len(reshaped_vectors) == 1 else None


def _match_nearest_index_vector(index: _GraphIndex, name: str) -> str | None:
    """Match floor((arange(2*d) + 0.5) * (d / (2*d)))."""
    cast_int = _producer(index, name, "Cast")
    if cast_int is None or int(_attribute(cast_int, "to", -1)) != int(onnx.TensorProto.INT64):
        return None
    scaled = _producer(index, cast_int.input[0], "Mul") if cast_int.input else None
    if scaled is None or len(scaled.input) != 2:
        return None

    left = index.producers.get(scaled.input[0])
    right = index.producers.get(scaled.input[1])
    if (
        left is not None
        and right is not None
        and left.op_type == "Add"
        and right.op_type == "Div"
        and _is_default_domain(left)
        and _is_default_domain(right)
    ):
        offset, ratio = left, right
    elif (
        left is not None
        and right is not None
        and left.op_type == "Div"
        and right.op_type == "Add"
        and _is_default_domain(left)
        and _is_default_domain(right)
    ):
        offset, ratio = right, left
    else:
        return None

    range_name = _other_input_with_constant(index, offset, 0.5)
    arange = _producer(index, range_name, "Range") if range_name else None
    if arange is None or len(arange.input) != 3:
        return None
    if _constant_scalar(index, arange.input[0]) != 0:
        return None
    if _constant_scalar(index, arange.input[2]) != 1:
        return None

    end_cast = _producer(index, arange.input[1], "Cast")
    if end_cast is None or int(_attribute(end_cast, "to", -1)) != int(onnx.TensorProto.FLOAT):
        return None
    doubled_size = _producer(index, end_cast.input[0], "Mul") if end_cast.input else None
    if doubled_size is None:
        return None
    dimension_scalar = _other_input_with_constant(index, doubled_size, 2.0)
    if dimension_scalar is None:
        return None
    shape_vector = _match_dimension_scalar(index, dimension_scalar)
    if shape_vector is None:
        return None

    if len(ratio.input) != 2:
        return None
    dimension_float = _producer(index, ratio.input[0], "Cast")
    if (
        dimension_float is None
        or int(_attribute(dimension_float, "to", -1)) != int(onnx.TensorProto.FLOAT)
        or not dimension_float.input
        or dimension_float.input[0] != dimension_scalar
    ):
        return None
    denominator = _producer(index, ratio.input[1], "Mul")
    if denominator is None:
        return None
    denominator_other = _other_input_with_constant(index, denominator, 2.0)
    if denominator_other != dimension_float.output[0]:
        return None
    return shape_vector


def _match_unsqueeze(
    index: _GraphIndex,
    tensor_name: str,
    axes: list[int],
) -> onnx.NodeProto | None:
    node = _producer(index, tensor_name, "Unsqueeze")
    if node is None or _constant_axes(index, node) != axes:
        return None
    return node


def _match_resize_indices(index: _GraphIndex, name: str) -> tuple[str, str] | None:
    """Match a broadcasted pair of exact 2x nearest-neighbor index vectors."""
    concat = _producer(index, name, "Concat")
    if concat is None or len(concat.input) != 2 or int(_attribute(concat, "axis", 0)) != -1:
        return None

    outer_h = _match_unsqueeze(index, concat.input[0], [-1])
    outer_w = _match_unsqueeze(index, concat.input[1], [-1])
    if outer_h is None or outer_w is None:
        return None
    expand_h = _producer(index, outer_h.input[0], "Expand")
    expand_w = _producer(index, outer_w.input[0], "Expand")
    if (
        expand_h is None
        or expand_w is None
        or len(expand_h.input) != 2
        or len(expand_w.input) != 2
        or expand_h.input[1] != expand_w.input[1]
    ):
        return None

    inner_h = _match_unsqueeze(index, expand_h.input[0], [-1])
    if inner_h is None or not inner_h.input:
        return None
    h_indices = inner_h.input[0]
    w_indices = expand_w.input[0]
    h_shape_vector = _match_nearest_index_vector(index, h_indices)
    w_shape_vector = _match_nearest_index_vector(index, w_indices)
    if h_shape_vector is None or w_shape_vector is None:
        return None

    grid_shape = _producer(index, expand_h.input[1], "Shape")
    grid_max = (
        _producer(index, grid_shape.input[0], "Max")
        if grid_shape is not None and grid_shape.input
        else None
    )
    if grid_max is None or len(grid_max.input) != 2:
        return None
    if set(grid_max.input) != {inner_h.output[0], w_indices}:
        return None
    return h_shape_vector, w_shape_vector


@dataclass(frozen=True)
class _ResizeMatch:
    pre_cast: onnx.NodeProto
    pre_transpose: onnx.NodeProto
    gather: onnx.NodeProto
    post_transpose: onnx.NodeProto
    post_cast: onnx.NodeProto
    source: str
    indices: str


def _match_resize_path(index: _GraphIndex, gather: onnx.NodeProto) -> _ResizeMatch | None:
    """Match the exact FP16/FP32 GatherND path used for 2x NCHW upsampling."""
    if (
        gather.op_type != "GatherND"
        or not _is_default_domain(gather)
        or len(gather.input) != 2
        or len(gather.output) != 1
        or int(_attribute(gather, "batch_dims", 0)) != 0
    ):
        return None

    pre_transpose = _producer(index, gather.input[0], "Transpose")
    if pre_transpose is None or len(pre_transpose.input) != 1:
        return None
    pre_cast = _producer(index, pre_transpose.input[0], "Cast")
    if (
        pre_cast is None
        or len(pre_cast.input) != 1
        or int(_attribute(pre_cast, "to", -1)) != int(onnx.TensorProto.FLOAT)
    ):
        return None

    post_transpose = _only_consumer(index, gather.output[0], "Transpose")
    if post_transpose is None:
        return None
    post_output = _node_output(post_transpose)
    post_cast = _only_consumer(index, post_output, "Cast") if post_output else None
    if post_cast is None or len(post_cast.output) != 1:
        return None

    permutation = [2, 3, 0, 1]
    if list(_attribute(pre_transpose, "perm", [])) != permutation:
        return None
    if list(_attribute(post_transpose, "perm", [])) != permutation:
        return None

    pre_cast_output = _node_output(pre_cast)
    pre_transpose_output = _node_output(pre_transpose)
    if (
        pre_cast_output is None
        or _only_consumer(index, pre_cast_output, "Transpose") is not pre_transpose
        or pre_transpose_output is None
        or _only_consumer(index, pre_transpose_output, "GatherND") is not gather
    ):
        return None

    source = pre_cast.input[0]
    source_type = index.element_types.get(source)
    if source_type not in (int(onnx.TensorProto.FLOAT), int(onnx.TensorProto.FLOAT16)):
        return None
    if int(_attribute(post_cast, "to", -1)) != source_type:
        return None

    intermediate_outputs = {
        pre_cast.output[0],
        pre_transpose.output[0],
        gather.output[0],
        post_transpose.output[0],
    }
    if intermediate_outputs & index.graph_outputs:
        return None

    shape_vectors = _match_resize_indices(index, gather.input[1])
    if shape_vectors is None:
        return None
    reshape = _producer(index, source, "Reshape")
    if reshape is None or len(reshape.input) < 2:
        return None
    shape_concat = _producer(index, reshape.input[1], "Concat")
    if (
        shape_concat is None
        or len(shape_concat.input) != 4
        or int(_attribute(shape_concat, "axis", 0)) != 0
        or tuple(shape_concat.input[2:4]) != shape_vectors
    ):
        return None

    return _ResizeMatch(
        pre_cast=pre_cast,
        pre_transpose=pre_transpose,
        gather=gather,
        post_transpose=post_transpose,
        post_cast=post_cast,
        source=source,
        indices=gather.input[1],
    )


# =============================================================================
# SURGERYPIPE
# =============================================================================


class SurgeryPipe(BasePipe[SurgeryPipeConfig]):
    """Surgery pipe for precise model modifications.

    This pipe performs targeted post-optimization graph transformations.

    Currently supported operations:
    - clamp-constant-values: Clamp extreme float constants (e.g., -inf → -1e3)
    """

    name: ClassVar[str] = "surgery"
    capabilities: ClassVar[dict[str, Any]] = POST_SURGERY_CAPABILITIES

    @classmethod
    def build_config(cls, **kwargs: Any) -> SurgeryPipeConfig:
        """Build surgery pipe config from kwargs.

        Args:
            **kwargs: User-provided configuration
                - clamp_constant_values: Enable/disable constant clamping
                - clamp_min: Minimum value for clamping (default: -1e3)
                - clamp_max: Maximum value for clamping (default: 1e3)
                - remove_isnan_in_attention_mask: Remove IsNaN guard patterns
                - simplify_l2_normalization: Remove redundant normalization ops
                - gathernd_to_resize: Replace exact GatherND upsampling grids
                - silu_to_quick_gelu: Emit QuickGelu(alpha=1) for exact SiLU
                - scaled_matmul_to_fused_matmul: Emit FusedMatMul for scaled MatMul
                - verbose: Enable verbose logging

        Returns:
            Configured SurgeryPipeConfig
        """
        return SurgeryPipeConfig(
            clamp_constant_values=kwargs.get("clamp_constant_values", False),
            clamp_min=kwargs.get("clamp_min", -1e3),
            clamp_max=kwargs.get("clamp_max", 1e3),
            remove_isnan_in_attention_mask=kwargs.get("remove_isnan_in_attention_mask", False),
            untie_constant_batched_matmul=kwargs.get("untie_constant_batched_matmul", False),
            simplify_l2_normalization=kwargs.get("simplify_l2_normalization", False),
            gathernd_to_resize=kwargs.get("gathernd_to_resize", False),
            silu_to_quick_gelu=kwargs.get("silu_to_quick_gelu", False),
            scaled_matmul_to_fused_matmul=kwargs.get("scaled_matmul_to_fused_matmul", False),
            verbose=kwargs.get("verbose", False),
        )

    @classmethod
    def should_process(cls, config: SurgeryPipeConfig) -> bool:
        """Check if surgery pipe should process the model.

        Args:
            config: Surgery pipe configuration

        Returns:
            True if any surgery operation is enabled
        """
        return (
            config.clamp_constant_values
            or config.remove_isnan_in_attention_mask
            or config.untie_constant_batched_matmul
            or config.simplify_l2_normalization
            or config.gathernd_to_resize
            or config.silu_to_quick_gelu
            or config.scaled_matmul_to_fused_matmul
        )

    def process(self, model: onnx.ModelProto, config: SurgeryPipeConfig) -> onnx.ModelProto:
        """Apply surgery operations to the model.

        Args:
            model: Input ONNX model (will not be modified)
            config: Surgery pipe configuration

        Returns:
            New model with surgery operations applied
        """
        if not self.should_process(config):
            return model
        # Import onnx inside method to avoid import errors
        import onnx

        # Create a copy of the model to avoid modifying the original
        model_copy = onnx.ModelProto()
        model_copy.CopyFrom(model)

        if config.clamp_constant_values:
            model_copy = self._clamp_constant_values(
                model_copy, config.clamp_min, config.clamp_max, config.verbose
            )

        if config.remove_isnan_in_attention_mask:
            model_copy = self._remove_isnan_in_attention_mask(model_copy, config.verbose)

        if config.untie_constant_batched_matmul:
            model_copy = self._untie_constant_batched_matmul(model_copy, config.verbose)

        if config.simplify_l2_normalization:
            model_copy = self._simplify_l2_normalization(model_copy, config.verbose)

        if config.gathernd_to_resize:
            model_copy = self._gathernd_to_resize(model_copy, config.verbose)

        if config.silu_to_quick_gelu:
            model_copy = self._silu_to_quick_gelu(model_copy, config.verbose)

        if config.scaled_matmul_to_fused_matmul:
            model_copy = self._scaled_matmul_to_fused_matmul(model_copy, config.verbose)

        return model_copy

    def _clamp_constant_values(
        self,
        model: onnx.ModelProto,
        clamp_min: float,
        clamp_max: float,
        verbose: bool = False,
    ) -> onnx.ModelProto:
        """Clamp extreme float constant values in the model.

        This operation modifies initializers (weights/constants) to clamp
        extreme values like -inf or very large floats to a reasonable range.
        This prevents quantization issues where inf values produce inf scales.

        Args:
            model: ONNX model (modified in place)
            clamp_min: Minimum allowed value
            clamp_max: Maximum allowed value
            verbose: Log details about clamped tensors

        Returns:
            Model with clamped constants
        """
        from onnx import TensorProto, numpy_helper

        clamped_count = 0
        clamped_tensors: list[str] = []

        for initializer in model.graph.initializer:
            # Only process float types
            if initializer.data_type not in (
                TensorProto.FLOAT,
                TensorProto.FLOAT16,
                TensorProto.DOUBLE,
            ):
                continue

            # Convert to numpy array
            tensor = numpy_helper.to_array(initializer)
            original_min = float(tensor.min())
            original_max = float(tensor.max())

            # Check if clamping is needed
            needs_clamp = original_min < clamp_min or original_max > clamp_max

            if needs_clamp:
                # Clamp the values (np.clip is equivalent to torch.clamp)
                clamped = np.clip(tensor, clamp_min, clamp_max)

                # Create new tensor proto with clamped values
                new_tensor = numpy_helper.from_array(clamped, initializer.name)

                # Copy over the initializer
                initializer.CopyFrom(new_tensor)

                clamped_count += 1
                clamped_tensors.append(initializer.name)

                if verbose:
                    logger.info(
                        "Clamped tensor '%s': [%.2e, %.2e] -> [%.2e, %.2e]",
                        initializer.name,
                        original_min,
                        original_max,
                        clamp_min,
                        clamp_max,
                    )

        if clamped_count > 0:
            logger.info(
                "SurgeryPipe: Clamped %d tensor(s) to range [%.2e, %.2e]",
                clamped_count,
                clamp_min,
                clamp_max,
            )
            if verbose:
                logger.debug("Clamped tensors: %s", clamped_tensors)

        return model

    # -----------------------------------------------------------------
    # remove-isnan-in-attention-mask
    # -----------------------------------------------------------------

    def _remove_isnan_in_attention_mask(
        self,
        model: onnx.ModelProto,
        verbose: bool = False,
    ) -> onnx.ModelProto:
        """Remove Softmax → IsNaN → Where NaN guard patterns in attention.

        Pattern: Softmax → IsNaN → Where(isnan, 0, softmax_out)
        Remove IsNaN + guard Where, use Softmax output directly.

        These guards are dead code when clamp_constant_values has already
        replaced -inf with a finite value (Softmax never produces NaN).

        Args:
            model: ONNX model (modified in place).
            verbose: Log details about each removal.

        Returns:
            Model with IsNaN guard patterns removed.
        """
        guard_count = 0

        # Build output→node map
        output_to_node: dict[str, onnx.NodeProto] = {}
        for node in model.graph.node:
            for out in node.output:
                output_to_node[out] = node

        nodes_to_remove: list[onnx.NodeProto] = []
        rewire_map: dict[str, str] = {}

        for node in list(model.graph.node):
            if node.op_type != "IsNaN":
                continue
            producer = output_to_node.get(node.input[0])
            if producer is None or producer.op_type != "Softmax":
                continue
            softmax_out = producer.output[0]
            isnan_out = node.output[0]

            # Find guard Where consuming IsNaN output
            guard_wheres = [
                n for n in model.graph.node if n.op_type == "Where" and isnan_out in n.input
            ]
            if len(guard_wheres) != 1:
                continue
            guard_where = guard_wheres[0]
            if softmax_out not in guard_where.input:
                continue

            guard_out = guard_where.output[0]
            nodes_to_remove.extend([node, guard_where])
            rewire_map[guard_out] = softmax_out
            guard_count += 1
            if verbose:
                logger.info(
                    "  remove-isnan: remove %s + %s, rewire %s -> %s",
                    node.name,
                    guard_where.name,
                    guard_out,
                    softmax_out,
                )

        # Apply rewiring
        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp in rewire_map:
                    node.input[i] = rewire_map[inp]
        for graph_out in model.graph.output:
            if graph_out.name in rewire_map:
                graph_out.name = rewire_map[graph_out.name]

        # Remove dead nodes
        remove_ids = {id(n) for n in nodes_to_remove}
        remaining = [n for n in model.graph.node if id(n) not in remove_ids]
        del model.graph.node[:]
        model.graph.node.extend(remaining)

        if guard_count:
            logger.info(
                "SurgeryPipe: remove-isnan-in-attention-mask: %d IsNaN+Where guards removed",
                guard_count,
            )

        return model

    # -----------------------------------------------------------------
    # untie-constant-batched-matmul
    # -----------------------------------------------------------------

    def _untie_constant_batched_matmul(
        self,
        model: onnx.ModelProto,
        verbose: bool = False,
    ) -> onnx.ModelProto:
        """Make a batched MatMul's constant operand runtime-valued.

        OpenVINO GPU's oneDNN gemm cannot select an implementation for a batched
        (rank >= 3) MatMul where an operand is a compile-time constant: the same
        gemm with a dynamic operand, and 2D constant gemm, both compile fine.
        Transformer disentangled-attention position terms depend only on weights,
        so they fold into 3D constants and hit this case.

        Fix: route each such constant operand through ``Add(const, zero)`` where
        ``zero`` is a runtime ``[1]`` tensor built from the first graph input's
        *data*: ``Cast(first_input -> float) -> Reshape([-1]) -> Slice([0:1])``
        yields a single element ``elem``, and ``zero = Sub(elem, elem) == 0.0``.
        ``zero`` is data-dependent, so OpenVINO's constant folder cannot collapse
        the Add back into a packed gemm weight, yet ``+ 0`` leaves the values
        unchanged and the single batched MatMul is preserved (no perf cost).

        Assumption: the first graph input has at least one element at runtime.
        The ``Slice([0:1])`` is out of bounds for a zero-sized input (e.g. a
        dynamic batch dimension fed an empty batch), which would raise at
        inference time rather than produce a zero.
        """
        from onnx import TensorProto, helper, numpy_helper

        graph = model.graph
        initializers = {init.name: init for init in graph.initializer}

        # Collect (matmul_node, operand_index) where the operand is a constant
        # initializer of rank >= 3. Skip MatMuls whose operands are all constant
        # (those fold away entirely and never reach gemm impl selection).
        targets: list[tuple[onnx.NodeProto, int]] = []
        for node in graph.node:
            if node.op_type != "MatMul" or len(node.input) != 2:
                continue
            const_idx = [i for i, name in enumerate(node.input) if name in initializers]
            if len(const_idx) != 1:
                continue
            idx = const_idx[0]
            if len(initializers[node.input[idx]].dims) >= 3:
                targets.append((node, idx))

        if not targets:
            return model

        if not graph.input:
            logger.warning(
                "SurgeryPipe: untie-constant-batched-matmul: no graph input to "
                "derive a runtime value from; skipping %d MatMul(s)",
                len(targets),
            )
            return model

        prefix = "winml_ovgpu_untie"
        first_input = graph.input[0].name
        new_nodes: list[onnx.NodeProto] = []
        new_inits: list[onnx.TensorProto] = []

        # Build a shape-[1] runtime zero from input *data* (not shape — input
        # shapes are static and would be folded). Only ubiquitous ops are used
        # so the static analyzer handles them: a single input element is sliced
        # out and subtracted from itself. A [1] tensor broadcasts against any
        # constant operand, regardless of its rank.
        xf = f"{prefix}_xf"
        new_nodes.append(
            helper.make_node("Cast", [first_input], [xf], to=TensorProto.FLOAT, name=xf)
        )
        flat = f"{prefix}_flat"
        new_inits.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), f"{prefix}_m1"))
        new_nodes.append(helper.make_node("Reshape", [xf, f"{prefix}_m1"], [flat], name=flat))
        elem = f"{prefix}_elem"
        # Slice(flat, starts=[0], ends=[1], axes=[0]) -> the first element.
        # starts and axes are distinct tensors even though both hold [0], so a
        # future edit to one role cannot silently corrupt the other.
        starts = f"{prefix}_slice_starts"
        ends = f"{prefix}_slice_ends"
        axis = f"{prefix}_slice_axis"
        new_inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), starts))
        new_inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), ends))
        new_inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), axis))
        new_nodes.append(helper.make_node("Slice", [flat, starts, ends, axis], [elem], name=elem))
        # zero = elem - elem == 0.0 (data-dependent, so it is not folded away).
        zero_f32 = f"{prefix}_zero_f32"
        new_nodes.append(helper.make_node("Sub", [elem, elem], [zero_f32], name=zero_f32))

        # A zero must match each operand's dtype (ONNX has no implicit promotion).
        zero_by_dtype: dict[int, str] = {int(TensorProto.FLOAT): zero_f32}

        def zero_for(dtype: int) -> str:
            name = zero_by_dtype.get(dtype)
            if name is None:
                name = f"{prefix}_zero_{dtype}"
                new_nodes.append(helper.make_node("Cast", [zero_f32], [name], to=dtype, name=name))
                zero_by_dtype[dtype] = name
            return name

        untied = 0
        # Index the loop rather than node.name: node names are optional in ONNX
        # and exporters routinely leave them blank or duplicated, so deriving
        # `dyn` from the name would collide and produce an invalid graph.
        for untie_idx, (node, idx) in enumerate(targets):
            const_name = node.input[idx]
            dtype = initializers[const_name].data_type
            if dtype not in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE):
                continue
            dyn = f"{prefix}_untied{untie_idx}_in{idx}"
            new_nodes.append(
                helper.make_node("Add", [const_name, zero_for(dtype)], [dyn], name=dyn)
            )
            node.input[idx] = dyn
            untied += 1
            if verbose:
                logger.info(
                    "  untie-constant-batched-matmul: %s input[%d] %s -> %s",
                    node.name,
                    idx,
                    const_name,
                    dyn,
                )

        if untied == 0:
            return model

        graph.initializer.extend(new_inits)
        # Prepend new nodes: their inputs are only graph inputs / initializers,
        # so placing them first keeps the graph topologically sorted.
        existing = list(graph.node)
        del graph.node[:]
        graph.node.extend(new_nodes + existing)

        logger.info(
            "SurgeryPipe: untie-constant-batched-matmul: untied %d batched "
            "MatMul constant operand(s)",
            untied,
        )

        return model

    # -----------------------------------------------------------------
    # simplify-l2-normalization
    # -----------------------------------------------------------------

    def _simplify_l2_normalization(
        self,
        model: onnx.ModelProto,
        verbose: bool = False,
    ) -> onnx.ModelProto:
        """Remove Clip(min=0) and Expand from exact ReduceL2 normalization."""
        index = _GraphIndex.build(model)
        remove_ids: set[int] = set()
        prune_seeds: set[str] = set()
        simplified = 0

        for reduce_l2 in list(model.graph.node):
            if (
                reduce_l2.op_type != "ReduceL2"
                or not _is_default_domain(reduce_l2)
                or not reduce_l2.input
                or _constant_axes(index, reduce_l2) != [1]
                or int(_attribute(reduce_l2, "keepdims", 1)) != 1
            ):
                continue
            norm_output = _node_output(reduce_l2)
            if norm_output is None:
                continue
            clip = _only_consumer(index, norm_output, "Clip")
            if clip is None or _node_output(clip) in index.graph_outputs:
                continue

            clip_min_values: list[float] = []
            attribute_min = _attribute(clip, "min")
            if attribute_min is not None:
                clip_min_values.append(float(attribute_min))
            if len(clip.input) > 1 and clip.input[1]:
                input_min = _constant_scalar(index, clip.input[1])
                if input_min is None:
                    continue
                clip_min_values.append(float(input_min))
            if not clip_min_values or any(value != 0.0 for value in clip_min_values):
                continue
            if _attribute(clip, "max") is not None:
                continue
            if len(clip.input) > 2 and clip.input[2]:
                continue

            clip_output = _node_output(clip)
            expand = _only_consumer(index, clip_output, "Expand") if clip_output else None
            if (
                expand is None
                or len(expand.input) != 2
                or _node_output(expand) in index.graph_outputs
                or not _expand_preserves_shape(index, expand, reduce_l2.input[0])
            ):
                continue
            expand_output = _node_output(expand)
            div = _only_consumer(index, expand_output, "Div") if expand_output else None
            if (
                div is None
                or len(div.input) != 2
                or div.input[0] != reduce_l2.input[0]
                or div.input[1] != expand_output
            ):
                continue

            div.input[1] = norm_output
            remove_ids.update((id(clip), id(expand)))
            prune_seeds.update(name for node in (clip, expand) for name in node.input if name)
            simplified += 1
            if verbose:
                logger.info(
                    "  simplify-l2-normalization: %s -> %s now broadcasts directly",
                    reduce_l2.name or norm_output,
                    div.name or div.output[0],
                )

        if not simplified:
            return model

        _remove_nodes(model, remove_ids)
        _prune_dead_ancestors(model, prune_seeds)
        _prune_stale_value_info(model)
        logger.info(
            "SurgeryPipe: simplify-l2-normalization: simplified %d normalization(s)",
            simplified,
        )
        return model

    # -----------------------------------------------------------------
    # gathernd-to-resize
    # -----------------------------------------------------------------

    def _gathernd_to_resize(
        self,
        model: onnx.ModelProto,
        verbose: bool = False,
    ) -> onnx.ModelProto:
        """Replace an exact generated 2x nearest-neighbor grid with Resize."""
        if _default_opset(model) < 11:
            logger.warning(
                "SurgeryPipe: gathernd-to-resize requires ONNX opset 11 or newer; skipping"
            )
            return model

        index = _GraphIndex.build(model)
        matches = [
            match
            for node in model.graph.node
            if node.op_type == "GatherND"
            for match in [_match_resize_path(index, node)]
            if match is not None
        ]
        if not matches:
            return model

        allocator = _NameAllocator(model)
        scales_name = allocator.new("surgery_resize_scales")
        model.graph.initializer.append(
            onnx.numpy_helper.from_array(
                np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
                scales_name,
            )
        )

        replacements: dict[int, onnx.NodeProto] = {}
        remove_ids: set[int] = set()
        prune_seeds: set[str] = set()
        for match in matches:
            resize = onnx.helper.make_node(
                "Resize",
                [match.source, "", scales_name],
                [match.post_cast.output[0]],
                name=allocator.new("surgery_nearest_resize"),
                mode="nearest",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor",
            )
            replacements[id(match.pre_cast)] = resize
            path_nodes = (
                match.pre_cast,
                match.pre_transpose,
                match.gather,
                match.post_transpose,
                match.post_cast,
            )
            remove_ids.update(id(node) for node in path_nodes)
            prune_seeds.update(name for node in path_nodes for name in node.input if name)
            prune_seeds.add(match.indices)
            if verbose:
                logger.info(
                    "  gathernd-to-resize: %s -> %s",
                    match.gather.name or match.gather.output[0],
                    resize.name,
                )

        rewritten: list[onnx.NodeProto] = []
        for node in model.graph.node:
            replacement = replacements.get(id(node))
            if replacement is not None:
                rewritten.append(replacement)
            elif id(node) not in remove_ids:
                rewritten.append(node)
        del model.graph.node[:]
        model.graph.node.extend(rewritten)

        _prune_dead_ancestors(model, prune_seeds)
        _prune_stale_value_info(model)
        logger.info(
            "SurgeryPipe: gathernd-to-resize: replaced %d upsampling path(s)",
            len(matches),
        )
        return model

    # -----------------------------------------------------------------
    # silu-to-quick-gelu
    # -----------------------------------------------------------------

    def _silu_to_quick_gelu(
        self,
        model: onnx.ModelProto,
        verbose: bool = False,
    ) -> onnx.ModelProto:
        """Replace exact x*Sigmoid(x) topology with QuickGelu(alpha=1)."""
        index = _GraphIndex.build(model)
        allocator = _NameAllocator(model)
        replacements: dict[int, onnx.NodeProto] = {}
        remove_ids: set[int] = set()

        for mul in list(model.graph.node):
            if (
                mul.op_type != "Mul"
                or not _is_default_domain(mul)
                or len(mul.input) != 2
                or len(mul.output) != 1
            ):
                continue
            candidates: list[onnx.NodeProto] = []
            for sigmoid_input_index in (0, 1):
                sigmoid = _producer(index, mul.input[sigmoid_input_index], "Sigmoid")
                other_input = mul.input[1 - sigmoid_input_index]
                sigmoid_output = _node_output(sigmoid) if sigmoid is not None else None
                if (
                    sigmoid is not None
                    and len(sigmoid.input) == 1
                    and sigmoid.input[0] == other_input
                    and sigmoid_output not in index.graph_outputs
                    and _only_consumer(index, sigmoid_output, "Mul") is mul
                ):
                    candidates.append(sigmoid)
            if len(candidates) != 1:
                continue

            sigmoid = candidates[0]
            quick_gelu = onnx.helper.make_node(
                "QuickGelu",
                [sigmoid.input[0]],
                [mul.output[0]],
                name=allocator.new("surgery_silu_quick_gelu"),
                domain="com.microsoft",
                alpha=1.0,
            )
            replacements[id(sigmoid)] = quick_gelu
            remove_ids.update((id(sigmoid), id(mul)))
            if verbose:
                logger.info(
                    "  silu-to-quick-gelu: %s + %s -> %s",
                    sigmoid.name or sigmoid.output[0],
                    mul.name or mul.output[0],
                    quick_gelu.name,
                )

        if not replacements:
            return model

        rewritten: list[onnx.NodeProto] = []
        for node in model.graph.node:
            replacement = replacements.get(id(node))
            if replacement is not None:
                rewritten.append(replacement)
            elif id(node) not in remove_ids:
                rewritten.append(node)
        del model.graph.node[:]
        model.graph.node.extend(rewritten)
        _ensure_ms_opset(model)
        _prune_stale_value_info(model)
        logger.info(
            "SurgeryPipe: silu-to-quick-gelu: fused %d SiLU activation(s)",
            len(replacements),
        )
        return model

    # -----------------------------------------------------------------
    # scaled-matmul-to-fused-matmul
    # -----------------------------------------------------------------

    def _scaled_matmul_to_fused_matmul(
        self,
        model: onnx.ModelProto,
        verbose: bool = False,
    ) -> onnx.ModelProto:
        """Replace MatMul followed by scalar Mul with FusedMatMul(alpha)."""
        index = _GraphIndex.build(model)
        allocator = _NameAllocator(model)
        replacements: dict[int, onnx.NodeProto] = {}
        remove_ids: set[int] = set()
        prune_seeds: set[str] = set()
        supported_types = {
            int(onnx.TensorProto.FLOAT),
            int(onnx.TensorProto.FLOAT16),
            int(onnx.TensorProto.DOUBLE),
            int(onnx.TensorProto.BFLOAT16),
        }

        for matmul in list(model.graph.node):
            if (
                matmul.op_type != "MatMul"
                or not _is_default_domain(matmul)
                or len(matmul.input) != 2
                or len(matmul.output) != 1
                or matmul.output[0] in index.graph_outputs
            ):
                continue
            input_types = [index.element_types.get(name) for name in matmul.input]
            if (
                input_types[0] is None
                or input_types[0] != input_types[1]
                or input_types[0] not in supported_types
            ):
                continue

            mul = _only_consumer(index, matmul.output[0], "Mul")
            if mul is None or len(mul.input) != 2 or len(mul.output) != 1:
                continue
            if mul.input[0] == matmul.output[0]:
                scale_name = mul.input[1]
            elif mul.input[1] == matmul.output[0]:
                scale_name = mul.input[0]
            else:
                continue
            scale_values = _constant_array(index, scale_name)
            scale_type = _constant_element_type(index, scale_name)
            if scale_values is None or scale_values.size != 1 or scale_type != input_types[0]:
                continue
            try:
                alpha = float(scale_values.reshape(-1)[0])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(alpha) or float(np.float32(alpha)) != alpha:
                continue

            fused_matmul = onnx.helper.make_node(
                "FusedMatMul",
                list(matmul.input),
                [mul.output[0]],
                name=allocator.new("surgery_scaled_fused_matmul"),
                domain="com.microsoft",
                alpha=alpha,
            )
            replacements[id(matmul)] = fused_matmul
            remove_ids.update((id(matmul), id(mul)))
            prune_seeds.add(scale_name)
            if verbose:
                logger.info(
                    "  scaled-matmul-to-fused-matmul: %s + %s -> %s (alpha=%s)",
                    matmul.name or matmul.output[0],
                    mul.name or mul.output[0],
                    fused_matmul.name,
                    alpha,
                )

        if not replacements:
            return model

        rewritten: list[onnx.NodeProto] = []
        for node in model.graph.node:
            replacement = replacements.get(id(node))
            if replacement is not None:
                rewritten.append(replacement)
            elif id(node) not in remove_ids:
                rewritten.append(node)
        del model.graph.node[:]
        model.graph.node.extend(rewritten)
        _ensure_ms_opset(model)
        _prune_dead_ancestors(model, prune_seeds)
        _prune_stale_value_info(model)
        logger.info(
            "SurgeryPipe: scaled-matmul-to-fused-matmul: fused %d MatMul(s)",
            len(replacements),
        )
        return model


class PreSurgeryPipe(SurgeryPipe):
    """Apply proof-dependent graph surgeries before ORT rewrites the graph."""

    name: ClassVar[str] = "pre-surgery"
    capabilities: ClassVar[dict[str, Any]] = PRE_SURGERY_CAPABILITIES

    @classmethod
    def build_config(cls, **kwargs: Any) -> SurgeryPipeConfig:
        """Build a config containing only pre-optimization surgeries."""
        return SurgeryPipeConfig(
            simplify_l2_normalization=kwargs.get("simplify_l2_normalization", False),
            gathernd_to_resize=kwargs.get("gathernd_to_resize", False),
            scaled_matmul_to_fused_matmul=kwargs.get("scaled_matmul_to_fused_matmul", False),
            verbose=kwargs.get("verbose", False),
        )
