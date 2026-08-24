# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Conservative static grouped-Conv graph surgery."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
from onnx import (
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    helper,
    numpy_helper,
)

from ..onnx import get_captured_tensor_names
from ..quant.hints import (
    QUANTIZATION_REGION_HINT_KEY,
    STANDARD_ONNX_DOMAINS,
    QuantizationHintError,
    QuantizationRegion,
    parse_quantization_region_hints,
    serialize_quantization_region_hints,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RewritePlan:
    conv: NodeProto
    tail_slice: NodeProto
    kernel_start: int
    kernel_end: int
    pad_begin: int
    pad_end: int
    group: int
    branch_names: tuple[str, str]
    split_name: str
    concat_name: str
    split_outputs: tuple[str, str]
    branch_outputs: tuple[str, str]
    weight_names: tuple[str, str]
    bias_names: tuple[str, str] | None


def _constant_vector(
    name: str,
    initializers: dict[str, TensorProto],
) -> list[int] | None:
    initializer = initializers.get(name)
    if initializer is None:
        return None
    values = np.asarray(numpy_helper.to_array(initializer))
    if values.ndim != 1:
        return None
    return [int(value) for value in values]


def _static_shapes(model: ModelProto) -> dict[str, tuple[int | None, ...]]:
    shapes: dict[str, tuple[int | None, ...]] = {}
    for value_info in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dimensions = [
            dimension.dim_value if dimension.HasField("dim_value") else None
            for dimension in tensor_type.shape.dim
        ]
        shapes[value_info.name] = tuple(dimensions)
    return shapes


def _unique_name(base: str, used_names: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _replace_attribute(node: NodeProto, name: str, value: object) -> None:
    attributes = [
        copy.deepcopy(attribute) for attribute in node.attribute if attribute.name != name
    ]
    attributes.append(helper.make_attribute(name, value))
    del node.attribute[:]
    node.attribute.extend(attributes)


def _referenced_tensor_names(graph: GraphProto) -> set[str]:
    names = {value.name for value in (*graph.input, *graph.output)}
    for node in graph.node:
        names.update(input_name for input_name in node.input if input_name)
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                names.update(_referenced_tensor_names(attribute.g))
            elif attribute.type == AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    names.update(_referenced_tensor_names(subgraph))
    return names


def _consumer_nodes(graph: GraphProto) -> dict[str, list[NodeProto]]:
    consumers: dict[str, list[NodeProto]] = {}
    for node in graph.node:
        consumed_names = {input_name for input_name in node.input if input_name}
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                consumed_names.update(get_captured_tensor_names(attribute.g))
            elif attribute.type == AttributeProto.GRAPHS:
                for nested_graph in attribute.graphs:
                    consumed_names.update(get_captured_tensor_names(nested_graph))
        for input_name in consumed_names:
            consumers.setdefault(input_name, []).append(node)
    return consumers


def _match_plan(
    conv: NodeProto,
    *,
    consumers: dict[str, list[NodeProto]],
    graph_outputs: set[str],
    initializers: dict[str, TensorProto],
    shapes: dict[str, tuple[int | None, ...]],
    used_names: set[str],
) -> _RewritePlan | None:
    if conv.op_type != "Conv" or conv.domain not in ("", "ai.onnx"):
        return None
    if len(conv.input) not in (2, 3) or len(conv.output) != 1:
        return None
    if conv.output[0] in graph_outputs:
        return None

    output_consumers = consumers.get(conv.output[0], [])
    if len(output_consumers) != 1:
        return None
    tail_slice = output_consumers[0]
    if tail_slice.op_type != "Slice" or tail_slice.domain not in ("", "ai.onnx"):
        return None
    if len(tail_slice.input) != 5 or len(tail_slice.output) != 1:
        return None

    conv_attributes = {attribute.name: attribute for attribute in conv.attribute}
    auto_pad = conv_attributes.get("auto_pad")
    if auto_pad is not None and (auto_pad.type != AttributeProto.STRING or auto_pad.s != b"NOTSET"):
        return None
    pads_attribute = conv_attributes.get("pads")
    if pads_attribute is None or pads_attribute.type != AttributeProto.INTS:
        return None
    pads = [int(value) for value in pads_attribute.ints]
    strides_attribute = conv_attributes.get("strides")
    if strides_attribute is not None and strides_attribute.type != AttributeProto.INTS:
        return None
    strides = (
        [int(value) for value in strides_attribute.ints] if strides_attribute is not None else [1]
    )
    dilations_attribute = conv_attributes.get("dilations")
    if dilations_attribute is not None and dilations_attribute.type != AttributeProto.INTS:
        return None
    dilations = (
        [int(value) for value in dilations_attribute.ints]
        if dilations_attribute is not None
        else [1]
    )
    group_attribute = conv_attributes.get("group")
    if group_attribute is not None and group_attribute.type != AttributeProto.INT:
        return None
    group = int(group_attribute.i) if group_attribute is not None else 1
    if len(pads) != 2 or strides != [1] or dilations != [1] or group < 2 or group % 2:
        return None

    input_shape = shapes.get(conv.input[0])
    if input_shape is None or len(input_shape) != 3:
        return None
    input_channels, input_length = input_shape[1:]
    if input_channels is None or input_length is None or input_channels <= 0 or input_length <= 0:
        return None

    weight_initializer = initializers.get(conv.input[1])
    if weight_initializer is None:
        return None
    weights = np.asarray(numpy_helper.to_array(weight_initializer))
    if weights.ndim != 3:
        return None
    output_channels, channels_per_group, kernel_size = weights.shape
    if (
        output_channels <= 0
        or output_channels % group
        or input_channels != channels_per_group * group
        or kernel_size <= 1
    ):
        return None
    kernel_shape_attribute = conv_attributes.get("kernel_shape")
    if kernel_shape_attribute is not None and kernel_shape_attribute.type != AttributeProto.INTS:
        return None
    kernel_shape = (
        [int(value) for value in kernel_shape_attribute.ints]
        if kernel_shape_attribute is not None
        else [kernel_size]
    )
    if kernel_shape != [kernel_size]:
        return None

    bias_names: tuple[str, str] | None = None
    if len(conv.input) == 3:
        bias_initializer = initializers.get(conv.input[2])
        if bias_initializer is None:
            return None
        bias = np.asarray(numpy_helper.to_array(bias_initializer))
        if bias.shape != (output_channels,):
            return None

    starts = _constant_vector(tail_slice.input[1], initializers)
    ends = _constant_vector(tail_slice.input[2], initializers)
    axes = _constant_vector(tail_slice.input[3], initializers)
    steps = _constant_vector(tail_slice.input[4], initializers)
    if starts != [0] or axes not in ([2], [-1]) or steps != [1] or ends is None or len(ends) != 1:
        return None

    output_length = input_length + pads[0] + pads[1] - kernel_size + 1
    if output_length <= 1:
        return None
    retained_length = ends[0] + output_length if ends[0] < 0 else ends[0]
    retained_length = min(max(retained_length, 0), output_length)
    if retained_length <= 0 or retained_length >= output_length:
        return None

    kernel_start = max(0, pads[0] - (retained_length - 1))
    kernel_end = min(kernel_size - 1, pads[0] + input_length - 1)
    if kernel_start > kernel_end or (kernel_start == 0 and kernel_end == kernel_size - 1):
        return None
    trimmed_kernel = kernel_end - kernel_start + 1
    pad_begin = pads[0] - kernel_start
    pad_end = retained_length - input_length - pad_begin + trimmed_kernel - 1
    if pad_begin < 0 or pad_end < 0:
        return None
    if input_length + pad_begin + pad_end - trimmed_kernel + 1 != retained_length:
        return None

    base_name = conv.name or conv.output[0] or "grouped_conv"
    split_name = _unique_name(f"{base_name}__winml_split", used_names)
    branch_names = (
        _unique_name(f"{base_name}__winml_part0", used_names),
        _unique_name(f"{base_name}__winml_part1", used_names),
    )
    concat_name = _unique_name(f"{base_name}__winml_concat", used_names)
    split_outputs = (
        _unique_name(f"{split_name}_output0", used_names),
        _unique_name(f"{split_name}_output1", used_names),
    )
    branch_outputs = (
        _unique_name(f"{branch_names[0]}_output", used_names),
        _unique_name(f"{branch_names[1]}_output", used_names),
    )
    weight_names = (
        _unique_name(f"{conv.input[1]}__winml_part0", used_names),
        _unique_name(f"{conv.input[1]}__winml_part1", used_names),
    )
    if len(conv.input) == 3:
        bias_names = (
            _unique_name(f"{conv.input[2]}__winml_part0", used_names),
            _unique_name(f"{conv.input[2]}__winml_part1", used_names),
        )

    return _RewritePlan(
        conv=conv,
        tail_slice=tail_slice,
        kernel_start=kernel_start,
        kernel_end=kernel_end,
        pad_begin=pad_begin,
        pad_end=pad_end,
        group=group,
        branch_names=branch_names,
        split_name=split_name,
        concat_name=concat_name,
        split_outputs=split_outputs,
        branch_outputs=branch_outputs,
        weight_names=weight_names,
        bias_names=bias_names,
    )


def _validated_existing_regions(model: ModelProto) -> tuple[QuantizationRegion, ...]:
    regions = parse_quantization_region_hints(model) or ()
    if not regions:
        return ()

    nodes_by_name: dict[str, list[NodeProto]] = {}
    for node in model.graph.node:
        nodes_by_name.setdefault(node.name, []).append(node)
    consumers = _consumer_nodes(model.graph)
    graph_outputs = {value.name for value in model.graph.output}
    for region in regions:
        concat_nodes = nodes_by_name.get(region.concat, [])
        if (
            len(concat_nodes) != 1
            or concat_nodes[0].op_type != "Concat"
            or concat_nodes[0].domain not in STANDARD_ONNX_DOMAINS
        ):
            raise QuantizationHintError(
                f"Existing hinted Concat {region.concat!r} does not identify one Concat node"
            )
        branch_outputs: list[str] = []
        for branch_name in region.branches:
            branch_nodes = nodes_by_name.get(branch_name, [])
            if (
                len(branch_nodes) != 1
                or branch_nodes[0].op_type != "Conv"
                or branch_nodes[0].domain not in STANDARD_ONNX_DOMAINS
                or len(branch_nodes[0].output) != 1
                or not branch_nodes[0].output[0]
            ):
                raise QuantizationHintError(
                    f"Existing hinted branch {branch_name!r} does not identify one Conv node"
                )
            branch_output = branch_nodes[0].output[0]
            if branch_output in graph_outputs or consumers.get(branch_output) != concat_nodes:
                raise QuantizationHintError(
                    f"Existing hinted branch {branch_name!r} does not exclusively feed its Concat"
                )
            branch_outputs.append(branch_output)
        if list(concat_nodes[0].input) != branch_outputs:
            raise QuantizationHintError(
                f"Existing hinted Concat {region.concat!r} does not consume its branches directly"
            )
    return regions


def _set_region_hints(
    model: ModelProto,
    regions: tuple[QuantizationRegion, ...],
) -> None:
    entry = next(
        (item for item in model.metadata_props if item.key == QUANTIZATION_REGION_HINT_KEY),
        None,
    )
    if entry is None:
        entry = model.metadata_props.add()
        entry.key = QUANTIZATION_REGION_HINT_KEY
    entry.value = serialize_quantization_region_hints(regions)


def trim_split_grouped_convs(
    model: ModelProto,
    *,
    verbose: bool = False,
) -> ModelProto:
    """Trim unreachable 1D kernels and split even grouped Conv regions."""
    existing_regions = _validated_existing_regions(model)
    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    consumers = _consumer_nodes(graph)
    graph_outputs = {value.name for value in graph.output}
    shapes = _static_shapes(model)
    used_names = {
        name for node in graph.node for name in (*node.input, *node.output, node.name) if name
    }
    used_names.update(initializer.name for initializer in graph.initializer)
    used_names.update(value.name for value in (*graph.input, *graph.value_info, *graph.output))

    plans = [
        plan
        for node in graph.node
        if (
            plan := _match_plan(
                node,
                consumers=consumers,
                graph_outputs=graph_outputs,
                initializers=initializers,
                shapes=shapes,
                used_names=used_names,
            )
        )
        is not None
    ]
    if not plans:
        return model

    replacements: dict[int, list[NodeProto]] = {}
    removed_node_ids: set[int] = set()
    new_initializers: list[TensorProto] = []
    regions: list[QuantizationRegion] = []
    removable_initializers: set[str] = set()
    for plan in plans:
        weights = np.asarray(numpy_helper.to_array(initializers[plan.conv.input[1]]))
        output_midpoint = weights.shape[0] // 2
        trimmed_weights = weights[:, :, plan.kernel_start : plan.kernel_end + 1]
        for index, values in enumerate(np.split(trimmed_weights, [output_midpoint], axis=0)):
            new_initializers.append(
                numpy_helper.from_array(
                    np.ascontiguousarray(values),
                    plan.weight_names[index],
                )
            )

        if plan.bias_names is not None:
            bias = np.asarray(numpy_helper.to_array(initializers[plan.conv.input[2]]))
            for index, values in enumerate(np.split(bias, [output_midpoint], axis=0)):
                new_initializers.append(
                    numpy_helper.from_array(
                        np.ascontiguousarray(values),
                        plan.bias_names[index],
                    )
                )

        split = helper.make_node(
            "Split",
            [plan.conv.input[0]],
            list(plan.split_outputs),
            name=plan.split_name,
            axis=1,
        )
        branches: list[NodeProto] = []
        for index in range(2):
            branch = copy.deepcopy(plan.conv)
            branch.name = plan.branch_names[index]
            del branch.input[:]
            branch.input.extend([plan.split_outputs[index], plan.weight_names[index]])
            if plan.bias_names is not None:
                branch.input.append(plan.bias_names[index])
            del branch.output[:]
            branch.output.append(plan.branch_outputs[index])
            _replace_attribute(branch, "group", plan.group // 2)
            _replace_attribute(branch, "kernel_shape", [plan.kernel_end - plan.kernel_start + 1])
            _replace_attribute(branch, "pads", [plan.pad_begin, plan.pad_end])
            branches.append(branch)
        concat = helper.make_node(
            "Concat",
            list(plan.branch_outputs),
            list(plan.tail_slice.output),
            name=plan.concat_name,
            axis=1,
        )
        replacements[id(plan.conv)] = [split, *branches, concat]
        removed_node_ids.update((id(plan.conv), id(plan.tail_slice)))
        removable_initializers.update(plan.conv.input[1:])
        removable_initializers.update(plan.tail_slice.input[1:])
        regions.append(
            QuantizationRegion(
                branches=plan.branch_names,
                concat=plan.concat_name,
            )
        )
        if verbose:
            logger.info(
                "trim-split-grouped-conv: %s kernel [%d:%d], groups %d -> %d + %d",
                plan.conv.name,
                plan.kernel_start,
                plan.kernel_end + 1,
                plan.group,
                plan.group // 2,
                plan.group // 2,
            )

    rebuilt: list[NodeProto] = []
    for node in graph.node:
        if id(node) in replacements:
            rebuilt.extend(replacements[id(node)])
        elif id(node) not in removed_node_ids:
            rebuilt.append(node)
    del graph.node[:]
    graph.node.extend(rebuilt)
    graph.initializer.extend(new_initializers)
    referenced_tensors = _referenced_tensor_names(graph)
    kept_initializers = [
        initializer
        for initializer in graph.initializer
        if initializer.name not in removable_initializers or initializer.name in referenced_tensors
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    _set_region_hints(model, (*existing_regions, *regions))
    logger.info("SurgeryPipe: trim-split-grouped-conv: rewrote %d region(s)", len(plans))
    return model


__all__ = [
    "QUANTIZATION_REGION_HINT_KEY",
    "trim_split_grouped_convs",
]
