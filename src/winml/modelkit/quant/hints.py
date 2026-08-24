# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Consume optimizer-authored quantization region hints."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import onnx

from ..onnx import get_captured_tensor_names


QUANTIZATION_REGION_HINT_KEY = "winml.quantization.region_hints"
STANDARD_ONNX_DOMAINS = frozenset(("", "ai.onnx"))
QDQ_ONNX_DOMAINS = STANDARD_ONNX_DOMAINS | {"com.microsoft"}


@dataclass(frozen=True)
class QuantizationRegion:
    """One optimizer-authored Conv-to-Concat quantization region."""

    branches: tuple[str, str]
    concat: str


class QuantizationHintError(ValueError):
    """A quantization region hint is malformed or does not match the graph."""


def _parse_payload(payload: object) -> tuple[QuantizationRegion, ...]:
    if not isinstance(payload, dict) or set(payload) != {"version", "regions"}:
        raise QuantizationHintError(f"Invalid {QUANTIZATION_REGION_HINT_KEY} schema")
    if type(payload["version"]) is not int or payload["version"] != 1:
        raise QuantizationHintError(
            f"Unsupported {QUANTIZATION_REGION_HINT_KEY} version: {payload['version']!r}"
        )
    raw_regions = payload["regions"]
    if not isinstance(raw_regions, list) or not raw_regions:
        raise QuantizationHintError(f"Invalid {QUANTIZATION_REGION_HINT_KEY} regions")

    regions: list[QuantizationRegion] = []
    branch_names: set[str] = set()
    concat_names: set[str] = set()
    for index, value in enumerate(raw_regions):
        if not isinstance(value, dict) or set(value) != {"kind", "branches", "concat"}:
            raise QuantizationHintError(f"Invalid quantization region {index}")
        if value["kind"] != "conv_concat":
            raise QuantizationHintError(f"Unsupported quantization region kind: {value['kind']!r}")
        raw_branches = value["branches"]
        concat = value["concat"]
        if (
            not isinstance(raw_branches, list)
            or len(raw_branches) != 2
            or any(not isinstance(name, str) or not name for name in raw_branches)
            or len(set(raw_branches)) != 2
            or not isinstance(concat, str)
            or not concat
        ):
            raise QuantizationHintError(f"Invalid conv_concat region {index}")
        if branch_names.intersection(raw_branches) or concat in concat_names:
            raise QuantizationHintError(f"Duplicate node name in quantization region {index}")
        if concat in raw_branches:
            raise QuantizationHintError(f"Concat is also a branch in quantization region {index}")
        branch_names.update(raw_branches)
        concat_names.add(concat)
        regions.append(
            QuantizationRegion(branches=(raw_branches[0], raw_branches[1]), concat=concat)
        )
    if branch_names.intersection(concat_names):
        raise QuantizationHintError("A hinted node has both branch and Concat roles")
    return tuple(regions)


def parse_quantization_region_hints(
    model: onnx.ModelProto,
) -> tuple[QuantizationRegion, ...] | None:
    """Parse and strictly validate model-level quantization region hints."""
    entries = [item for item in model.metadata_props if item.key == QUANTIZATION_REGION_HINT_KEY]
    if not entries:
        return None
    if len(entries) != 1:
        raise QuantizationHintError(f"Duplicate {QUANTIZATION_REGION_HINT_KEY} metadata")
    try:
        payload = json.loads(entries[0].value)
    except (json.JSONDecodeError, TypeError) as error:
        raise QuantizationHintError(f"Invalid {QUANTIZATION_REGION_HINT_KEY} JSON") from error
    return _parse_payload(payload)


def serialize_quantization_region_hints(
    regions: tuple[QuantizationRegion, ...],
) -> str:
    """Validate and serialize quantization regions in the versioned schema."""
    payload = {
        "version": 1,
        "regions": [
            {
                "kind": "conv_concat",
                "branches": list(region.branches),
                "concat": region.concat,
            }
            for region in regions
        ],
    }
    _parse_payload(payload)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _one_node_index(
    indexes_by_name: dict[str, list[int]],
    name: str,
    *,
    role: str,
) -> int:
    indexes = indexes_by_name.get(name, [])
    if len(indexes) != 1:
        raise QuantizationHintError(
            f"Hinted {role} {name!r} must identify exactly one node; found {len(indexes)}"
        )
    return indexes[0]


def _one_output(node: onnx.NodeProto, *, context: str) -> str:
    if len(node.output) != 1 or not node.output[0]:
        raise QuantizationHintError(f"{context} must have exactly one output")
    return node.output[0]


def _one_consumer(
    consumers: dict[str, list[int]],
    output_name: str,
    *,
    context: str,
) -> int:
    indexes = consumers.get(output_name, [])
    if len(indexes) != 1:
        raise QuantizationHintError(
            f"{context} must have exactly one consumer; found {len(indexes)}"
        )
    return indexes[0]


def canonicalize_quantization_region_hints(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove exact hinted branch-output QDQ pairs from a copy of *model*."""
    regions = parse_quantization_region_hints(model)
    if regions is None:
        return model

    nodes = list(model.graph.node)
    indexes_by_name: dict[str, list[int]] = {}
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        indexes_by_name.setdefault(node.name, []).append(index)
        consumed_names = {input_name for input_name in node.input if input_name}
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                consumed_names.update(get_captured_tensor_names(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested_graph in attribute.graphs:
                    consumed_names.update(get_captured_tensor_names(nested_graph))
        for input_name in consumed_names:
            consumers.setdefault(input_name, []).append(index)
    graph_outputs = {value.name for value in model.graph.output}

    remove_indexes: set[int] = set()
    rewrites_by_concat: dict[int, dict[str, str]] = {}
    for region in regions:
        concat_index = _one_node_index(indexes_by_name, region.concat, role="Concat")
        concat = nodes[concat_index]
        if concat.op_type != "Concat":
            raise QuantizationHintError(f"Hinted Concat {region.concat!r} is {concat.op_type}")
        if concat.domain not in STANDARD_ONNX_DOMAINS:
            raise QuantizationHintError(
                f"Hinted Concat {region.concat!r} has unsupported domain {concat.domain!r}"
            )

        region_dq_outputs: list[str] = []
        region_rewrites: dict[str, str] = {}
        for branch_name in region.branches:
            branch_index = _one_node_index(indexes_by_name, branch_name, role="branch")
            branch = nodes[branch_index]
            if branch.op_type != "Conv":
                raise QuantizationHintError(f"Hinted branch {branch_name!r} is {branch.op_type}")
            if branch.domain not in STANDARD_ONNX_DOMAINS:
                raise QuantizationHintError(
                    f"Hinted branch {branch_name!r} has unsupported domain {branch.domain!r}"
                )
            branch_output = _one_output(branch, context=f"Hinted branch {branch_name!r}")
            if branch_output in graph_outputs:
                raise QuantizationHintError(
                    f"Hinted branch {branch_name!r} output is a graph output"
                )

            q_index = _one_consumer(
                consumers,
                branch_output,
                context=f"Hinted branch {branch_name!r} output",
            )
            q_node = nodes[q_index]
            if q_node.op_type != "QuantizeLinear":
                raise QuantizationHintError(
                    f"Hinted branch {branch_name!r} output consumer is {q_node.op_type}"
                )
            if q_node.domain not in QDQ_ONNX_DOMAINS:
                raise QuantizationHintError(
                    f"Branch {branch_name!r} QuantizeLinear has unsupported "
                    f"domain {q_node.domain!r}"
                )
            if not q_node.input or q_node.input[0] != branch_output:
                raise QuantizationHintError(
                    f"Hinted branch {branch_name!r} output is not the QuantizeLinear data input"
                )
            q_output = _one_output(q_node, context=f"Branch {branch_name!r} QuantizeLinear")
            if q_output in graph_outputs:
                raise QuantizationHintError(
                    f"Branch {branch_name!r} QuantizeLinear output is a graph output"
                )

            dq_index = _one_consumer(
                consumers,
                q_output,
                context=f"Branch {branch_name!r} QuantizeLinear output",
            )
            dq_node = nodes[dq_index]
            if dq_node.op_type != "DequantizeLinear":
                raise QuantizationHintError(
                    f"Branch {branch_name!r} QuantizeLinear consumer is {dq_node.op_type}"
                )
            if dq_node.domain not in QDQ_ONNX_DOMAINS:
                raise QuantizationHintError(
                    f"Branch {branch_name!r} DequantizeLinear has unsupported "
                    f"domain {dq_node.domain!r}"
                )
            if dq_node.domain != q_node.domain:
                raise QuantizationHintError(
                    f"Branch {branch_name!r} Q/DQ nodes use different domains"
                )
            if not dq_node.input or dq_node.input[0] != q_output:
                raise QuantizationHintError(
                    f"Branch {branch_name!r} QuantizeLinear output is not the "
                    "DequantizeLinear data input"
                )
            dq_output = _one_output(
                dq_node,
                context=f"Branch {branch_name!r} DequantizeLinear",
            )
            if dq_output in graph_outputs:
                raise QuantizationHintError(
                    f"Branch {branch_name!r} DequantizeLinear output is a graph output"
                )
            downstream_index = _one_consumer(
                consumers,
                dq_output,
                context=f"Branch {branch_name!r} DequantizeLinear output",
            )
            if downstream_index != concat_index:
                raise QuantizationHintError(
                    f"Branch {branch_name!r} does not feed hinted Concat {region.concat!r}"
                )
            if q_index in remove_indexes or dq_index in remove_indexes:
                raise QuantizationHintError(
                    f"Quantization nodes are shared by branch {branch_name!r}"
                )
            remove_indexes.update((q_index, dq_index))
            region_dq_outputs.append(dq_output)
            region_rewrites[dq_output] = branch_output

        if list(concat.input) != region_dq_outputs:
            raise QuantizationHintError(
                f"Hinted Concat {region.concat!r} inputs do not exactly match its branches"
            )
        rewrites_by_concat[concat_index] = region_rewrites

    result = onnx.ModelProto()
    result.CopyFrom(model)
    for concat_index, rewrites in rewrites_by_concat.items():
        concat = result.graph.node[concat_index]
        for input_index, input_name in enumerate(concat.input):
            concat.input[input_index] = rewrites[input_name]
    kept_nodes = [
        node for index, node in enumerate(result.graph.node) if index not in remove_indexes
    ]
    del result.graph.node[:]
    result.graph.node.extend(kept_nodes)

    kept_metadata = [
        copy.deepcopy(item)
        for item in result.metadata_props
        if item.key != QUANTIZATION_REGION_HINT_KEY
    ]
    del result.metadata_props[:]
    result.metadata_props.extend(kept_metadata)
    return result


__all__ = [
    "QDQ_ONNX_DOMAINS",
    "QUANTIZATION_REGION_HINT_KEY",
    "STANDARD_ONNX_DOMAINS",
    "QuantizationHintError",
    "QuantizationRegion",
    "canonicalize_quantization_region_hints",
    "parse_quantization_region_hints",
    "serialize_quantization_region_hints",
]
