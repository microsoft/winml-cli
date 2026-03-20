# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Fix QDQ node dtype info after ORT quantization.

ORT's quantize() inserts scale and zero_point tensors as initializers but
does not always populate their type info in graph.input/value_info entries.
This module repairs those entries using the initializer's data_type as
ground truth.

The fix is necessary because ORT's SymbolicShapeInference directly accesses
``known_vi_[name].type.tensor_type.elem_type`` for QDQ inputs — if that
returns 0 (UNDEFINED), shape inference fails with
"Incomplete symbolic shape inference".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import onnx
from onnx import TensorProto, helper

from ..compiler import QDQ_OP_TYPES


logger = logging.getLogger(__name__)


# Defaults when initializer is not available
_DEFAULT_SCALE_TYPE = TensorProto.FLOAT
_DEFAULT_ZERO_POINT_TYPE = TensorProto.UINT8


@dataclass
class QdqFixResult:
    """Result of QDQ dtype fix operation."""

    inputs_fixed: int = 0
    value_info_fixed: int = 0
    value_info_added: int = 0
    tensors_examined: int = 0
    warnings: list[str] = field(default_factory=list)


def fix_qdq_dtype_info(model: onnx.ModelProto) -> QdqFixResult:
    """Fix UNDEFINED dtype on QDQ node scale/zero_point tensors.

    Modifies the model **in-place**:
    - Fixes graph.input entries with UNDEFINED elem_type
    - Fixes value_info entries with UNDEFINED elem_type
    - Adds missing value_info entries for tensors not found anywhere

    Type resolution priority:
    1. Initializer data_type (ground truth — the actual data is always correct)
    2. Defaults: FLOAT for scale, UINT8 for zero_point

    Args:
        model: ONNX ModelProto to fix (modified in-place).

    Returns:
        QdqFixResult with counts of fixes applied.
    """
    result = QdqFixResult()
    graph = model.graph

    # Step 1: Collect all scale/zp tensor names from QDQ nodes,
    # tracking whether each is a scale or zero_point
    qdq_params = _collect_qdq_params(graph)
    result.tensors_examined = len(qdq_params)

    if not qdq_params:
        logger.debug("No QDQ parameters found — nothing to fix")
        return result

    # Step 2: Build initializer lookup: name -> (data_type, dims)
    init_lookup = {init.name: (init.data_type, list(init.dims)) for init in graph.initializer}

    # Step 3: Determine correct type for each QDQ parameter
    type_map: dict[str, tuple[int, list[int]]] = {}
    for name, is_scale in qdq_params.items():
        dtype, dims = _resolve_type(name, init_lookup, is_scale=is_scale)
        type_map[name] = (dtype, dims)

    # Step 4: Fix graph.input entries
    for inp in graph.input:
        if inp.name not in type_map:
            continue
        if _has_undefined_type(inp):
            dtype, dims = type_map[inp.name]
            new_vi = helper.make_tensor_value_info(inp.name, dtype, dims or None)
            inp.CopyFrom(new_vi)
            result.inputs_fixed += 1
            logger.debug(
                "Fixed graph input '%s': UNDEFINED -> %s",
                inp.name,
                TensorProto.DataType.Name(dtype),
            )

    # Step 5: Fix value_info entries
    for vi in graph.value_info:
        if vi.name not in type_map:
            continue
        if _has_undefined_type(vi):
            dtype, dims = type_map[vi.name]
            new_vi = helper.make_tensor_value_info(vi.name, dtype, dims or None)
            vi.CopyFrom(new_vi)
            result.value_info_fixed += 1
            logger.debug(
                "Fixed value_info '%s': UNDEFINED -> %s",
                vi.name,
                TensorProto.DataType.Name(dtype),
            )

    # Step 6: Add missing value_info for tensors not in inputs/value_info/initializer
    existing_names = {inp.name for inp in graph.input}
    existing_names |= {vi.name for vi in graph.value_info}
    init_names = set(init_lookup.keys())

    for name, (dtype, dims) in type_map.items():
        if name not in existing_names and name not in init_names:
            new_vi = helper.make_tensor_value_info(name, dtype, dims or None)
            graph.value_info.append(new_vi)
            result.value_info_added += 1
            logger.debug(
                "Added value_info for '%s': %s",
                name,
                TensorProto.DataType.Name(dtype),
            )

    total_fixes = result.inputs_fixed + result.value_info_fixed + result.value_info_added
    if total_fixes > 0:
        logger.info(
            "QDQ dtype fix: %d inputs fixed, %d value_info fixed, %d value_info added",
            result.inputs_fixed,
            result.value_info_fixed,
            result.value_info_added,
        )
    else:
        logger.debug(
            "QDQ dtype fix: no fixes needed (%d tensors examined)",
            result.tensors_examined,
        )

    return result


def _collect_qdq_params(graph: onnx.GraphProto) -> dict[str, bool]:
    """Collect scale and zero_point tensor names from QDQ nodes.

    Args:
        graph: ONNX GraphProto.

    Returns:
        Dict mapping tensor name -> is_scale (True for scale, False for zero_point).
        If a tensor is used as both (unlikely but possible), scale takes precedence.
    """
    params: dict[str, bool] = {}

    for node in graph.node:
        if node.op_type not in QDQ_OP_TYPES:
            continue

        # input[1] = scale (required)
        if len(node.input) > 1 and node.input[1]:
            params.setdefault(node.input[1], True)

        # input[2] = zero_point (optional)
        if len(node.input) > 2 and node.input[2]:
            params.setdefault(node.input[2], False)

    return params


def _resolve_type(
    tensor_name: str,
    init_lookup: dict[str, tuple[int, list[int]]],
    *,
    is_scale: bool,
) -> tuple[int, list[int]]:
    """Resolve the correct dtype and dims for a QDQ parameter tensor.

    Priority:
    1. Initializer data_type + dims (ground truth)
    2. Default: FLOAT for scale, UINT8 for zero_point

    Args:
        tensor_name: Name of the tensor.
        init_lookup: Initializer name -> (data_type, dims).
        is_scale: True if this is a scale tensor, False for zero_point.

    Returns:
        Tuple of (onnx_elem_type, dims).
    """
    if tensor_name in init_lookup:
        return init_lookup[tensor_name]

    default_type = _DEFAULT_SCALE_TYPE if is_scale else _DEFAULT_ZERO_POINT_TYPE
    return default_type, []


def _has_undefined_type(vi: onnx.ValueInfoProto) -> bool:
    """Check if a ValueInfoProto has UNDEFINED elem_type.

    Args:
        vi: An ONNX ValueInfoProto.

    Returns:
        True if elem_type is UNDEFINED or type info is missing entirely.
    """
    if not vi.type.HasField("tensor_type"):
        return True
    return vi.type.tensor_type.elem_type == TensorProto.UNDEFINED
