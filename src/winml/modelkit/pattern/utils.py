# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared utility functions for pattern matching and validation.

Combines utilities from:
- modelkit.analyze.pattern.pattern_utils
- modelkit.analyze.utils.model_utils (pattern-relevant functions)
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from google.protobuf import json_format
from onnx import AttributeProto, ModelProto, TensorProto, ValueInfoProto
from onnx.defs import OpSchema

from ..onnx import ONNXDomain, SupportedONNXType


# ---------------------------------------------------------------------------
# From pattern_utils.py
# ---------------------------------------------------------------------------


def validate_scale_bias_shape_for_axis(
    scale_or_bias_shape: tuple,
    input_shape: tuple,
    axis: int,
) -> bool:
    """Validate that Scale/B shape is compatible with the given axis for broadcasting.

    For ONNX Mul/Add broadcasting to work correctly at the specified axis:
    - If 1D (normalized_dim,): only valid when axis is the last dimension (broadcast aligns right)
    - If multi-dim: the only non-1 dimension must be at the axis position when right-aligned

    Args:
        scale_or_bias_shape: Shape of Scale or B tensor.
        input_shape: Shape of input X tensor.
        axis: Normalization axis (can be negative).

    Returns:
        True if shape is compatible, False otherwise.
    """
    rank = len(input_shape)
    # Normalize negative axis
    normalized_axis = axis if axis >= 0 else rank + axis

    if normalized_axis < 0 or normalized_axis >= rank:
        return False

    normalized_dim = input_shape[normalized_axis]

    # Handle symbolic dimensions - can't validate, assume valid
    if isinstance(normalized_dim, str):
        return True

    sb_shape = scale_or_bias_shape
    sb_rank = len(sb_shape)

    # 1D shape: only valid when axis is last (broadcast aligns from right)
    if sb_rank == 1:
        if sb_shape[0] != normalized_dim:
            return False
        return normalized_axis == rank - 1

    # Multi-dim: must have exactly one non-1 dimension at axis position (right-aligned)
    if sb_rank > rank:
        return False

    non_one_positions = [i for i, d in enumerate(sb_shape) if d != 1]
    if len(non_one_positions) != 1:
        return False

    non_one_pos_in_sb = non_one_positions[0]
    non_one_pos_in_input = non_one_pos_in_sb + (rank - sb_rank)

    if non_one_pos_in_input != normalized_axis:
        return False

    return sb_shape[non_one_pos_in_sb] == normalized_dim


def get_tensor_shape(tensor_name: str, matcher: Any) -> tuple | None:
    """Get tensor shape from constants or shape inference.

    Args:
        tensor_name: Name of the tensor in the ONNX graph.
        matcher: PatternMatcher instance with tensor_values and tensor_shapes.

    Returns:
        Shape tuple if available, None otherwise.
    """
    if tensor_name in matcher.tensor_values:
        return matcher.tensor_values[tensor_name].shape
    return matcher.tensor_shapes.get(tensor_name)


# ---------------------------------------------------------------------------
# From model_utils.py (pattern-relevant functions)
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    TensorProto.FLOAT: "FLOAT",
    TensorProto.UINT4: "UINT4",
    TensorProto.UINT8: "UINT8",
    TensorProto.INT4: "INT4",
    TensorProto.INT8: "INT8",
    TensorProto.UINT16: "UINT16",
    TensorProto.INT16: "INT16",
    TensorProto.INT32: "INT32",
    TensorProto.INT64: "INT64",
    TensorProto.STRING: "STRING",
    TensorProto.BOOL: "BOOL",
    TensorProto.FLOAT16: "FLOAT16",
    TensorProto.DOUBLE: "DOUBLE",
    TensorProto.COMPLEX64: "COMPLEX64",
    TensorProto.COMPLEX128: "COMPLEX128",
    TensorProto.BFLOAT16: "BFLOAT16",
}


def dtype_from_tensorproto_enum(tp: int) -> str:
    """Convert a TensorProto data type enum to its string name."""
    return DTYPE_MAP.get(tp, f"unknown({tp})")


def shape_and_dtype_from_valueinfo(vi: ValueInfoProto) -> tuple[list | None, str | None]:
    """Extract shape and dtype from a ValueInfoProto."""
    if not vi.type.HasField("tensor_type"):
        return (None, None)
    tt = vi.type.tensor_type
    dtype = dtype_from_tensorproto_enum(tt.elem_type)

    shape = []
    if tt.HasField("shape"):
        for d in tt.shape.dim:
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            elif d.HasField("dim_param"):
                shape.append(d.dim_param)
            else:
                shape.append(None)
    else:
        shape = None
    return (tuple(shape) if shape is not None else None, dtype)


def collect_valueinfo_dict(model: ModelProto) -> dict[str, ValueInfoProto]:
    """Collect all ValueInfoProto entries from a model into a dict keyed by name."""
    vid = {}
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        vid[vi.name] = vi
    return vid


def collect_initializers(model: ModelProto) -> dict[str, TensorProto]:
    """Collect all initializer tensors from a model into a dict keyed by name."""
    return {init.name: init for init in model.graph.initializer}


def get_op_input_properties(schema: OpSchema):
    """Get operator input properties from OpSchema.

    Args:
        schema: OpSchema object for the operator

    Returns:
        Tuple of (input_names, variadic_input_name, attribute_names, type_annotations)
    """
    op_input_names = []
    type_annotations = {}
    op_variadic_input_name = None

    # Extract inputs from schema
    supported_onnx_types = {x.onnx_type for x in SupportedONNXType}
    for input_param in schema.inputs:
        # legacy compatibility: onnxscript type string or TypeVar_OpName
        if input_param.type_str in supported_onnx_types:
            type_str = SupportedONNXType.from_onnx_type(input_param.type_str).annotation
        else:
            type_str = f"{input_param.type_str}_{schema.name}"
        if input_param.option == OpSchema.FormalParameterOption.Variadic:
            if op_variadic_input_name is not None:
                raise ValueError(
                    f"Multiple variadic inputs not supported: "
                    f"{op_variadic_input_name} and {input_param.name}"
                )
            op_variadic_input_name = input_param.name
            type_annotations[input_param.name] = type_str
        elif input_param.option == OpSchema.FormalParameterOption.Optional:
            op_input_names.append(input_param.name)
            type_annotations[input_param.name] = f"Optional[{type_str}]"  # legacy compatibility
        else:
            op_input_names.append(input_param.name)
            type_annotations[input_param.name] = type_str

        for name, attribute in schema.attributes.items():
            type_annotations[name] = attribute.type.name

    op_attribute_names = list(schema.attributes.keys())

    return op_input_names, op_variadic_input_name, op_attribute_names, type_annotations


def get_op_since_version(op_name: str, model_opset_version: int, op_domain: str) -> int:
    """Get the since_version for an operator using onnx.defs.get_schema.

    Args:
        op_name: Name of the operator
        model_opset_version: The model's opset version to look up
        op_domain: The domain of the operator (empty string for ai.onnx)

    Returns:
        The since_version of the operator
    """
    schema = ONNXDomain.from_str(op_domain).get_op_schema(op_name, model_opset_version)
    return schema.since_version


# TODO: wrap in class

DUMMY_FLOAT = -999.9


def make_hashable(value: Any, replace_float_with_dummy: bool = True) -> Any:
    """Convert value to hashable form, replacing floats with DUMMY_FLOAT.

    Recursively processes:
    - Floats -> DUMMY_FLOAT
    - Lists/Tuples -> Tuple of processed elements
    - Dicts -> Tuple of sorted (key, processed_value) items
    - ndarrays -> Tuple of processed elements (converted via tolist())
    - Others -> Original value
    """
    # Fast path: type identity checks avoid isinstance MRO traversal
    val_type = type(value)
    if val_type is int or val_type is str or val_type is bool or value is None:
        return value
    if val_type is float:
        return DUMMY_FLOAT if replace_float_with_dummy else value
    if val_type is list or val_type is tuple:
        return _make_hashable_sequence(value, replace_float_with_dummy)
    if val_type is dict:
        return tuple(
            sorted([(k, make_hashable(v, replace_float_with_dummy)) for k, v in value.items()])
        )
    if isinstance(value, np.ndarray):
        return make_hashable(value.tolist(), replace_float_with_dummy)
    if isinstance(value, np.floating):
        return DUMMY_FLOAT if replace_float_with_dummy else float(value)
    return value


_SIMPLE_TYPES = frozenset({int, str, bool, type(None)})


def _make_hashable_sequence(value: list | tuple, replace_float_with_dummy: bool) -> tuple:
    """Fast-path for converting sequences: avoids recursion when elements are simple types."""
    # Fast path: if all elements are simple types, just convert to tuple directly
    needs_conversion = False
    for x in value:
        if type(x) not in _SIMPLE_TYPES:
            needs_conversion = True
            break
    if not needs_conversion:
        return tuple(value) if type(value) is list else value
    return tuple([make_hashable(x, replace_float_with_dummy) for x in value])


def get_attribute_proto_value(a: Any, replace_float_with_dummy: bool = True) -> Any:
    """Extract a Python value from an AttributeProto."""
    # if floating, replace with DUMMY_FLOAT to effectively ignore float values in matching
    if a.type == AttributeProto.FLOAT:
        return make_hashable(a.f, replace_float_with_dummy)
    if a.type == AttributeProto.INT:
        return a.i
    if a.type == AttributeProto.STRING:
        return a.s.decode("utf-8")
    if a.type == AttributeProto.TENSOR:
        return make_hashable(json.loads(json_format.MessageToJson(a.t)), replace_float_with_dummy)
    if a.type == AttributeProto.INTS:
        return tuple(a.ints)
    if a.type == AttributeProto.FLOATS:
        return tuple(make_hashable(f, replace_float_with_dummy) for f in a.floats)
    raise ValueError(f"Unsupported attribute type {a.type} for attribute {a.name}.")
