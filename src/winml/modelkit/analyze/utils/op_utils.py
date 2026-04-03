# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Utility functions for operator check result management."""

import hashlib
import json
from typing import Any

import numpy as np
import onnx
from google.protobuf import json_format

from ...pattern.op_input_gen import normalize_constraint_dict


def compute_case_signature(case: dict, *, namespace: str) -> str:
    """Compute a signature for a test case based on its content.

    The signature is used to match test cases across different runs,
    allowing delta detection when the input generator changes.

    Args:
        case: Test case dictionary containing type_vars, attrs, input_constraints, etc.

    Returns:
        A string signature that uniquely identifies the test case.
    """
    # Extract the key fields that define a test case
    sig_parts = []

    if namespace:
        # Namespacing keeps case_index stable per output file when signatures collide across files
        sig_parts.append(f"ns:{namespace}")

    def _safe_dump(obj: Any) -> str:
        def _default(o: Any):
            if isinstance(o, onnx.TensorProto):
                return json.loads(json_format.MessageToJson(o))
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.generic):
                return o.item()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        return json.dumps(obj, sort_keys=True, default=_default)

    def _is_empty_top_level(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (dict, list, tuple, set)):
            return len(value) == 0
        return False

    # Type variables (e.g., T=FLOAT)
    if "type_vars" in case:
        type_vars = case["type_vars"]
        sig_parts.append(f"types:{_safe_dump(type_vars)}")

    # Attributes
    if "attrs" in case:
        attrs = case["attrs"]
        if not _is_empty_top_level(attrs):
            sig_parts.append(f"attrs:{_safe_dump(attrs)}")

    # Input constraints (shapes/values)
    if "input_constraints" in case:
        constraints = {
            k: normalize_constraint_dict(v) if isinstance(v, dict) else v
            for k, v in case["input_constraints"].items()
        }
        sig_parts.append(f"inputs:{_safe_dump(constraints)}")

    # Input is constant flags
    if "input_is_constant" in case:
        is_const = case["input_is_constant"]
        sig_parts.append(f"const:{_safe_dump(is_const)}")

    # Dynamic axes configuration
    if "dynamic_axes" in case:
        dynamic_axes = case["dynamic_axes"]
        if not _is_empty_top_level(dynamic_axes):
            sig_parts.append(f"dynamic:{_safe_dump(dynamic_axes)}")

    # QDQ configuration: include only when present to keep non-QDQ signatures stable.
    if "qdq_types" in case:
        qdq_types = case["qdq_types"]
        if not _is_empty_top_level(qdq_types):
            sig_parts.append(f"qdq:{_safe_dump(qdq_types)}")

    return "|".join(sig_parts)


def hash_case_signature(signature: str) -> str:
    """Return a stable hash value for a case signature."""
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()
