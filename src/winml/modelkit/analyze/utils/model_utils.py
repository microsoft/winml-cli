# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Model utilities for analyze.

Functions that are shared with modelkit.pattern are re-exported from there.
The node_to_pattern_match function stays here as it uses OperatorPattern (analyzer-specific).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from onnx import NodeProto

from winml.modelkit.pattern.models import OperatorPattern, PatternType

# Re-export shared utilities from winml.modelkit.pattern.utils
from winml.modelkit.pattern.utils import (  # noqa: F401
    DUMMY_FLOAT,
    collect_initializers,
    collect_valueinfo_dict,
    dtype_from_tensorproto_enum,
    get_attribute_proto_value,
    get_op_input_properties,
    get_op_since_version,
    make_hashable,
    shape_and_dtype_from_valueinfo,
)


if TYPE_CHECKING:
    from winml.modelkit.pattern.match import PatternMatchResult


def node_to_pattern_match(
    node: NodeProto,
) -> PatternMatchResult:
    """Convert an ONNX node to a PatternMatchResult object.

    Creates an operator-level PatternMatchResult from a single ONNX node.

    Args:
        node: ONNX NodeProto to convert

    Returns:
        PatternMatchResult object representing the operator pattern

    Process:
        1. Extract op_type and namespace from node
        2. Create OperatorPattern with pattern_id (e.g., "OP/ai.onnx/Conv")
        3. Extract node attributes, dtypes, and shapes
        4. Create and return PatternMatchResult with the pattern and node info

    Note:
        This is useful for converting individual operators to patterns
        for runtime support checking.
    """
    from winml.modelkit.pattern.match import PatternMatchResult, SkeletonMatchResult

    # Detect namespace
    namespace = "ai.onnx"
    if node.domain:
        if node.domain == "com.microsoft":
            namespace = "com.microsoft"
        elif node.domain != "":
            namespace = node.domain

    # Create pattern_id
    pattern_id = f"OP/{namespace}/{node.op_type}"

    # Create OperatorPattern
    operator_pattern = OperatorPattern(
        pattern_id=pattern_id,
        pattern_type=PatternType.OPERATOR,
        namespace=namespace,
        op_type=node.op_type,
        description=f"{node.op_type} operator",
    )

    # Create minimal SkeletonMatchResult for API compatibility
    # This is a single-node match without full skeleton topology
    skeleton_result = SkeletonMatchResult(
        pattern=operator_pattern,
        matched_nodes=[node],
        matcher=None,  # type: ignore
        inputs=[],
        output="",
        removable=False,
    )

    # Create PatternMatchResult
    return PatternMatchResult(
        skeleton_match_result=skeleton_result,
        schema_input_to_value={},
        schema_output_to_value={},
        type_param_to_type={},
        attributes={},
        input_infos={},
    )
