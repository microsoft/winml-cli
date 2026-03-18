# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Operator Mapping Checkers (ADR-002)

Wrapper functions for operator mapping decision logic.
These checkers reuse the base constraint checkers from ADR-001 and adapt them
for the specific use case of selecting QNN operators based on ONNX node properties.

Only 2 checker types are needed for conditional operator mapping:
1. check_input_rank: Shape-based decisions (14 rules, 7 operators)
2. check_attribute: Attribute value comparison (5 rules, 2 operators)

Note: Previously included always_true and check_attribute_in, but these were
removed in final design optimization (see ADR-002).
"""

from typing import Any

# Import from same op_checker package
from .shape_checker import ShapeConstraintChecker
from .value_checker import ValueConstraintChecker


def check_input_rank(node, input_index: int, rank: int, valueinfo=None, **kwargs) -> bool:
    """Check if input at specified index has expected rank.

    Used to distinguish between 2D and 3D operator variants:
    - Conv → Conv2d (rank=4) vs Conv3d (rank=5)
    - Pooling → 2D variants (rank=4) vs 3D variants (rank=5)

    This function wraps ShapeConstraintChecker.check_exact_rank() from ADR-001
    for use in operator mapping context.

    Args:
        node: ONNX NodeProto or node object
        input_index: Which input to check (0-based)
        rank: Expected rank value
        valueinfo: Optional dict mapping tensor names to ValueInfoProto for ONNX NodeProto
        **kwargs: Additional parameters (ignored, for flexibility)

    Returns:
        True if input rank matches expected value, False otherwise

    Examples:
        >>> # Conv with 4D input (NCHW) → Conv2d
        >>> node = create_conv_node(input_shape=[1, 3, 224, 224])
        >>> check_input_rank(node, input_index=0, rank=4)
        True

        >>> # Conv with 5D input (NCDHW) → Conv3d
        >>> node = create_conv_node(input_shape=[1, 3, 16, 112, 112])
        >>> check_input_rank(node, input_index=0, rank=5)
        True
    """
    # For ONNX NodeProto, use valueinfo to get shape
    if hasattr(node, "input") and valueinfo:
        if input_index >= len(node.input):
            return False

        input_name = node.input[input_index]
        if input_name not in valueinfo:
            return False

        vi = valueinfo[input_name]
        if not vi.type.tensor_type.HasField("shape"):
            return False

        input_shape = [
            d.dim_value if d.HasField("dim_value") else -1 for d in vi.type.tensor_type.shape.dim
        ]

        success, _ = ShapeConstraintChecker.check_exact_rank(input_shape, rank)
        return success

    # Fallback to original logic for test/mock objects with inputs attribute
    if not hasattr(node, "inputs") or input_index >= len(node.inputs):
        return False

    # Get input shape
    input_tensor = node.inputs[input_index]
    if not hasattr(input_tensor, "shape"):
        return False

    input_shape = input_tensor.shape

    # Reuse shared shape checker from ADR-001
    success, _ = ShapeConstraintChecker.check_exact_rank(input_shape, rank)
    return success


def check_attribute(node, attribute_name: str, expected_value: Any | list[Any], **kwargs) -> bool:
    """Check if attribute matches expected value(s).

    Supports both single value comparison and multi-value matching.
    Uses ValueConstraintChecker.check_allowed_values() from ADR-001.

    Used for attribute-based operator selection:
    - Resize → ResizeNearestNeighbor (mode="nearest") vs ResizeBilinear (mode="linear")
    - Mod → ElementWiseFmod (fmod=1) vs ElementWiseMod (fmod=0)
    - DepthToSpace → DepthToSpace (mode="DCR")

    Args:
        node: ONNX NodeProto or node object
        attribute_name: Name of the attribute to check
        expected_value: Single value or list of acceptable values
        **kwargs: Additional parameters (ignored, for flexibility)

    Returns:
        True if attribute matches (single value) or is in list (multiple values)

    Examples:
        >>> # Single value check
        >>> node = create_resize_node(mode="nearest")
        >>> check_attribute(node, "mode", "nearest")
        True

        >>> # Multi-value check
        >>> node = create_resize_node(mode="cubic")
        >>> check_attribute(node, "mode", ["cubic", ""])
        True

        >>> # Missing attribute with default "" in expected_value list
        >>> node = create_resize_node()  # mode not set
        >>> check_attribute(node, "mode", ["cubic", ""])
        True
    """
    # Normalize to list for unified processing
    expected_values = expected_value if isinstance(expected_value, list) else [expected_value]

    # Handle missing attribute container
    if not hasattr(node, "attribute"):
        # If attribute not present, check if "" (empty/default) is acceptable
        success, _ = ValueConstraintChecker.check_allowed_values("", expected_values)
        return success

    # Find and extract attribute value
    for attr in node.attribute:
        if attr.name == attribute_name:
            # Extract attribute value based on type
            if hasattr(attr, "s"):  # String
                value = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
            elif hasattr(attr, "i"):  # Integer
                value = attr.i
            elif hasattr(attr, "f"):  # Float
                value = attr.f
            else:
                # Unknown type, check if default is acceptable
                success, _ = ValueConstraintChecker.check_allowed_values("", expected_values)
                return success

            # Use shared checker from ADR-001
            success, _ = ValueConstraintChecker.check_allowed_values(value, expected_values)
            return success

    # Attribute not found - check if default ("") is acceptable
    success, _ = ValueConstraintChecker.check_allowed_values("", expected_values)
    return success


# Checker function registry for operator mapping (only conditional mappings)
# Note: Direct mappings don't need checkers - they map 1:1 without conditions
CHECKER_REGISTRY = {
    "check_input_rank": check_input_rank,
    "check_attribute": check_attribute,
}


def get_qnn_op_for_onnx_node(onnx_node, mapping_config, valueinfo=None):
    """Determine QNN operator for an ONNX node using conditional checker framework.

    This is the main entry point for operator mapping resolution. It handles:
    1. Direct mappings (90.4% of operators) - simple 1:1 lookup
    2. Conditional mappings (9.6% of operators) - evaluate checker rules
    3. Unsupported operators - return error

    Args:
        onnx_node: ONNX NodeProto object
        mapping_config: Loaded JSON mapping configuration
        valueinfo: Optional dict mapping tensor names to ValueInfoProto for shape inference

    Returns:
        tuple: (qnn_op_name, rule_description) or (None, error_message)

    Examples:
        >>> # Direct mapping
        >>> node = create_relu_node()
        >>> get_qnn_op_for_onnx_node(node, mapping_config)
        ('Relu', 'Direct mapping')

        >>> # Conditional mapping (rank-based)
        >>> node = create_conv_node(input_shape=[1, 3, 224, 224])
        >>> get_qnn_op_for_onnx_node(node, mapping_config)
        ('Conv2d', 'NCHW format: [batch, channels, height, width]')

        >>> # Conditional mapping (attribute-based)
        >>> node = create_resize_node(mode="nearest")
        >>> get_qnn_op_for_onnx_node(node, mapping_config)
        ('ResizeNearestNeighbor', 'Nearest neighbor interpolation')
    """
    onnx_op = onnx_node.op_type

    # Check if operator exists in mapping
    if onnx_op not in mapping_config.get("mappings", {}):
        return None, f"ONNX operator '{onnx_op}' not found in mapping"

    op_mapping = mapping_config["mappings"][onnx_op]
    mapping_type = op_mapping.get("mapping_type", "unknown")

    # Handle unsupported operators
    if mapping_type == "unsupported":
        return None, f"ONNX operator '{onnx_op}' is not supported by QNN"

    # Handle direct mappings (91.3% of operators)
    if mapping_type == "direct":
        qnn_op = op_mapping.get("qnn_op")
        if not qnn_op:
            return None, f"Direct mapping for '{onnx_op}' missing qnn_op field"
        return qnn_op, "Direct mapping"

    # Handle conditional mappings (8.7% of operators)
    if mapping_type == "conditional":
        rules = op_mapping.get("rules", [])

        # Evaluate rules in order
        for rule in rules:
            checker_name = rule.get("checker")
            condition = rule.get("condition", {})
            qnn_op = rule.get("qnn_op")
            description = condition.get("description", "No description")

            # Get checker function
            checker_func = CHECKER_REGISTRY.get(checker_name)
            if not checker_func:
                continue

            # Evaluate condition
            try:
                # Pass valueinfo to checker if available
                condition_with_valueinfo = dict(condition)
                if valueinfo is not None:
                    condition_with_valueinfo["valueinfo"] = valueinfo

                if checker_func(onnx_node, **condition_with_valueinfo):
                    return qnn_op, description
            except Exception as e:
                # Log error but continue checking other rules
                print(f"Error evaluating checker '{checker_name}': {e}")
                continue

        # No rule matched
        return None, f"No matching rule found for ONNX operator '{onnx_op}'"

    return None, f"Unknown mapping type '{mapping_type}'"


__all__ = [
    "CHECKER_REGISTRY",
    "check_attribute",
    "check_input_rank",
    "get_qnn_op_for_onnx_node",
]
