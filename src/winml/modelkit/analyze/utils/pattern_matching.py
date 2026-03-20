# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pattern matching with wildcard support."""

from typing import Any


def match_pattern_with_wildcards(pattern: dict[str, Any], attributes: dict[str, Any]) -> bool:
    """Match detected pattern attributes against rule with wildcard support.

    Implements universal wildcard matching where "*" in pattern matches any value.

    Args:
        pattern: Pattern/rule attributes (may contain "*" wildcards)
        attributes: Actual attributes from detected pattern

    Returns:
        True if pattern matches attributes, False otherwise

    Examples:
        >>> match_pattern_with_wildcards(
        ...     {"kernel_shape": [3, 3]}, {"kernel_shape": [3, 3]}
        ... )
        True
        >>> match_pattern_with_wildcards(
        ...     {"kernel_shape": "*"}, {"kernel_shape": [3, 3]}
        ... )
        True
        >>> match_pattern_with_wildcards(
        ...     {"kernel_shape": [3, 3]}, {"kernel_shape": [5, 5]}
        ... )
        False
    """
    # Iterate through each attribute in pattern
    for attr_name, expected_value in pattern.items():
        # Wildcard matches any value
        if expected_value == "*":
            continue

        # Get actual value from detected pattern
        actual_value = attributes.get(attr_name)

        # If attribute missing or doesn't match, fail
        if actual_value != expected_value:
            return False

    return True


def match_type_vars_with_wildcards(pattern: dict[str, str], types: dict[str, str]) -> bool:
    """Match detected type variables against pattern with wildcard support.

    Supports:
    - Exact match: "float32" matches only "float32"
    - Wildcard: "*" matches any type
    - Alternatives: "float32|float16" matches either "float32" or "float16"

    Args:
        pattern: Pattern type constraints (may contain "*" wildcards or "|"
            alternatives)
        types: Actual data types from detected pattern (e.g., {"T": "float32"})

    Returns:
        True if types match pattern, False otherwise

    Examples:
        >>> match_type_vars_with_wildcards({"T": "float32"}, {"T": "float32"})
        True
        >>> match_type_vars_with_wildcards({"T": "*"}, {"T": "float32"})
        True
        >>> match_type_vars_with_wildcards({"T": "float32|float16"}, {"T": "float16"})
        True
        >>> match_type_vars_with_wildcards({"T": "float32"}, {"T": "int8"})
        False
    """
    for type_var, expected_type in pattern.items():
        # Wildcard matches any type
        if expected_type == "*":
            continue

        actual_type = types.get(type_var)

        # Check if expected_type contains alternatives (pipe-separated)
        if "|" in expected_type:
            allowed_types = [t.strip() for t in expected_type.split("|")]
            if actual_type not in allowed_types:
                return False
        else:
            # Exact match
            if actual_type != expected_type:
                return False

    return True


def match_version_with_wildcard(actual_version: str, rule_version: str) -> bool:
    """Match version string with wildcard support.

    Args:
        actual_version: Actual version string
        rule_version: Rule version ("*" matches any)

    Returns:
        True if versions match, False otherwise

    Examples:
        >>> match_version_with_wildcard("2.3.1", "2.3.1")
        True
        >>> match_version_with_wildcard("2.3.1", "*")
        True
        >>> match_version_with_wildcard("2.3.1", "2.3.0")
        False
    """
    if rule_version == "*":
        return True
    return actual_version == rule_version
