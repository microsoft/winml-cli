# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pattern matching with wildcard support."""

from typing import Any


def match_pattern_with_wildcards(pattern: dict[str, Any], attributes: dict[str, Any]) -> bool:
    """Match pattern attributes, treating ``"*"`` as any value."""
    for attr_name, expected_value in pattern.items():
        if expected_value != "*" and attributes.get(attr_name) != expected_value:
            return False
    return True


def match_type_vars_with_wildcards(pattern: dict[str, str], types: dict[str, str]) -> bool:
    """Match type variables with wildcard and pipe-separated alternatives."""
    for type_var, expected_type in pattern.items():
        if expected_type == "*":
            continue
        actual_type = types.get(type_var)
        if "|" in expected_type:
            allowed_types = [value.strip() for value in expected_type.split("|")]
            if actual_type not in allowed_types:
                return False
        elif actual_type != expected_type:
            return False
    return True


def match_version_with_wildcard(actual_version: str, rule_version: str) -> bool:
    """Match an exact version or the universal ``"*"`` rule."""
    return rule_version == "*" or actual_version == rule_version
