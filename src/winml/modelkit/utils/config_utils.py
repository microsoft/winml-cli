# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Configuration utilities for WinML pipeline configs.

Provides recursive merge functionality for dataclasses and dict-like configs.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, TypeVar


T = TypeVar("T")


def merge_config(base: T, overrides: dict[str, Any] | T | None) -> T:
    """Recursively merge overrides into a base config.

    Works with dataclasses, dict-like configs, and nested structures.
    Only fields explicitly provided in overrides are applied.

    Args:
        base: Base configuration (dataclass or dict-like)
        overrides: User overrides as dict or config object

    Returns:
        New config with overrides applied (base is not modified)

    Example:
        from winml.modelkit.config import WinMLBuildConfig, merge_config

        base = WinMLBuildConfig()
        merged = merge_config(base, {
            "quant": {"samples": 100, "weight_type": "int8"},
            "export": {"opset_version": 18},
        })

        # Or merge two configs
        merged = merge_config(base, other_config)
    """
    if overrides is None:
        return base

    # Convert overrides to dict if it's a config object
    if hasattr(overrides, "to_dict"):
        overrides = overrides.to_dict()
    elif dataclasses.is_dataclass(overrides) and not isinstance(overrides, type):
        overrides = dataclasses.asdict(overrides)
    elif isinstance(overrides, dict):
        overrides = dict(overrides)  # Copy to avoid mutation
    else:
        raise TypeError(f"overrides must be dict or config, got {type(overrides)}")

    return _merge_into(base, overrides)


def _merge_into(base: T, overrides: dict[str, Any]) -> T:
    """Internal recursive merge implementation."""
    if dataclasses.is_dataclass(base) and not isinstance(base, type):
        # Handle dataclass
        return _merge_dataclass(base, overrides)

    if isinstance(base, dict):
        # Handle dict-like config (e.g., WinMLOptimizationConfig)
        result = type(base)(**base)  # type: ignore[call-arg]
        result.update(overrides)
        return result  # type: ignore[return-value]

    # Primitive or unknown type - just return override
    return base


def _merge_dataclass(base: T, overrides: dict[str, Any]) -> T:
    """Merge overrides into a dataclass, handling nested configs."""
    # Get current field values
    current = {}
    for f in dataclasses.fields(base):
        current[f.name] = getattr(base, f.name)

    # Apply overrides recursively
    for key, value in overrides.items():
        if key not in current:
            continue  # Skip unknown fields

        current_val = current[key]

        if value is None:
            # Explicit None - set to None
            current[key] = None
        elif current_val is None:
            # Base is None, override has value - use override
            # Try to construct from dict if nested config
            field_type = _get_field_type(base, key)
            if field_type and isinstance(value, dict):
                if hasattr(field_type, "from_dict"):
                    current[key] = field_type.from_dict(value)
                elif dataclasses.is_dataclass(field_type):
                    current[key] = field_type(**value)
                else:
                    current[key] = value
            else:
                current[key] = value
        elif isinstance(value, dict) and (
            dataclasses.is_dataclass(current_val) or isinstance(current_val, dict)
        ):
            # Nested config - recurse
            current[key] = _merge_into(current_val, value)
        elif isinstance(value, list) and isinstance(current_val, list):
            # List - replace entirely (no merge)
            current[key] = value
        else:
            # Primitive - override
            current[key] = value

    # Create new instance
    return type(base)(**current)  # type: ignore[return-value]


def _get_field_type(obj: Any, field_name: str) -> type | None:
    """Get the type annotation for a dataclass field.

    Handles both runtime type objects and PEP 563 string annotations
    (from ``from __future__ import annotations``).
    """
    if not dataclasses.is_dataclass(obj):
        return None

    # Resolve string annotations to actual types
    try:
        hints = typing.get_type_hints(type(obj))
    except (NameError, AttributeError):
        hints = {}

    resolved = hints.get(field_name)
    if resolved is None:
        # Fallback to raw field.type if get_type_hints fails
        for f in dataclasses.fields(obj):
            if f.name == field_name:
                resolved = f.type
                break
        if resolved is None:
            return None

    # Handle Optional[X] / X | None / Union[X, None] -> X
    args = getattr(resolved, "__args__", ())
    if args and type(None) in args:
        # It's an Optional/Union with None — extract the non-None type
        for arg in args:
            if arg is not type(None):
                return arg  # type: ignore[return-value]
    return resolved if isinstance(resolved, type) else None
