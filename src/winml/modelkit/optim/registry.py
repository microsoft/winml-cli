"""Capability definitions for ONNX Runtime graph optimization.

This module provides capability definitions for managing optimization settings.
Capabilities are simple data classes that define optimization options with
CLI generation and validation support.

The module supports:
- Boolean flags (enable/disable)
- Integer parameters with validation
- Choice parameters with fixed options
- Automatic CLI flag generation
- Configuration validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CapabilityCategory(Enum):
    """Capability category for organization and filtering."""

    GELU = "gelu"
    ATTENTION = "attention"
    LAYER_NORM = "layer_norm"
    ACTIVATION = "activation"
    MATMUL = "matmul"
    CONVOLUTION = "convolution"
    GEMM = "gemm"
    ELIMINATION = "elimination"
    GRAPH = "graph"
    LAYOUT = "layout"
    MISC = "misc"
    CONTROL = "control"
    SURGERY = "surgery"
    REWRITE = "rewrite"


@dataclass(frozen=True)
class CapabilityDef:
    """Base capability definition.

    Capabilities are immutable data classes that define optimization options.
    Each pipe owns its capabilities directly via class-level dicts.

    Attributes:
        name: Kebab-case capability name (e.g., "gelu-fusion")
        ort_name: ONNX Runtime optimization name(s). Can be a single string or tuple
                  of strings for optimizers with level-specific names
        description: Human-readable description
        category: Capability category for organization
        default: Default value for the capability
        depends_on: Tuple of capability names this capability depends on
        conflicts_with: Tuple of capability names that conflict with this one
        ep_constraint: Tuple of supported Execution Providers, or None if universal.
                       Examples: ("CPU",), ("CPU", "CUDA", "DML"), None (all EPs)
    """

    name: str
    ort_name: str | tuple[str, ...] | None
    description: str
    category: CapabilityCategory
    default: Any
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    conflicts_with: tuple[str, ...] = field(default_factory=tuple)
    ep_constraint: tuple[str, ...] | None = None

    @property
    def config_name(self) -> str:
        """Get configuration key name (same as capability name)."""
        return self.name

    @property
    def python_name(self) -> str:
        """Convert kebab-case name to snake_case Python identifier."""
        return self.name.replace("-", "_")


@dataclass(frozen=True)
class BoolCapability(CapabilityDef):
    """Boolean capability (enable/disable flag).

    Generates --enable-X and --disable-X CLI flags for toggling.
    Default must be a boolean value.
    """

    default: bool = False

    def __post_init__(self) -> None:
        """Validate boolean default."""
        if not isinstance(self.default, bool):
            msg = f"BoolCapability '{self.name}' must have bool default, got {type(self.default)}"
            raise TypeError(msg)

    def cli_flags(self) -> tuple[str, str]:
        """Generate enable/disable CLI flag pair.

        Returns:
            Tuple of (enable_flag, disable_flag)
        """
        return (f"--enable-{self.name}", f"--disable-{self.name}")


@dataclass(frozen=True)
class IntCapability(CapabilityDef):
    """Integer capability with range validation.

    Generates --X=<value> CLI flag with validation.
    Default must be within [min_value, max_value].
    """

    default: int = 0
    min_value: int = 0
    max_value: int = 100

    def __post_init__(self) -> None:
        """Validate integer constraints."""
        if not isinstance(self.default, int):
            msg = f"IntCapability '{self.name}' must have int default, got {type(self.default)}"
            raise TypeError(msg)
        if not (self.min_value <= self.default <= self.max_value):
            msg = (
                f"IntCapability '{self.name}' default {self.default} "
                f"outside range [{self.min_value}, {self.max_value}]"
            )
            raise ValueError(msg)

    def cli_flag(self) -> str:
        """Generate CLI flag with value syntax.

        Returns:
            Flag string like "--opt-level=<value>"
        """
        return f"--{self.name}=<value>"


@dataclass(frozen=True)
class ChoiceCapability(CapabilityDef):
    """Choice capability with fixed options.

    Generates --X=<choice> CLI flag with validation.
    Default must be one of the valid choices.
    """

    default: str = ""
    choices: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate choice constraints."""
        if not isinstance(self.default, str):
            msg = f"ChoiceCapability '{self.name}' must have str default, got {type(self.default)}"
            raise TypeError(msg)
        if not self.choices:
            msg = f"ChoiceCapability '{self.name}' must have at least one choice"
            raise ValueError(msg)
        if self.default not in self.choices:
            msg = (
                f"ChoiceCapability '{self.name}' default '{self.default}' "
                f"not in choices {self.choices}"
            )
            raise ValueError(msg)

    def cli_flag(self) -> str:
        """Generate CLI flag with choice syntax.

        Returns:
            Flag string like "--layout={NCHW,NHWC}"
        """
        choices_str = ",".join(self.choices)
        return f"--{self.name}={{{choices_str}}}"


# Standalone validation functions


def defaults(capabilities: dict[str, CapabilityDef]) -> dict[str, Any]:
    """Get default values for all capabilities.

    Args:
        capabilities: Dictionary of capability definitions

    Returns:
        Dictionary mapping capability names to default values
    """
    return {name: cap.default for name, cap in capabilities.items()}


def validate(config: dict[str, Any], capabilities: dict[str, CapabilityDef]) -> list[str]:
    """Validate configuration against capability definitions.

    Args:
        config: Configuration dictionary to validate
        capabilities: Dictionary of capability definitions

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for key, value in config.items():
        cap = capabilities.get(key)
        if cap is None:
            errors.append(f"Unknown capability '{key}'")
            continue

        # Type-specific validation
        if isinstance(cap, BoolCapability):
            if not isinstance(value, bool):
                errors.append(f"Capability '{key}' expects bool, got {type(value).__name__}")
        elif isinstance(cap, IntCapability):
            if not isinstance(value, int):
                errors.append(f"Capability '{key}' expects int, got {type(value).__name__}")
            elif not (cap.min_value <= value <= cap.max_value):
                errors.append(
                    f"Capability '{key}' value {value} outside range "
                    f"[{cap.min_value}, {cap.max_value}]"
                )
        elif isinstance(cap, ChoiceCapability):
            if not isinstance(value, str):
                errors.append(f"Capability '{key}' expects str, got {type(value).__name__}")
            elif value not in cap.choices:
                errors.append(f"Capability '{key}' value '{value}' not in choices {cap.choices}")

    return errors


def validate_dependencies(
    config: dict[str, Any], capabilities: dict[str, CapabilityDef]
) -> list[str]:
    """Validate capability dependencies and conflicts.

    Checks that:
    1. All enabled capabilities have their dependencies satisfied
    2. No conflicting capabilities are both enabled

    Args:
        config: Configuration dictionary with capability values
        capabilities: Dictionary of capability definitions

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for name, cap in capabilities.items():
        # Get effective enabled state
        if isinstance(cap, BoolCapability):
            enabled = config.get(name, cap.default)
        else:
            enabled = name in config

        if not enabled:
            continue

        # Check dependencies are satisfied
        for dep_name in cap.depends_on:
            dep_cap = capabilities.get(dep_name)
            if dep_cap is None:
                errors.append(f"'{name}' depends on unknown capability '{dep_name}'")
            elif isinstance(dep_cap, BoolCapability):
                dep_enabled = config.get(dep_name, dep_cap.default)
                if not dep_enabled:
                    errors.append(f"'{name}' requires '{dep_name}' to be enabled")
            else:
                dep_enabled = dep_name in config
                if not dep_enabled:
                    errors.append(f"'{name}' requires '{dep_name}' to be enabled")

        # Check no conflicts
        for conflict_name in cap.conflicts_with:
            conflict_cap = capabilities.get(conflict_name)
            if conflict_cap is None:
                errors.append(f"'{name}' conflicts with unknown capability '{conflict_name}'")
            elif isinstance(conflict_cap, BoolCapability):
                conflict_enabled = config.get(conflict_name, conflict_cap.default)
                if conflict_enabled:
                    errors.append(f"'{name}' conflicts with '{conflict_name}'")
            else:
                conflict_enabled = conflict_name in config
                if conflict_enabled:
                    errors.append(f"'{name}' conflicts with '{conflict_name}'")

    return errors


def auto_enable_dependencies(
    config: dict[str, Any], capabilities: dict[str, CapabilityDef]
) -> dict[str, Any]:
    """Automatically enable required dependencies.

    Iteratively enables dependencies until no more changes are needed.

    Args:
        config: Configuration dictionary with capability values
        capabilities: Dictionary of capability definitions

    Returns:
        Updated config with dependencies enabled
    """
    result = config.copy()
    changed = True

    while changed:
        changed = False
        for name, cap in capabilities.items():
            # Get effective enabled state
            if isinstance(cap, BoolCapability):
                enabled = result.get(name, cap.default)
            else:
                enabled = name in result

            if not enabled:
                continue

            for dep_name in cap.depends_on:
                if not result.get(dep_name, False):
                    result[dep_name] = True
                    changed = True

    return result
