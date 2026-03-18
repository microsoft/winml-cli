# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ORT Mapping Verification Tests.

This module verifies the correctness of ORT optimizer name mappings in the
capability registry. These tests ensure that all registered capabilities
have valid ORT names following correct conventions.

Following Cardinal Rules:
- CARDINAL RULE #1: No hardcoded model architectures
- CARDINAL RULE #2: All tests use pytest with code-generated results
- CARDINAL RULE #3: Tests must run and pass

Test Strategy:
    1. Verify all ort_names are non-empty strings
    2. Verify ort_names follow naming convention (PascalCase, no spaces)
    3. Verify no duplicate ort_names across capabilities
    4. Verify tuple ort_names are properly structured

ORT Naming Convention:
    - PascalCase: GeluFusionL2, MatMulAddFusion, LayerNormFusionL2
    - No spaces or special characters
    - May include level suffix: L1, L2
    - May include descriptive suffix: Fusion, Approximation
"""

from __future__ import annotations

import re

import pytest

# Trigger capability auto-discovery by importing capabilities
import winml.modelkit.optim.capabilities  # noqa: F401
from ..capabilities.conftest import get_all_ort_names


# Note: populated_registry fixture was removed (was no-op)


def test_all_ort_names_are_strings() -> None:
    """Verify all ORT names are non-empty strings.

    Each capability must have a valid ort_name that is:
    - Non-empty string
    - Not None
    - Not whitespace-only

    This test validates the basic data type and content of all ort_names.
    """
    all_ort_names = get_all_ort_names()

    # Verify we have at least some capabilities registered
    assert len(all_ort_names) > 0, "No ORT names found in capability registry"

    # Verify each name is a non-empty string
    for ort_name in all_ort_names:
        assert isinstance(ort_name, str), f"ORT name '{ort_name}' is not a string: {type(ort_name)}"

        assert len(ort_name) > 0, "Found empty ORT name"

        assert ort_name.strip() == ort_name, (
            f"ORT name '{ort_name}' has leading/trailing whitespace"
        )

        assert len(ort_name.strip()) > 0, f"ORT name '{ort_name}' is whitespace-only"


def test_ort_names_follow_convention() -> None:
    """Verify ORT names follow naming convention with no spaces.

    Valid ORT names should:
    - Start with uppercase letter
    - Use PascalCase or PascalCase_With_Underscores
    - Contain no spaces
    - May include numbers (e.g., L1, L2)
    - Underscores are allowed (e.g., MatMul_BatchNormalization_Fusion)

    Examples:
        Valid: GeluFusionL2, MatMulAddFusion, MatMul_BatchNormalization_Fusion
        Invalid: gelu_fusion, matmul-add-fusion, Gelu Fusion
    """
    all_ort_names = get_all_ort_names()

    # Pattern: starts with uppercase, alphanumeric and underscores allowed
    # (no hyphens, no spaces)
    valid_pattern = re.compile(r"^[A-Z][a-zA-Z0-9_]*$")

    invalid_names = []

    for ort_name in all_ort_names:
        # Check for spaces
        if " " in ort_name:
            invalid_names.append((ort_name, "contains spaces"))
            continue

        # Check for hyphens
        if "-" in ort_name:
            invalid_names.append((ort_name, "contains hyphens"))
            continue

        # Check valid pattern
        if not valid_pattern.match(ort_name):
            invalid_names.append((ort_name, "invalid format"))
            continue

    # Report all invalid names
    if invalid_names:
        error_msg = "Invalid ORT names found:\n"
        for name, reason in invalid_names:
            error_msg += f"  - '{name}': {reason}\n"
        pytest.fail(error_msg)


def test_no_duplicate_ort_names_within_pipe() -> None:
    """Verify no duplicate ORT names within a single pipe.

    Each ORT optimizer name should be unique within a single pipe.
    Duplicate names within a pipe would indicate:
    - Multiple capabilities trying to control same optimizer within one pipe
    - Copy-paste errors in capability definitions

    Note: Different pipes (e.g., ORTGraphPipe and ORTFusionPipe) may have
    capabilities with the same ORT name as they represent different optimization
    passes that may target similar patterns.

    This test ensures per-pipe integrity by checking for uniqueness within each pipe.
    """
    from winml.modelkit.optim.pipes import PIPES
    from winml.modelkit.optim.registry import BoolCapability

    all_duplicates: dict[str, list[tuple[str, int]]] = {}

    for pipe_class in PIPES:
        pipe_name = pipe_class.name

        # Collect ort_names from this pipe
        ort_names: list[str] = []
        for capability in pipe_class.capabilities.values():
            if not isinstance(capability, BoolCapability):
                continue

            if isinstance(capability.ort_name, tuple):
                ort_names.extend([name for name in capability.ort_name if name])
            elif isinstance(capability.ort_name, str) and capability.ort_name:
                ort_names.append(capability.ort_name)

        # Count occurrences
        name_counts: dict[str, int] = {}
        for name in ort_names:
            name_counts[name] = name_counts.get(name, 0) + 1

        # Find duplicates within this pipe
        duplicates = {name: count for name, count in name_counts.items() if count > 1}

        if duplicates:
            all_duplicates[pipe_name] = list(duplicates.items())

    # Report all duplicates
    if all_duplicates:
        error_msg = "Duplicate ORT names found within pipes:\n"
        for pipe_name, dups in all_duplicates.items():
            error_msg += f"  {pipe_name}:\n"
            for name, count in dups:
                error_msg += f"    - '{name}': appears {count} times\n"
        pytest.fail(error_msg)


def test_ort_names_registry_integrity() -> None:
    """Verify ORT name registry integrity and structure.

    This test performs comprehensive validation of capabilities in all pipes:
    - All BoolCapability instances have ort_name attribute
    - ort_name is string, tuple of strings, or None (for custom implementations)
    - Tuple ort_names are properly structured
    - No empty tuples
    """
    from winml.modelkit.optim.pipes import PIPES
    from winml.modelkit.optim.registry import BoolCapability

    # Collect all capabilities from all pipes
    all_capabilities: dict[str, BoolCapability] = {
        cap_name: capability
        for pipe_class in PIPES
        for cap_name, capability in pipe_class.capabilities.items()
        if isinstance(capability, BoolCapability)
    }

    # Verify we have capabilities registered
    assert len(all_capabilities) > 0, "No capabilities found in pipes"

    invalid_capabilities = []

    for cap_name, capability in all_capabilities.items():
        # Check ort_name attribute exists
        if not hasattr(capability, "ort_name"):
            invalid_capabilities.append((cap_name, "missing ort_name attribute"))
            continue

        ort_name = capability.ort_name

        # None is valid for custom implementations (e.g., SurgeryPipe capabilities)
        if ort_name is None:
            continue

        # Check ort_name is string or tuple
        if isinstance(ort_name, str):
            # Validate single string
            if not ort_name or not ort_name.strip():
                invalid_capabilities.append((cap_name, f"empty ort_name: '{ort_name}'"))
        elif isinstance(ort_name, tuple):
            # Validate tuple structure
            if len(ort_name) == 0:
                invalid_capabilities.append((cap_name, "empty ort_name tuple"))
            else:
                for idx, name in enumerate(ort_name):
                    if not isinstance(name, str):
                        invalid_capabilities.append(
                            (cap_name, f"ort_name[{idx}] is not string: {type(name)}")
                        )
                    elif not name or not name.strip():
                        invalid_capabilities.append((cap_name, f"ort_name[{idx}] is empty"))
        else:
            invalid_capabilities.append(
                (cap_name, f"ort_name has invalid type: {type(ort_name)}")
            )

    # Report all invalid capabilities
    if invalid_capabilities:
        error_msg = "Invalid capability ort_names found:\n"
        for cap_name, reason in invalid_capabilities:
            error_msg += f"  - '{cap_name}': {reason}\n"
        pytest.fail(error_msg)


def test_level_suffixes_are_valid() -> None:
    """Verify level suffixes (L1, L2) are valid when present.

    Some ORT optimizers have level-specific variants:
    - L1: Level 1 optimization
    - L2: Level 2 optimization

    This test ensures level suffixes at the end of names are properly formatted.
    Names like "LayerNormFusion" or "SkipLayerNormFusion" are valid as the "L"
    is part of a word, not a level indicator.
    """
    all_ort_names = get_all_ort_names()

    # Pattern for level suffix at end: must end with L followed by digit(s)
    level_suffix_pattern = re.compile(r"L\d+$")

    for ort_name in all_ort_names:
        # Only validate if name ends with L followed by characters and doesn't
        # match the valid pattern (L followed by digits)
        match = re.search(r"L\d*$", ort_name)
        if match and not level_suffix_pattern.search(ort_name):
            suffix = match.group(0)
            pytest.fail(
                f"ORT name '{ort_name}' has invalid level suffix. "
                f"Expected format: L1, L2, etc. Got: {suffix}"
            )
