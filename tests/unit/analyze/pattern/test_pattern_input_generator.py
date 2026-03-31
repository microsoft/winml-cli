# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for PatternInputGenerator classes.

Tests verify:
- All pattern input generators are registered
- Each generator can be instantiated
- Each generator's input validation passes
- Registry functions work correctly
"""

import pytest

from winml.modelkit.pattern import (
    get_pattern_input_generator,
    get_registered_pattern_input_generators,
)

from .conftest import (
    PATTERNS_REQUIRING_NEWER_OPSET,
    SKIP_VALIDATION_PATTERNS,
    TEST_DOMAIN_VERSIONS,
)


class TestPatternInputGeneratorRegistry:
    """Test pattern input generator registration."""

    def test_all_patterns_registered(self) -> None:
        """Test that all patterns are registered."""
        registered = get_registered_pattern_input_generators()
        assert len(registered) == 16

    def test_get_pattern_input_generator(self) -> None:
        """Test retrieving pattern generators by name."""
        registered_patterns = get_registered_pattern_input_generators()
        for pattern_name in registered_patterns:
            generator_class = get_pattern_input_generator(pattern_name)
            assert generator_class is not None
            assert generator_class.registration_name == pattern_name

    def test_get_unregistered_pattern_raises_error(self) -> None:
        """Test that retrieving unregistered pattern raises KeyError."""
        with pytest.raises(KeyError, match="No PatternInputGenerator registered"):
            get_pattern_input_generator("NonexistentPattern")


class TestPatternInputGeneratorValidation:
    """Test validation of pattern input generators."""

    @pytest.mark.parametrize("pattern_name", get_registered_pattern_input_generators())
    def test_pattern_validation(self, pattern_name: str) -> None:
        """Test that each pattern's input generator validates successfully."""
        if pattern_name in SKIP_VALIDATION_PATTERNS:
            pytest.skip(f"Skipping validation for {pattern_name} (known runtime limitation)")

        generator_class = get_pattern_input_generator(pattern_name)
        domain_versions = PATTERNS_REQUIRING_NEWER_OPSET.get(pattern_name, TEST_DOMAIN_VERSIONS)
        gen = generator_class(domain_versions=domain_versions)
        gen.validate_inputs()

    @pytest.mark.parametrize("pattern_name", get_registered_pattern_input_generators())
    def test_pattern_instantiation(self, pattern_name: str) -> None:
        """Test that each pattern's input generator can be instantiated."""
        generator_class = get_pattern_input_generator(pattern_name)
        domain_versions = PATTERNS_REQUIRING_NEWER_OPSET.get(pattern_name, TEST_DOMAIN_VERSIONS)
        gen = generator_class(domain_versions=domain_versions)
        assert gen.registration_name == pattern_name
