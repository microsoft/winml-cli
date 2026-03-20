# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Unit tests for pattern_matching.py wildcard matching algorithms.

Tests verify:
- match_pattern_with_wildcards for attribute matching
- match_type_vars_with_wildcards for data type matching
- "*" wildcard support (matches any value)
- Exact match support
- Type alternatives with pipe-separated syntax (e.g., "float32|float16")
"""

from winml.modelkit.analyze.utils.pattern_matching import (
    match_pattern_with_wildcards,
    match_type_vars_with_wildcards,
)


class TestMatchPatternWithWildcards:
    """Test wildcard matching for operator attributes."""

    def test_exact_match_no_wildcards(self):
        """Test that exact values match without wildcards."""
        pattern = {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}
        attributes = {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}

        assert match_pattern_with_wildcards(pattern, attributes) is True

    def test_exact_mismatch(self):
        """Test that non-matching exact values return False."""
        pattern = {"kernel_shape": [3, 3]}
        attributes = {"kernel_shape": [5, 5]}

        assert match_pattern_with_wildcards(pattern, attributes) is False

    def test_wildcard_matches_any_value(self):
        """Test that '*' wildcard matches any value."""
        pattern = {"kernel_shape": "*", "pads": "*"}
        attributes = {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}

        assert match_pattern_with_wildcards(pattern, attributes) is True

        # Different values should also match
        attributes2 = {"kernel_shape": [5, 5], "pads": [0, 0, 0, 0]}
        assert match_pattern_with_wildcards(pattern, attributes2) is True

    def test_mixed_wildcard_and_exact(self):
        """Test pattern with both wildcards and exact values."""
        pattern = {"kernel_shape": "*", "pads": [1, 1, 1, 1], "strides": "*"}

        # Matches: kernel_shape wildcard, pads exact match, strides wildcard
        attributes1 = {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [1, 1]}
        assert match_pattern_with_wildcards(pattern, attributes1) is True

        # Mismatch: pads don't match
        attributes2 = {"kernel_shape": [3, 3], "pads": [0, 0, 0, 0], "strides": [1, 1]}
        assert match_pattern_with_wildcards(pattern, attributes2) is False

    def test_missing_attribute_in_pattern(self):
        """Test that attributes not in pattern are ignored."""
        pattern = {"kernel_shape": [3, 3]}
        attributes = {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "extra_attr": 42}

        # Only kernel_shape is checked, other attributes ignored
        assert match_pattern_with_wildcards(pattern, attributes) is True

    def test_missing_attribute_in_attributes(self):
        """Test that pattern keys missing in attributes return False."""
        pattern = {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}
        attributes = {"kernel_shape": [3, 3]}  # Missing 'pads'

        assert match_pattern_with_wildcards(pattern, attributes) is False

    def test_empty_pattern_matches_any(self):
        """Test that empty pattern matches any attributes."""
        pattern = {}
        attributes = {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]}

        assert match_pattern_with_wildcards(pattern, attributes) is True

    def test_nested_list_exact_match(self):
        """Test that nested list values match exactly."""
        pattern = {"kernel_shape": [3, 3], "dilations": [1, 1]}
        attributes = {"kernel_shape": [3, 3], "dilations": [1, 1]}

        assert match_pattern_with_wildcards(pattern, attributes) is True

        # Order matters
        attributes_wrong_order = {"kernel_shape": [3, 3], "dilations": [1, 2]}
        assert match_pattern_with_wildcards(pattern, attributes_wrong_order) is False

    def test_string_attribute_matching(self):
        """Test matching with string attribute values."""
        pattern = {"auto_pad": "SAME_UPPER", "mode": "*"}

        attributes1 = {"auto_pad": "SAME_UPPER", "mode": "CONSTANT"}
        assert match_pattern_with_wildcards(pattern, attributes1) is True

        attributes2 = {"auto_pad": "VALID", "mode": "CONSTANT"}
        assert match_pattern_with_wildcards(pattern, attributes2) is False

    def test_integer_attribute_matching(self):
        """Test matching with integer attribute values."""
        pattern = {"group": 1, "output_padding": "*"}

        attributes = {"group": 1, "output_padding": 0}
        assert match_pattern_with_wildcards(pattern, attributes) is True


class TestMatchTypeVarsWithWildcards:
    """Test wildcard matching for type variables."""

    def test_exact_type_match(self):
        """Test that exact type variables match."""
        pattern = {"T": "float32"}
        types = {"T": "float32"}

        assert match_type_vars_with_wildcards(pattern, types) is True

    def test_exact_type_mismatch(self):
        """Test that non-matching types return False."""
        pattern = {"T": "float32"}
        types = {"T": "int64"}

        assert match_type_vars_with_wildcards(pattern, types) is False

    def test_wildcard_matches_any_type(self):
        """Test that '*' wildcard matches any type."""
        pattern = {"T": "*"}

        types1 = {"T": "float32"}
        assert match_type_vars_with_wildcards(pattern, types1) is True

        types2 = {"T": "int64"}
        assert match_type_vars_with_wildcards(pattern, types2) is True

        types3 = {"T": "uint8"}
        assert match_type_vars_with_wildcards(pattern, types3) is True

    def test_mixed_wildcard_and_exact_types(self):
        """Test pattern with both wildcard and exact type constraints."""
        pattern = {"T": "*", "U": "float32"}

        # Matches: T wildcard, U exact
        types1 = {"T": "int64", "U": "float32"}
        assert match_type_vars_with_wildcards(pattern, types1) is True

        # Mismatch: U doesn't match
        types2 = {"T": "int64", "U": "int32"}
        assert match_type_vars_with_wildcards(pattern, types2) is False

    def test_missing_type_var_in_pattern(self):
        """Test that type vars not in pattern are ignored."""
        pattern = {"T": "float32"}
        types = {"T": "float32", "U": "int64"}

        # Only T is checked
        assert match_type_vars_with_wildcards(pattern, types) is True

    def test_missing_type_var_in_types(self):
        """Test that pattern type vars missing in types return False."""
        pattern = {"T": "float32", "U": "int64"}
        types = {"T": "float32"}  # Missing 'U'

        assert match_type_vars_with_wildcards(pattern, types) is False

    def test_empty_pattern_matches_any_types(self):
        """Test that empty pattern matches any types."""
        pattern = {}
        types = {"T": "float32", "U": "int64"}

        assert match_type_vars_with_wildcards(pattern, types) is True

    def test_multiple_type_vars_all_exact(self):
        """Test matching with multiple exact type variables."""
        pattern = {"T": "float32", "U": "int64", "V": "uint8"}
        types = {"T": "float32", "U": "int64", "V": "uint8"}

        assert match_type_vars_with_wildcards(pattern, types) is True

    def test_multiple_type_vars_all_wildcards(self):
        """Test matching with all wildcard type variables."""
        pattern = {"T": "*", "U": "*", "V": "*"}
        types = {"T": "float16", "U": "int32", "V": "bool"}

        assert match_type_vars_with_wildcards(pattern, types) is True

    def test_single_alternative_match(self):
        """Test that single type in alternatives matches."""
        pattern = {"T": "float32|float16"}

        types1 = {"T": "float32"}
        assert match_type_vars_with_wildcards(pattern, types1) is True

        types2 = {"T": "float16"}
        assert match_type_vars_with_wildcards(pattern, types2) is True

    def test_single_alternative_mismatch(self):
        """Test that type not in alternatives doesn't match."""
        pattern = {"T": "float32|float16"}
        types = {"T": "int64"}

        assert match_type_vars_with_wildcards(pattern, types) is False

    def test_multiple_alternatives(self):
        """Test pattern with many type alternatives."""
        pattern = {"T": "float32|float16|int32|int64"}

        assert match_type_vars_with_wildcards(pattern, {"T": "float32"}) is True
        assert match_type_vars_with_wildcards(pattern, {"T": "float16"}) is True
        assert match_type_vars_with_wildcards(pattern, {"T": "int32"}) is True
        assert match_type_vars_with_wildcards(pattern, {"T": "int64"}) is True
        assert match_type_vars_with_wildcards(pattern, {"T": "uint8"}) is False

    def test_mixed_alternatives_and_exact(self):
        """Test pattern with both alternatives and exact types."""
        pattern = {"T": "float32|float16", "U": "int64"}

        # T matches alternative, U exact match
        types1 = {"T": "float32", "U": "int64"}
        assert match_type_vars_with_wildcards(pattern, types1) is True

        types2 = {"T": "float16", "U": "int64"}
        assert match_type_vars_with_wildcards(pattern, types2) is True

        # T doesn't match any alternative
        types3 = {"T": "int32", "U": "int64"}
        assert match_type_vars_with_wildcards(pattern, types3) is False

        # U doesn't match
        types4 = {"T": "float32", "U": "int32"}
        assert match_type_vars_with_wildcards(pattern, types4) is False

    def test_mixed_alternatives_and_wildcard(self):
        """Test pattern with alternatives, wildcard, and exact types."""
        pattern = {"T": "float32|float16", "U": "*", "V": "int64"}

        types1 = {"T": "float32", "U": "anything", "V": "int64"}
        assert match_type_vars_with_wildcards(pattern, types1) is True

        types2 = {"T": "float16", "U": "something_else", "V": "int64"}
        assert match_type_vars_with_wildcards(pattern, types2) is True

    def test_alternatives_with_spaces(self):
        """Test that spaces in alternatives are handled correctly."""
        pattern = {"T": "float32 | float16 | int32"}

        # Should strip spaces and match
        types1 = {"T": "float32"}
        assert match_type_vars_with_wildcards(pattern, types1) is True

        types2 = {"T": "float16"}
        assert match_type_vars_with_wildcards(pattern, types2) is True

        types3 = {"T": "int32"}
        assert match_type_vars_with_wildcards(pattern, types3) is True

    def test_single_type_no_pipe(self):
        """Test that single types without pipe still work."""
        pattern = {"T": "float32"}

        types1 = {"T": "float32"}
        assert match_type_vars_with_wildcards(pattern, types1) is True

        types2 = {"T": "float16"}
        assert match_type_vars_with_wildcards(pattern, types2) is False

    def test_empty_alternative_components(self):
        """Test alternatives don't break with edge cases."""
        # This tests robustness, though such patterns shouldn't normally occur
        pattern = {"T": "float32|"}
        types = {"T": "float32"}
        # Should still match float32
        assert match_type_vars_with_wildcards(pattern, types) is True

    def test_conv_weight_type_alternatives(self):
        """Test Conv operator with flexible weight types."""
        # Conv might accept float32 or float16 weights
        pattern = {
            "T": "float32",  # Input type exact
            "W": "float32|float16|bfloat16",  # Weight type alternatives
        }

        types1 = {"T": "float32", "W": "float32"}
        assert match_type_vars_with_wildcards(pattern, types1) is True

        types2 = {"T": "float32", "W": "float16"}
        assert match_type_vars_with_wildcards(pattern, types2) is True

        types3 = {"T": "float32", "W": "bfloat16"}
        assert match_type_vars_with_wildcards(pattern, types3) is True

        # Input type mismatch
        types4 = {"T": "float16", "W": "float32"}
        assert match_type_vars_with_wildcards(pattern, types4) is False

        # Weight type not in alternatives
        types5 = {"T": "float32", "W": "int8"}
        assert match_type_vars_with_wildcards(pattern, types5) is False

    def test_quantized_operator_types(self):
        """Test quantized operators with restricted integer types."""
        pattern = {
            "T": "int8|uint8",  # Quantized types
            "T_Scale": "float32",  # Scale must be float32
        }

        types1 = {"T": "int8", "T_Scale": "float32"}
        assert match_type_vars_with_wildcards(pattern, types1) is True

        types2 = {"T": "uint8", "T_Scale": "float32"}
        assert match_type_vars_with_wildcards(pattern, types2) is True

        # Wrong quantized type
        types3 = {"T": "int32", "T_Scale": "float32"}
        assert match_type_vars_with_wildcards(pattern, types3) is False

    def test_any_float_any_int(self):
        """Test pattern accepting any float or any int."""
        pattern = {
            "T_float": "float32|float16|bfloat16",
            "T_int": "int8|int16|int32|int64|uint8|uint16|uint32|uint64",
        }

        types1 = {"T_float": "float32", "T_int": "int32"}
        assert match_type_vars_with_wildcards(pattern, types1) is True

        types2 = {"T_float": "bfloat16", "T_int": "uint8"}
        assert match_type_vars_with_wildcards(pattern, types2) is True

        # Mixed up
        types3 = {"T_float": "int32", "T_int": "float32"}
        assert match_type_vars_with_wildcards(pattern, types3) is False


class TestEdgeCases:
    """Test edge cases for pattern matching."""

    def test_none_values_in_pattern(self):
        """Test behavior with None values in pattern."""
        pattern = {"kernel_shape": None}
        attributes = {"kernel_shape": None}

        # None should match None exactly
        assert match_pattern_with_wildcards(pattern, attributes) is True

    def test_boolean_attribute_matching(self):
        """Test matching with boolean attribute values."""
        pattern = {"transA": True, "transB": False}
        attributes = {"transA": True, "transB": False}

        assert match_pattern_with_wildcards(pattern, attributes) is True

        attributes_mismatch = {"transA": False, "transB": False}
        assert match_pattern_with_wildcards(pattern, attributes_mismatch) is False

    def test_float_attribute_matching(self):
        """Test matching with float attribute values."""
        pattern = {"alpha": 1.0, "beta": 0.5}
        attributes = {"alpha": 1.0, "beta": 0.5}

        assert match_pattern_with_wildcards(pattern, attributes) is True
