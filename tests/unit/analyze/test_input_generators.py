# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for InputGenerator classes.

Tests verify:
- All operator input generators are registered
- Each generator can be instantiated with opset22
- Each generator's input validation passes
- Registry functions work correctly
- filter_kwargs_by_opset filters correctly for operators with/without variadic inputs
"""

import numpy as np
import pytest
from onnx.defs import SchemaError

from winml.modelkit.onnx import ONNXDomain
from winml.modelkit.pattern.op_input_gen.op_input_gen import (
    get_registered_operators,
    get_runtime_checker_op,
)


class TestInputGeneratorRegistry:
    """Test unary operator input generator registration."""

    def test_all_operators_registered(self) -> None:
        """Test that all operators are registered."""
        # Verify count
        assert len(get_registered_operators()) == 117

    def test_get_runtime_checker_op(self) -> None:
        """Test retrieving operator generators by name."""
        registered_ops = get_registered_operators()
        for op_name in registered_ops:
            generator_class = get_runtime_checker_op(op_name)
            assert generator_class is not None
            assert generator_class.op_name == op_name

    def test_get_unregistered_operator_raises_error(self) -> None:
        """Test that retrieving unregistered operator raises KeyError."""
        with pytest.raises(KeyError, match="No OpInputGenerator registered"):
            get_runtime_checker_op("NonexistentOperator")


class TestInputGeneratorValidation:
    """Test validation of unary operator input generators."""

    @pytest.mark.parametrize("op_name", get_registered_operators())
    @pytest.mark.parametrize("opset_version", [17, 22, 23])
    def test_operator_validation(self, op_name: str, opset_version: int) -> None:
        """Test that each operator's input generator validates successfully.

        Args:
            op_name: Name of the operator to test
            opset_version: The ONNX opset version to test with
        """
        # Get OpSchema for this operator and opset version
        domain = ONNXDomain.AI_ONNX
        try:
            schema = domain.get_op_schema(op_name, opset_version)
        except SchemaError:
            # Operator doesn't exist in this opset version, skip
            return

        generator_class = get_runtime_checker_op(op_name)
        gen = generator_class(schema)

        # Should not raise any exceptions
        gen.validate_inputs()

    @pytest.mark.parametrize("op_name", get_registered_operators())
    def test_operator_instantiation(self, op_name: str) -> None:
        """Test that each operator's input generator can be instantiated.

        Args:
            op_name: Name of the operator to test
        """
        # Opset 23 is a superset of 22 in terms of ops
        domain = ONNXDomain.AI_ONNX
        schema = domain.get_op_schema(op_name, 23)

        generator_class = get_runtime_checker_op(op_name)
        gen = generator_class(schema)

        assert gen.op_name == op_name
        assert gen.schema == schema


class TestFilterKwargsByOpset:
    """Test filter_kwargs_by_opset for operators with and without variadic inputs."""

    @pytest.fixture()
    def abs_generator(self):
        """Create an Abs operator generator (no variadic inputs)."""
        domain = ONNXDomain.AI_ONNX
        schema = domain.get_op_schema("Abs", 22)
        generator_class = get_runtime_checker_op("Abs")
        return generator_class(schema)

    @pytest.fixture()
    def concat_generator(self):
        """Create a Concat operator generator (has variadic input 'inputs')."""
        domain = ONNXDomain.AI_ONNX
        schema = domain.get_op_schema("Concat", 22)
        generator_class = get_runtime_checker_op("Concat")
        return generator_class(schema)

    def test_non_variadic_keeps_supported_keys(self, abs_generator):
        """Supported input/attribute keys are kept."""
        kwargs = {"X": 1}
        result = abs_generator.filter_kwargs_by_opset(kwargs)
        assert result == {"X": 1}

    def test_non_variadic_removes_unsupported_keys(self, abs_generator):
        """Keys not in the schema are removed."""
        kwargs = {"X": 1, "unsupported_key": 2, "another_bad": 3}
        result = abs_generator.filter_kwargs_by_opset(kwargs)
        assert result == {"X": 1}

    def test_non_variadic_empty_kwargs(self, abs_generator):
        """Empty dict returns empty dict."""
        assert abs_generator.filter_kwargs_by_opset({}) == {}

    def test_non_variadic_all_unsupported(self, abs_generator):
        """All unsupported keys returns empty dict."""
        kwargs = {"foo": 1, "bar": 2}
        result = abs_generator.filter_kwargs_by_opset(kwargs)
        assert result == {}

    def test_variadic_keeps_supported_keys(self, concat_generator):
        """Regular input/attribute keys are kept for variadic ops."""
        kwargs = {"axis": 0}
        result = concat_generator.filter_kwargs_by_opset(kwargs)
        assert result == {"axis": 0}

    def test_variadic_keeps_variadic_input_name(self, concat_generator):
        """The variadic input name itself is kept."""
        kwargs = {"inputs": [1, 2], "axis": 0}
        result = concat_generator.filter_kwargs_by_opset(kwargs)
        assert result == {"inputs": [1, 2], "axis": 0}

    def test_variadic_keeps_expanded_variadic_names(self, concat_generator):
        """Expanded variadic names like 'inputs__0', 'inputs__1' are kept."""
        kwargs = {"inputs__0": 1, "inputs__1": 2, "inputs__2": 3, "axis": 0}
        result = concat_generator.filter_kwargs_by_opset(kwargs)
        assert result == {"inputs__0": 1, "inputs__1": 2, "inputs__2": 3, "axis": 0}

    def test_variadic_removes_unsupported_keys(self, concat_generator):
        """Unsupported keys are removed for variadic ops."""
        kwargs = {"inputs__0": 1, "axis": 0, "bad_key": 99}
        result = concat_generator.filter_kwargs_by_opset(kwargs)
        assert result == {"inputs__0": 1, "axis": 0}

    def test_variadic_mixed_keys(self, concat_generator):
        """Mix of variadic name, expanded names, attributes, and unsupported keys."""
        kwargs = {
            "inputs": [1, 2],
            "inputs__0": 1,
            "inputs__1": 2,
            "axis": 0,
            "unknown": "x",
            "extra": 42,
        }
        result = concat_generator.filter_kwargs_by_opset(kwargs)
        assert "inputs" in result
        assert "inputs__0" in result
        assert "inputs__1" in result
        assert "axis" in result
        assert "unknown" not in result
        assert "extra" not in result

    def test_variadic_empty_kwargs(self, concat_generator):
        """Empty dict returns empty dict for variadic ops."""
        assert concat_generator.filter_kwargs_by_opset({}) == {}


class TestSqueezeDeriveProperties:
    """Test Squeeze derive_properties for input and attribute axes."""

    @pytest.fixture()
    def squeeze_generator_opset11(self):
        """Create a Squeeze generator for an older opset that uses attr axes."""
        domain = ONNXDomain.AI_ONNX
        schema = domain.get_op_schema("Squeeze", 11)
        generator_class = get_runtime_checker_op("Squeeze")
        return generator_class(schema)

    @pytest.fixture()
    def squeeze_generator_opset22(self):
        """Create a Squeeze generator for a newer opset that uses input axes."""
        domain = ONNXDomain.AI_ONNX
        schema = domain.get_op_schema("Squeeze", 22)
        generator_class = get_runtime_checker_op("Squeeze")
        return generator_class(schema)

    def test_squeeze_derive_properties_supports_attr_axes(self, squeeze_generator_opset11):
        """Older opsets should derive properties from attr_axes without KeyError."""
        result = squeeze_generator_opset11.derive_properties(
            {"data_shape": (1, 2, 1), "attr_axes": [0, -1]}
        )

        assert result["data_dim"] == 3
        assert result["axes_is_empty"] is False
        assert result["axes_len_greater_than_one"] is True
        assert result["data_single_entry"] is False

    def test_squeeze_derive_properties_supports_axes_value(self, squeeze_generator_opset22):
        """Newer opsets should continue deriving properties from axes_value."""
        result = squeeze_generator_opset22.derive_properties(
            {"data_shape": (1,), "axes_value": np.array([0], dtype=np.int64)}
        )

        assert result["data_dim"] == 1
        assert result["axes_is_empty"] is False
        assert result["axes_len_greater_than_one"] is False
        assert result["data_single_entry"] is True


class TestSplitInfiniteProperties:
    """Regression tests for Split matching properties."""

    @pytest.fixture()
    def split_generator_opset12(self):
        """Create a Split generator for an opset that uses attr_split."""
        domain = ONNXDomain.AI_ONNX
        schema = domain.get_op_schema("Split", 12)
        generator_class = get_runtime_checker_op("Split")
        return generator_class(schema)

    def test_split_attr_split_is_treated_as_infinite_property(self, split_generator_opset12):
        """Older-opset Split should not require an exact attr_split tuple match."""
        infinite_properties = split_generator_opset12.get_infinite_property_names()

        assert "attr_split" in infinite_properties
