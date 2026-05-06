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


def _parse_registry_key(registry_key: str) -> tuple[ONNXDomain, str]:
    """Parse a registry key into (domain, op_type).

    "Gelu"                → (ONNXDomain.AI_ONNX, "Gelu")
    "com.microsoft::Gelu" → (ONNXDomain.COM_MICROSOFT, "Gelu")
    """
    if "::" in registry_key:
        domain_str, op_type = registry_key.split("::", 1)
        return ONNXDomain.from_str(domain_str), op_type
    return ONNXDomain.AI_ONNX, registry_key


class TestInputGeneratorRegistry:
    """Test unary operator input generator registration."""

    def test_all_operators_registered(self) -> None:
        """Test that all operators are registered."""
        # Verify count
        assert len(get_registered_operators()) == 119

    def test_get_runtime_checker_op(self) -> None:
        """Test retrieving operator generators by name."""
        registered_ops = get_registered_operators()
        for registry_key in registered_ops:
            _, op_type = _parse_registry_key(registry_key)
            generator_class = get_runtime_checker_op(registry_key)
            assert generator_class is not None
            assert generator_class.op_name == op_type

    def test_get_unregistered_operator_raises_error(self) -> None:
        """Test that retrieving unregistered operator raises KeyError."""
        with pytest.raises(KeyError, match="No OpInputGenerator registered"):
            get_runtime_checker_op("NonexistentOperator")


class TestInputGeneratorValidation:
    """Test validation of unary operator input generators."""

    @pytest.mark.parametrize("op_name", get_registered_operators())
    @pytest.mark.parametrize("opset_version", [1, 17, 22, 23])
    def test_operator_validation(self, op_name: str, opset_version: int) -> None:
        """Test that each operator's input generator instantiates and validates successfully.

        Args:
            op_name: Registry key (e.g. "Relu" or "com.microsoft::Gelu")
            opset_version: The ONNX opset version to test with
        """
        domain, op_type = _parse_registry_key(op_name)
        # ai.onnx opset 1 schemas predate current input signatures; generators target modern opsets
        if domain == ONNXDomain.AI_ONNX and opset_version == 1:
            return
        try:
            schema = domain.get_op_schema(op_type, opset_version)
        except SchemaError:
            # Operator doesn't exist in this opset version, skip
            return

        generator_class = get_runtime_checker_op(op_name)
        gen = generator_class(schema)

        assert gen.op_name == op_type
        assert gen.schema == schema

        if domain == ONNXDomain.AI_ONNX and op_type == "LpNormalization" and opset_version >= 22:
            # LpNormalization >= 22 is not supported by onnnxruntime 1.23
            return

        gen.validate_inputs()


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


class TestSplitDerivedProperties:
    """Regression tests for Split derive_properties coverage."""

    @pytest.fixture()
    def split_generator_opset18(self):
        """Create a Split generator for opset with num_outputs support."""
        domain = ONNXDomain.AI_ONNX
        schema = domain.get_op_schema("Split", 18)
        generator_class = get_runtime_checker_op("Split")
        return generator_class(schema)

    @pytest.fixture()
    def split_generator_opset12(self):
        """Create a Split generator for older opset with attr_split."""
        domain = ONNXDomain.AI_ONNX
        schema = domain.get_op_schema("Split", 12)
        generator_class = get_runtime_checker_op("Split")
        return generator_class(schema)

    def test_split_axis_flags_are_derived(
        self, split_generator_opset18, split_generator_opset12
    ):
        """Split should expose finite axis-sign and axis-zero flags for rule matching."""
        common = {
            "input_shape": (6, 3, 4),
            "n_outputs": 2,
        }

        result_axis_zero = split_generator_opset18.derive_properties(
            {
                **common,
                "attr_axis": 0,
                "split_value": np.array([3, 3], dtype=np.int64),
            }
        )
        result_axis_negative = split_generator_opset18.derive_properties(
            {
                **common,
                "attr_axis": -1,
                "split_value": np.array([2, 2], dtype=np.int64),
            }
        )
        result_axis_positive_nonzero = split_generator_opset18.derive_properties(
            {
                **common,
                "attr_axis": 2,
                "attr_num_outputs": 3,
            }
        )
        result_attr_split_path = split_generator_opset12.derive_properties(
            {
                **common,
                "attr_axis": 0,
                "attr_split": (1, 1, 1),
            }
        )
        result_missing_axis = split_generator_opset18.derive_properties(
            {
                **common,
            }
        )

        assert result_axis_zero["axis_is_negative"] is False
        assert result_axis_zero["axis_is_zero"] is True
        assert result_axis_zero["num_outputs"] == 2

        assert result_axis_negative["axis_is_negative"] is True
        assert result_axis_negative["axis_is_zero"] is False
        assert result_axis_negative["num_outputs"] == 2

        assert result_axis_positive_nonzero["axis_is_negative"] is False
        assert result_axis_positive_nonzero["axis_is_zero"] is False
        assert result_axis_positive_nonzero["num_outputs"] == 3

        assert result_attr_split_path["axis_is_negative"] is False
        assert result_attr_split_path["axis_is_zero"] is True
        assert result_attr_split_path["num_outputs"] == 3

        assert result_missing_axis["axis_is_negative"] is None
        assert result_missing_axis["axis_is_zero"] is None
        assert result_missing_axis["num_outputs"] == 2

    def test_split_generated_cases_cover_axis_flag_states(self, split_generator_opset18):
        """Generated combinations should cover all reachable finite axis-flag states."""
        combinations = split_generator_opset18.get_input_and_infinite_attribute_combinations()

        state_set = set()
        has_axis_minus1_case = False
        has_axis_positive_nonzero_case = False
        for comb in combinations:
            axis = comb.get("axis")
            if axis is None:
                continue

            state_set.add((axis < 0, axis == 0))

            split_constraint = comb.get("split")
            input_constraint = comb.get("input")
            if split_constraint is None or input_constraint is None:
                continue

            split_values = tuple(np.array(split_constraint.value).tolist())
            input_shape = tuple(input_constraint.shape)
            if input_shape == (6, 3, 4) and split_values == (2, 2):
                if axis == -1:
                    has_axis_minus1_case = True
                if axis == 2:
                    has_axis_positive_nonzero_case = True

        assert state_set == {
            (True, False),
            (False, True),
            (False, False),
        }
        assert has_axis_minus1_case
        assert has_axis_positive_nonzero_case


class TestScatterNDDerivedProperties:
    """Regression tests for ScatterND derive_properties coverage."""

    @pytest.fixture()
    def scatternd_generator_opset18(self):
        """Create a ScatterND generator for current ONNX opset behavior."""
        domain = ONNXDomain.AI_ONNX
        schema = domain.get_op_schema("ScatterND", 18)
        generator_class = get_runtime_checker_op("ScatterND")
        return generator_class(schema)

    def test_scatternd_k_is_two_is_derived(self, scatternd_generator_opset18):
        """k_is_two should distinguish k=2 from other non-edge k values."""
        data_shape = (2, 2, 2, 2, 2, 3)

        result_q1_k2 = scatternd_generator_opset18.derive_properties(
            {
                "data_shape": data_shape,
                "indices_value": np.array([0, 1], dtype=np.int64),
                "updates_shape": data_shape[2:],
            }
        )
        result_q1_k3 = scatternd_generator_opset18.derive_properties(
            {
                "data_shape": data_shape,
                "indices_value": np.array([0, 1, 0], dtype=np.int64),
                "updates_shape": data_shape[3:],
            }
        )
        result_q2_k2 = scatternd_generator_opset18.derive_properties(
            {
                "data_shape": data_shape,
                "indices_value": np.array([[0, 1], [1, 0]], dtype=np.int64),
                "updates_shape": (2, *data_shape[2:]),
            }
        )

        assert result_q1_k2["q_is_one"] is True
        assert result_q1_k2["k_is_two"] is True
        assert result_q1_k2["k_is_one"] is False
        assert result_q1_k2["k_is_dim_minus_one"] is False
        assert result_q1_k2["k_is_dim"] is False

        assert result_q1_k3["q_is_one"] is True
        assert result_q1_k3["k_is_two"] is False

        assert result_q2_k2["q_is_one"] is False
        assert result_q2_k2["k_is_two"] is True

    def test_scatternd_generated_cases_cover_k_is_two_states(self, scatternd_generator_opset18):
        """Generated ScatterND combinations should cover k_is_two True/False states."""
        combinations = scatternd_generator_opset18.get_input_and_infinite_attribute_combinations()

        k_is_two_states = set()
        has_dim6_q1_k2 = False
        has_dim6_q1_k3 = False
        for comb in combinations:
            data_constraint = comb.get("data")
            indices_constraint = comb.get("indices")
            if data_constraint is None or indices_constraint is None:
                continue

            data_shape = tuple(data_constraint.shape)
            indices_shape = tuple(np.array(indices_constraint.value).shape)
            if len(indices_shape) == 0:
                continue

            q = len(indices_shape)
            k = indices_shape[-1]
            k_is_two_states.add(k == 2)

            if len(data_shape) == 6 and q == 1 and k == 2:
                has_dim6_q1_k2 = True
            if len(data_shape) == 6 and q == 1 and k == 3:
                has_dim6_q1_k3 = True

        assert k_is_two_states == {False, True}
        assert has_dim6_q1_k2
        assert has_dim6_q1_k3


class TestConvTransposeDerivedProperties:
    """Regression tests for ConvTranspose derive_properties coverage."""

    @pytest.fixture()
    def conv_transpose_generator_opset22(self):
        """Create a ConvTranspose generator for current ONNX opset behavior."""
        domain = ONNXDomain.AI_ONNX
        schema = domain.get_op_schema("ConvTranspose", 22)
        generator_class = get_runtime_checker_op("ConvTranspose")
        return generator_class(schema)

    def test_conv_transpose_kernel_equals_stride_is_derived(self, conv_transpose_generator_opset22):
        """kernel_equals_stride should be a finite boolean derived from infinite attrs."""
        common = {
            "X_shape": (2, 6, 10, 10),
            "W_shape": (6, 6, 3, 3),
            "attr_dilations": (1, 1),
            "attr_pads": (0, 0, 0, 0),
            "attr_group": 1,
        }

        result_true = conv_transpose_generator_opset22.derive_properties(
            {**common, "attr_strides": (3, 3), "attr_kernel_shape": (3, 3)}
        )
        result_false = conv_transpose_generator_opset22.derive_properties(
            {**common, "attr_strides": (2, 2), "attr_kernel_shape": (3, 3)}
        )
        result_even_kernel = conv_transpose_generator_opset22.derive_properties(
            {**common, "attr_strides": (2, 2), "attr_kernel_shape": (2, 2)}
        )
        result_even_mismatch = conv_transpose_generator_opset22.derive_properties(
            {**common, "attr_strides": (1, 1), "attr_kernel_shape": (2, 2)}
        )
        result_missing = conv_transpose_generator_opset22.derive_properties(common)

        assert result_true["kernel_equals_stride"] is True
        assert result_false["kernel_equals_stride"] is False
        assert result_false["kernel_all_even"] is False
        assert result_even_kernel["kernel_all_even"] is True
        assert result_even_mismatch["kernel_equals_stride"] is False
        assert result_even_mismatch["kernel_all_even"] is True
        assert result_missing["kernel_equals_stride"] is None
        assert result_missing["kernel_all_even"] is None

    def test_conv_transpose_generated_cases_cover_derived_state_combinations(
        self, conv_transpose_generator_opset22
    ):
        """Generated combinations should cover all finite derived-property combinations."""
        combinations = (
            conv_transpose_generator_opset22.get_input_and_infinite_attribute_combinations()
        )

        eq_states = set()
        kernel_even_states = set()
        combined_states = set()
        has_odd_exact_stride_case = False
        has_even_non_exact_stride_case = False
        for comb in combinations:
            kernel_shape = comb.get("kernel_shape")
            strides = comb.get("strides")
            if kernel_shape is None or strides is None:
                continue

            kernel_tuple = tuple(kernel_shape)
            stride_tuple = tuple(strides)
            eq = kernel_tuple == stride_tuple
            even = all((int(dim) % 2) == 0 for dim in kernel_tuple)
            eq_states.add(eq)
            kernel_even_states.add(even)
            combined_states.add((eq, even))

            if (
                kernel_tuple == (3, 3)
                and stride_tuple == (3, 3)
                and comb.get("group") == 1
                and comb.get("auto_pad") == "NOTSET"
            ):
                has_odd_exact_stride_case = True

            if (
                kernel_tuple == (2, 2)
                and stride_tuple == (1, 1)
                and comb.get("group") == 1
                and comb.get("auto_pad") == "NOTSET"
            ):
                has_even_non_exact_stride_case = True

        assert eq_states == {False, True}
        assert kernel_even_states == {False, True}
        assert combined_states >= {
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        }
        assert has_odd_exact_stride_case
        assert has_even_non_exact_stride_case
