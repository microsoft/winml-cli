# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generators for unary-like ONNX operators.

Unary-like operators have one primary input tensor but may have additional
attributes or optional inputs. This module provides input generators for operators
that can reuse the UnaryInputGenerator base shapes while adding operator-specific
attributes or input handling.
"""

import itertools
from typing import Any

import numpy as np

from winml.modelkit.onnx.dtypes import SupportedONNXType
from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    InputValueConstraint,
    QDQParameterConfig,
    register_runtime_checker_op,
)
from .unary_input_generator import UnaryInputGenerator


# ============================================================================
# Category 1: Single input + float attributes
# These can directly extend UnaryInputGenerator and override get_finite_attribute_sets
# ============================================================================


@register_runtime_checker_op
class CeluInputGenerator(UnaryInputGenerator):
    """Input generator for Celu operator.

    Signature: Celu(X, *, alpha=1.0) -> Y
    Single input with one float attribute for controlling the activation shape.
    """

    op_name = "Celu"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return alpha values to test."""
        return {"alpha": [1.0]}


@register_runtime_checker_op
class EluInputGenerator(UnaryInputGenerator):
    """Input generator for Elu operator.

    Signature: Elu(X, *, alpha=1.0) -> Y
    Single input with one float attribute for controlling the activation.
    """

    op_name = "Elu"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return alpha values to test."""
        return {"alpha": [1.0]}


@register_runtime_checker_op
class LeakyReluInputGenerator(UnaryInputGenerator):
    """Input generator for LeakyRelu operator.

    Signature: LeakyRelu(X, *, alpha=0.01) -> Y
    Single input with one float attribute for controlling the negative slope.
    """

    op_name = "LeakyRelu"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return alpha values to test."""
        return {"alpha": [0.01]}


@register_runtime_checker_op
class SeluInputGenerator(UnaryInputGenerator):
    """Input generator for Selu operator.

    Signature: Selu(X, *, alpha=1.67..., gamma=1.05...) -> Y
    Single input with two float attributes for SELU activation.
    """

    op_name = "Selu"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return alpha and gamma values to test.

        Using default values which are mathematically derived constants.
        """
        return {
            "alpha": [1.6732631921768188],
            "gamma": [1.0507010221481323],
        }


@register_runtime_checker_op
class ShrinkInputGenerator(UnaryInputGenerator):
    """Input generator for Shrink operator.

    Signature: Shrink(input, *, bias=0.0, lambd=0.5) -> Y
    Single input with two float attributes for soft thresholding.
    """

    op_name = "Shrink"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return bias and lambd values to test."""
        return {
            "bias": [0.1],
            "lambd": [0.5],
        }


@register_runtime_checker_op
class HardSigmoidInputGenerator(UnaryInputGenerator):
    """Input generator for HardSigmoid operator.

    Signature: HardSigmoid(X, *, alpha=0.2, beta=0.5) -> Y
    Single input with two float attributes for piecewise linear approximation.
    """

    op_name = "HardSigmoid"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return alpha and beta values to test."""
        return {
            "alpha": [0.2],
            "beta": [0.5],
        }


@register_runtime_checker_op
class ThresholdedReluInputGenerator(UnaryInputGenerator):
    """Input generator for ThresholdedRelu operator.

    Signature: ThresholdedRelu(X, *, alpha=1.0) -> Y
    Single input with one float attribute for threshold value.
    """

    op_name = "ThresholdedRelu"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return alpha values to test."""
        return {"alpha": [1.0]}


# ============================================================================
# Category 2: Single input + string/int attributes
# ============================================================================


@register_runtime_checker_op
class GeluInputGenerator(UnaryInputGenerator):
    """Input generator for Gelu operator.

    Signature: Gelu(X, *, approximate='none') -> Y
    Single input with string attribute for approximation method.
    """

    op_name = "Gelu"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return approximate values to test."""
        return {"approximate": ["none", "tanh"]}


@register_runtime_checker_op
class IsInfInputGenerator(UnaryInputGenerator):
    """Input generator for IsInf operator.

    Signature: IsInf(X, *, detect_negative=1, detect_positive=1) -> Y
    Single input with two int attributes for detecting infinities.
    """

    op_name = "IsInf"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return detect_negative and detect_positive values to test."""
        return {
            "detect_negative": [0, 1],
            "detect_positive": [0, 1],
        }


# ============================================================================
# Category 3: Single input + axis attribute (axis operations)
# ============================================================================


@register_runtime_checker_op
class SoftmaxInputGenerator(UnaryInputGenerator):
    """Input generator for Softmax operator.

    Signature: Softmax(input, *, axis=-1) -> Y
    Single input with axis attribute for softmax dimension.
    """

    op_name = "Softmax"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return axis values to test.

        Testing various axes including negative indexing.
        """
        return {"axis": list(range(-5, 6))}

    def derive_properties(self, properties: dict) -> dict:
        item = super().derive_properties(properties)
        input_param_name = self.op_input_names[0]
        shape = item.get(f"{input_param_name}_shape")
        item["axis_size_is_one"] = shape[item["attr_axis"]] == 1 if shape else False
        return item

    def get_qdq_config(self):
        return {
            "input": QDQParameterConfig(support_activation=True),
        }


@register_runtime_checker_op
class LogSoftmaxInputGenerator(UnaryInputGenerator):
    """Input generator for LogSoftmax operator.

    Signature: LogSoftmax(input, *, axis=-1) -> Y
    Single input with axis attribute for log-softmax dimension.
    """

    op_name = "LogSoftmax"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return axis values to test."""
        return {"axis": list(range(-5, 6))}


@register_runtime_checker_op
class HardmaxInputGenerator(UnaryInputGenerator):
    """Input generator for Hardmax operator.

    Signature: Hardmax(input, *, axis=-1) -> Y
    Single input with axis attribute for hardmax dimension.
    """

    op_name = "Hardmax"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return axis values to test."""
        return {"axis": [-1, 0, 1, 2, 3, 4, 5]}


class ArgExtremaInputGenerator(UnaryInputGenerator):
    """Base class for ArgMax and ArgMin input generators.

    Provides common functionality for handling axis, keepdims, and select_last_index attributes.
    """

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return attribute combinations to test.

        Test both keepdims modes and both select_last_index modes.
        """
        return {
            "keepdims": [0, 1],  # Test both modes: prune dimension vs keep dimension
            "select_last_index": [0, 1],  # Test both: first occurrence vs last occurrence
        }

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for ArgMax/ArgMin.

        Iterate over all possible axis values for each input shape.
        Include negative axis values (e.g., -1 for last dimension) first
        as they are commonly used in real models.
        """
        input_name = self.op_input_names[0]

        # Get shapes from parent class
        parent_combinations = super().get_input_and_infinite_attribute_combinations()

        combinations = []
        for combo in parent_combinations:
            if input_name in combo and isinstance(combo[input_name], InputShapeConstraint):
                shape = combo[input_name].shape
                data_dim = len(shape)
                # Generate axis values: negative (-1) first, then positive (0, 1, ..., data_dim-1)
                # Negative axis is more common in real models (e.g., axis=-1 for last dim)
                axis_values = [-1] + list(range(data_dim))
                for axis in axis_values:
                    new_combo = combo.copy()
                    new_combo["axis"] = InputValueConstraint(axis)
                    combinations.append(new_combo)
            else:
                combinations.append(combo)

        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for ArgExtrema operator."""
        item = properties.copy()
        item["data_dim"] = len(item["data_shape"])
        item["attr_axis_is_zero"] = int(item["attr_axis"] == 0)
        item["attr_axis_is_one"] = int(item["attr_axis"] == 1)
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes/values with infinite possibilities
        """
        return super().get_infinite_property_names() + ["attr_axis"]


@register_runtime_checker_op
class ArgMaxInputGenerator(ArgExtremaInputGenerator):
    """Input generator for ArgMax operator.

    Signature: ArgMax(data, *, axis=0, keepdims=1, select_last_index=0) -> INT64
    Single input with axis, keepdims, and select_last_index attributes.

    Computes the indices of the max elements along the provided axis.
    The output has the same rank as input if keepdims=1, otherwise the
    reduced dimension is pruned.
    """

    op_name = "ArgMax"


@register_runtime_checker_op
class ArgMinInputGenerator(ArgExtremaInputGenerator):
    """Input generator for ArgMin operator.

    Signature: ArgMin(data, *, axis=0, keepdims=1, select_last_index=0) -> INT64
    Single input with axis, keepdims, and select_last_index attributes.

    Computes the indices of the min elements along the provided axis.
    The output has the same rank as input if keepdims=1, otherwise the
    reduced dimension is pruned.
    """

    op_name = "ArgMin"


# ============================================================================
# Category 4: Single input + required attributes with specific constraints
# ============================================================================


@register_runtime_checker_op
class LRNInputGenerator(UnaryInputGenerator):
    """Input generator for LRN (Local Response Normalization) operator.

    Signature: LRN(X, *, alpha, beta, bias, size) -> Y
    Single input with size as required attribute and other optional attributes.
    LRN requires 4D or 5D input (N x C x H x W) or (N x C x D x H x W).
    """

    op_name = "LRN"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return attribute values to test.

        size is required and must be odd. Using default values for other attributes.
        """
        return {
            "size": [1, 3, 5, 7],
            "alpha": [0.0001],
            "beta": [0.75],
            "bias": [1.0],
        }

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for LRN.

        LRN requires at least 4D input with format (N x C x H x W) or higher dimensions.
        Testing with 4D inputs which are most common for image processing.
        """
        # Get the input parameter name from the operator schema
        input_param_name = self.op_input_names[0]

        return [
            # 4D inputs (N x C x H x W)
            {input_param_name: InputShapeConstraint((2, 4, 3, 3))},
        ]


# ============================================================================
# Category 5: Multiple inputs (cannot simply extend UnaryInputGenerator)
# ============================================================================


@register_runtime_checker_op
class ClipInputGenerator(UnaryInputGenerator):
    """Input generator for Clip operator.

    Signature: Clip(input, min=None, max=None) -> Y
    One required input with two optional scalar inputs for clipping range.
    """

    op_name = "Clip"
    expand_optionals = False

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Clip has no attributes."""
        return {}

    def _get_clip_param_names(self) -> tuple[str, str]:
        """Get parameter names for Clip operator.

        Returns:
            Tuple of (min_name, max_name)
        """
        return self.op_input_names[1], self.op_input_names[2]

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for Clip.

        Test with various tensor shapes and optional min/max scalars.
        """
        min_name, max_name = self._get_clip_param_names()

        # Get shapes from parent class
        parent_combinations = super().get_input_and_infinite_attribute_combinations()

        # Define min/max combinations
        min_max_combinations = [
            # Both min and max
            {
                min_name: InputValueConstraint(np.array(-1.0, dtype=np.float32)),
                max_name: InputValueConstraint(np.array(1.0, dtype=np.float32)),
            },
            # Only min
            {min_name: InputValueConstraint(np.array(0.0, dtype=np.float32))},
            # Only max
            {max_name: InputValueConstraint(np.array(1.0, dtype=np.float32))},
            # Neither min nor max
            {},
        ]

        # Create product of shapes and min/max combinations
        combinations = []
        for shape_constraint in parent_combinations:
            for min_max_combo in min_max_combinations:
                combo = shape_constraint.copy()
                combo.update(min_max_combo)
                combinations.append(combo)

        return combinations

    def get_infinite_property_names(self) -> list[str]:
        """Get list of attribute names and input names that have infinite value sets.

        To call in result processing. To be overridden by subclasses.

        Returns:
            List of attribute and input names with infinite value sets.
        """
        min_name, max_name = self._get_clip_param_names()
        parent_infinite_props = super().get_infinite_property_names()
        return parent_infinite_props + [min_name + "_value", max_name + "_value"]


@register_runtime_checker_op
class CumSumInputGenerator(UnaryInputGenerator):
    """Input generator for CumSum operator.

    Signature: CumSum(x, axis, *, exclusive=0, reverse=0) -> Y
    Two required inputs (data and axis scalar) with two int attributes.
    """

    op_name = "CumSum"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return attribute combinations to test."""
        return {
            "exclusive": [0, 1],
            "reverse": [0, 1],
        }

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for CumSum.

        Test with various tensor shapes and axis values.
        """
        # Get parameter names from the operator schema
        x_name = self.op_input_names[0]
        axis_name = self.op_input_names[1]

        shape_constraints = super().get_input_and_infinite_attribute_combinations()

        combinations = []
        for constraint in shape_constraints:
            shape = constraint[x_name].shape
            axis_values = range(-len(shape), len(shape))
            # Only add valid axis values for the shape
            for axis in axis_values:
                combinations.append(
                    {
                        x_name: InputShapeConstraint(shape),
                        axis_name: InputValueConstraint(np.array(axis, dtype=np.int64)),
                    }
                )
        return combinations

    def get_qdq_config(self):
        return {
            "x": QDQParameterConfig(support_activation=True),
            "axis": QDQParameterConfig(),
        }


@register_runtime_checker_op
class DropoutInputGenerator(UnaryInputGenerator):
    """Input generator for Dropout operator.

    Signature: Dropout(data, ratio=None, training_mode=None, *, seed=None) -> (output, mask)
    One required input with two optional inputs and one optional attribute.
    Note: Dropout has two outputs but we only test the execution.
    """

    op_name = "Dropout"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return attribute combinations to test."""
        return {"seed": [42]}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for Dropout.

        Test with various tensor shapes and ratio/training_mode.
        """
        ratio_name = self.op_input_names[1]
        training_mode_name = self.op_input_names[2]
        # Get shapes from parent class
        parent_combinations = super().get_input_and_infinite_attribute_combinations()

        # Define ratio and training_mode combinations
        training_mode_combinations = [
            {training_mode_name: InputValueConstraint(np.array(False, dtype=np.bool_))},
            {training_mode_name: InputValueConstraint(np.array(True, dtype=np.bool_))},
        ]
        ratio_training_combinations = [
            {ratio_name: InputValueConstraint(np.array(0.5, dtype=np.float32))}
        ]
        # Create product of shapes and ratio/training combinations
        combinations = []
        for shape_constraint in parent_combinations:
            for ratio_training_combo in ratio_training_combinations:
                for training_mode_combo in training_mode_combinations:
                    combo = shape_constraint.copy()
                    combo.update(ratio_training_combo)
                    combo.update(training_mode_combo)
                    combinations.append(combo)

        return combinations


# ============================================================================
# Category 6: Type casting operators
# ============================================================================


@register_runtime_checker_op
class CastInputGenerator(UnaryInputGenerator):
    """Input generator for Cast operator.

    Signature: Cast(input, *, saturate=1, to) -> Y
    Single input with required 'to' attribute specifying target dtype.
    """

    op_name = "Cast"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return attribute combinations to test.

        'to' is required and specifies target dtype as int (TensorProto.DataType).
        We'll test common type conversions.
        """
        from onnx import TensorProto

        return {
            "to": [
                int(TensorProto.BOOL),
                int(TensorProto.DOUBLE),
                int(TensorProto.FLOAT),
                int(TensorProto.FLOAT16),
                int(TensorProto.INT8),
                int(TensorProto.INT16),
                int(TensorProto.INT32),
                int(TensorProto.INT64),
                int(TensorProto.UINT8),
                int(TensorProto.UINT16),
                int(TensorProto.UINT32),
                int(TensorProto.UINT64),
            ],
            "saturate": [0, 1],
            "rounding_mode": ["up", "nearest"],
        }

    def infer_output_types(
        self, kwargs: dict[str, Any], tags: dict[str, Any], required_outputs_only: bool = True
    ) -> list[str]:
        """Infer output type from 'to' attribute.

        Args:
            kwargs: Operator input arguments containing 'to' attribute
            tags: Tags containing type_vars and other metadata
            required_outputs_only: If True, only infer types for required outputs.
                                   Optional outputs are skipped.

        Returns:
            List containing the output type specified by 'to' attribute as ONNX type string
        """
        # The 'to' attribute directly specifies the output type as TensorProto enum
        output_type_enum = kwargs["to"]
        onnx_type = SupportedONNXType.from_tensor_proto_type(output_type_enum)
        return [onnx_type.annotation]


@register_runtime_checker_op
class CastLikeInputGenerator(UnaryInputGenerator):
    """Input generator for CastLike operator.

    Signature: CastLike(input, target_type, *, saturate=1) -> Y
    Two inputs: data to cast and target type tensor.
    """

    op_name = "CastLike"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return attribute combinations to test."""
        return {"saturate": [0, 1], "rounding_mode": ["up", "nearest"]}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for CastLike.

        Test with various source and target types.
        """
        input_name = self.op_input_names[0]
        target_type_name = self.op_input_names[1]
        # Get shapes from parent class
        parent_combinations = super().get_input_and_infinite_attribute_combinations()

        # Create target arrays with different dtypes
        target_arrays = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.zeros((1, 2, 3, 4, 5, 6)),
        ]

        # Create combinations of input shapes and target types
        return [
            {
                input_name: shape_constraint[input_name],
                # TODO : use InputValueConstraint or InputShapeConstraint for target_type input?
                target_type_name: InputValueConstraint(target_array),
            }
            for shape_constraint in parent_combinations
            for target_array in target_arrays
        ]

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for CastLike operator testing."""
        item = properties.copy()
        item["input_dim"] = len(item["input_shape"])
        item["target_type_dim"] = len(item["target_type_value"])
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Get list of attribute names and input names that have infinite value sets.

        To call in result processing. To be overridden by subclasses.

        Returns:
            List of attribute and input names with infinite value sets.
        """
        return ["input_shape", "target_type_value"]
