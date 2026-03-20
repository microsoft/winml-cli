# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generators for reduction ONNX operators.

Reduction operators reduce tensor elements along specified axes to produce
output tensors with reduced dimensions.

All Reduce* operators share the same signature:
- Input: data (tensor to reduce)
- Optional input: axes (which axes to reduce along)
- Attributes: keepdims, noop_with_empty_axes
"""

from typing import Any

import numpy as np

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    InputValueConstraint,
    OpInputGenerator,
    QDQParameterConfig,
    register_runtime_checker_op,
)


class ReductionInputGenerator(OpInputGenerator):
    """Base class for reduction operator input generators.

    All Reduce* operators (ReduceSum, ReduceMean, ReduceMax, etc.) share:
    - Same signature: data, axes, keepdims, noop_with_empty_axes
    - Same input shape requirements (1D through 6D)
    - Same axis patterns to test

    This base class provides common shapes and axis combinations.
    """

    def get_common_data_shapes(self) -> list[tuple[int, ...]]:
        """Return common shapes for reduction testing.

        Test shapes from 1D through 6D to cover all dimensions.
        """
        return [
            (6,),  # 1D
            (3, 6),  # 2D
            (2, 4, 6),  # 3D
            (2, 4, 5, 5),  # 4D
            (2, 4, 3, 4, 4),  # 5D
            (2, 2, 2, 2, 2, 3),  # 6D
        ]

    def get_common_axes_combinations(self, shape: tuple[int, ...]) -> list[np.ndarray]:
        """Return common axes patterns to test for given shape.

        For each shape, test:
        1. All axes - reduce all axes (empty array triggers default behavior)
        2. Reduce last axis
        3. Reduce first axis
        4. Reduce multiple axes (for 3D+)

        Args:
            shape: Input tensor shape

        Returns:
            List of axes arrays (always explicit, never None)
        """
        rank = len(shape)
        axes_combinations = [
            list(range(rank)),  # All axes (explicit)
            [-1],  # Reduce last axis
            [0],  # Reduce first axis
        ]
        if rank >= 3:
            axes_combinations.append([0, -1])  # First and last axes
            axes_combinations.append([0, 1])  # First 2 axes
            axes_combinations.append([-2, -1])  # Last 2 axes

        # basically all Reduce* ops have "axes" as attrribute for opset <=17
        # and as input for opset >=18, EXCEPT ReduceSum, which has "axes" as input since opset 13
        if "axes" in self.op_input_names:
            axes_combinations = [np.array(axes, dtype=np.int64) for axes in axes_combinations]

        return axes_combinations

    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """Return finite attribute combinations for Reduce* operators.

        All Reduce* operators have:
        - keepdims: 0 or 1 (whether to keep reduced dimensions)
        - noop_with_empty_axes: 0 or 1 (behavior when axes is empty)
        """
        return {
            "keepdims": [0, 1],
            "noop_with_empty_axes": [0, 1],
        }

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for reduction operators.

        Strategy:
        - Test all common shapes (1D through 6D)
        - For each shape, test different axes patterns
        - axes is always explicitly provided (never omitted)
        """
        # CRITICAL: Always provide explicit values for all inputs,
        # even optional ones, to ensure consistent querying
        return [
            {
                "data": InputShapeConstraint(shape),
                "axes": InputValueConstraint(axes),
            }
            for shape in self.get_common_data_shapes()
            for axes in self.get_common_axes_combinations(shape)
        ]

    def derive_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Derive additional properties for reduction operator testing.

        Args:
            properties: Base properties containing data_shape and axes

        Returns:
            Updated properties with reduction-specific derived values
        """
        item = properties.copy()
        input_name = self.op_input_names[0]  # "data" for Reduce* operators
        item[f"{input_name}_dim"] = len(item[f"{input_name}_shape"])
        axes = item.get("attr_axes")
        if axes is None:
            axes = item.get("axes_value")
        if axes is not None:
            item["single_axis"] = len(axes) == 1
            item["has_first_axis"] = 0 in axes or -item[f"{input_name}_dim"] in axes
            # item["has_last_axis"] = (item[f"{input_name}_dim"] - 1) in axes or -1 in axes
            item["full_reduction"] = (len(axes) == item[f"{input_name}_dim"])
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes/axes with infinite possibilities
        """
        return (
            [f"{input_name}_value" for input_name in self.op_input_names]
            + [f"{input_name}_shape" for input_name in self.op_input_names]
            + ["attr_axes"]
        )

    def get_qdq_config(self):
        return {
            "data": QDQParameterConfig(support_activation=True),
            "axes": QDQParameterConfig(),
        }


# ============================================================================
# Reduce* operators - All share the same base implementation
# ============================================================================


@register_runtime_checker_op
class ReduceL1InputGenerator(ReductionInputGenerator):
    """Input generator for ReduceL1 operator.

    Computes L1 norm (sum of absolute values) along specified axes.
    """

    op_name = "ReduceL1"


@register_runtime_checker_op
class ReduceL2InputGenerator(ReductionInputGenerator):
    """Input generator for ReduceL2 operator.

    Computes L2 norm (square root of sum of squares) along specified axes.
    """

    op_name = "ReduceL2"


@register_runtime_checker_op
class ReduceLogSumInputGenerator(ReductionInputGenerator):
    """Input generator for ReduceLogSum operator.

    Computes log of sum along specified axes.
    """

    op_name = "ReduceLogSum"


@register_runtime_checker_op
class ReduceLogSumExpInputGenerator(ReductionInputGenerator):
    """Input generator for ReduceLogSumExp operator.

    Computes log of sum of exponentials along specified axes.
    Numerically stable version of log(sum(exp(x))).
    """

    op_name = "ReduceLogSumExp"


@register_runtime_checker_op
class ReduceMaxInputGenerator(ReductionInputGenerator):
    """Input generator for ReduceMax operator.

    Computes maximum values along specified axes.
    """

    op_name = "ReduceMax"


@register_runtime_checker_op
class ReduceMeanInputGenerator(ReductionInputGenerator):
    """Input generator for ReduceMean operator.

    Computes mean (average) values along specified axes.
    """

    op_name = "ReduceMean"


@register_runtime_checker_op
class ReduceMinInputGenerator(ReductionInputGenerator):
    """Input generator for ReduceMin operator.

    Computes minimum values along specified axes.
    """

    op_name = "ReduceMin"


@register_runtime_checker_op
class ReduceProdInputGenerator(ReductionInputGenerator):
    """Input generator for ReduceProd operator.

    Computes product of values along specified axes.
    """

    op_name = "ReduceProd"


@register_runtime_checker_op
class ReduceSumInputGenerator(ReductionInputGenerator):
    """Input generator for ReduceSum operator.

    Computes sum of values along specified axes.
    """

    op_name = "ReduceSum"


@register_runtime_checker_op
class ReduceSumSquareInputGenerator(ReductionInputGenerator):
    """Input generator for ReduceSumSquare operator.

    Computes sum of squared values along specified axes.
    """

    op_name = "ReduceSumSquare"


# ============================================================================
# TopK operator - Different signature, implemented separately
# ============================================================================


@register_runtime_checker_op
class TopKInputGenerator(OpInputGenerator):
    """Input generator for TopK operator.

    TopK retrieves the top-K largest or smallest elements along a specified axis.

    Signature:
    - Inputs: X (tensor), K (number of elements to retrieve)
    - Attributes: axis, largest, sorted
    - Outputs: Values (top K values), Indices (their indices)

    This operator is different from Reduce* operators as it:
    1. Takes K as an input (not an attribute)
    2. Returns TWO outputs (values and indices)
    3. Has different attributes (axis, largest, sorted)
    """

    op_name = "TopK"

    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """Return finite attribute combinations for TopK.

        Attributes:
        - axis: Test default (-1, last axis) and first axis (0)
        - largest: Test both largest (1) and smallest (0)
        - sorted: Test both sorted (1) and unsorted (0)
        """
        return {
            "axis": [-1, 0],  # Last axis and first axis
            "largest": [0, 1],  # Smallest and largest
            "sorted": [0, 1],  # Sorted and unsorted
        }

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for TopK operator.

        Strategy:
        - Test shapes from 1D through 6D
        - For each shape, K should be smaller than the dimension being reduced
        - K is an input (not an attribute)
        """
        combinations = []

        # Test shapes from 1D through 6D
        test_shapes = [
            (6,),  # 1D
            (3, 6),  # 2D
            (2, 4, 6),  # 3D
            (2, 4, 5, 5),  # 4D
            (2, 4, 3, 4, 4),  # 5D
            (2, 2, 2, 2, 2, 3),  # 6D
        ]

        for shape in test_shapes:
            # K should be valid for both axis=-1 and axis=0
            # Choose K to be smaller than both first and last dimension
            min_dim = min(shape[0], shape[-1])

            for k_value in (1, min_dim):
                combinations.append(
                    {
                        "X": InputShapeConstraint(shape),
                        "K": InputValueConstraint(np.array([k_value], dtype=np.int64)),
                    }
                )

        return combinations

    def derive_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Derive additional properties for TopK operator testing.

        Args:
            properties: Base properties containing X_shape, K, and axis

        Returns:
            Updated properties with TopK-specific derived values
        """
        item = properties.copy()
        input_name = self.op_input_names[0]  # "X" for TopK

        item[f"{input_name}_dim"] = len(item[f"{input_name}_shape"])
        item["K_is_max"] = item["K_value"][0] == item[f"{input_name}_shape"][item["attr_axis"]]
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes/K with infinite possibilities
        """
        return [f"{input_name}_value" for input_name in self.op_input_names] + [
            f"{input_name}_shape" for input_name in self.op_input_names
        ]
