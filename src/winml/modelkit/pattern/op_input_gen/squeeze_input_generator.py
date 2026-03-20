# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generator for Squeeze operator."""

from typing import Any, ClassVar

import numpy as np

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    InputValueConstraint,
    OpInputGenerator,
    register_runtime_checker_op,
)


@register_runtime_checker_op
class SqueezeInputGenerator(OpInputGenerator):
    """Input generator for Squeeze operator.

    Squeeze operator documentation:
    - Input (data): Tensors with at least max(dims) dimensions
    - Input (axes): Optional list of integers indicating the dimensions to squeeze.
      Negative values mean counting from the back. Accepted range is [-r, r-1]
      where r = rank(data).

    Constraints:
    - If axes is not provided, all single dimensions (size=1) will be removed
    - If axes is provided, only those specified dimensions are squeezed
    - Axes must point to dimensions with size=1, otherwise an error is raised

    Coverage strategy:
    - Input dimensions: 1D through 6D
    - Test shapes with 1s in various positions
    - Test different axes combinations (single axis, multiple axes, negative axes)
    - Always provide explicit axes input (never omit optional inputs)
    """

    op_name = "Squeeze"

    # Base dimension values for non-squeezable dimensions
    _BASE_DIMS: ClassVar[list[int]] = [2, 3, 4, 5, 6, 7]

    def _make_shape_with_ones_at(self, ndim: int, one_positions: list[int]) -> tuple[int, ...]:
        """Create a shape with 1s at specified positions and base values elsewhere.

        Args:
            ndim: Number of dimensions
            one_positions: List of indices where shape should be 1

        Returns:
            Tuple representing the shape
        """
        shape = []
        base_idx = 0
        for i in range(ndim):
            if i in one_positions:
                shape.append(1)
            else:
                shape.append(self._BASE_DIMS[base_idx % len(self._BASE_DIMS)])
                base_idx += 1
        return tuple(shape)

    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """Returns finite attribute sets for Squeeze.

        Squeeze has no finite attributes.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Returns comprehensive input combinations for Squeeze operator."""
        combinations = []

        # ===== Systematic generation for 0D through 6D =====
        # Test 1: 0D tensor
        # 0D tensor (scalar), axes must be empty
        combinations.append(
            {
                "data": InputShapeConstraint(()),
                "axes": InputValueConstraint(np.array([], dtype=np.int64)),
            }
        )

        for ndim in range(1, 7):
            # Test 2: Squeeze single axis at each position (positive axis)
            # Shape has 1 at position 0, other dims are non-1
            shape = self._make_shape_with_ones_at(ndim, [0])
            combinations.append(
                {
                    "data": InputShapeConstraint(shape),
                    "axes": InputValueConstraint(np.array([0], dtype=np.int64)),
                }
            )

            # Test 3: Squeeze first and last dimensions with mixed positive/negative axes
            # Only add if ndim >= 2 (need at least 2 dims for first and last)
            if ndim >= 2:
                shape = self._make_shape_with_ones_at(ndim, [0, ndim - 1])
                combinations.append(
                    {
                        "data": InputShapeConstraint(shape),
                        "axes": InputValueConstraint(np.array([0, -1], dtype=np.int64)),
                    }
                )

            # Test 5: No squeezable dims (all non-1), empty axes
            no_ones_shape = self._make_shape_with_ones_at(ndim, [])
            combinations.append(
                {
                    "data": InputShapeConstraint(no_ones_shape),
                    "axes": InputValueConstraint(np.array([], dtype=np.int64)),
                }
            )

        return combinations

    def derive_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Derive additional properties for Squeeze operator testing.

        Args:
            properties: Base properties containing data_shape and axes_value

        Returns:
            Updated properties with Squeeze-specific derived values
        """
        item = properties.copy()
        item["data_dim"] = len(item["data_shape"])

        # Get axes value
        axes_value = item["axes_value"]
        if isinstance(axes_value, np.ndarray):
            axes_list = axes_value.tolist()
        else:
            axes_list = list(axes_value) if axes_value is not None else []

        item["axes_is_empty"] = len(axes_list) == 0
        item["axes_len_greater_than_one"] = len(axes_list) > 1

        item["data_single_entry"] = all(dim == 1 for dim in item["data_shape"])

        # Commented out for now; can be enabled if needed
        # Count number of 1s in input shape
        # data_shape = item["data_shape"]

        # Check if squeezing batch dimension (first dim with size 1)
        # if len(data_shape) > 0:
        #     item["squeeze_batch"] = 0 in axes_list or -len(data_shape) in axes_list
        # else:
        #     item["squeeze_batch"] = False

        return item

    def get_infinite_property_names(self) -> list[str]:
        """Returns names of infinite properties for Squeeze operator."""
        return ["data_shape", "data_value", "axes_value"]
