# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generator for Slice ONNX operator.

This module contains the input generator for the Slice operator which
produces a slice of the input tensor along multiple axes.
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


# Shared data shapes for Slice operator (1D through 6D)
_SLICE_DATA_SHAPES: list[tuple[int, ...]] = [
    # (6,),  # 1D
    # (4, 5),  # 2D
    (3, 4, 5),  # 3D
    (3, 4, 5, 6),  # 4D
    # (3, 3, 4, 5, 6),  # 5D
    # (3, 3, 3, 4, 5, 6),  # 6D
]


@register_runtime_checker_op
class SliceInputGenerator(OpInputGenerator):
    """Input generator for Slice operator.

    Slice operator documentation:
    - Input 1 (data): Tensor of data to extract slices from
    - Input 2 (starts): 1-D INT64 tensor of starting indices
    - Input 3 (ends): 1-D INT64 tensor of ending indices (exclusive)
    - Input 4 (axes, optional): 1-D INT64 tensor of axes to slice
    - Input 5 (steps, optional): 1-D INT64 tensor of slice steps

    Constraints:
    - steps cannot be 0
    - For positive stepping, starts is clamped to [0, dims[axes[i]]]
    - For negative stepping, starts is clamped to [0, dims[axes[i]]-1]
    - All index values use standard Python/NumPy slicing semantics

    Coverage strategy (focused subset):
    - axes: [0] or [0, ..., r-1] (full range)
    - steps: [1, ..., 1], [-1, ..., -1], or [1, 2, 1, 1, ..., 1]
    - starts/ends: positive [0, r-1] or [1, r-2] patterns
    """

    op_name = "Slice"
    expand_optionals = False

    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """Slice has no attributes, all parameters are inputs.

        Returns:
            Empty dictionary as Slice has no attributes
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for Slice operator.

        Focused coverage strategy:
        - Test data shapes from 1D through 6D
        - axes: [0] (single axis) or [0, ..., r-1] (all axes)
        - steps: [1,...,1], [-1,...,-1], [1,2,1,1,...,1]
        - starts/ends: positive values covering [0, dim-1] and [1, dim-2]
        """
        combinations = []

        for data_shape in _SLICE_DATA_SHAPES:
            rank = len(data_shape)

            # Axes patterns: single axis [0] or all axes [0, 1, ..., r-1]
            axes_patterns = [np.array([0], dtype=np.int64)]
            if rank > 1:
                axes_patterns.append(np.array(list(range(rank)), dtype=np.int64))

            for axes in axes_patterns:
                num_axes = len(axes)

                # Get dimension sizes for the axes being sliced
                axis_dims = [data_shape[int(ax)] for ax in axes]

                # Step patterns: all 1s, all -1s, and mixed [1,2,1,...]
                steps_patterns = [
                    np.ones(num_axes, dtype=np.int64),
                    -np.ones(num_axes, dtype=np.int64),
                ]
                # Add step=2 pattern for single axis if dimension allows
                if num_axes == 1 and axis_dims[0] >= 3:
                    steps_patterns.append(np.array([2], dtype=np.int64))
                # Add mixed step pattern [1, 2, 1, 1, ...] for multi-axis if applicable
                elif num_axes >= 2 and axis_dims[1] >= 3:
                    steps_with_two = np.ones(num_axes, dtype=np.int64)
                    steps_with_two[1] = 2
                    steps_patterns.append(steps_with_two)

                for steps in steps_patterns:
                    is_all_forward = np.all(steps >= 1)
                    is_all_backward = np.all(steps <= -1)

                    # Determine start/end patterns based on step direction
                    starts_ends_patterns = []

                    if is_all_forward:
                        # Forward slicing patterns
                        starts_ends_patterns.append((
                            np.zeros(num_axes, dtype=np.int64),
                            np.array([d for d in axis_dims], dtype=np.int64)
                        ))
                        if all(d >= 3 for d in axis_dims):
                            starts_ends_patterns.append((
                                np.ones(num_axes, dtype=np.int64),
                                np.array([d - 1 for d in axis_dims], dtype=np.int64)
                            ))
                    elif is_all_backward:
                    # Backward slicing patterns
                        starts_ends_patterns.append((
                            np.array([d - 1 for d in axis_dims], dtype=np.int64),
                            np.zeros(num_axes, dtype=np.int64)
                        ))
                        if all(d >= 3 for d in axis_dims):
                            starts_ends_patterns.append((
                                np.array([d - 2 for d in axis_dims], dtype=np.int64),
                                np.zeros(num_axes, dtype=np.int64)
                            ))
                    for starts, ends in starts_ends_patterns:
                        combinations.append({
                            "data": InputShapeConstraint(data_shape),
                            "starts": InputValueConstraint(starts),
                            "ends": InputValueConstraint(ends),
                            "axes": InputValueConstraint(axes),
                            "steps": InputValueConstraint(steps),
                        })

            # Add combinations without axes or steps (default behavior)
            combinations.append({
                "data": InputShapeConstraint(data_shape),
                "starts": InputValueConstraint(np.zeros(rank, dtype=np.int64)),
                "ends": InputValueConstraint(np.array([d for d in data_shape], dtype=np.int64)),
                "axes": None,
                "steps": None,
            })

        return combinations

    def derive_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Derive additional properties for Slice operator testing.

        Args:
            properties: Base properties containing data_shape, axes_value, etc.

        Returns:
            Updated properties with Slice-specific derived values

        Real-world patterns observed in models (CLIP, DETR, SAM):
        - Pattern 1: axes=[3], starts=[0], ends=[77], steps=[1] (single axis, last dim)
        - Pattern 2: axes=[0], starts=[0], ends=[2] (single axis, first dim, no steps)
        - Pattern 3: starts=[-1], ends=[INT64_MAX] (last element to end)
        - Pattern 4: starts=[-1], ends=[-INT64_MAX], axes=[0], steps=[-1] (reverse)
        - Pattern 5: starts=[0], ends=[INT64_MAX], axes=[3], steps=[2] (even indices)
        - Pattern 6: starts=[1], ends=[INT64_MAX], axes=[3], steps=[2] (odd indices)
        - Pattern 7: starts=[0], ends=[0], axes=[0] (empty slice)
        - Pattern 8: starts=[2], ends=[5], axes=[0] (middle slice)
        """
        item = properties.copy()
        data_shape = item["data_shape"]
        data_dim = len(data_shape)
        item["data_dim"] = data_dim

        # Derive axes-related properties
        axes_value = item.get("axes_value")
        if isinstance(axes_value, np.ndarray):
            axes_array = axes_value
        elif axes_value is None:
            # When axes is None, default to all axes [0, 1, ..., rank-1]
            axes_array = np.arange(data_dim, dtype=np.int64)
        else:
            axes_array = np.array(axes_value, dtype=np.int64)

        num_axes = len(axes_array)

        item["axes_is_single"] = num_axes == 1
        # after we add more cases of different axes, we can uncomment this
        # item["num_axes"] = num_axes
        # item["axes_is_all"] = num_axes == data_dim

        # Normalize negative axes to positive for comparison
        normalized_axes = np.where(axes_array < 0, axes_array + data_dim, axes_array)
        # Get dimension sizes for the sliced axes
        axis_dims = np.array([data_shape[int(ax)] for ax in normalized_axes], dtype=np.int64)

        # Derive starts-related properties
        starts_value = item.get("starts_value")
        if isinstance(starts_value, np.ndarray):
            starts_array = starts_value
        else:
            starts_array = np.array(starts_value, dtype=np.int64)

        # Normalize negative starts: add axis_dim if negative
        normalized_starts = np.where(
            starts_array < 0, starts_array + axis_dims, starts_array
        )

        # Derive ends-related properties
        ends_value = item.get("ends_value")
        if isinstance(ends_value, np.ndarray):
            ends_array = ends_value
        else:
            ends_array = np.array(ends_value, dtype=np.int64)

        # Normalize negative ends: add axis_dim if negative
        normalized_ends = np.where(ends_array < 0, ends_array + axis_dims, ends_array)

        # Check if this is a full slice (starts at 0, ends at dimension size)
        item["slice_all"] = bool(np.all(normalized_starts == 0) and np.all(normalized_ends >= axis_dims))

        # Derive steps-related properties
        steps_value = item.get("steps_value")
        if isinstance(steps_value, np.ndarray):
            steps_array = steps_value
        elif steps_value is None:
            # When steps is None, default to all 1s
            steps_array = np.ones(num_axes, dtype=np.int64)
        else:
            steps_array = np.array(steps_value, dtype=np.int64)
        item["steps_all_ones"] = bool(np.all(steps_array == 1))
        item["steps_is_neg_one"] = bool(np.all(steps_array == -1))

        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes/values with infinite possibilities
        """
        return [
            "data_shape",
            "starts_value",
            "ends_value",
            "axes_value",
            "steps_value"
        ]

    def get_qdq_config(self) -> dict[str, QDQParameterConfig]:
        return {
            self.op_input_names[0]: QDQParameterConfig(support_activation=True),  # data
            self.op_input_names[1]: QDQParameterConfig(),  # starts
            self.op_input_names[2]: QDQParameterConfig(),  # ends
            self.op_input_names[3]: QDQParameterConfig(),  # axes
            self.op_input_names[4]: QDQParameterConfig(),  # steps
        }
