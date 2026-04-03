# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generators for indexing and shape manipulation ONNX operators.

This module contains input generators for operators that perform indexing
and shape manipulation operations:
- Gather: Gathers entries along an axis using indices
- ScatterND: Scatters updates into a copy of data at specified indices
- Unsqueeze: Inserts single-dimensional entries to shape
- Split: Splits a tensor into multiple outputs
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


# Shared data shapes for indexing operators (1D through 6D)
_INDEXING_DATA_SHAPES: list[tuple[int, ...]] = [
    (6,),  # 1D
    (3, 4),  # 2D
    (2, 3, 4),  # 3D
    (2, 3, 4, 5),  # 4D
    (2, 2, 2, 3, 2),  # 5D
    (2, 2, 2, 2, 2, 3),  # 6D
]


@register_runtime_checker_op
class GatherInputGenerator(OpInputGenerator):
    """Input generator for Gather operator.

    Gather operator documentation:
    - Input 1 (data): Tensor of rank r >= 1
    - Input 2 (indices): INT32/INT64 tensor of any rank q
    - Attribute (axis): Which axis to gather on (default 0)
      Negative values count from the back. Range: [-r, r-1]

    Constraints:
    - data must have rank >= 1
    - indices can be any rank (including scalar)
    - All index values must be within bounds [-s, s-1] for axis of size s
    - Output rank = q + (r - 1)

    Coverage strategy:
    - Test data shapes from 1D through 6D
    - Test indices shapes: scalar, 1D, 2D
    - Test axis values: 0 (default), -1 (last), middle axis for higher dims
    - Test both positive and negative indices
    """

    op_name = "Gather"

    def get_finite_attribute_sets(self) -> dict[str, list[int]]:
        """Gather has no simple finite attribute sets.

        The axis attribute depends on input rank, so it's handled
        in get_input_and_infinite_attribute_combinations.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for Gather operator.

        Strategy:
        - Use representative shapes from 1D through 6D
        - For each data shape, test multiple index configurations
        - Test different axis values appropriate for each shape
        """
        combinations = []

        # Generate test cases using nested loops
        # Coverage: data 1D-6D, all axes per dimension + [-1], indices 0D-7D
        for data_dim in range(1, 7):
            data_shape = _INDEXING_DATA_SHAPES[data_dim - 1]
            # Test all axes (0 through data_dim-1) plus negative axis (-1)
            # TODO list to 0, 1, -1 as attr_axis is ignored
            for axis in [*list(range(min(2, data_dim) if self.qdq_generator else data_dim)), -1]:
                # Generate indices of varying dimensions (0D through 7D)
                # Constraint: total rank cannot exceed 7, so indices_dim <= 8 - data_dim
                for indices_dim in range(8 - data_dim):
                    # Create indices shape with dimension 2 in each axis
                    # For 0D (scalar), shape is ()
                    indices_shape = (2,) * indices_dim if indices_dim > 0 else ()
                    # Use InputShapeConstraint with min_max to generate valid indices
                    # The values will be in range [0, axis_size-1] for the given axis
                    axis_size = data_shape[axis]
                    indices = InputShapeConstraint(indices_shape, min_max=(0, axis_size - 1))
                    combinations.append(
                        {
                            "data": InputShapeConstraint(data_shape),
                            "indices": indices,
                            "axis": axis,
                        }
                    )

        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for Gather operator testing.

        Args:
            properties: Base properties containing data_shape and indices_shape

        Returns:
            Updated properties with Gather-specific derived values
        """
        item = properties.copy()
        item["data_dim"] = len(item["data_shape"])
        item["indices_dim"] = len(item["indices_shape"])
        item["output_dim"] = item["data_dim"] + item["indices_dim"] - 1
        # add this for QNN EP
        item["attr_axis_is_zero"] = int(item["attr_axis"] == 0)
        item["attr_axis_is_one"] = int(item["attr_axis"] == 1)
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes/values with infinite possibilities
        """
        return ["data_shape", "indices_shape", "attr_axis"]

    def get_qdq_config(self):
        """Return QDQ configuration for Gather operator inputs."""
        return {
            "data": QDQParameterConfig(support_activation=True),
            "indices": QDQParameterConfig(support_non_qdq=True),
        }


@register_runtime_checker_op
class GatherElementsInputGenerator(OpInputGenerator):
    """Input generator for GatherElements operator.

    GatherElements operator documentation:
    - Input 1 (data): Tensor of rank r >= 1
    - Input 2 (indices): Tensor of rank r (same as data) with int32/int64 indices
    - Attribute (axis): Which axis to gather on (default 0)

    Constraints:
    - data and indices must have same rank r
    - indices values must be within bounds [-s, s-1] along axis of size s
    - Output shape is same as indices shape

    Coverage strategy:
    - Test data shapes from 1D through 6D
    - Test indices with same shape as data and with different shape (but same rank)
    - Test all axis values including negative
    """

    op_name = "GatherElements"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """GatherElements has no simple finite attribute sets."""
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for GatherElements operator."""
        combinations = []

        for data_dim in range(1, 7):
            data_shape = _INDEXING_DATA_SHAPES[data_dim - 1]

            for axis in [*list(range(data_dim)), -1]:
                axis_idx = axis if axis >= 0 else axis + data_dim
                axis_size = data_shape[axis_idx]

                # Case 1: indices shape same as data shape
                # Values must be within [0, axis_size-1]
                indices_val = InputShapeConstraint(data_shape, min_max=(0, axis_size - 1))
                combinations.append(
                    {
                        "data": InputShapeConstraint(data_shape),
                        "indices": indices_val,
                        "axis": axis,
                    }
                )

                # Case 2: indices shape different but same rank
                # The gathered axis can have any size (determines output size).
                # Other axes must match data shape.
                indices_shape_2 = list(data_shape)
                indices_shape_2[axis_idx] = (
                    indices_shape_2[axis_idx] * 2 if indices_shape_2[axis_idx] > 0 else 2
                )
                indices_shape_2 = tuple(indices_shape_2)

                indices_val_2 = InputShapeConstraint(indices_shape_2, min_max=(0, axis_size - 1))
                combinations.append(
                    {
                        "data": InputShapeConstraint(data_shape),
                        "indices": indices_val_2,
                        "axis": axis,
                    }
                )

        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for GatherElements operator testing."""
        item = properties.copy()
        item["data_dim"] = len(item["data_shape"])
        item["indices_dim"] = len(item["indices_shape"])
        # add this for QNN EP
        item["attr_axis_is_zero"] = int(item["attr_axis"] == 0)
        item["attr_axis_is_one"] = int(item["attr_axis"] == 1)
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values."""
        return ["data_shape", "indices_shape", "attr_axis"]

    def get_qdq_config(self):
        """Return QDQ configuration for GatherElements operator inputs."""
        return {
            "data": QDQParameterConfig(support_activation=True),
            "indices": QDQParameterConfig(support_non_qdq=True),
        }


@register_runtime_checker_op
class GatherNDInputGenerator(OpInputGenerator):
    """Input generator for GatherND operator.

    GatherND operator documentation:
    - Input 1 (data): Tensor of rank r >= 1
    - Input 2 (indices): Tensor of rank q >= 1
    - Attribute (batch_dims): Number of batch dimensions b

    Constraints:
    - 0 <= batch_dims < min(q, r)
    - indices.shape[:batch_dims] == data.shape[:batch_dims]
    - indices.shape[-1] = k where 1 <= k <= r - batch_dims
    - All index values must be within bounds

    Coverage strategy:
    - Test data shapes from 1D through 6D
    - Test batch_dims 0 and 1
    - Test various k values
    - Test indices rank q > batch_dims
    """

    op_name = "GatherND"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """GatherND has no simple finite attribute sets."""
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for GatherND operator."""
        combinations = []

        for data_dim in range(1, 7):
            data_shape = _INDEXING_DATA_SHAPES[data_dim - 1]

            # Test batch_dims 0, and 1 if data rank allows
            possible_batch_dims = [0]
            if data_dim > 1:
                possible_batch_dims.append(1)

            for b in possible_batch_dims:
                # k is last dim of indices. 1 <= k <= r - b
                max_k = data_dim - b
                # Test a few k values: 1, max_k, and maybe middle
                k_values = {1, max_k}
                if max_k > 2:
                    k_values.add(max_k // 2)

                for k in sorted(k_values):
                    for extra_dims in range(2):  # 0 and 1 extra dim
                        # Construct indices shape
                        # First b dims match data
                        indices_shape_prefix = data_shape[:b]
                        # Middle dims (arbitrary size, say 2)
                        indices_shape_middle = (2,) * extra_dims
                        # Last dim is k
                        indices_shape = indices_shape_prefix + indices_shape_middle + (k,)

                        # Generate valid indices
                        indices_val = self._generate_valid_indices(data_shape, indices_shape, k, b)

                        combinations.append(
                            {
                                "data": InputShapeConstraint(data_shape),
                                "indices": InputValueConstraint(indices_val),
                                "batch_dims": b,
                            }
                        )

        return combinations

    def _generate_valid_indices(
        self,
        data_shape: tuple[int, ...],
        indices_shape: tuple[int, ...],
        k: int,
        batch_dims: int,
    ) -> np.ndarray:
        """Generate valid indices for GatherND."""
        indices = np.zeros(indices_shape, dtype=np.int64)

        # The dimensions of data being indexed are data_shape[batch_dims : batch_dims + k]
        # Why?
        # GatherND maps index-tuples to slices.
        # "indices is an q-dimensional ... best thought of as
        # (q-1)-dimensional tensor of index-tuples".
        # "Each element defines a slice of data".
        # k = indices.shape[-1]. So each "element" is a vector of size k.
        # This vector indexes into `data` starting from `batch_dims`.
        # index[0] -> data dim b
        # index[1] -> data dim b+1
        # ...
        # index[k-1] -> data dim b+k-1

        target_dims = data_shape[batch_dims : batch_dims + k]

        for i in range(k):
            max_val = target_dims[i]
            # Generate values cycling through [0, max_val-1]
            flat_size = int(np.prod(indices_shape[:-1]))
            values = np.arange(flat_size, dtype=np.int64) % max_val
            indices[..., i] = values.reshape(indices_shape[:-1])

        return indices

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for GatherND operator testing."""
        item = properties.copy()
        item["data_dim"] = len(item["data_shape"])
        indices_array = np.array(item["indices_value"])
        item["indices_dim"] = len(indices_array.shape)
        item["last_index_dimension"] = indices_array.shape[-1]
        # add this for QNN EP
        item["attr_batch_dims_is_zero"] = int(item["attr_batch_dims"] == 0)
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values."""
        return ["data_shape", "indices_value", "attr_batch_dims"]


@register_runtime_checker_op
class ScatterNDInputGenerator(OpInputGenerator):
    """Input generator for ScatterND operator (inverse of Gather).

    ScatterND operator documentation:
    - Input 1 (data): Tensor of rank r >= 1
    - Input 2 (indices): INT64 tensor of rank q >= 1
      - indices.shape[-1] = k, where k <= rank(data)
      - Treated as (q-1)-dimensional tensor of k-tuples
    - Input 3 (updates): Tensor of rank q + r - k - 1
      - Shape must equal indices.shape[0:q-1] ++ data.shape[k:r]
    - Attribute (reduction): Type of reduction to apply
      - "none" (default): no reduction
      - "add": addition
      - "mul": multiplication
      - "max": maximum
      - "min": minimum

    Constraints:
    - data must have rank >= 1
    - indices must have rank >= 1
    - indices.shape[-1] (k) <= rank(data)
    - updates.shape = indices.shape[:-1] + data.shape[k:]
    - Index values must be in valid range for data dimensions

    Coverage strategy (optimized for performance):
    - Test data shapes from 1D through 6D (same as Gather)
    - Test k values from 1 to (data_rank - 1) for focused coverage
    - Test q values 1-2 only (covers key indexing scenarios)
    - Test all reduction modes

    Optimization notes:
    - Reduced k range excludes k=data_rank case to focus on partial indexing
    - Limited q to 1-2 to cover scalar and basic multi-dimensional indices
    - These changes reduce test cases by ~80% while maintaining core coverage
    """

    op_name = "ScatterND"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Return finite attribute combinations for ScatterND.

        Returns:
            Dictionary with reduction attribute options
        """
        return {
            "reduction": ["none", "add", "mul", "max", "min"],
        }

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for ScatterND operator.

        Optimized strategy for performance:
        - Use representative data shapes from 1D through 6D
        - For each data shape, test k values from 1 to (data_rank - 1)
          where k = indices.shape[-1] determines indexed dimensions
        - For each k, test indices ranks q = 1, 2 only
        - Ensure updates shape matches: updates.shape = indices.shape[:-1] + data.shape[k:]

        Performance optimizations:
        - Excludes k = data_rank to focus on partial dimension indexing
        - Limits q to essential cases: scalar indices (q=1) and 2D indices (q=2)
        - Reduces total test combinations from ~500+ to ~100 cases
        """
        combinations = []

        # Generate optimized test cases using nested loops
        # Coverage: data 1D-6D, focused k values, limited q values for performance
        for data_dim in range(1, 7):
            data_shape = _INDEXING_DATA_SHAPES[data_dim - 1]

            # k ranges from 1 to (data_dim - 1) - excludes full dimension indexing
            # This focuses on partial indexing scenarios which are most common
            for k in range(1, data_dim + 1):
                # indices rank q limited to 1, 2 for performance optimization
                # q=1: scalar indices, q=2: basic multi-dimensional indices
                # Higher q values provide diminishing test value
                for q in range(1, 3):
                    # indices_shape = (q-1 dims of size 2) + (k,)
                    indices_shape = (2,) * (q - 1) + (k,)

                    updates_shape = indices_shape[:-1] + data_shape[k:]

                    # Generate valid indices values
                    # Each index tuple must be valid for the first k dimensions of data
                    indices_values = self._generate_valid_indices(data_shape, indices_shape, k)

                    combinations.append(
                        {
                            "data": InputShapeConstraint(data_shape),
                            "indices": InputValueConstraint(indices_values),
                            "updates": InputShapeConstraint(updates_shape),
                        }
                    )

        return combinations

    def _generate_valid_indices(
        self,
        data_shape: tuple[int, ...],
        indices_shape: tuple[int, ...],
        k: int,
    ) -> np.ndarray:
        """Generate valid indices for ScatterND.

        Args:
            data_shape: Shape of the data tensor
            indices_shape: Shape of the indices tensor
            k: Number of dimensions being indexed (indices.shape[-1])

        Returns:
            numpy array of valid indices
        """
        # Create indices array with the specified shape
        indices = np.zeros(indices_shape, dtype=np.int64)

        # Fill with valid index values
        # For each position in indices[..., i], the value must be in [0, data_shape[i]-1]
        for i in range(k):
            # Create a view of the i-th index component
            # and fill with values cycling through valid range
            max_val = data_shape[i]
            # Use modulo to cycle through valid values
            flat_size = int(np.prod(indices_shape[:-1]))
            values = np.arange(flat_size, dtype=np.int64) % max_val
            indices[..., i] = values.reshape(indices_shape[:-1])

        return indices

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for ScatterND operator testing.

        Args:
            properties: Base properties containing data_shape, indices_value, updates_shape

        Returns:
            Updated properties with ScatterND-specific derived values
        """
        item = properties.copy()
        item["data_dim"] = len(item["data_shape"])
        indices_array = np.array(item["indices_value"])
        k = indices_array.shape[-1]  # Number of dimensions indexed
        q = len(indices_array.shape)  # indices rank
        item["q_is_one"] = q == 1
        item["k_is_one"] = k == 1
        item["k_is_dim_minus_one"] = k == (item["data_dim"] - 1)
        item["k_is_dim"] = k == item["data_dim"]
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes/values with infinite possibilities
        """
        return ["data_shape", "indices_value", "updates_shape"]

    def get_qdq_config(self):
        """Return QDQ configuration for ScatterND operator inputs."""
        return {
            self.op_input_names[0]: QDQParameterConfig(support_activation=True),  # data
            self.op_input_names[1]: QDQParameterConfig(support_non_qdq=True),  # indices
            self.op_input_names[2]: QDQParameterConfig(support_activation=True),  # updates
        }


@register_runtime_checker_op
class UnsqueezeInputGenerator(OpInputGenerator):
    """Input generator for Unsqueeze operator.

    Unsqueeze operator documentation:
    - Input 1 (data): Original tensor to unsqueeze
    - Input 2 (axes): INT64 tensor with list of dimension indices to insert
    - No attributes in opset 21+

    Constraints:
    - axes should not contain duplicates
    - Each value in axes should be in range [-output_rank, output_rank-1]
    - output_rank = rank(data) + len(axes)
    - The order of values in axes does not matter

    Coverage strategy:
    - Test data shapes from 1D through 6D
    - Test adding dimensions at different positions
    - Test adding multiple dimensions at once
    - Test negative axis values
    """

    op_name = "Unsqueeze"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Unsqueeze has no attributes in opset 21+.

        The axes input is required and handled in
        get_input_and_infinite_attribute_combinations.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for Unsqueeze operator.

        Strategy:
        - Use representative shapes from 1D through 6D
        - For each data shape, test different axes patterns
        - Test single axis and multiple axes
        - Test both positive and negative axis values
        """
        combinations = []

        # Define (data_shape, axes_values) test cases
        # Format: (data_shape, axes_array)
        test_cases = [
            # 0D data (rank 0) -> 1D or higher
            ((), np.array([0], dtype=np.int64)),  # Add dim at front: () -> (1,)
            ((), np.array([0, 1], dtype=np.int64)),  # Multiple dims: () -> (1, 1)
            # 1D data (rank 1) -> 2D or higher
            ((6,), np.array([0], dtype=np.int64)),  # Add dim at front: (6,) -> (1, 6)
            ((6,), np.array([1], dtype=np.int64)),  # Add dim at end: (6,) -> (6, 1)
            ((6,), np.array([-1], dtype=np.int64)),  # Negative index
            ((6,), np.array([0, 2], dtype=np.int64)),  # Multiple dims: (6,) -> (1, 6, 1)
            # 2D data (rank 2) -> 3D or higher
            ((3, 4), np.array([0], dtype=np.int64)),  # (3, 4) -> (1, 3, 4)
            ((3, 4), np.array([1], dtype=np.int64)),  # (3, 4) -> (3, 1, 4)
            ((3, 4), np.array([2], dtype=np.int64)),  # (3, 4) -> (3, 4, 1)
            ((3, 4), np.array([-1], dtype=np.int64)),  # Negative index
            ((3, 4), np.array([0, 3], dtype=np.int64)),  # Multiple dims: (3, 4) -> (1, 3, 4, 1)
            # 3D data (rank 3) -> 4D or higher
            ((2, 3, 4), np.array([0], dtype=np.int64)),  # Add at front
            ((2, 3, 4), np.array([2], dtype=np.int64)),  # Add in middle
            ((2, 3, 4), np.array([-1], dtype=np.int64)),  # Add at end (negative)
            ((2, 3, 4), np.array([0, 4], dtype=np.int64)),  # Multiple dims
            # 4D data (rank 4) -> 5D or higher
            ((2, 3, 4, 5), np.array([0], dtype=np.int64)),  # NCHW -> (1, N, C, H, W)
            ((2, 3, 4, 5), np.array([1], dtype=np.int64)),  # Insert after batch
            ((2, 3, 4, 5), np.array([-1], dtype=np.int64)),  # Add at end
            ((2, 3, 4, 5), np.array([0, 2], dtype=np.int64)),  # Multiple dims
            # 5D data (rank 5) -> 6D or higher
            ((2, 2, 2, 3, 2), np.array([0], dtype=np.int64)),  # Add at front
            ((2, 2, 2, 3, 2), np.array([3], dtype=np.int64)),  # Add in middle
            ((2, 2, 2, 3, 2), np.array([-1], dtype=np.int64)),  # Add at end
            ((2, 2, 2, 3, 2), np.array([0, 5], dtype=np.int64)),  # Multiple dims
            # 6D data (rank 6) - note: result will be 7D or higher
            ((2, 2, 2, 2, 2, 3), np.array([0], dtype=np.int64)),  # Add at front
            ((2, 2, 2, 2, 2, 3), np.array([-1], dtype=np.int64)),  # Add at end
            ((2, 2, 2, 2, 2, 3), np.array([0, 3], dtype=np.int64)),  # Multiple dims
        ]

        for data_shape, axes in test_cases:
            combinations.append(
                {
                    "data": InputShapeConstraint(data_shape),
                    "axes": InputValueConstraint(axes),
                }
            )

        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for Unsqueeze operator testing.

        Args:
            properties: Base properties containing data_shape and axes_value

        Returns:
            Updated properties with Unsqueeze-specific derived values
        """
        item = properties.copy()
        item["data_dim"] = len(item["data_shape"])

        # axes may come from input (axes_value) in newer opsets or attr_axes in older ones
        axes_raw = None
        if "axes_value" in item and item["axes_value"] is not None:
            axes_raw = item["axes_value"]
        elif "attr_axes" in item and item["attr_axes"] is not None:
            axes_raw = item["attr_axes"]

        if axes_raw is None:
            raise ValueError("Unsqueeze requires axes input or attribute to derive output shape")

        axes_array = np.array(axes_raw)
        item["num_axes"] = len(axes_array)
        item["output_dim"] = item["data_dim"] + item["num_axes"]
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes/values with infinite possibilities
        """
        return ["data_shape", "axes_value"]

    def get_qdq_config(self):
        """Return QDQ configuration for Unsqueeze operator inputs."""
        return {
            "data": QDQParameterConfig(support_activation=True),
            "axes": QDQParameterConfig(support_non_qdq=True),
        }


@register_runtime_checker_op
class SplitInputGenerator(OpInputGenerator):
    """Input generator for Split operator.

    Split operator documentation:
    - Input 1 (input): The tensor to split
    - Input 2 (split): Optional INT64 tensor with sizes of each output
    - Attribute (axis): Which axis to split on (default 0)
      Negative values count from the back. Range: [-rank, rank-1]
    - Attribute (num_outputs): Number of outputs (alternative to split input)

    Constraints:
    - Either split input OR num_outputs attribute should be specified (not both)
    - If num_outputs is used, tensor is split into equal parts
      (last chunk may be smaller if not evenly divisible)
    - If split input is used, sum of values must equal dim size at axis
    - All values in split should be >= 0

    Coverage strategy:
    - Test input shapes from 1D through 6D
    - Test both split input and num_outputs approaches
    - Test different axis values
    - Test equal and unequal splits
    """

    op_name = "Split"
    expand_optionals = False

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Split has no simple finite attribute sets.

        Both axis and num_outputs depend on input shape, so they're handled
        in get_input_and_infinite_attribute_combinations.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for Split operator.

        Strategy:
        - Use representative shapes from 1D through 6D
        - For each shape, test both split and num_outputs approaches
        - Test different axis values appropriate for each shape
        - Test equal and unequal splits

        Note: For opset 13+, Split uses an optional 'split' input (not attribute).
        When split is omitted, the number of outputs must be known at graph
        construction time, which is not possible with onnxscript function calls.
        Therefore, we only test cases with explicit split values.
        """
        combinations = []

        # Test cases format: (input_shape, split_input, axis)
        # split_input is a numpy array with explicit split sizes
        test_cases = [
            # 1D input (rank 1) - only axis 0
            ((6,), np.array([3, 3], dtype=np.int64), 0),  # Equal split into 2 parts
            ((6,), np.array([2, 4], dtype=np.int64), 0),  # Unequal split
            ((7,), np.array([3, 2, 2], dtype=np.int64), 0),  # Split into 3 parts
            ((6,), np.array([1, 1, 1, 1, 1, 1], dtype=np.int64), 0),  # All ones split
            # 2D input (rank 2) - test axis 0 and 1
            ((6, 4), np.array([3, 3], dtype=np.int64), 0),  # Split rows equally
            ((6, 4), np.array([2, 2], dtype=np.int64), 1),  # Split columns equally
            ((6, 4), np.array([2, 4], dtype=np.int64), 0),  # Unequal row split
            ((6, 4), np.array([1, 3], dtype=np.int64), 1),  # Unequal column split
            ((6, 4), np.array([1, 1, 1, 1], dtype=np.int64), -1),  # Negative axis (last), 4 splits
            ((6, 4), np.array([1, 1, 1, 1, 1, 1], dtype=np.int64), 0),  # All ones split axis 0
            # 3D input (rank 3) - test different axes
            ((6, 3, 4), np.array([3, 3], dtype=np.int64), 0),  # Split first axis
            ((6, 3, 4), np.array([1, 1, 1], dtype=np.int64), 1),  # Split second axis into 3
            ((6, 3, 4), np.array([2, 2], dtype=np.int64), 2),  # Split third axis
            ((6, 3, 4), np.array([2, 4], dtype=np.int64), 0),  # Unequal split
            ((6, 3, 4), np.array([2, 2], dtype=np.int64), -1),  # Negative axis
            ((6, 3, 4), np.array([1, 1, 1, 1, 1, 1], dtype=np.int64), 0),  # All ones split axis 0
            # 4D input (rank 4) - NCHW format example
            ((4, 3, 6, 5), np.array([2, 2], dtype=np.int64), 0),  # Split batches
            ((4, 3, 6, 5), np.array([1, 1, 1], dtype=np.int64), 1),  # Split channels into 3
            ((4, 3, 6, 5), np.array([2, 2, 2], dtype=np.int64), 2),  # Split height into 3
            ((4, 3, 6, 5), np.array([2, 3, 1], dtype=np.int64), 2),  # Unequal split
            (
                (4, 3, 6, 5),
                np.array([1, 1, 1, 1, 1], dtype=np.int64),
                -1,
            ),  # Split width, negative axis
            ((4, 3, 6, 5), np.array([1, 1, 1, 1], dtype=np.int64), 0),  # All ones split axis 0
            # 5D input (rank 5)
            ((4, 2, 2, 3, 2), np.array([2, 2], dtype=np.int64), 0),  # Split first axis
            ((4, 2, 2, 3, 2), np.array([1, 1, 1], dtype=np.int64), 3),  # Split middle axis into 3
            ((4, 2, 2, 3, 4), np.array([2, 1, 1], dtype=np.int64), -1),  # Unequal split
            ((4, 2, 2, 3, 2), np.array([1, 1, 1, 1], dtype=np.int64), 0),  # All ones split axis 0
            # 6D input (rank 6)
            ((6, 2, 2, 2, 2, 3), np.array([2, 2, 2], dtype=np.int64), 0),  # Split first axis into 3
            ((6, 2, 2, 2, 2, 3), np.array([1, 1, 1], dtype=np.int64), -1),  # Split last axis into 3
            ((6, 2, 2, 2, 2, 3), np.array([1, 1], dtype=np.int64), 3),  # Equal split
            (
                (6, 2, 2, 2, 2, 3),
                np.array([1, 1, 1, 1, 1, 1], dtype=np.int64),
                0,
            ),  # All ones split axis 0
        ]

        for input_shape, split_input, axis in test_cases:
            # Add test case with explicit split input
            combinations.append(
                {
                    "input": InputShapeConstraint(input_shape),
                    "split": InputValueConstraint(split_input),
                    "axis": axis,
                }
            )

            # Also add num_outputs test case if the attribute exists (opset 18+)
            # Note: Although ONNX spec says tensor doesn't need to be evenly splittable,
            # onnxruntime has issues when axis_size % num_outputs != 0, so we only add
            # num_outputs test cases when evenly divisible
            # See: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/tensor/split.cc
            if "num_outputs" in self.op_attribute_names:
                num_outputs = split_input.size
                axis_size = input_shape[axis]
                if axis_size % num_outputs == 0:
                    combinations.append(
                        {
                            "input": InputShapeConstraint(input_shape),
                            "axis": axis,
                            "num_outputs": num_outputs,
                        }
                    )

        print(f"Generated {len(combinations)} input combinations for Split operator.")
        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for Split operator testing.

        Args:
            properties: Base properties containing input_shape and either
                split_value or attr_num_outputs

        Returns:
            Updated properties with Split-specific derived values
        """
        item = properties.copy()
        item["input_dim"] = len(item["input_shape"])
        # Handle both cases: explicit split input or num_outputs attribute
        if "split_value" in item and item["split_value"] is not None:
            split_array = np.array(item["split_value"])
            item["num_outputs"] = len(split_array)
        elif "attr_num_outputs" in item:
            item["num_outputs"] = item["attr_num_outputs"]
        elif "attr_split" in item and item["attr_split"] is not None:
            # Older opsets use split as an attribute instead of an input
            split_array = np.array(item["attr_split"])
            item["num_outputs"] = len(split_array)
        else:
            # caller is get_query_conditions_for_*, "n_outputs" must be present
            item["num_outputs"] = item["n_outputs"]
        # num_outputs may already be set from the combination
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes/values with infinite possibilities
        """
        return [
            "input_shape",
            "split_value",
            "attr_num_outputs",
            "num_outputs",
            "attr_axis",
            "attr_split",
        ]

    def infer_output_types(
        self, kwargs: dict[str, Any], tags: dict[str, Any], required_outputs_only: bool = True
    ) -> list[str]:
        """Infer output types for Split operator based on number of outputs."""
        if "split" in kwargs:
            num_output = kwargs["split"].size
        elif "num_outputs" in kwargs:
            num_output = kwargs["num_outputs"]
        else:
            raise ValueError("Number of outputs cannot be determined for Split operator.")
        type_var_key = self.schema.outputs[0].type_str
        annotation = tags["type_vars"][f"{type_var_key}_{self.op_name}"]
        return [annotation] * num_output

    def get_qdq_config(self):
        """Return QDQ configuration for Split operator inputs."""
        return {
            "input": QDQParameterConfig(support_activation=True),
            "split": QDQParameterConfig(support_non_qdq=True),
        }
