# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generator for Transpose ONNX operator."""

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    OpInputGenerator,
    QDQParameterConfig,
    register_runtime_checker_op,
)


@register_runtime_checker_op
class TransposeInputGenerator(OpInputGenerator):
    """Input generator for Transpose operator.

    Transpose operator documentation:
    - Input (data): Input tensor to be transposed
    - Attribute (perm): List of integers specifying the permutation of axes
      If not provided, reverses the axes (default behavior)

    Constraints:
    - perm must be a permutation of [0, 1, ..., rank-1]
    - All values in perm must be unique
    - Length of perm must equal input rank

    Coverage strategy:
    - Test shapes from 1D through 6D (following Reshape generator)
    - 2 permutations per shape:
      1. Default permutation (None) - reverses axes
      2. Custom permutation - varies by shape
    """

    op_name = "Transpose"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Transpose has no simple finite attribute sets.

        The perm attribute depends on input rank, so it's handled
        in get_input_and_infinite_attribute_combinations.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Return input combinations for Transpose operator.

        Strategy:
        - Use representative shapes from Reshape generator (1D-6D)
        - For each shape, test 2 permutations:
          1. Default (no perm attribute) - reverses axes
          2. Custom permutation that's commonly used
        """
        combinations = []

        # Define (shape, custom_perm) pairs
        # Each shape gets 2 tests: one with no perm, one with custom perm
        test_cases = [
            # 1D - only one permutation possible
            ((6,), [0]),  # Identity
            # 2D - common: transpose matrix
            ((2, 3), [1, 0]),  # Swap axes
            # 2D - another shape
            ((4, 5), [1, 0]),  # Swap axes
            # 3D - common: move last axis to first
            ((2, 3, 4), [2, 0, 1]),  # Rotate axes
            # 3D - another pattern: swap first two axes
            ((3, 4, 5), [1, 0, 2]),  # Swap first two
            # 4D - common: NCHW to NHWC (batch, channels, height, width)
            ((2, 3, 4, 5), [0, 2, 3, 1]),  # channels last
            # 4D - another pattern: swap middle axes
            ((2, 3, 4, 5), [0, 2, 1, 3]),  # Swap axes 1 and 2
            # 5D - rotate last 3 axes
            ((2, 2, 2, 3, 2), [0, 1, 3, 4, 2]),  # Move axis 2 to end
            # 5D - another pattern
            ((2, 2, 3, 2, 2), [0, 2, 1, 3, 4]),  # Swap axes 1 and 2
            # 6D - complex permutation
            ((2, 2, 2, 2, 2, 3), [0, 1, 3, 2, 4, 5]),  # Swap middle axes
            # 6D - another pattern
            ((2, 2, 2, 2, 2, 2), [1, 0, 2, 3, 4, 5]),  # Swap first two axes
        ]

        if self.qdq_generator:
            # TODO: in QNN, for ALL permutations, they are not supported, not sure why
            #  [0, 2, 3, 1]  # NCHW -> NHWC
            #  [0, 3, 1, 2]  # NHWC -> NCHW
            test_cases[5] = ((2, 3, 4, 5), [0, 1, 3, 2])

        for shape, custom_perm in test_cases:
            # Test 1: Default permutation (reverses axes)
            # No perm attribute means default behavior
            # combinations.append({
            #     "data": InputShapeConstraint(shape),
            # })

            # Test 2: Custom permutation
            # Add perm as a finite attribute value
            combinations.append(
                {
                    "data": InputShapeConstraint(shape),
                    "perm": custom_perm,
                }
            )

        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties based on input shape."""
        item = properties.copy()
        input_param_name = self.op_input_names[0]
        shape = item[f"{input_param_name}_shape"]
        item[f"{input_param_name}_dim"] = len(shape)
        # temp workaround for AMD QDQ edge cases with input shape (2, 3)
        item[f"{input_param_name}_shape_is_2_3"] = shape == (2, 3)
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite value ranges."""
        input_param_name = self.op_input_names[0]
        return [f"{input_param_name}_shape", "attr_perm"]

    def get_qdq_config(self):
        """Return QDQ configuration for Transpose operator inputs."""
        return {
            "data": QDQParameterConfig(support_activation=True),
        }
