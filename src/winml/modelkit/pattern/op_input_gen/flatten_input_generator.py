# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generator for Flatten operator."""

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    OpInputGenerator,
    QDQParameterConfig,
    register_runtime_checker_op,
)


@register_runtime_checker_op
class FlattenInputGenerator(OpInputGenerator):
    """Input generator for Flatten operator.

    Flatten operator documentation:
    - Input: input tensor of rank >= axis
    - Attribute (axis): Indicate up to which input dimensions (exclusive) should be
      flattened to the outer dimension. Must be in range [-r, r] where r is the rank.
      Default is 1. Negative values mean counting from the back.

    Output shape:
    - Input shape (d_0, d_1, ... d_n) becomes (d_0 X d_1 ... d_(axis-1), d_axis X ... X d_n)
    - When axis=0, output is (1, total_elements)
    - When axis=rank, output is (total_elements, 1)

    Coverage strategy:
    - Input dimensions: 1D through 6D
    - Axis values: test different positions including 0, 1, -1, middle positions
    - Edge cases: axis at boundaries (0 and rank), negative axis values
    """

    op_name = "Flatten"

    def get_finite_attribute_sets(self) -> dict[str, list[int]]:
        """Returns finite attribute sets for Flatten.

        Note: axis is infinite (depends on input rank), so it returns empty dict.
        Each input combination will specify its own valid axis value.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint | int]]:
        """Returns comprehensive input combinations for Flatten operator.

        Coverage strategy:
        - Input dimensions: 0D through 6D (dim in range(7))
        - Axis positions: 0 to dim inclusive (axis in range(dim + 1))
        - Shape: predefined shapes for each dimension
        """
        # Predefined shapes for each dimension
        shapes = [
            (),  # 0D
            (6,),  # 1D
            (2, 3),  # 2D
            (2, 3, 4),  # 3D
            (2, 3, 4, 5),  # 4D
            (2, 2, 2, 3, 2),  # 5D
            (2, 2, 2, 2, 2, 2),  # 6D
        ]

        combinations = []
        for dim, shape in enumerate(shapes):  # 0D -> 6D
            for axis in range(dim + 1):
                combinations.append(
                    {
                        "input": InputShapeConstraint(shape),
                        "axis": axis,
                    }
                )

        return combinations

    def derive_properties(self, properties: dict[str, any]) -> dict[str, any]:
        """Derive additional properties for Flatten operator testing.

        Args:
            properties: Base properties containing input_shape and axis

        Returns:
            Updated properties with Flatten-specific derived values
        """
        item = properties.copy()
        item["input_dim"] = len(item["input_shape"])
        item["is_first_axis"] = item["attr_axis"] in (0, -item["input_dim"])
        item["is_last_axis"] = item["attr_axis"] in (item["input_dim"], -1)
        normalized_axis = (
            item["attr_axis"] if item["attr_axis"] >= 0 else item["input_dim"] + item["attr_axis"]
        )
        item["normalized_axis"] = normalized_axis

        return item

    def get_infinite_property_names(self) -> list[str]:
        """Returns names of infinite properties for Flatten operator."""
        return ["input_shape", "input_value", "attr_axis"]

    def get_qdq_config(self):
        return {
            "input": QDQParameterConfig(support_activation=True),
        }
