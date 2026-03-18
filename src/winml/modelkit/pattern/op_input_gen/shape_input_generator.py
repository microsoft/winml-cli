# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generator for Shape operator."""

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    OpInputGenerator,
    register_runtime_checker_op,
)


@register_runtime_checker_op
class ShapeInputGenerator(OpInputGenerator):
    """Input generator for Shape operator.

    Shape operator documentation:
    - Input: data - An input tensor of any rank
    - Attribute (start): Starting axis for slicing the shape. Default is 0.
      Negative values mean counting dimensions from the back.
    - Attribute (end): Optional ending axis for slicing the shape (exclusive).
      Negative values mean counting dimensions from the back.
      If omitted, sizes of all axes up to (including) the last one will be included.

    Output:
    - 1D int64 tensor containing the shape (or slice of shape) of the input tensor

    Examples:
    - Input shape [2, 3, 4], no attributes -> Output: [2, 3, 4]
    - Input shape [2, 3, 4], start=-1 -> Output: [4]
    - Input shape [2, 3, 4], end=-1 -> Output: [2, 3]
    - Input shape [2, 3, 4], start=1, end=2 -> Output: [3]

    Coverage strategy:
    - Input dimensions: 1D through 6D
    - Start values: 0, 1, middle, last, negative values
    - End values: None (default), explicit values, negative values
    - Edge cases: start > end (empty result), boundaries
    """

    op_name = "Shape"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Returns finite attribute sets for Shape.

        Note: start and end are infinite (depend on input rank), so return empty dict.
        Each input combination will specify its own valid start/end values.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint | int | None]]:
        """Returns comprehensive input combinations for Shape operator.

        Coverage strategy:
        - Input dimensions: 1D through 6D
        - Start positions: 0, 1, middle, last, negative values
        - End positions: explicit values, negative values
        - Edge cases: start at boundaries, end at boundaries
        - Ordered from smallest to largest dimensions

        Note: We always provide explicit values for both start and end attributes
        to ensure consistent querying across all combinations.
        """
        combinations = []

        # ===== 1D Input (rank 1) =====
        # start=0, end=1: (6,) -> [6] (full shape)
        combinations.append(
            {
                "data": InputShapeConstraint((6,)),
                "start": 0,
                "end": 1,
            }
        )
        # start=0, end=-1: (6,) -> [] (empty, since end=-1 means axis 0 exclusive)
        combinations.append(
            {
                "data": InputShapeConstraint((6,)),
                "start": 0,
                "end": -1,
            }
        )
        # start=-1, end=1: (6,) -> [6] (last dimension)
        combinations.append(
            {
                "data": InputShapeConstraint((6,)),
                "start": -1,
                "end": 1,
            }
        )

        # ===== 2D Input (rank 2) =====
        # start=0, end=2: (2, 3) -> [2, 3] (full shape)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3)),
                "start": 0,
                "end": 2,
            }
        )
        # start=0, end=1: (2, 3) -> [2] (first dimension only)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3)),
                "start": 0,
                "end": 1,
            }
        )
        # start=1, end=2: (2, 3) -> [3] (second dimension only)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3)),
                "start": 1,
                "end": 2,
            }
        )
        # start=-1, end=2: (2, 3) -> [3] (last dimension)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3)),
                "start": -1,
                "end": 2,
            }
        )
        # start=0, end=-1: (2, 3) -> [2] (exclude last)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3)),
                "start": 0,
                "end": -1,
            }
        )
        # start=-2, end=-1: (2, 3) -> [2] (second to last only)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3)),
                "start": -2,
                "end": -1,
            }
        )

        # ===== 3D Input (rank 3) =====
        # start=0, end=3: (2, 3, 4) -> [2, 3, 4] (full shape)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "start": 0,
                "end": 3,
            }
        )
        # start=0, end=1: (2, 3, 4) -> [2] (first only)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "start": 0,
                "end": 1,
            }
        )
        # start=1, end=2: (2, 3, 4) -> [3] (middle only)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "start": 1,
                "end": 2,
            }
        )
        # start=1, end=3: (2, 3, 4) -> [3, 4] (skip first)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "start": 1,
                "end": 3,
            }
        )
        # start=0, end=2: (2, 3, 4) -> [2, 3] (exclude last)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "start": 0,
                "end": 2,
            }
        )
        # start=-1, end=3: (2, 3, 4) -> [4] (last only)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "start": -1,
                "end": 3,
            }
        )
        # start=0, end=-1: (2, 3, 4) -> [2, 3] (exclude last)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "start": 0,
                "end": -1,
            }
        )
        # start=-2, end=-1: (2, 3, 4) -> [3] (second to last only)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "start": -2,
                "end": -1,
            }
        )

        # ===== 4D Input (rank 4) =====
        # start=0, end=4: (2, 3, 4, 5) -> [2, 3, 4, 5] (full shape)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4, 5)),
                "start": 0,
                "end": 4,
            }
        )
        # start=0, end=2: (2, 3, 4, 5) -> [2, 3] (first two)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4, 5)),
                "start": 0,
                "end": 2,
            }
        )
        # start=2, end=4: (2, 3, 4, 5) -> [4, 5] (last two)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4, 5)),
                "start": 2,
                "end": 4,
            }
        )
        # start=1, end=3: (2, 3, 4, 5) -> [3, 4] (middle two)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4, 5)),
                "start": 1,
                "end": 3,
            }
        )
        # start=-2, end=4: (2, 3, 4, 5) -> [4, 5] (last two via negative start)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4, 5)),
                "start": -2,
                "end": 4,
            }
        )
        # start=0, end=-2: (2, 3, 4, 5) -> [2, 3] (exclude last two)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4, 5)),
                "start": 0,
                "end": -2,
            }
        )
        # start=-3, end=-1: (2, 3, 4, 5) -> [3, 4] (middle via negative)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 3, 4, 5)),
                "start": -3,
                "end": -1,
            }
        )

        # ===== 5D Input (rank 5) =====
        # start=0, end=5: (2, 2, 3, 4, 5) -> [2, 2, 3, 4, 5] (full shape)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 2, 3, 4, 5)),
                "start": 0,
                "end": 5,
            }
        )
        # start=1, end=4: (2, 2, 3, 4, 5) -> [2, 3, 4] (middle three)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 2, 3, 4, 5)),
                "start": 1,
                "end": 4,
            }
        )
        # start=0, end=3: (2, 2, 3, 4, 5) -> [2, 2, 3] (first three)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 2, 3, 4, 5)),
                "start": 0,
                "end": 3,
            }
        )
        # start=2, end=5: (2, 2, 3, 4, 5) -> [3, 4, 5] (last three)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 2, 3, 4, 5)),
                "start": 2,
                "end": 5,
            }
        )
        # start=-1, end=5: (2, 2, 3, 4, 5) -> [5] (last only)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 2, 3, 4, 5)),
                "start": -1,
                "end": 5,
            }
        )
        # start=0, end=-1: (2, 2, 3, 4, 5) -> [2, 2, 3, 4] (exclude last)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 2, 3, 4, 5)),
                "start": 0,
                "end": -1,
            }
        )

        # ===== 6D Input (rank 6 - maximum) =====
        # start=0, end=6: (2, 2, 2, 2, 2, 3) -> [2, 2, 2, 2, 2, 3] (full shape)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 2, 2, 2, 2, 3)),
                "start": 0,
                "end": 6,
            }
        )
        # start=0, end=3: (2, 2, 2, 2, 2, 3) -> [2, 2, 2] (first three)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 2, 2, 2, 2, 3)),
                "start": 0,
                "end": 3,
            }
        )
        # start=3, end=6: (2, 2, 2, 2, 2, 3) -> [2, 2, 3] (last three)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 2, 2, 2, 2, 3)),
                "start": 3,
                "end": 6,
            }
        )
        # start=2, end=4: (2, 2, 2, 2, 2, 3) -> [2, 2] (middle two)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 2, 2, 2, 2, 3)),
                "start": 2,
                "end": 4,
            }
        )
        # start=-3, end=6: (2, 2, 2, 2, 2, 3) -> [2, 2, 3] (last three via negative)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 2, 2, 2, 2, 3)),
                "start": -3,
                "end": 6,
            }
        )
        # start=0, end=-3: (2, 2, 2, 2, 2, 3) -> [2, 2, 2] (exclude last three)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 2, 2, 2, 2, 3)),
                "start": 0,
                "end": -3,
            }
        )
        # start=-4, end=-2: (2, 2, 2, 2, 2, 3) -> [2, 2] (middle via negative)
        combinations.append(
            {
                "data": InputShapeConstraint((2, 2, 2, 2, 2, 3)),
                "start": -4,
                "end": -2,
            }
        )

        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for Shape operator testing.

        Args:
            properties: Base properties containing data_shape, start, end

        Returns:
            Updated properties with Shape-specific derived values
        """
        item = properties.copy()
        input_name = self.op_input_names[0]
        data_shape = item[f"{input_name}_shape"]
        data_dim = len(data_shape)
        item[f"{input_name}_dim"] = data_dim
        attr_start = item.get("attr_start", 0)
        if attr_start is None:
            attr_start = 0
        attr_end = item.get("attr_end", data_dim)
        if attr_end is None:
            attr_end = data_dim
        item["attr_start_less_than_end"] = attr_start < attr_end

        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes or values
            with infinite possibilities
        """
        input_name = self.op_input_names[0]
        return [f"{input_name}_shape", f"{input_name}_value", "attr_start", "attr_end"]
