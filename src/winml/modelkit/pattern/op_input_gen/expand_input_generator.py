# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generator for Expand operator."""

import numpy as np

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    InputValueConstraint,
    OpInputGenerator,
    register_runtime_checker_op,
)


@register_runtime_checker_op
class ExpandInputGenerator(OpInputGenerator):
    """Input generator for Expand operator.

    Expand operator documentation:
    - Input 1 (input): Input tensor to be broadcasted
    - Input 2 (shape): Target shape as 1-D INT64 tensor

    Broadcast rules (similar to numpy.broadcast_to):
    - Dimensions are right-aligned
    - Two corresponding dimensions must have the same value, or one of them is 1
    - Output.shape may not equal target shape when:
      - Some dimensions in shape equal 1
      - shape.ndim < input.shape.ndim

    Key difference from numpy.broadcast_to:
    - Expand allows shape to be smaller than input.size()

    Coverage strategy:
    - Input dimensions: 0D (scalar) through 6D
    - Broadcasting patterns:
      - Same shape (no broadcast)
      - Broadcast single dimension (1 -> n)
      - Broadcast multiple dimensions
      - Add dimensions (prepend 1s to input shape)
      - Larger output rank than input
      - Bidirectional broadcast (both tensors have size-1 dims)
    - Edge cases:
      - Empty shape array with scalar input
      - Shape with 1s that don't broadcast
      - Multiple dimensions broadcast simultaneously
      - Real-world patterns (e.g., ConvNeXt model shapes)

    Derived properties:
    - input_dim: Number of dimensions in input tensor
    - shape_len: Length of shape tensor
    - input_broadcasting_to_shape: True if input broadcasts to shape
    - shape_broadcasting_to_input: True if shape broadcasts to input
    """

    op_name = "Expand"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Returns finite attribute sets for Expand.

        Expand has no attributes, so return empty dict.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Returns comprehensive input combinations for Expand operator.

        Coverage strategy:
        - Input dimensions: 1D through 6D
        - Broadcasting patterns: various broadcast scenarios
        - Ordered from smallest to largest dimensions
        """
        return [
            # ===== 0D Inputs (dimension 0) =====
            {
                "input": InputShapeConstraint(()),
                "shape": InputValueConstraint(np.array([], dtype=np.int64)),
            },
            {
                "input": InputShapeConstraint(()),
                "shape": InputValueConstraint(np.array([6], dtype=np.int64)),
            },
            {
                "input": InputShapeConstraint((6,)),
                "shape": InputValueConstraint(np.array([], dtype=np.int64)),
            },
            {
                "input": InputShapeConstraint(()),
                "shape": InputValueConstraint(np.array([4, 5], dtype=np.int64)),
            },
            {
                "input": InputShapeConstraint((4, 5)),
                "shape": InputValueConstraint(np.array([], dtype=np.int64)),
            },
            {
                "input": InputShapeConstraint(()),
                "shape": InputValueConstraint(np.array([3, 2, 5], dtype=np.int64)),
            },
            {
                "input": InputShapeConstraint((3, 2, 5)),
                "shape": InputValueConstraint(np.array([], dtype=np.int64)),
            },
            {
                "input": InputShapeConstraint(()),
                "shape": InputValueConstraint(np.array([2, 3, 4, 5], dtype=np.int64)),
            },
            {
                "input": InputShapeConstraint((2, 3, 4, 5)),
                "shape": InputValueConstraint(np.array([], dtype=np.int64)),
            },
            # ===== 1D Inputs (dimension 1) =====
            # Equal shapes - no broadcasting
            {
                "input": InputShapeConstraint((6,)),
                "shape": InputValueConstraint(np.array([6], dtype=np.int64)),
            },
            # A broadcasts to B (scalar to vector)
            {
                "input": InputShapeConstraint((1,)),
                "shape": InputValueConstraint(np.array([6], dtype=np.int64)),
            },
            # B broadcasts to A (scalar to vector)
            {
                "input": InputShapeConstraint((6,)),
                "shape": InputValueConstraint(np.array([1], dtype=np.int64)),
            },
            # Convnext
            {
                "input": InputShapeConstraint((1, 56, 56, 96)),
                "shape": InputValueConstraint(np.array([96], dtype=np.int64)),
            },
            # ===== 2D Inputs (dimension 2) =====
            # Equal shapes
            {
                "input": InputShapeConstraint((4, 5)),
                "shape": InputValueConstraint(np.array([4, 5], dtype=np.int64)),
            },
            # A broadcasts to B (1D to 2D)
            {
                "input": InputShapeConstraint((5,)),
                "shape": InputValueConstraint(np.array([4, 5], dtype=np.int64)),
            },
            # B broadcasts to A (1D to 2D)
            {
                "input": InputShapeConstraint((4, 5)),
                "shape": InputValueConstraint(np.array([5], dtype=np.int64)),
            },
            # 1D to 3D broadcasting patterns from P0 models
            {
                "input": InputShapeConstraint((768,)),
                "shape": InputValueConstraint(np.array([1, 77, 768], dtype=np.int64)),
            },
            # Bidirectional broadcast (both have size-1 dims)
            {
                "input": InputShapeConstraint((4, 1)),
                "shape": InputValueConstraint(np.array([1, 5], dtype=np.int64)),
            },
            # A broadcasts to B (column vector to matrix)
            {
                "input": InputShapeConstraint((4, 1)),
                "shape": InputValueConstraint(np.array([4, 5], dtype=np.int64)),
            },
            # B broadcasts to A (row vector to matrix)
            {
                "input": InputShapeConstraint((4, 5)),
                "shape": InputValueConstraint(np.array([1, 5], dtype=np.int64)),
            },
            # ===== 3D Inputs (dimension 3) =====
            # Equal shapes
            {
                "input": InputShapeConstraint((3, 2, 5)),
                "shape": InputValueConstraint(np.array([3, 2, 5], dtype=np.int64)),
            },
            # A broadcasts to B (2D to 3D)
            {
                "input": InputShapeConstraint((2, 5)),
                "shape": InputValueConstraint(np.array([3, 2, 5], dtype=np.int64)),
            },
            # B broadcasts to A (2D to 3D)
            {
                "input": InputShapeConstraint((3, 2, 5)),
                "shape": InputValueConstraint(np.array([2, 5], dtype=np.int64)),
            },
            # Bidirectional broadcast in 3D
            {
                "input": InputShapeConstraint((3, 1, 5)),
                "shape": InputValueConstraint(np.array([1, 2, 1], dtype=np.int64)),
            },
            # A broadcasts to B (with size-1 dims)
            {
                "input": InputShapeConstraint((3, 1, 1)),
                "shape": InputValueConstraint(np.array([3, 2, 5], dtype=np.int64)),
            },
            # B broadcasts to A (with size-1 dims)
            {
                "input": InputShapeConstraint((3, 2, 5)),
                "shape": InputValueConstraint(np.array([1, 1, 5], dtype=np.int64)),
            },
            # ===== 4D Inputs (dimension 4) =====
            # Equal shapes (batch, channels, height, width)
            {
                "input": InputShapeConstraint((2, 4, 5, 6)),
                "shape": InputValueConstraint(np.array([2, 4, 5, 6], dtype=np.int64)),
            },
            # A broadcasts to B (3D to 4D)
            {
                "input": InputShapeConstraint((4, 5, 6)),
                "shape": InputValueConstraint(np.array([2, 4, 5, 6], dtype=np.int64)),
            },
            # B broadcasts to A (3D to 4D)
            {
                "input": InputShapeConstraint((2, 4, 5, 6)),
                "shape": InputValueConstraint(np.array([4, 5, 6], dtype=np.int64)),
            },
            # Bidirectional broadcast in 4D
            {
                "input": InputShapeConstraint((2, 1, 5, 1)),
                "shape": InputValueConstraint(np.array([1, 4, 1, 6], dtype=np.int64)),
            },
            # A broadcasts to B (channel-wise operation)
            {
                "input": InputShapeConstraint((1, 4, 1, 1)),
                "shape": InputValueConstraint(np.array([2, 4, 5, 6], dtype=np.int64)),
            },
            # B broadcasts to A (spatial broadcast)
            {
                "input": InputShapeConstraint((2, 4, 5, 6)),
                "shape": InputValueConstraint(np.array([1, 1, 5, 6], dtype=np.int64)),
            },
            # Convnext
            {
                "input": InputShapeConstraint((96,)),
                "shape": InputValueConstraint(np.array([1, 56, 56, 96], dtype=np.int64)),
            },
            # ===== 5D Inputs (dimension 5) =====
            # Equal shapes (batch, channels, depth, height, width)
            {
                "input": InputShapeConstraint((2, 2, 3, 4, 5)),
                "shape": InputValueConstraint(np.array([2, 2, 3, 4, 5], dtype=np.int64)),
            },
            # A broadcasts to B (4D to 5D)
            {
                "input": InputShapeConstraint((2, 3, 4, 5)),
                "shape": InputValueConstraint(np.array([2, 2, 3, 4, 5], dtype=np.int64)),
            },
            # B broadcasts to A (4D to 5D)
            {
                "input": InputShapeConstraint((2, 2, 3, 4, 5)),
                "shape": InputValueConstraint(np.array([2, 3, 4, 5], dtype=np.int64)),
            },
            # Bidirectional broadcast in 5D
            {
                "input": InputShapeConstraint((2, 1, 3, 1, 5)),
                "shape": InputValueConstraint(np.array([1, 2, 1, 4, 1], dtype=np.int64)),
            },
            # ===== 6D Inputs (dimension 6 - maximum) =====
            # Equal shapes
            {
                "input": InputShapeConstraint((2, 2, 2, 2, 3, 3)),
                "shape": InputValueConstraint(np.array([2, 2, 2, 2, 3, 3], dtype=np.int64)),
            },
            # A broadcasts to B (5D to 6D)
            {
                "input": InputShapeConstraint((2, 2, 2, 3, 3)),
                "shape": InputValueConstraint(np.array([2, 2, 2, 2, 3, 3], dtype=np.int64)),
            },
            # B broadcasts to A (5D to 6D)
            {
                "input": InputShapeConstraint((2, 2, 2, 2, 3, 3)),
                "shape": InputValueConstraint(np.array([2, 2, 2, 3, 3], dtype=np.int64)),
            },
            # Bidirectional broadcast in 6D
            {
                "input": InputShapeConstraint((2, 1, 2, 1, 3, 1)),
                "shape": InputValueConstraint(np.array([1, 2, 1, 2, 1, 3], dtype=np.int64)),
            },
        ]

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for Expand operator testing.

        Args:
            properties: Base properties containing input_shape, shape_value

        Returns:
            Updated properties with Expand-specific derived values
        """
        item = properties.copy()

        # Handle case where shape_value is unknown
        # This occurs when the shape input is dynamic (e.g., comes from model input
        # or another node's output) rather than being a constant initializer or
        # Constant node. In such cases, the static analyzer cannot determine the
        # actual shape values at compile time.
        if "shape_value" not in item:
            item["input_dim"] = 0
            item["shape_len"] = -1  # Unknown length
            item["input_broadcasting_to_shape"] = False
            item["shape_broadcasting_to_input"] = False
            return item

        input_shape = tuple(item["input_shape"])
        shape_value = tuple(item["shape_value"])

        # Check if broadcasting occurs due to different dimensionality
        dim_input = len(input_shape)
        dim_shape = len(shape_value)

        item["input_dim"] = dim_input
        item["shape_len"] = dim_shape
        input_broadcasting_to_shape = dim_input < dim_shape
        shape_broadcasting_to_input = dim_shape < dim_input

        # make shapes same length by prepending 1s
        if len(input_shape) < len(shape_value):
            input_shape = (1,) * (len(shape_value) - len(input_shape)) + input_shape
        elif len(shape_value) < len(input_shape):
            shape_value = (1,) * (len(input_shape) - len(shape_value)) + shape_value

        # Check if broadcasting occurs due to different axis sizes
        input_broadcasting_to_shape = input_broadcasting_to_shape or any(
            xs != 0 and xs < ys for xs, ys in zip(input_shape, shape_value, strict=False)
        )
        shape_broadcasting_to_input = shape_broadcasting_to_input or any(
            ys != 0 and ys < xs for xs, ys in zip(input_shape, shape_value, strict=False)
        )

        item["input_broadcasting_to_shape"] = input_broadcasting_to_shape
        item["shape_broadcasting_to_input"] = shape_broadcasting_to_input
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes or values
            with infinite possibilities
        """
        return ["input_shape", "input_value", "shape_shape", "shape_value"]
