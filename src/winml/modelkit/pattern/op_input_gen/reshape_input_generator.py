# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    InputValueConstraint,
    OpInputGenerator,
    QDQParameterConfig,
    register_runtime_checker_op,
)


@register_runtime_checker_op
class ReshapeInputGenerator(OpInputGenerator):
    """Complete input generator for the Reshape operator.

    Reshape operator documentation:
    - Input 1 (data): Input tensor to be reshaped
    - Input 2 (shape): Target shape (INT64 tensor)
    - Attribute (allowzero): By default (0), zero in shape means copy dimension
      from input. When set to 1, zero in shape means explicit zero dimension.

    Constraints:
    - At most one dimension in shape can be -1 (inferred from total elements)
    - Total number of elements must be preserved (except with -1 inference)
    - If allowzero=1, shape cannot contain both 0 and -1
    """

    op_name = "Reshape"

    def get_finite_attribute_sets(self) -> dict[str, list[int]]:
        """Returns finite attribute sets for Reshape.

        allowzero: [0, 1]
            - 0: zero in shape copies dimension from input (default behavior)
            - 1: zero in shape means explicit zero dimension
        """
        return {"allowzero": [0, 1]}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Returns comprehensive input combinations for Reshape operator.

        Coverage strategy:
        - Input dimensions: 1D through 6D (as per spec: max 6 dimensions)
        - Shape patterns: regular reshape, -1 (inferred), 0 (copy/explicit),
          scalar output
        - Axis sizes: limited to max 6 per axis (as per spec)
        - Ordered from smallest to largest dimensions
        """
        return [
            # ===== 1D Input (dimension 1) =====
            # Scalar to 1D
            {
                "data": InputShapeConstraint((1,)),
                "shape": InputValueConstraint(np.array([1], dtype=np.int64)),
            },
            # 1D to scalar (empty shape)
            {
                "data": InputShapeConstraint((1,)),
                "shape": InputValueConstraint(np.array([], dtype=np.int64)),
            },
            # 1D to 1D (different size)
            {
                "data": InputShapeConstraint((6,)),
                "shape": InputValueConstraint(np.array([6], dtype=np.int64)),
            },
            # 1D to 2D
            {
                "data": InputShapeConstraint((6,)),
                "shape": InputValueConstraint(np.array([2, 3], dtype=np.int64)),
            },
            # 1D to 2D with -1 (inferred dimension)
            {
                "data": InputShapeConstraint((12,)),
                "shape": InputValueConstraint(np.array([3, -1], dtype=np.int64)),
            },
            # ===== 2D Input (dimension 2) =====
            # 2D to 1D
            {
                "data": InputShapeConstraint((2, 3)),
                "shape": InputValueConstraint(np.array([6], dtype=np.int64)),
            },
            # 2D to 2D (different shape)
            {
                "data": InputShapeConstraint((2, 3)),
                "shape": InputValueConstraint(np.array([3, 2], dtype=np.int64)),
            },
            # 2D to 2D with -1
            {
                "data": InputShapeConstraint((4, 5)),
                "shape": InputValueConstraint(np.array([-1, 10], dtype=np.int64)),
            },
            # 2D to 3D
            {
                "data": InputShapeConstraint((6, 4)),
                "shape": InputValueConstraint(np.array([2, 3, 4], dtype=np.int64)),
            },
            # 2D to 3D with -1
            {
                "data": InputShapeConstraint((2, 6)),
                "shape": InputValueConstraint(np.array([3, 2, -1], dtype=np.int64)),
            },
            # 2D to 4D, Convnext
            {
                "data": InputShapeConstraint((49, 768)),
                "shape": InputValueConstraint(np.array([1, 7, 7, 768], dtype=np.int64)),
            },
            # ===== 3D Input (dimension 3) =====
            # 3D to 1D (flatten)
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "shape": InputValueConstraint(np.array([24], dtype=np.int64)),
            },
            # 3D to 2D
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "shape": InputValueConstraint(np.array([6, 4], dtype=np.int64)),
            },
            # 3D to 3D (different shape)
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "shape": InputValueConstraint(np.array([4, 3, 2], dtype=np.int64)),
            },
            # 3D to 3D with -1
            {
                "data": InputShapeConstraint((3, 4, 5)),
                "shape": InputValueConstraint(np.array([5, -1, 3], dtype=np.int64)),
            },
            # 3D to 4D
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "shape": InputValueConstraint(np.array([2, 2, 3, 2], dtype=np.int64)),
            },
            # 3D to 4D with -1
            {
                "data": InputShapeConstraint((3, 2, 4)),
                "shape": InputValueConstraint(np.array([2, 3, -1, 2], dtype=np.int64)),
            },
            # 3D to 5D
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "shape": InputValueConstraint(np.array([2, 1, 3, 2, 2], dtype=np.int64)),
            },
            # ===== 4D Input (dimension 4) =====
            # 4D to 1D, Whisper
            {
                "data": InputShapeConstraint((2, 3, 4, 5)),
                "shape": InputValueConstraint(np.array([120], dtype=np.int64)),
            },
            # 4D to 2D, Convnext
            {
                "data": InputShapeConstraint((1, 56, 56, 96)),
                "shape": InputValueConstraint(np.array([3136, 96], dtype=np.int64)),
            },
            # 4D to 3D
            {
                "data": InputShapeConstraint((2, 3, 4, 5)),
                "shape": InputValueConstraint(np.array([2, 12, 5], dtype=np.int64)),
            },
            # 4D to 4D with -1
            {
                "data": InputShapeConstraint((2, 3, 4, 5)),
                "shape": InputValueConstraint(np.array([6, 2, -1, 5], dtype=np.int64)),
            },
            # 4D to 4D with 0
            {
                "data": InputShapeConstraint((2, 3, 4, 5)),
                "shape": InputValueConstraint(np.array([0, 0, 20, 1], dtype=np.int64)),
            },
            # 4D to 5D
            {
                "data": InputShapeConstraint((2, 3, 2, 2)),
                "shape": InputValueConstraint(np.array([2, 3, 2, 1, 2], dtype=np.int64)),
            },
            # 4D to 6D
            {
                "data": InputShapeConstraint((2, 2, 2, 3)),
                "shape": InputValueConstraint(np.array([2, 2, 2, 1, 1, 3], dtype=np.int64)),
            },
            # ===== 5D Input (dimension 5) =====
            # 5D to 3D
            {
                "data": InputShapeConstraint((2, 2, 2, 3, 2)),
                "shape": InputValueConstraint(np.array([4, 6, 2], dtype=np.int64)),
            },
            # 5D to 4D with -1
            {
                "data": InputShapeConstraint((2, 2, 3, 2, 2)),
                "shape": InputValueConstraint(np.array([4, 3, -1, 2], dtype=np.int64)),
            },
            # 5D to 5D with 0
            {
                "data": InputShapeConstraint((2, 3, 2, 2, 2)),
                "shape": InputValueConstraint(np.array([0, 0, 0, 4, 1], dtype=np.int64)),
            },
            # 5D to 6D
            {
                "data": InputShapeConstraint((2, 2, 2, 2, 3)),
                "shape": InputValueConstraint(np.array([2, 2, 2, 2, 1, 3], dtype=np.int64)),
            },
            # ===== 6D Input (dimension 6 - maximum) =====
            # 6D to 3D
            {
                "data": InputShapeConstraint((2, 2, 2, 2, 2, 2)),
                "shape": InputValueConstraint(np.array([8, 4, 2], dtype=np.int64)),
            },
            # 6D to 4D with -1
            {
                "data": InputShapeConstraint((2, 2, 2, 2, 2, 3)),
                "shape": InputValueConstraint(np.array([4, 4, -1, 3], dtype=np.int64)),
            },
            # 6D to 6D with -1
            {
                "data": InputShapeConstraint((2, 2, 2, 2, 2, 3)),
                "shape": InputValueConstraint(np.array([2, 2, 2, -1, 2, 3], dtype=np.int64)),
            },
            # ===== Edge Cases =====
            # Large single dimension
            {
                "data": InputShapeConstraint((120,)),
                "shape": InputValueConstraint(np.array([5, 4, 6], dtype=np.int64)),
            },
            # Multiple zeros in shape (for allowzero testing)
            {
                "data": InputShapeConstraint((2, 3, 4)),
                "shape": InputValueConstraint(np.array([0, 0, 0], dtype=np.int64)),
            },
            # -1 at different positions
            {
                "data": InputShapeConstraint((3, 4, 5)),
                "shape": InputValueConstraint(np.array([-1, 12], dtype=np.int64)),
            },
            # Complex reshape with mix of 0 and regular values
            {
                "data": InputShapeConstraint((4, 3, 2)),
                "shape": InputValueConstraint(np.array([0, 6, 1], dtype=np.int64)),
            },
        ]

    def derive_properties(self, properties: dict[str, any]) -> dict[str, any]:
        """Derive additional properties for Reshape operator testing.

        Args:
            properties: Base properties from parent class

        Returns:
            Updated properties with Reshape-specific derived values
        """
        item = properties.copy()
        item["data_dim"] = len(item["data_shape"])

        # Handle case where shape_value is unknown
        # This occurs when the shape input is dynamic (e.g., comes from model input
        # or another node's output) rather than being a constant initializer or
        # Constant node. In such cases, the static analyzer cannot determine the
        # actual shape values at compile time.
        if "shape_value" not in item:
            item["shape_len"] = -1  # Unknown length
            item["shape_has_zero"] = False
            item["shape_is_constant_or_empty"] = item.get("shape_is_constant", True)
            item["shape_all_zeros"] = False
            return item

        array = np.array(item["shape_value"])
        item["shape_len"] = len(array)  # shape of "shape" input tensor
        # item["shape_has_minus_one"] = bool(np.any(array == -1))
        item["shape_has_zero"] = bool(np.any(array == 0))
        item["shape_is_constant_or_empty"] = item["shape_is_constant"] or (item["shape_len"] == 0)
        item["shape_all_zeros"] = bool(np.all(array == 0))

        return item

    def get_infinite_property_names(self) -> list[str]:
        """Returns names of infinite properties for Reshape operator."""
        return ["data_shape", "shape_value"]

    def get_qdq_config(self) -> dict[str, QDQParameterConfig]:
        """Returns QDQ configuration for Reshape operator."""
        return {
                "data": QDQParameterConfig(support_activation=True),
                "shape": QDQParameterConfig(),
            }
