from typing import Any

import numpy as np

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    InputValueConstraint,
    OpInputGenerator,
    register_runtime_checker_op,
)


@register_runtime_checker_op
class PadInputGenerator(OpInputGenerator):
    """Input generator for the Pad operator.

    Pad operator documentation:
    - Input 1 (data): Input tensor to pad
    - Input 2 (pads): INT64 tensor of padding amounts
      [begin_dim0, begin_dim1, ..., end_dim0, end_dim1, ...]
    - Input 3 (constant_value, optional): Constant value to use for padding when mode is "constant"
    - Input 4 (axes, optional): Axes to pad (default: all axes)
    - Attribute (mode): Padding mode - "constant", "reflect", "edge", or "wrap"

    Constraints:
    - For "reflect" mode, pad amounts cannot exceed (dimension_size - 1)
    - For "wrap" mode, pad amounts cannot exceed dimension_size
    - Pads tensor must have length 2 * rank(data)
    """

    op_name = "Pad"

    def get_finite_attribute_sets(self) -> dict[str, list[str]]:
        """Returns finite attribute sets for Pad.

        mode: ["constant", "reflect", "edge", "wrap"]
            - constant: Pads with a constant value (default)
            - reflect: Pads with reflection of the tensor mirrored on the border
            - edge: Pads with edge values of the tensor
            - wrap: Pads with wrapping of the tensor
        """
        return {"mode": ["constant", "reflect", "edge", "wrap"]}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputConstraint]]:
        """Returns comprehensive input combinations for Pad operator.

        Coverage strategy:
        - Input dimensions: 1D through 6D
        - Pad amounts: symmetric, asymmetric, zero padding
        - Optional inputs: constant_value
        - Values kept within reflect/wrap mode constraints (pad <= dim_size - 1)

        Note: 'axes' input is omitted as it was added in opset 18 and tests
        validate against opset 17.
        """
        return [
            # ===== 1D Input =====
            {
                "data": InputShapeConstraint((6,)),
                "pads": InputValueConstraint(np.array([1, 1], dtype=np.int64)),
                "constant_value": InputValueConstraint(np.array(0.0, dtype=np.float32)),
            },
            # ===== 2D Input =====
            {
                "data": InputShapeConstraint((4, 5)),
                "pads": InputValueConstraint(np.array([1, 2, 1, 2], dtype=np.int64)),
                "constant_value": InputValueConstraint(np.array(0.0, dtype=np.float32)),
            },
            # ===== 3D Input =====
            {
                "data": InputShapeConstraint((3, 4, 5)),
                "pads": InputValueConstraint(np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)),
                "constant_value": InputValueConstraint(np.array(0.0, dtype=np.float32)),
            },
            # ===== 4D Input (common for images/convolutions) =====
            {
                "data": InputShapeConstraint((2, 3, 4, 5)),
                "pads": InputValueConstraint(np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)),
                "constant_value": InputValueConstraint(np.array(0.0, dtype=np.float32)),
            },
            # ===== 5D Input =====
            {
                "data": InputShapeConstraint((2, 3, 4, 4, 5)),
                "pads": InputValueConstraint(
                    np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1], dtype=np.int64)
                ),
                "constant_value": InputValueConstraint(np.array(0.0, dtype=np.float32)),
            },
            # ===== 6D Input =====
            {
                "data": InputShapeConstraint((2, 2, 3, 3, 4, 4)),
                "pads": InputValueConstraint(
                    np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=np.int64)
                ),
                "constant_value": InputValueConstraint(np.array(0.0, dtype=np.float32)),
            },
            # ===== Edge cases: zero padding =====
            {
                "data": InputShapeConstraint((3, 4)),
                "pads": InputValueConstraint(np.array([0, 0, 0, 0], dtype=np.int64)),
                "constant_value": InputValueConstraint(np.array(0.0, dtype=np.float32)),
            },
            # ===== Asymmetric padding =====
            {
                "data": InputShapeConstraint((4, 4)),
                "pads": InputValueConstraint(np.array([2, 0, 0, 2], dtype=np.int64)),
                "constant_value": InputValueConstraint(np.array(1.0, dtype=np.float32)),
            },
        ]

    def derive_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Derive additional properties for Pad operator testing.

        Args:
            properties: Base properties containing data_shape, pads_value, etc.

        Returns:
            Updated properties with Pad-specific derived values
        """
        item = properties.copy()
        item["data_dim"] = len(item["data_shape"])

        # TODO: Enable these derived properties if needed in future tests
        # Get pads value
        pads_value = item["pads_value"]
        if isinstance(pads_value, np.ndarray):
            pads_array = pads_value
        else:
            pads_array = np.array(pads_value, dtype=np.int64)

        # need for NVidia TRT RTX execution provider tests
        item["pads_all_zeros"] = bool(np.all(pads_array == 0))
        # item["pads_is_symmetric"] = bool(
        #     np.array_equal(pads_array[: len(pads_array) // 2],
        #                    pads_array[len(pads_array) // 2 :])
        # )

        return item

    def get_infinite_property_names(self) -> list[str]:
        """Returns names of infinite properties for Pad operator."""
        return ["data_shape", "pads_value", "constant_value_value"]
