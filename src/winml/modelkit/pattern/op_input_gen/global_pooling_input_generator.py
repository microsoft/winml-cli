# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generators for ONNX pooling operators.

This module provides input generators for global pooling operators that
reduce spatial dimensions by computing statistics across channels.
"""

from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    OpInputGenerator,
    QDQParameterConfig,
    register_runtime_checker_op,
)


class GlobalPoolingInputGenerator(OpInputGenerator):
    """Base class for global pooling operator input generators.

    Global pooling operators (GlobalAveragePool, GlobalMaxPool) apply pooling
    across all spatial dimensions of the input tensor, equivalent to using a
    kernel size equal to the spatial dimensions.

    Operator characteristics:
    - Input: Single tensor X with shape (N, C, D1, D2, ..., Dn)
      where N is batch size, C is channels, and D1...Dn are spatial dimensions
    - Output: Tensor with shape (N, C, 1, 1, ..., 1) - spatial dims reduced to 1
    - Attributes: None
    - Constraint: Input must have at least 3 dimensions (N, C, at least one spatial dim)

    Test coverage strategy:
    - Input dimensions: 3D through 5D (3D is minimum, 5D is ONNXRuntime maximum)
    - Various spatial dimension configurations
    - Common image (4D: NxCxHxW) and video (5D: NxCxDxHxW) formats
    """

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Returns finite attribute sets for global pooling operators.

        Global pooling operators have no attributes.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, object]]:
        """Returns input combinations for global pooling operators.

        Coverage strategy:
        - Input dimensions: 3D through 6D (minimum 3D required)
        - Various spatial configurations (square, rectangular, etc.)
        - Common formats: image (4D), video (5D)

        Note: Input parameter is 'X' for global pooling operators.
        Note: ONNXRuntime supports up to 5D inputs for global pooling operators.
        """
        # Get the input parameter name from the operator schema
        input_param_name = self.op_input_names[0]

        return [
            # Input dimension less than 3 are not supported on CPU
            # ===== 1D Input (dimension 1 - invalid) =====
            # {input_param_name: InputShapeConstraint((4,))},
            # ===== 2D Input (dimension 2 - invalid) =====
            # {input_param_name: InputShapeConstraint((2, 4))},
            # ===== 3D Input (dimension 3 - minimum) =====
            # Basic 3D: (N, C, L) - temporal/sequence data
            {input_param_name: InputShapeConstraint((2, 4, 6))},
            # ===== 4D Input (dimension 4) =====
            # Common image format: (N, C, H, W)
            {input_param_name: InputShapeConstraint((2, 3, 4, 4))},
            # ===== 5D Input (dimension 5) =====
            # Video format: (N, C, D, H, W)
            {input_param_name: InputShapeConstraint((2, 4, 3, 4, 4))},
            # Input dimension more than 5 are not supported on CPU
            # {input_param_name: InputShapeConstraint((1, 8, 2, 5, 8, 7))},
        ]

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties from input constraints.

        Adds dimension information for the input tensor.
        """
        input_param_name = self.op_input_names[0]
        item = properties.copy()
        item[f"{input_param_name}_dim"] = len(item[f"{input_param_name}_shape"])
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        For global pooling operators, the input shape can vary infinitely.
        """
        input_param_name = self.op_input_names[0]
        return [f"{input_param_name}_shape"]


@register_runtime_checker_op
class GlobalAveragePoolInputGenerator(GlobalPoolingInputGenerator):
    """Input generator for GlobalAveragePool operator.

    GlobalAveragePool computes the average value for each channel across
    all spatial dimensions. This is equivalent to AveragePool with kernel
    size equal to the spatial dimensions.

    Operator signature (from onnxscript.opset22):
    - Input: X (differentiable tensor)
    - Output: Tensor with spatial dimensions reduced to 1
    - Attributes: None
    """

    op_name = "GlobalAveragePool"

    def get_qdq_config(self) -> dict[str, QDQParameterConfig]:
        """Return QDQ configuration for GlobalAveragePool operator inputs."""
        return {
            "X": QDQParameterConfig(support_activation=True),
            "Y": QDQParameterConfig(support_activation=True, support_non_qdq=True),
        }


@register_runtime_checker_op
class GlobalMaxPoolInputGenerator(GlobalPoolingInputGenerator):
    """Input generator for GlobalMaxPool operator.

    GlobalMaxPool computes the maximum value for each channel across
    all spatial dimensions. This is equivalent to MaxPool with kernel
    size equal to the spatial dimensions.

    Operator signature (from onnxscript.opset22):
    - Input: X (differentiable tensor)
    - Output: Tensor with spatial dimensions reduced to 1
    - Attributes: None
    """

    op_name = "GlobalMaxPool"


# @register_runtime_checker_op # completely unsupported on CPU
# class GlobalLpPoolInputGenerator(GlobalPoolingInputGenerator):
#     """Input generator for GlobalLpPool operator.

#     NOTE: GlobalLpPool is defined in ONNX opset 22 specification,
#     but NOT IMPLEMENTED in ONNXRuntime as of December 2024.
#     Error: "Could not find an implementation for GlobalLpPool(22)"

#     Keeping this class commented for future implementation.

#     GlobalLpPool computes the Lp norm for each channel across all
#     spatial dimensions. This is equivalent to LpPool with kernel size
#     equal to the spatial dimensions.

#     Operator signature (from onnxscript.opset22):
#     - Input: X (differentiable tensor)
#     - Output: Tensor with spatial dimensions reduced to 1
#     - Attributes: p (int, default=2) - p value of the Lp norm
#     """

#     op_name = "GlobalLpPool"

#     def get_finite_attribute_sets(self) -> dict[str, list]:
#         """Returns finite attribute sets for GlobalLpPool operator.

#         Returns different p values to test various Lp norms:
#         - p=1: L1 norm (Manhattan/taxicab norm)
#         - p=2: L2 norm (Euclidean norm, default)
#         """
#         return {
#             "p": [1, 2],
#         }
