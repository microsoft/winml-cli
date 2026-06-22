# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from .op_input_gen import (
    InputConstraint,
    InputShapeConstraint,
    OpInputGenerator,
    QDQParameterConfig,
    register_runtime_checker_op,
)


class UnaryInputGenerator(OpInputGenerator):
    """Universal input generator for unary ONNX operators.

    Unary operators are element-wise operations that take a single input tensor
    and produce a single output tensor of the same shape.

    Supported operators:
    - Abs, Acos, Acosh, Asin, Asinh, Atan, Atanh
    - BitwiseNot, Ceil, Cos, Cosh
    - Erf, Exp, Floor
    - HardSwish
    - Identity
    - Log
    - Mish
    - Neg, NonZero, Not
    - Reciprocal, Relu, Round
    - Sigmoid, Sign, Sin, Sinh, Softplus, Softsign, Sqrt
    - Tan, Tanh

    Operator characteristics (based on Abs documentation):
    - Input: Single tensor X of type T
    - Output: Single tensor of same type T and shape as input
    - Attributes: None (most unary ops have no attributes)
    - Operation: Element-wise transformation

    Test coverage strategy:
    - Input dimensions: 1D through 6D (max per spec)
    - Various shapes to test different tensor configurations
    - Ordered from smallest to largest dimensions
    """

    support_0d_input = True  # Support scalar inputs

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Returns finite attribute sets for unary operators.

        Most unary operators have no attributes, so return empty dict.
        Subclasses can override if specific operators have attributes.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, object]]:
        """Returns comprehensive input combinations for unary operators.

        Coverage strategy:
        - Input dimensions: 1D through 6D (as per spec: max 6 dimensions)
        - Axis sizes: limited to max 6 per axis (as per spec)
        - Various shape patterns to cover different use cases
        - Ordered from smallest to largest dimensions

        Since unary operators are element-wise and work on any shape,
        we test a representative set of shapes across different dimensions.

        Note: The input parameter name varies across operators (X or input).
        This method automatically detects the correct parameter name.
        """
        # Get the input parameter name from the operator schema
        input_param_name = self.op_input_names[0]

        result: list[dict[str, object]] = [
            # ===== 1D Input (dimension 1) =====
            # Scalar (single element)
            {input_param_name: InputShapeConstraint((1,))},
            # Small 1D tensor
            {input_param_name: InputShapeConstraint((6,))},
            # ===== 2D Input (dimension 2) =====
            # Different 2D shape
            {input_param_name: InputShapeConstraint((4, 5))},
            # ===== 3D Input (dimension 3) =====
            {input_param_name: InputShapeConstraint((3, 2, 5))},
            # ===== 4D Input (dimension 4) =====
            # Common batch-like shape (batch, channels, height, width)
            {input_param_name: InputShapeConstraint((2, 4, 5, 6))},
            # ===== 5D Input (dimension 5) =====
            # 5D tensor (e.g., batch, channels, depth, height, width)
            {input_param_name: InputShapeConstraint((2, 2, 3, 4, 5))},
            # ===== 6D Input (dimension 6 - maximum) =====
            {input_param_name: InputShapeConstraint((6, 6, 2, 2, 2, 2))},
        ]

        if self.support_0d_input:
            result.insert(0, {input_param_name: InputShapeConstraint(())})

        return result

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties from input shapes."""
        input_param_name = self.op_input_names[0]
        item = properties.copy()
        shape = item[f"{input_param_name}_shape"]
        item[f"{input_param_name}_dim"] = len(shape)
        item[f"{input_param_name}_is_single_element"] = (
            all(d == 1 for d in shape) or len(shape) == 0
        )
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite value ranges."""
        input_param_name = self.op_input_names[0]
        return [f"{input_param_name}_shape"]

    def get_qdq_config(self) -> dict[str, QDQParameterConfig]:
        """Return QDQ configuration for unary operator inputs."""
        return {
            self.op_input_names[0]: QDQParameterConfig(support_activation=True),
        }


# Create specific classes for each unary operator
@register_runtime_checker_op
class AbsInputGenerator(UnaryInputGenerator):
    """Input generator for Abs operator."""

    op_name = "Abs"


@register_runtime_checker_op
class AcosInputGenerator(UnaryInputGenerator):
    """Input generator for Acos operator."""

    op_name = "Acos"


@register_runtime_checker_op
class AcoshInputGenerator(UnaryInputGenerator):
    """Input generator for Acosh operator."""

    op_name = "Acosh"


@register_runtime_checker_op
class AsinInputGenerator(UnaryInputGenerator):
    """Input generator for Asin operator."""

    op_name = "Asin"


@register_runtime_checker_op
class AsinhInputGenerator(UnaryInputGenerator):
    """Input generator for Asinh operator."""

    op_name = "Asinh"


@register_runtime_checker_op
class AtanInputGenerator(UnaryInputGenerator):
    """Input generator for Atan operator."""

    op_name = "Atan"


@register_runtime_checker_op
class AtanhInputGenerator(UnaryInputGenerator):
    """Input generator for Atanh operator."""

    op_name = "Atanh"


@register_runtime_checker_op
class BitwiseNotInputGenerator(UnaryInputGenerator):
    """Input generator for BitwiseNot operator."""

    op_name = "BitwiseNot"


@register_runtime_checker_op
class CeilInputGenerator(UnaryInputGenerator):
    """Input generator for Ceil operator."""

    op_name = "Ceil"


@register_runtime_checker_op
class CosInputGenerator(UnaryInputGenerator):
    """Input generator for Cos operator."""

    op_name = "Cos"


@register_runtime_checker_op
class CoshInputGenerator(UnaryInputGenerator):
    """Input generator for Cosh operator."""

    op_name = "Cosh"


@register_runtime_checker_op
class ErfInputGenerator(UnaryInputGenerator):
    """Input generator for Erf operator."""

    op_name = "Erf"


@register_runtime_checker_op
class ExpInputGenerator(UnaryInputGenerator):
    """Input generator for Exp operator."""

    op_name = "Exp"


@register_runtime_checker_op
class FloorInputGenerator(UnaryInputGenerator):
    """Input generator for Floor operator."""

    op_name = "Floor"


@register_runtime_checker_op
class HardSwishInputGenerator(UnaryInputGenerator):
    """Input generator for HardSwish operator."""

    op_name = "HardSwish"


@register_runtime_checker_op
class IdentityInputGenerator(UnaryInputGenerator):
    """Input generator for Identity operator."""

    op_name = "Identity"


@register_runtime_checker_op
class IsNaNInputGenerator(UnaryInputGenerator):
    """Input generator for IsNaN operator.

    Signature: IsNaN(X) -> Y
    Pure unary operator with no attributes.
    """

    op_name = "IsNaN"

    def get_qdq_config(self) -> dict[str, QDQParameterConfig]:
        """Return QDQ configuration for IsNaN operator inputs."""
        return {
            self.op_input_names[0]: QDQParameterConfig(support_activation=True),
            "Y": QDQParameterConfig(
                support_non_qdq=True,  # Output can be non-quantized (boolean)
            ),
        }


@register_runtime_checker_op
class LogInputGenerator(UnaryInputGenerator):
    """Input generator for Log operator."""

    op_name = "Log"


@register_runtime_checker_op
class MishInputGenerator(UnaryInputGenerator):
    """Input generator for Mish operator."""

    op_name = "Mish"


@register_runtime_checker_op
class NegInputGenerator(UnaryInputGenerator):
    """Input generator for Neg operator."""

    op_name = "Neg"


@register_runtime_checker_op
class NonZeroInputGenerator(UnaryInputGenerator):
    """Input generator for NonZero operator."""

    op_name = "NonZero"


@register_runtime_checker_op
class NotInputGenerator(UnaryInputGenerator):
    """Input generator for Not operator."""

    op_name = "Not"


@register_runtime_checker_op
class ReciprocalInputGenerator(UnaryInputGenerator):
    """Input generator for Reciprocal operator."""

    op_name = "Reciprocal"


@register_runtime_checker_op
class ReluInputGenerator(UnaryInputGenerator):
    """Input generator for Relu operator."""

    op_name = "Relu"

    def get_qdq_config(self) -> dict[str, QDQParameterConfig]:
        """Return QDQ configuration for Relu operator inputs."""
        # From p1 model MobileNet
        return {
            self.op_input_names[0]: QDQParameterConfig(
                support_non_qdq=True, support_activation=True
            ),
        }


@register_runtime_checker_op
class RoundInputGenerator(UnaryInputGenerator):
    """Input generator for Round operator."""

    op_name = "Round"


@register_runtime_checker_op
class SigmoidInputGenerator(UnaryInputGenerator):
    """Input generator for Sigmoid operator."""

    op_name = "Sigmoid"


@register_runtime_checker_op
class SignInputGenerator(UnaryInputGenerator):
    """Input generator for Sign operator."""

    op_name = "Sign"


@register_runtime_checker_op
class SinInputGenerator(UnaryInputGenerator):
    """Input generator for Sin operator."""

    op_name = "Sin"


@register_runtime_checker_op
class SinhInputGenerator(UnaryInputGenerator):
    """Input generator for Sinh operator."""

    op_name = "Sinh"


@register_runtime_checker_op
class SoftplusInputGenerator(UnaryInputGenerator):
    """Input generator for Softplus operator."""

    op_name = "Softplus"


@register_runtime_checker_op
class SoftsignInputGenerator(UnaryInputGenerator):
    """Input generator for Softsign operator."""

    op_name = "Softsign"


@register_runtime_checker_op
class SqrtInputGenerator(UnaryInputGenerator):
    """Input generator for Sqrt operator."""

    op_name = "Sqrt"


@register_runtime_checker_op
class TanInputGenerator(UnaryInputGenerator):
    """Input generator for Tan operator."""

    op_name = "Tan"


@register_runtime_checker_op
class TanhInputGenerator(UnaryInputGenerator):
    """Input generator for Tanh operator."""

    op_name = "Tanh"
