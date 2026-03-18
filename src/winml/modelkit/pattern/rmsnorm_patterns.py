# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""RMSNormalization pattern definitions for ONNX pattern matching.

This module provides patterns for matching RMSNormalization subgraphs:
1. RMSNormalizationPowPattern: Multi-node pattern using Pow(x, 2) for squaring
2. RMSNormalizationMulPattern: Multi-node pattern using Mul(x, x) for squaring
3. TransposedSingleRMSNormalizationPattern: Transpose-wrapped RMSNorm for any axis

All patterns compute: Y = X / sqrt(Mean(X^2) + epsilon) * Scale
All patterns share the same schema to enable pattern rewriting.

Schema follows the ONNX RMSNormalization specification (opset 23):
- Two type constraints: T (input X) and V (output Y, Scale)
- Attributes: axis, epsilon
"""

from abc import abstractmethod
from typing import Any

import numpy as np
from onnx.defs import OpSchema

from winml.modelkit.onnx.domains import ONNXDomain
from winml.modelkit.pattern.op_input_gen import InputShapeConstraint, InputValueConstraint
from winml.modelkit.pattern.utils import (
    get_attribute_proto_value,
    get_tensor_shape,
    validate_scale_bias_shape_for_axis,
)
from winml.modelkit.pattern.base import (
    Pattern,
    PatternInputGenerator,
    PatternMatchResult,
    PatternMismatchedException,
    PatternSchema,
    Skeleton,
    SkeletonMatchResult,
    register_pattern_input_generator,
)


def _validate_rmsnorm_scale(
    skeleton_match_result: SkeletonMatchResult,
    pattern_result: PatternMatchResult,
) -> bool:
    """Validate Scale shape is compatible with the normalization axis."""
    matcher = skeleton_match_result.matcher
    scale_tensor = skeleton_match_result.inputs[1]

    input_shape = pattern_result.input_infos["X"].shape
    if input_shape is None or len(input_shape) < 2:
        return False

    axis = pattern_result.attributes["axis"]
    rank = len(input_shape)
    normalized_axis = axis if axis >= 0 else rank + axis
    if normalized_axis < 0 or normalized_axis >= rank:
        return False

    normalized_dim = input_shape[normalized_axis]
    if isinstance(normalized_dim, str):
        return True

    scale_shape = get_tensor_shape(scale_tensor, matcher)
    return scale_shape is None or validate_scale_bias_shape_for_axis(
        scale_shape, input_shape, axis
    )


# Shared schema for RMSNormalization patterns (follows ONNX opset 23 spec)
_RMSNORM_SCHEMA = PatternSchema(
    name="RMSNormalizationPattern",
    doc=(
        "RMS Normalization pattern.\n"
        "Computes Y = X / sqrt(Mean(X^2) + epsilon) * Scale.\n"
        "Normalization is performed along the specified axis.\n"
        "\n"
        "Shape constraints:\n"
        "- X: N-dimensional tensor (any rank >= 2)\n"
        "- Scale: Tensor with total elements equal to X.shape[axis]\n"
        "\n"
        "All patterns support any axis with properly shaped Scale:\n"
        "- axis=-1: 1D Scale (normalized_dim,) is valid\n"
        "- axis!=-1: multi-dim Scale with non-1 dim at axis position\n"
    ),
    type_constraints=[
        OpSchema.TypeConstraintParam(
            type_param_str="T",
            allowed_type_strs=[
                "tensor(float16)",
                "tensor(float)",
                "tensor(double)",
                "tensor(bfloat16)",
            ],
            description="Constrain input X type to float tensors.",
        ),
        OpSchema.TypeConstraintParam(
            type_param_str="V",
            allowed_type_strs=[
                "tensor(float16)",
                "tensor(float)",
                "tensor(double)",
                "tensor(bfloat16)",
            ],
            description="Constrain output Y and Scale type to float tensors.",
        ),
    ],
    inputs=[
        OpSchema.FormalParameter(
            name="X",
            type_str="T",
            description="Input tensor to be normalized.",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        ),
        OpSchema.FormalParameter(
            name="Scale",
            type_str="V",
            description="Scale tensor.",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="Y",
            type_str="V",
            description="Normalized output tensor.",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        )
    ],
    attributes={
        "axis": OpSchema.Attribute(
            name="axis",
            description="The axis along which to normalize. Default is -1 (last axis).",
            type=OpSchema.AttrType.INT,
            required=True,
        ),
        "epsilon": OpSchema.Attribute(
            name="epsilon",
            description="Small constant for numerical stability. Default is 1e-5.",
            type=OpSchema.AttrType.FLOAT,
            required=True,
        ),
    },
)


class RMSNormalizationPatternBase(Pattern):
    """Abstract base class for all RMSNormalization patterns (Pow, Mul).

    Provides shared schema and Scale validation logic.
    """

    def get_schema(self) -> PatternSchema:
        """Return the schema definition for RMSNormalization pattern."""
        return _RMSNORM_SCHEMA

    def check_skeleton_result(
        self, skeleton_match_result: SkeletonMatchResult
    ) -> PatternMatchResult | None:
        """Check skeleton match result and validate Scale shape."""
        pattern_result = super().check_skeleton_result(skeleton_match_result)
        if pattern_result is None:
            return None

        if not _validate_rmsnorm_scale(skeleton_match_result, pattern_result):
            return None

        return pattern_result


class _RMSNormalizationExpandedPatternBase(RMSNormalizationPatternBase):
    """Base class for expanded RMSNorm patterns (Pow and Mul variants).

    Graph: Pow/Mul → ReduceMean → Add → Sqrt → Div → Mul
    """

    @abstractmethod
    def _get_squaring_internal_constants(
        self, dtype: np.dtype
    ) -> list[tuple[int, int, np.ndarray]]:
        """Return internal constants specific to the squaring operation.

        Args:
            dtype: The numpy dtype to use for constants.

        Returns:
            List of (node_idx, slot, value) tuples for squaring-specific constants.
        """

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants and attributes for model generation."""
        dtype = inputs["X"].dtype
        epsilon = attributes["epsilon"]
        axis = attributes["axis"]

        internal_constants = self._get_squaring_internal_constants(dtype)
        # Epsilon is at node 2 (Add), slot 1
        internal_constants.append((2, 1, np.array(epsilon, dtype=dtype)))

        opset_version = domain_versions[ONNXDomain.AI_ONNX]
        internal_attributes: dict[tuple[int, str], Any] = {}

        # ReduceMean is at node 1 (only one, unlike LayerNorm's two)
        if opset_version >= 18:
            axes_value = np.array([axis], dtype=np.int64)
            internal_constants.append((1, 1, axes_value))
        else:
            internal_attributes[(1, "axes")] = [axis]

        return internal_constants, internal_attributes

    def _infer_schema_attributes(
        self, skeleton_match_result: SkeletonMatchResult
    ) -> dict[str, Any]:
        """Infer axis and epsilon from matched ReduceMean and Add nodes."""
        matcher = skeleton_match_result.matcher
        nodes = skeleton_match_result.matched_nodes
        opset_version = matcher.domain_versions[ONNXDomain.AI_ONNX]

        # ReduceMean is at node index 1
        reducemean_node = nodes[1]
        if opset_version >= 18:
            if len(reducemean_node.input) < 2:
                raise PatternMismatchedException("ReduceMean missing axes input")
            axes_tensor = reducemean_node.input[1]
            if axes_tensor not in matcher.tensor_values:
                raise PatternMismatchedException(f"Axes tensor {axes_tensor} not found")
            axes_value = matcher.tensor_values[axes_tensor]
        else:
            axes_value = None
            for attr in reducemean_node.attribute:
                if attr.name == "axes":
                    axes_value = np.array(list(attr.ints), dtype=np.int64)
                    break
            if axes_value is None:
                raise PatternMismatchedException("ReduceMean missing axes attribute")

        if len(axes_value) != 1:
            raise PatternMismatchedException(
                f"Only single-axis normalization supported, got axes={axes_value}"
            )
        axis = int(axes_value[0])

        # Epsilon Add is at node index 2
        epsilon_node = nodes[2]
        if len(epsilon_node.input) < 2:
            raise PatternMismatchedException("Epsilon node has fewer than 2 inputs")
        epsilon_tensor = epsilon_node.input[1]
        if epsilon_tensor not in matcher.tensor_values:
            raise PatternMismatchedException(f"Epsilon tensor {epsilon_tensor} not found")
        epsilon_value = float(matcher.tensor_values[epsilon_tensor].flat[0])

        return {"axis": axis, "epsilon": epsilon_value}


class RMSNormalizationPowPattern(_RMSNormalizationExpandedPatternBase):
    """Pattern definition for RMSNormalization using Pow for squaring.

    This translates to the following node topology (6 nodes):
        X -> Pow(2) -> ReduceMean -> Add(eps) -> Sqrt -> Div -> Mul(Scale) -> Y
        |                                                 ^
        +-------------------------------------------------+
    """

    def _get_squaring_internal_constants(
        self, dtype: np.dtype
    ) -> list[tuple[int, int, np.ndarray]]:
        """Return Pow exponent constant (2.0)."""
        return [(0, 1, np.array(2.0, dtype=dtype))]

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for RMSNormalization pattern (Pow variant).

        Returns:
            Skeleton defining the RMSNormalization computation graph topology.
        """
        # Node indices: 0=Pow, 1=ReduceMean, 2=Add, 3=Sqrt, 4=Div, 5=Mul
        node_op_types = [
            "Pow",  # 0: square input (X^2)
            "ReduceMean",  # 1: mean of squared (Mean(X^2))
            "Add",  # 2: add epsilon
            "Sqrt",  # 3: sqrt(Mean(X^2) + eps)
            "Div",  # 4: X / sqrt(...)
            "Mul",  # 5: scale by weight
        ]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        # Edges: (src, src_slot, dst, dst_slot)
        # -1 represents input X, -2 represents input Scale
        edges = [
            (-1, 0, 0, 0),  # X -> Pow[0] (base)
            (0, 0, 1, 0),  # Pow -> ReduceMean[0]
            (1, 0, 2, 0),  # ReduceMean -> Add[0] (augend)
            (2, 0, 3, 0),  # Add -> Sqrt[0]
            (-1, 0, 4, 0),  # X -> Div[0] (dividend, skip connection)
            (3, 0, 4, 1),  # Sqrt -> Div[1] (divisor)
            (4, 0, 5, 0),  # Div -> Mul[0] (multiplicand)
            (-2, 0, 5, 1),  # Scale -> Mul[1] (multiplier/weight)
        ]

        exit_nodes = [5]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=exit_nodes,
            n_inputs=2,
        )


class RMSNormalizationMulPattern(_RMSNormalizationExpandedPatternBase):
    """Pattern definition for RMSNormalization using Mul for squaring.

    This variant uses Mul(x, x) instead of Pow(x, 2) for the squaring operation.
    Otherwise identical to RMSNormalizationPowPattern.

    This translates to the following node topology (6 nodes):
        X -> Mul(X,X) -> ReduceMean -> Add(eps) -> Sqrt -> Div -> Mul(Scale) -> Y
        |    ^                                              ^
        +----+----------------------------------------------+
    """

    def _get_squaring_internal_constants(
        self, dtype: np.dtype
    ) -> list[tuple[int, int, np.ndarray]]:
        """Return empty list - Mul variant has no squaring-specific constants."""
        return []

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for RMSNormalization pattern (Mul variant).

        Returns:
            Skeleton defining the RMSNormalization computation graph topology.
        """
        # Node indices: 0=Mul, 1=ReduceMean, 2=Add, 3=Sqrt, 4=Div, 5=Mul
        node_op_types = [
            "Mul",  # 0: square input (X*X)
            "ReduceMean",  # 1: mean of squared
            "Add",  # 2: add epsilon
            "Sqrt",  # 3: sqrt(Mean(X^2) + eps)
            "Div",  # 4: X / sqrt(...)
            "Mul",  # 5: scale by weight
        ]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        # Edges: (src, src_slot, dst, dst_slot)
        # -1 represents input X, -2 represents input Scale
        edges = [
            (-1, 0, 0, 0),  # X -> Mul[0] (first input for X*X)
            (-1, 0, 0, 1),  # X -> Mul[1] (second input for X*X)
            (0, 0, 1, 0),  # Mul -> ReduceMean[0]
            (1, 0, 2, 0),  # ReduceMean -> Add[0] (augend)
            (2, 0, 3, 0),  # Add -> Sqrt[0]
            (-1, 0, 4, 0),  # X -> Div[0] (dividend, skip connection)
            (3, 0, 4, 1),  # Sqrt -> Div[1] (divisor)
            (4, 0, 5, 0),  # Div -> Mul[0] (multiplicand)
            (-2, 0, 5, 1),  # Scale -> Mul[1] (multiplier/weight)
        ]

        exit_nodes = [5]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=exit_nodes,
            n_inputs=2,
        )


class TransposedSingleRMSNormalizationPattern(RMSNormalizationPatternBase):
    """Transpose-wrapped RMSNormalization pattern for arbitrary axis.

    Topology: X → Transpose → RMSNormalization(axis=-1) → Transpose → Y
              Scale → Reshape (to 1D) → RMSNormalization

    Universal rewrite target for Pow/Mul patterns.
    """

    def _compute_transpose_permutation(
        self, axis: int, rank: int
    ) -> tuple[list[int], list[int]]:
        """Compute permutations to move axis to last position and back."""
        if axis < 0:
            axis = rank + axis

        if axis < 0 or axis >= rank:
            raise ValueError(f"axis {axis} out of range for rank {rank}")

        if axis == rank - 1:
            return list(range(rank)), list(range(rank))

        perm_forward = list(range(rank))
        perm_forward.append(perm_forward.pop(axis))

        perm_inverse = [0] * rank
        for i, p in enumerate(perm_forward):
            perm_inverse[p] = i

        return perm_forward, perm_inverse

    def get_skeleton(self) -> Skeleton:
        """Return skeleton: Transpose → Reshape → RMSNormalization → Transpose."""
        return Skeleton(
            node_op_types=[
                "Transpose",
                "Reshape",
                "RMSNormalization",
                "Transpose",
            ],
            node_domains=[ONNXDomain.AI_ONNX] * 4,
            edges=[
                (-1, 0, 0, 0),  # X -> Transpose1[0]
                (0, 0, 2, 0),  # Transpose1 -> RMSNorm[0]
                (-2, 0, 1, 0),  # Scale -> Reshape[0]
                (1, 0, 2, 1),  # Reshape -> RMSNorm[1] (scale)
                (2, 0, 3, 0),  # RMSNorm -> Transpose2[0]
            ],
            exit_nodes=[3],
            n_inputs=2,
        )

    def _infer_schema_attributes(
        self, skeleton_match_result: SkeletonMatchResult
    ) -> dict[str, Any]:
        """Infer axis from Transpose perm and epsilon from RMSNormalization node."""
        nodes = skeleton_match_result.matched_nodes

        # Extract perm from first Transpose (node 0) to infer original axis
        transpose1_node = nodes[0]
        perm_forward = None
        for attr in transpose1_node.attribute:
            if attr.name == "perm":
                perm_forward = list(attr.ints)
                break
        if perm_forward is None:
            raise PatternMismatchedException("Transpose missing perm attribute")

        # perm_forward[-1] tells us which original axis is now at the end
        axis = perm_forward[-1]

        # Extract epsilon from RMSNormalization node (node 2)
        rmsnorm_node = nodes[2]
        epsilon = None
        for attr in rmsnorm_node.attribute:
            if attr.name == "epsilon":
                epsilon = get_attribute_proto_value(
                    attr, replace_float_with_dummy=False
                )
                break
        if epsilon is None:
            raise PatternMismatchedException("RMSNormalization missing epsilon")

        return {"axis": axis, "epsilon": epsilon}

    def _get_normalized_dim(
        self, inputs: dict[str, np.ndarray], attributes: dict[str, Any]
    ) -> int:
        """Get the size of the dimension being normalized."""
        x_shape = inputs["X"].shape
        axis = attributes["axis"]
        rank = len(x_shape)
        normalized_axis = axis if axis >= 0 else rank + axis
        return x_shape[normalized_axis]

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return transpose permutations and reshape shapes for model generation."""
        axis = attributes["axis"]
        epsilon = attributes["epsilon"]
        rank = len(inputs["X"].shape)
        normalized_dim = self._get_normalized_dim(inputs, attributes)

        perm_forward, perm_inverse = self._compute_transpose_permutation(axis, rank)

        # Reshape target shape: 1D with normalized_dim
        reshape_target_shape = np.array([normalized_dim], dtype=np.int64)

        # Internal constants: Reshape shape input
        internal_constants: list[tuple[int, int, np.ndarray]] = [
            (1, 1, reshape_target_shape),  # Reshape (Scale) shape input
        ]

        # Internal attributes
        internal_attributes: dict[tuple[int, str], Any] = {
            (0, "perm"): perm_forward,  # Transpose1 perm
            (2, "axis"): -1,  # RMSNorm always axis=-1
            (2, "epsilon"): epsilon,  # RMSNorm epsilon
            (3, "perm"): perm_inverse,  # Transpose2 perm
        }

        return internal_constants, internal_attributes


class RMSNormalizationPatternInputGenerator(PatternInputGenerator):
    """Base PatternInputGenerator for RMSNormalization patterns.

    Since RMSNormalization opset 23 is not available yet, this generator
    provides its own input combinations rather than inheriting from a
    runtime checker op generator.
    """

    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """Return finite attribute sets (empty for this pattern)."""
        return {}

    def get_input_and_infinite_attribute_combinations(self) -> list[dict[str, Any]]:
        """Return input combinations with broadcast-compatible Scale shapes."""
        common_shapes = [
            (6,),
            (3, 6),
            (2, 4, 6),
            (2, 4, 5, 5),
            (2, 4, 3, 4, 4),
            (2, 3, 2, 4, 3, 4),
        ]
        axis_options = [-1, 0]

        adapted: list[dict[str, Any]] = []
        for shape in common_shapes:
            for axis in axis_options:
                rank = len(shape)
                normalized_axis = axis if axis >= 0 else rank + axis
                normalized_dim = shape[normalized_axis]

                if axis == -1 or normalized_axis == rank - 1:
                    scale = InputValueConstraint(
                        np.ones((normalized_dim,), dtype=np.float32)
                    )
                else:
                    broadcast_shape = [1] * rank
                    broadcast_shape[normalized_axis] = normalized_dim
                    scale = InputValueConstraint(
                        np.ones((normalized_dim,), dtype=np.float32).reshape(
                            broadcast_shape
                        )
                    )

                adapted.append(
                    {
                        "X": InputShapeConstraint(shape),
                        "Scale": scale,
                        "axis": axis,
                        "epsilon": 1e-5,
                    }
                )

        return adapted


@register_pattern_input_generator
class RMSNormalizationPowPatternInputGenerator(RMSNormalizationPatternInputGenerator):
    """PatternInputGenerator for RMSNormalization pattern (Pow variant)."""

    pattern = RMSNormalizationPowPattern()
    registration_name = "RMSNormalizationPow"


@register_pattern_input_generator
class RMSNormalizationMulPatternInputGenerator(RMSNormalizationPatternInputGenerator):
    """PatternInputGenerator for RMSNormalization pattern (Mul variant)."""

    pattern = RMSNormalizationMulPattern()
    registration_name = "RMSNormalizationMul"


@register_pattern_input_generator
class TransposedSingleRMSNormalizationPatternInputGenerator(
    RMSNormalizationPatternInputGenerator
):
    """PatternInputGenerator for TransposedSingleRMSNormalizationPattern."""

    pattern = TransposedSingleRMSNormalizationPattern()
    registration_name = "TransposedSingleRMSNormalization"
