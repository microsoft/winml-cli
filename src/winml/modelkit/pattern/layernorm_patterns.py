# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""LayerNormalization pattern definitions for ONNX pattern matching.

This module provides patterns for matching LayerNormalization subgraphs:
1. LayerNormalizationPowPattern: Multi-node pattern using Pow(x, 2) for squaring
2. LayerNormalizationMulPattern: Multi-node pattern using Mul(x, x) for squaring
3. TransposedSingleLayerNormalizationPattern: Transpose-wrapped LayerNorm for any axis

All patterns compute: Y = (X - Mean) / sqrt(Var + epsilon) * Scale + Bias
All patterns share the same schema to enable pattern rewriting.
"""

from abc import abstractmethod
from typing import Any

import numpy as np
from onnx.defs import OpSchema

from winml.modelkit.pattern.base import (
    Pattern,
    PatternInputGenerator,
    PatternMatchResult,
    PatternMismatchedError,
    PatternSchema,
    Skeleton,
    SkeletonMatchResult,
    register_pattern_input_generator,
)
from winml.modelkit.pattern.op_input_gen import get_runtime_checker_op
from winml.modelkit.pattern.utils import (
    get_attribute_proto_value,
    get_tensor_shape,
    validate_scale_bias_shape_for_axis,
)

from ..onnx import ONNXDomain


def _validate_layernorm_scale_bias(
    skeleton_match_result: SkeletonMatchResult,
    pattern_result: PatternMatchResult,
) -> bool:
    """Validate Scale and B shapes are compatible with the normalization axis."""
    matcher = skeleton_match_result.matcher
    scale_tensor = skeleton_match_result.inputs[1]
    bias_tensor = skeleton_match_result.inputs[2]

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
    if scale_shape is not None and not validate_scale_bias_shape_for_axis(
        scale_shape, input_shape, axis
    ):
        return False

    bias_shape = get_tensor_shape(bias_tensor, matcher)
    return bias_shape is None or validate_scale_bias_shape_for_axis(bias_shape, input_shape, axis)


# Shared schema for LayerNormalization patterns
_LAYERNORM_SCHEMA = PatternSchema(
    name="LayerNormalizationPattern",
    doc=(
        "Layer Normalization pattern.\n"
        "Computes Y = (X - Mean) / sqrt(Var + epsilon) * Scale + Bias.\n"
        "Normalization is performed along the specified axis.\n"
        "\n"
        "Shape constraints:\n"
        "- X: N-dimensional tensor (any rank >= 2)\n"
        "- Scale: Tensor with total elements equal to X.shape[axis]\n"
        "- Bias: Tensor with total elements equal to X.shape[axis]\n"
        "\n"
        "All patterns support any axis with properly shaped Scale/Bias:\n"
        "- axis=-1: 1D Scale/Bias (normalized_dim,) is valid\n"
        "- axis!=-1: multi-dim Scale/Bias with non-1 dim at axis position\n"
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
            description="Constrain input and output types to float tensors.",
        )
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
            type_str="T",
            description="Scale tensor.",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        ),
        OpSchema.FormalParameter(
            name="B",
            type_str="T",
            description="Bias tensor.",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="Y",
            type_str="T",
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


class LayerNormalizationPatternBase(Pattern):
    """Abstract base class for all LayerNormalization patterns (Pow, Mul, Transposed).

    Provides shared schema and Scale/B validation logic.
    """

    def get_schema(self) -> PatternSchema:
        """Return the schema definition for LayerNormalization pattern."""
        return _LAYERNORM_SCHEMA

    def check_skeleton_result(
        self, skeleton_match_result: SkeletonMatchResult
    ) -> PatternMatchResult | None:
        """Check skeleton match result and validate Scale/B shapes."""
        pattern_result = super().check_skeleton_result(skeleton_match_result)
        if pattern_result is None:
            return None

        if not _validate_layernorm_scale_bias(skeleton_match_result, pattern_result):
            return None

        return pattern_result


class _LayerNormalizationExpandedPatternBase(LayerNormalizationPatternBase):
    """Base class for expanded LayerNorm patterns (Pow and Mul variants).

    Graph: ReduceMean → Sub → Pow/Mul → ReduceMean → Add → Sqrt → Div → Mul → Add
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
        internal_constants.append((4, 1, np.array(epsilon, dtype=dtype)))

        opset_version = domain_versions[ONNXDomain.AI_ONNX]
        internal_attributes: dict[tuple[int, str], Any] = {}

        if opset_version >= 18:
            axes_value = np.array([axis], dtype=np.int64)
            internal_constants.append((0, 1, axes_value))
            internal_constants.append((3, 1, axes_value))
        else:
            internal_attributes[(0, "axes")] = [axis]
            internal_attributes[(3, "axes")] = [axis]

        return internal_constants, internal_attributes

    def _infer_schema_attributes(
        self, skeleton_match_result: SkeletonMatchResult
    ) -> dict[str, Any]:
        """Infer axis and epsilon from matched ReduceMean and Add nodes."""
        matcher = skeleton_match_result.matcher
        nodes = skeleton_match_result.matched_nodes
        opset_version = matcher.domain_versions[ONNXDomain.AI_ONNX]

        reducemean_node = nodes[0]
        if opset_version >= 18:
            if len(reducemean_node.input) < 2:
                raise PatternMismatchedError("ReduceMean missing axes input")
            axes_tensor = reducemean_node.input[1]
            if axes_tensor not in matcher.tensor_values:
                raise PatternMismatchedError(f"Axes tensor {axes_tensor} not found")
            axes_value = matcher.tensor_values[axes_tensor]
        else:
            axes_value = None
            for attr in reducemean_node.attribute:
                if attr.name == "axes":
                    axes_value = np.array(list(attr.ints), dtype=np.int64)
                    break
            if axes_value is None:
                raise PatternMismatchedError("ReduceMean missing axes attribute")

        if len(axes_value) != 1:
            raise PatternMismatchedError(
                f"Only single-axis normalization supported, got axes={axes_value}"
            )
        axis = int(axes_value[0])

        epsilon_node = nodes[4]
        if len(epsilon_node.input) < 2:
            raise PatternMismatchedError("Epsilon node has fewer than 2 inputs")
        epsilon_tensor = epsilon_node.input[1]
        if epsilon_tensor not in matcher.tensor_values:
            raise PatternMismatchedError(f"Epsilon tensor {epsilon_tensor} not found")
        epsilon_value = float(matcher.tensor_values[epsilon_tensor].flat[0])

        return {"axis": axis, "epsilon": epsilon_value}


class LayerNormalizationPowPattern(_LayerNormalizationExpandedPatternBase):
    """Pattern definition for LayerNormalization using Pow for squaring.

    This translates to the following node topology (9 nodes):
        Input -> ReduceMean -> Sub -> Pow(2) -> ReduceMean -> Add(eps) ->
                 Sqrt -> Div -> Mul(weight) -> Add(bias)
           |                    ^                       ^
           +--------------------+-----------------------+
    """

    def _get_squaring_internal_constants(
        self, dtype: np.dtype
    ) -> list[tuple[int, int, np.ndarray]]:
        """Return Pow exponent constant (2.0)."""
        return [(2, 1, np.array(2.0, dtype=dtype))]

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for LayerNormalization pattern.

        Returns:
            Skeleton defining the LayerNormalization computation graph topology.
        """
        # Node indices: 0=ReduceMean, 1=Sub, 2=Pow, 3=ReduceMean, 4=Add, 5=Sqrt, 6=Div, 7=Mul, 8=Add
        node_op_types = [
            "ReduceMean",  # 0: compute mean
            "Sub",  # 1: center the input
            "Pow",  # 2: square (exp=2)
            "ReduceMean",  # 3: compute variance
            "Add",  # 4: add epsilon
            "Sqrt",  # 5: standard deviation
            "Div",  # 6: normalize
            "Mul",  # 7: scale by weight
            "Add",  # 8: add bias
        ]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        # Edges: (src, src_slot, dst, dst_slot)
        # -1 represents input X, -2 represents input Scale, -3 represents input B
        edges = [
            (-1, 0, 0, 0),  # X -> ReduceMean1[0]
            (-1, 0, 1, 0),  # X -> Sub[0] (minuend)
            (0, 0, 1, 1),  # ReduceMean1 -> Sub[1] (subtrahend)
            (1, 0, 2, 0),  # Sub -> Pow[0] (base)
            (2, 0, 3, 0),  # Pow -> ReduceMean2[0]
            (3, 0, 4, 0),  # ReduceMean2 -> Add[0] (augend)
            (4, 0, 5, 0),  # Add -> Sqrt[0]
            (5, 0, 6, 1),  # Sqrt -> Div[1] (divisor)
            (1, 0, 6, 0),  # Sub -> Div[0] (dividend, skip connection)
            (6, 0, 7, 0),  # Div -> Mul[0] (multiplicand)
            (-2, 0, 7, 1),  # Scale -> Mul[1] (multiplier/weight)
            (7, 0, 8, 0),  # Mul -> Add[0] (augend)
            (-3, 0, 8, 1),  # B (bias) -> Add[1] (addend)
        ]

        exit_nodes = [8]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=exit_nodes,
            n_inputs=3,
        )


class LayerNormalizationMulPattern(_LayerNormalizationExpandedPatternBase):
    """Pattern definition for LayerNormalization using Mul for squaring.

    This variant uses Mul(x, x) instead of Pow(x, 2) for the squaring operation.
    Otherwise identical to LayerNormalizationPowPattern.

    This translates to the following node topology (9 nodes):
        Input -> ReduceMean -> Sub -> Mul(x,x) -> ReduceMean -> Add(eps) ->
                 Sqrt -> Div -> Mul(weight) -> Add(bias)
           |                    ^  ^                     ^
           +--------------------+--+---------------------+
    """

    def _get_squaring_internal_constants(
        self, dtype: np.dtype
    ) -> list[tuple[int, int, np.ndarray]]:
        """Return empty list - Mul variant has no squaring-specific constants."""
        return []

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for LayerNormalization pattern (Mul variant).

        Returns:
            Skeleton defining the LayerNormalization computation graph topology.
        """
        # Node indices: 0=ReduceMean, 1=Sub, 2=Mul, 3=ReduceMean, 4=Add, 5=Sqrt, 6=Div, 7=Mul, 8=Add
        node_op_types = [
            "ReduceMean",  # 0: compute mean
            "Sub",  # 1: center the input
            "Mul",  # 2: square (x*x)
            "ReduceMean",  # 3: compute variance
            "Add",  # 4: add epsilon
            "Sqrt",  # 5: standard deviation
            "Div",  # 6: normalize
            "Mul",  # 7: scale by weight
            "Add",  # 8: add bias
        ]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        # Edges: (src, src_slot, dst, dst_slot)
        # -1 represents input X, -2 represents input Scale, -3 represents input B
        edges = [
            (-1, 0, 0, 0),  # X -> ReduceMean1[0]
            (-1, 0, 1, 0),  # X -> Sub[0] (minuend)
            (0, 0, 1, 1),  # ReduceMean1 -> Sub[1] (subtrahend)
            (1, 0, 2, 0),  # Sub -> Mul[0] (first input)
            (1, 0, 2, 1),  # Sub -> Mul[1] (second input, for x*x)
            (2, 0, 3, 0),  # Mul -> ReduceMean2[0]
            (3, 0, 4, 0),  # ReduceMean2 -> Add[0] (augend)
            (4, 0, 5, 0),  # Add -> Sqrt[0]
            (5, 0, 6, 1),  # Sqrt -> Div[1] (divisor)
            (1, 0, 6, 0),  # Sub -> Div[0] (dividend, skip connection)
            (6, 0, 7, 0),  # Div -> Mul[0] (multiplicand)
            (-2, 0, 7, 1),  # Scale -> Mul[1] (multiplier/weight)
            (7, 0, 8, 0),  # Mul -> Add[0] (augend)
            (-3, 0, 8, 1),  # B (bias) -> Add[1] (addend)
        ]

        exit_nodes = [8]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=exit_nodes,
            n_inputs=3,
        )


class TransposedSingleLayerNormalizationPattern(LayerNormalizationPatternBase):
    """Transpose-wrapped LayerNormalization pattern for arbitrary axis.

    Topology: X → Transpose → LayerNorm(axis=-1) → Transpose → Y
              Scale/B → Reshape (to 1D) → LayerNorm

    Universal rewrite target for Pow/Mul patterns.
    """

    def _compute_transpose_permutation(self, axis: int, rank: int) -> tuple[list[int], list[int]]:
        """Compute permutations to move axis to last position and back."""
        # Normalize negative axis
        if axis < 0:
            axis = rank + axis

        # Validate axis range
        if axis < 0 or axis >= rank:
            raise ValueError(f"axis {axis} out of range for rank {rank}")

        # If axis already last, return identity
        if axis == rank - 1:
            return list(range(rank)), list(range(rank))

        # Build forward permutation: move axis to end
        # [0, 1, ..., axis-1, axis+1, ..., rank-1, axis]
        perm_forward = list(range(rank))
        perm_forward.append(perm_forward.pop(axis))

        # Build inverse permutation
        perm_inverse = [0] * rank
        for i, p in enumerate(perm_forward):
            perm_inverse[p] = i

        return perm_forward, perm_inverse

    def get_skeleton(self) -> Skeleton:
        """Return skeleton: Transpose → LayerNorm → Transpose with Reshape for Scale/B."""
        return Skeleton(
            node_op_types=["Transpose", "Reshape", "Reshape", "LayerNormalization", "Transpose"],
            node_domains=[ONNXDomain.AI_ONNX] * 5,
            edges=[
                (-1, 0, 0, 0),  # X -> Transpose1[0]
                (0, 0, 3, 0),  # Transpose1 -> LayerNorm[0]
                (-2, 0, 1, 0),  # Scale -> Reshape1[0]
                (1, 0, 3, 1),  # Reshape1 -> LayerNorm[1]
                (-3, 0, 2, 0),  # B -> Reshape2[0]
                (2, 0, 3, 2),  # Reshape2 -> LayerNorm[2]
                (3, 0, 4, 0),  # LayerNorm -> Transpose2[0]
            ],
            exit_nodes=[4],
            n_inputs=3,
        )

    def _infer_schema_attributes(
        self, skeleton_match_result: SkeletonMatchResult
    ) -> dict[str, Any]:
        """Infer axis from Transpose perm and epsilon from LayerNorm node."""
        nodes = skeleton_match_result.matched_nodes

        # Extract perm from first Transpose (node 0) to infer original axis
        transpose1_node = nodes[0]
        perm_forward = None
        for attr in transpose1_node.attribute:
            if attr.name == "perm":
                perm_forward = list(attr.ints)
                break
        if perm_forward is None:
            raise PatternMismatchedError("Transpose missing perm attribute")

        # Infer original axis: the position that was moved to -1
        # perm_forward[-1] tells us which original axis is now at the end
        axis = perm_forward[-1]

        # Extract epsilon from LayerNormalization node (node 3)
        ln_node = nodes[3]
        epsilon = None
        for attr in ln_node.attribute:
            if attr.name == "epsilon":
                epsilon = get_attribute_proto_value(attr, replace_float_with_dummy=False)
                break
        if epsilon is None:
            raise PatternMismatchedError("LayerNormalization missing epsilon")

        return {"axis": axis, "epsilon": epsilon}

    def _get_normalized_dim(self, inputs: dict[str, np.ndarray], attributes: dict[str, Any]) -> int:
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

        # Compute transpose permutations
        perm_forward, perm_inverse = self._compute_transpose_permutation(axis, rank)

        # Reshape target shape: 1D with normalized_dim
        reshape_target_shape = np.array([normalized_dim], dtype=np.int64)

        # Internal constants: Reshape shape inputs
        internal_constants: list[tuple[int, int, np.ndarray]] = [
            (1, 1, reshape_target_shape),  # Reshape1 (Scale) shape input
            (2, 1, reshape_target_shape),  # Reshape2 (B) shape input
        ]

        # Internal attributes
        internal_attributes: dict[tuple[int, str], Any] = {
            (0, "perm"): perm_forward,  # Transpose1 perm
            (3, "axis"): -1,  # LayerNorm always axis=-1
            (3, "epsilon"): epsilon,  # LayerNorm epsilon
            (4, "perm"): perm_inverse,  # Transpose2 perm
        }

        return internal_constants, internal_attributes


class LayerNormalizationPatternInputGenerator(
    PatternInputGenerator, get_runtime_checker_op("LayerNormalization")
):
    """Base PatternInputGenerator for LayerNormalization patterns.

    Adapts combinations from parent generator with proper Scale/B shapes for all axes.
    """

    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """Return finite attribute sets (empty for this pattern)."""
        return {}

    def get_input_and_infinite_attribute_combinations(self) -> list[dict[str, Any]]:
        """Return input combinations with broadcast-compatible Scale/B shapes."""
        from winml.modelkit.pattern.op_input_gen import InputValueConstraint

        combinations = super().get_input_and_infinite_attribute_combinations()

        adapted = []
        for combo in combinations:
            axis = combo["axis"]
            x_shape = combo["X"].shape
            rank = len(x_shape)
            normalized_axis = axis if axis >= 0 else rank + axis
            normalized_dim = x_shape[normalized_axis]

            if axis == -1 or normalized_axis == rank - 1:
                adapted_combo = dict(combo.items())
                if "epsilon" not in adapted_combo:
                    adapted_combo["epsilon"] = 1e-5
                adapted.append(adapted_combo)
            else:
                # Reshape Scale/B to [1, ..., normalized_dim, ..., 1]
                broadcast_shape = [1] * rank
                broadcast_shape[normalized_axis] = normalized_dim

                scale_value = combo["Scale"].value
                bias_value = combo["B"].value
                new_scale = np.ones((normalized_dim,), dtype=scale_value.dtype).reshape(
                    broadcast_shape
                )
                new_bias = np.zeros((normalized_dim,), dtype=bias_value.dtype).reshape(
                    broadcast_shape
                )

                adapted_combo = {
                    "X": combo["X"],
                    "Scale": InputValueConstraint(new_scale),
                    "B": InputValueConstraint(new_bias),
                    "axis": axis,
                    "epsilon": combo.get("epsilon", 1e-5),
                }
                adapted.append(adapted_combo)

        return adapted


@register_pattern_input_generator
class LayerNormalizationPowPatternInputGenerator(LayerNormalizationPatternInputGenerator):
    """PatternInputGenerator for LayerNormalization pattern (Pow variant)."""

    pattern = LayerNormalizationPowPattern()
    registration_name = "LayerNormalizationPow"


@register_pattern_input_generator
class LayerNormalizationMulPatternInputGenerator(LayerNormalizationPatternInputGenerator):
    """PatternInputGenerator for LayerNormalization pattern (Mul variant)."""

    pattern = LayerNormalizationMulPattern()
    registration_name = "LayerNormalizationMul"


@register_pattern_input_generator
class TransposedSingleLayerNormalizationPatternInputGenerator(
    LayerNormalizationPatternInputGenerator
):
    """PatternInputGenerator for TransposedSingleLayerNormalizationPattern."""

    pattern = TransposedSingleLayerNormalizationPattern()
    registration_name = "TransposedSingleLayerNormalization"
