# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any

import numpy as np
from onnx.defs import OpSchema

from winml.modelkit.onnx.domains import ONNXDomain
from winml.modelkit.pattern.base import (
    Pattern,
    PatternInputGenerator,
    PatternSchema,
    Skeleton,
    register_pattern_input_generator,
)
from winml.modelkit.pattern.match import PatternMatchResult, SkeletonMatchResult
from winml.modelkit.pattern.op_input_gen import InputShapeConstraint, InputValueConstraint


# Shared schema for MatMulAdd and ReshapeGemmReshape patterns
# Both patterns perform the same computation: Y = MatMul(A, B) + C
_MATMUL_ADD_SCHEMA = PatternSchema(
    name="GemmPattern",
    doc=(
        "MatMul followed by Add pattern (common in linear layers).\n"
        "Computes Y = MatMul(A, B) + C where A is an N-dimensional tensor, "
        "B is a 2D weight matrix, and C is a 1D bias vector.\n"
        "\n"
        "Shape constraints (similar to Gemm but more flexible for A):\n"
        "- A: N-dimensional tensor (any rank >= 2)\n"
        "- B: 2D matrix (required)\n"
        "- C: 1D vector (required)\n"
        "\n"
        "Unlike Gemm which requires A to be 2D, this pattern allows A to be N-dimensional, "
        "enabling batched matrix multiplication followed by bias addition."
    ),
    type_constraints=[
        OpSchema.TypeConstraintParam(
            type_param_str="T",
            allowed_type_strs=[
                "tensor(float16)",
                "tensor(float)",
                "tensor(double)",
                "tensor(uint32)",
                "tensor(uint64)",
                "tensor(int32)",
                "tensor(int64)",
                "tensor(bfloat16)",
            ],
            description="Constrain input and output types to float/int tensors.",
        )
    ],
    inputs=[
        OpSchema.FormalParameter(
            name="A",
            type_str="T",
            description="N-dimensional input tensor (N >= 2). ",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        ),
        OpSchema.FormalParameter(
            name="B",
            type_str="T",
            description="2D weight matrix (required to be 2D, like in Gemm).",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        ),
        OpSchema.FormalParameter(
            name="C",
            type_str="T",
            description="1D bias vector (required to be 1D, like in Gemm).",
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
            description="Output tensor with the same rank as A.",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        )
    ],
)


class MatMulAddPattern(Pattern):
    """Pattern definition for MatMul followed by Add (common in linear layers).

    This pattern represents: output = MatMul(A, B) + C
    where A is an N-dimensional tensor, B is a 2D weight matrix, and C is a 1D bias vector.

    Shape constraints (similar to Gemm but more flexible):
    - A: N-dimensional tensor (any rank >= 2)
    - B: 2D matrix (required, like in Gemm)
    - C: 1D vector (required, like in Gemm)

    Unlike Gemm which requires A to be 2D, this pattern allows A to be N-dimensional,
    enabling batched matrix multiplication followed by bias addition.

    This translates to the following node topology:
    - MatMul: matrix multiplication of A and B
    - Add: adds bias C to MatMul output
    """

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for MatMulAdd pattern.

        Returns:
            Skeleton defining the MatMul->Add computation graph topology.
        """
        # MatMulAdd pattern: MatMul(A, B) + bias
        # Node indices: 0=MatMul, 1=Add
        node_op_types = ["MatMul", "Add"]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        # Edges: (src, src_slot, dst, dst_slot)
        # -1, -2, -3 represent the inputs to the subgraph
        edges = [
            (-1, 0, 0, 0),  # input A -> MatMul[0]
            (-2, 0, 0, 1),  # input B -> MatMul[1]
            (-3, 0, 1, 0),  # bias -> Add[0]
            (0, 0, 1, 1),  # MatMul output -> Add[1]
        ]

        # Exit node that produces the final output
        exit_nodes = [1]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=exit_nodes,
            n_inputs=3,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants and attributes for MatMulAdd pattern.

        MatMulAdd pattern has no internal constants or attributes.

        Args:
            inputs: Dictionary mapping input names to numpy array values.
            attributes: Dictionary of attribute values for the pattern.
            is_constant_map: Dict mapping input_name -> is_constant (bool).
            domain_versions: Dict mapping ONNXDomain to opset version.

        Returns:
            Tuple of (empty list, empty dict).
        """
        return [], {}

    def check_skeleton_result(
        self, skeleton_match_result: SkeletonMatchResult
    ) -> PatternMatchResult | None:
        """Check if skeleton match result satisfies MatMulAdd constraints.

        Validates shape constraints:
        - B must be 2D (like in Gemm)
        - C must be 1D (like in Gemm)
        - A can be N-dimensional (unlike Gemm which requires 2D)

        Args:
            skeleton_match_result: The skeleton match result to validate.

        Returns:
            PatternMatchResult if validation passes, None otherwise.
        """
        # Call base implementation for constant/attribute validation
        pattern_result = super().check_skeleton_result(skeleton_match_result)
        if pattern_result is None:
            return None

        # Validate shape constraints
        # A (input 0): can be N-dimensional, no constraint
        # B (input 1): must be 2D
        # C (input 2): must be 1D
        input_infos = pattern_result.input_infos

        if "B" in input_infos:
            b_shape = input_infos["B"].shape
            if b_shape is not None and len(b_shape) != 2:
                return None  # B must be 2D

        if "C" in input_infos:
            c_shape = input_infos["C"].shape
            if c_shape is not None and len(c_shape) != 1:
                return None  # C must be 1D

        return pattern_result

    def get_schema(self) -> PatternSchema:
        """Return the schema definition for MatMulAdd pattern.

        Returns:
            PatternSchema defining the MatMulAdd pattern's input/output types.
        """
        return _MATMUL_ADD_SCHEMA


class ReshapeGemmReshapePattern(Pattern):
    """Pattern definition for Reshape -> Gemm -> Reshape (linear layer implementation).

    This pattern represents: output = Reshape(Gemm(Reshape(A), B, C))
    where:
    - A is an N-dimensional tensor reshaped to 2D for Gemm
    - B is a 2D weight matrix
    - C is a 1D bias vector
    - Output is reshaped back to N-dimensional

    This pattern performs the same computation as MatMulAdd but uses Gemm operator
    with explicit Reshape operations to handle N-dimensional inputs.

    Gemm attribute constraints (must be default values):
    - alpha: 1.0 (default)
    - beta: 1.0 (default)
    - transA: 0 (default)
    - transB: 0 (default)

    Shape constraints (same as MatMulAdd):
    - A: N-dimensional tensor (any rank >= 2)
    - B: 2D matrix (required)
    - C: 1D vector (required)
    """

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for ReshapeGemmReshape pattern.

        Returns:
            Skeleton defining the Reshape->Gemm->Reshape computation graph topology.
        """
        # Pattern: Reshape(A) -> Gemm(reshaped_A, B, C) -> Reshape(output)
        # Node indices: 0=Reshape, 1=Gemm, 2=Reshape
        node_op_types = ["Reshape", "Gemm", "Reshape"]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        # Edges: (src, src_slot, dst, dst_slot)
        # -1, -2, -3 represent the inputs to the subgraph (A, B, C)
        # Note: Reshape nodes also need shape inputs which come from constants/initializers
        edges = [
            (-1, 0, 0, 0),  # input A -> Reshape[0]
            (0, 0, 1, 0),  # Reshape output -> Gemm[0]
            (-2, 0, 1, 1),  # input B -> Gemm[1]
            (-3, 0, 1, 2),  # input C -> Gemm[2]
            (1, 0, 2, 0),  # Gemm output -> Reshape[0]
        ]

        # Exit node that produces the final output
        exit_nodes = [2]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=exit_nodes,
            n_inputs=3,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants and attributes for ReshapeGemmReshape pattern.

        ReshapeGemmReshape pattern requires shape inputs for the Reshape nodes:
        - Node 0 (first Reshape): shape to flatten A to 2D for Gemm
        - Node 2 (second Reshape): shape to restore original dimensions

        The Gemm attributes are constrained to default values.

        Args:
            inputs: Dictionary mapping input names to numpy array values.
            attributes: Dictionary of attribute values for the pattern.
            is_constant_map: Dict mapping input_name -> is_constant (bool).
            domain_versions: Dict mapping ONNXDomain to opset version.

        Returns:
            Tuple of (internal_constants, internal_attributes).
        """
        internal_constants = []

        # Compute shape constants for Reshape nodes based on input A
        if "A" in inputs and inputs["A"] is not None:
            a_shape = inputs["A"].shape
            # First Reshape (node 0, slot 1): flatten to 2D for Gemm
            # Shape is [product_of_batch_dims, last_dim] where last_dim is the last dimension of A
            batch_size = int(np.prod(a_shape[:-1]))
            first_reshape_shape = np.array([batch_size, a_shape[-1]], dtype=np.int64)
            internal_constants.append((0, 1, first_reshape_shape))

            # Second Reshape (node 2, slot 1): restore to original shape
            # The output shape is A's shape with the last dim replaced by B's output dim
            if "B" in inputs and inputs["B"] is not None:
                b_shape = inputs["B"].shape
                # Output shape: A's batch dims + B's output dim
                output_shape = [*list(a_shape[:-1]), b_shape[-1]]
                second_reshape_shape = np.array(output_shape, dtype=np.int64)
            else:
                raise ValueError(
                    "Input 'B' is required to determine output "
                    "shape for ReshapeGemmReshape pattern."
                )
            internal_constants.append((2, 1, second_reshape_shape))

        # Gemm attributes must be default values (node 1 is Gemm)
        internal_attributes: dict[tuple[int, str], Any] = {
            (1, "alpha"): 1.0,
            (1, "beta"): 1.0,
            (1, "transA"): 0,
            (1, "transB"): 0,
        }
        return internal_constants, internal_attributes

    def check_skeleton_result(
        self, skeleton_match_result: SkeletonMatchResult
    ) -> PatternMatchResult | None:
        """Check if skeleton match result satisfies ReshapeGemmReshape constraints.

        Validates:
        - Gemm attributes are default values (alpha=1.0, beta=1.0, transA=0, transB=0)
          Note: Attribute validation is handled by the base class using
          get_internal_constants_and_attributes() for attributes that are explicitly set.
          This method additionally validates non-default explicit values.
        - B must be 2D (like in Gemm)
        - C must be 1D (like in Gemm)

        Args:
            skeleton_match_result: The skeleton match result to validate.

        Returns:
            PatternMatchResult if validation passes, None otherwise.
        """
        # Call base implementation for constant/attribute validation
        pattern_result = super().check_skeleton_result(skeleton_match_result)
        if pattern_result is None:
            return None

        # Validate shape constraints (same as MatMulAdd)
        # A (input 0): can be N-dimensional, no constraint
        # B (input 1): must be 2D
        # C (input 2): must be 1D
        input_infos = pattern_result.input_infos

        if "B" in input_infos:
            b_shape = input_infos["B"].shape
            if b_shape is not None and len(b_shape) != 2:
                return None  # B must be 2D

        if "C" in input_infos:
            c_shape = input_infos["C"].shape
            if c_shape is not None and len(c_shape) != 1:
                return None  # C must be 1D

        return pattern_result

    def get_schema(self) -> PatternSchema:
        """Return the schema definition for ReshapeGemmReshape pattern.

        Returns the shared MatMulAdd schema since both patterns perform the same computation.

        Returns:
            PatternSchema defining the pattern's input/output types.
        """
        return _MATMUL_ADD_SCHEMA


class GemmPatternInputGenerator(PatternInputGenerator):
    """PatternInputGenerator for Gemm patterns."""

    pattern = ReshapeGemmReshapePattern()

    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """Return finite attribute sets for ReshapeGemmReshape (none)."""
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, Any]]:
        """Return input and infinite attribute combinations for ReshapeGemmReshape."""
        combinations = []
        # Use m=3, k=4, n=5 as base dimensions
        # Output shape will be (3, 5) after multiplication
        m, k, n = 3, 4, 5
        a_shapes = [(2,) * i + (m, k) for i in range(5)]
        b_shapes = [
            (k, n),
        ]

        # C shapes: full output, 1D, scalar, or None
        # Output after A@B is always (m, n) = (3, 5)
        c_options = [
            InputShapeConstraint((n,)),  # 1D bias (broadcasts to (3, 5)): (5,)
            InputShapeConstraint(()),  # Scalar broadcast: ()
            InputValueConstraint(np.array(0)),  # default is 0 scalar
        ]

        # Generate all 16 combinations: 2 A shapes x 2 B shapes x 4 C options
        for a_shape in a_shapes:
            for b_shape in b_shapes:
                for c_option in c_options:
                    combination = {
                        "A": InputShapeConstraint(a_shape),
                        "B": InputShapeConstraint(b_shape),
                    }
                    combination["C"] = c_option

                    combinations.append(combination)

        return combinations

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for Gemm operator testing.

        Args:
            properties: Base properties containing A_shape, B_shape, and optionally C_shape

        Returns:
            Updated properties with Gemm-specific derived values (A_dim, B_dim, C_dim)
        """
        item = properties.copy()
        item["A_dim"] = len(item["A_shape"])
        item["B_dim"] = len(item["B_shape"])
        # C is optional, only add C_dim if C_shape exists
        if "C_shape" in item:
            item["C_dim"] = len(item["C_shape"])
        else:
            item["C_dim"] = 0
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values.

        Returns:
            List of property names that represent shapes with infinite possibilities
        """
        return ["A_shape", "B_shape", "C_shape", "C_value"]


@register_pattern_input_generator
class ReshapeGemmReshapePatternInputGenerator(GemmPatternInputGenerator):
    """PatternInputGenerator for ReshapeGemmReshape pattern."""

    pattern = ReshapeGemmReshapePattern()
    registration_name = "ReshapeGemmReshape"


@register_pattern_input_generator
class MatMulAddPatternInputGenerator(GemmPatternInputGenerator):
    """PatternInputGenerator for MatMulAdd pattern."""

    pattern = MatMulAddPattern()
    registration_name = "MatMulAdd"
