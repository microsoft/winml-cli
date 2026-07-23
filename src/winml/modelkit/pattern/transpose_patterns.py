# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Transpose-related patterns for ONNX models.

This module provides patterns for matching Reshape-Transpose-Reshape sequences
commonly found in attention mechanisms and tensor manipulation operations.
"""

from typing import Any

import numpy as np
from onnx.defs import OpSchema

from ..onnx import ONNXDomain
from .base import (
    Pattern,
    PatternInputGenerator,
    PatternMismatchedError,
    PatternSchema,
    Skeleton,
    register_pattern_input_generator,
)
from .match import PatternMatchResult, SkeletonMatchResult
from .op_input_gen import InputShapeConstraint


# Schema for ReshapeTransposeReshape pattern
_RESHAPE_TRANSPOSE_RESHAPE_SCHEMA = PatternSchema(
    name="ReshapeTransposeReshapePattern",
    doc=(
        "Reshape followed by Transpose followed by Reshape pattern.\n"
        "This pattern is common in attention mechanisms where tensors are reshaped "
        "for multi-head attention, transposed to rearrange dimensions, "
        "and then reshaped back.\n"
        "\n"
        "Computes: output = Reshape(Transpose(Reshape(data, transpose_shape), perm), "
        "output_shape)\n"
        "\n"
        "Attributes:\n"
        "- transpose_shape: Shape for the first Reshape (before Transpose)\n"
        "- perm: Permutation for the Transpose operation\n"
        "- output_shape: Shape for the final Reshape (after Transpose)\n"
    ),
    type_constraints=[
        OpSchema.TypeConstraintParam(
            type_param_str="T",
            allowed_type_strs=[
                "tensor(float16)",
                "tensor(float)",
                "tensor(double)",
                "tensor(uint8)",
                "tensor(uint16)",
                "tensor(uint32)",
                "tensor(uint64)",
                "tensor(int8)",
                "tensor(int16)",
                "tensor(int32)",
                "tensor(int64)",
                "tensor(bfloat16)",
                "tensor(bool)",
            ],
            description="Constrain input and output types to all numeric tensors.",
        )
    ],
    inputs=[
        OpSchema.FormalParameter(
            name="data",
            type_str="T",
            description="Input tensor to be reshaped and transposed.",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="output",
            type_str="T",
            description="Output tensor after reshape-transpose-reshape operations.",
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        )
    ],
    attributes={
        "transpose_shape": OpSchema.Attribute(
            name="transpose_shape",
            description="Shape for the first Reshape (before Transpose). "
            "This determines the intermediate tensor shape for transposition.",
            type=OpSchema.AttrType.INTS,
            required=True,
        ),
        "perm": OpSchema.Attribute(
            name="perm",
            description="Permutation of dimensions for the Transpose operation. "
            "Lists the new order of dimensions.",
            type=OpSchema.AttrType.INTS,
            required=True,
        ),
        "output_shape": OpSchema.Attribute(
            name="output_shape",
            description="Shape for the final Reshape (after Transpose). "
            "This determines the final output tensor shape.",
            type=OpSchema.AttrType.INTS,
            required=True,
        ),
    },
)


class ReshapeTransposeReshapeOverlyHighDimPattern(Pattern):
    """Pattern for Reshape -> Transpose -> Reshape with overly high intermediate dimensionality.

    This pattern represents: Y = Reshape(Transpose(Reshape(X, transpose_shape), perm), output_shape)
    where the intermediate transpose operates on >= 6 dimensions, indicating an opportunity
    to merge consecutive axes and reduce the Transpose dimensionality.

    This is commonly found in:
    - Attention mechanisms (reshaping for multi-head attention)
    - Tensor dimension rearrangement operations
    - View/permute operations in deep learning models

    Dimension constraint: intermediate transpose_shape must have >= 6 dimensions.

    Attributes (inferred from matched nodes via _infer_schema_attributes):
    - transpose_shape: Shape constant for the first Reshape (node 0, slot 1)
    - perm: Permutation attribute of the Transpose node (node 1)
    - output_shape: Shape constant for the final Reshape (node 2, slot 1)

    Node topology:
    - Node 0 (Reshape): Reshape(X, transpose_shape)
    - Node 1 (Transpose): Transpose(reshape_output, perm=perm)
    - Node 2 (Reshape): Reshape(transpose_output, output_shape)

    Validation:
    The base class check_skeleton_result() handles validation by:
    1. Calling _infer_schema_attributes() to extract attributes from matched nodes
    2. Calling get_internal_constants_and_attributes() with those attributes
    3. Validating that internal constants match (Reshape shape inputs)
    4. Validating that internal attributes match (Transpose perm attribute)
    """

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for ReshapeTransposeReshape pattern.

        Returns:
            Skeleton defining the Reshape->Transpose->Reshape computation graph topology.
        """
        # Pattern: Reshape(X, shape1) -> Transpose(perm) -> Reshape(output_shape)
        # Node indices: 0=Reshape, 1=Transpose, 2=Reshape
        node_op_types = ["Reshape", "Transpose", "Reshape"]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        # Edges: (src, src_slot, dst, dst_slot)
        # -1 represents the input X to the subgraph
        # Note: Reshape nodes have shape as second input (slot 1)
        edges = [
            (-1, 0, 0, 0),  # input X -> Reshape[0] (data input)
            (0, 0, 1, 0),  # Reshape output -> Transpose[0]
            (1, 0, 2, 0),  # Transpose output -> Reshape[0] (data input)
        ]

        # Exit node that produces the final output
        exit_nodes = [2]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=exit_nodes,
            n_inputs=1,
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants and attributes for ReshapeTransposeReshape pattern.

        This method is called by the base class check_skeleton_result() for validation
        and by get_onnx_model() for model generation. The attributes dict is populated
        by _infer_schema_attributes() during matching.

        Internal constants (Reshape shape inputs):
        - Node 0 (first Reshape) slot 1: transpose_shape constant (int64 array)
        - Node 2 (second Reshape) slot 1: output_shape constant (int64 array)

        Internal attributes (Transpose node):
        - Node 1, 'perm': Permutation for the Transpose operation

        Args:
            inputs: Dictionary mapping input names to numpy array values.
                    Not used for this pattern as shapes come from attributes.
            attributes: Dictionary of attribute values for the pattern.
                Expected keys: 'transpose_shape', 'perm', 'output_shape'.
                During matching, these are populated by _infer_schema_attributes().
            is_constant_map: Dict mapping input_name -> is_constant (bool).
                    Not used for this pattern.
            domain_versions: Dict mapping ONNXDomain to opset version.

        Returns:
            Tuple of (internal_constants, internal_attributes):
            - internal_constants: List of (node_idx, slot, np.ndarray) for shape constants
            - internal_attributes: Dict of (node_idx, attr_name) -> value for perm attribute
        """
        internal_constants = []

        transpose_shape = np.array(attributes["transpose_shape"], dtype=np.int64)
        internal_constants.append((0, 1, transpose_shape))

        # Get output_shape from attributes for second Reshape (node 2, slot 1)
        output_shape = np.array(attributes["output_shape"], dtype=np.int64)
        internal_constants.append((2, 1, output_shape))

        # Transpose perm attribute (node 1)
        internal_attributes: dict[tuple[int, str], Any] = {}
        internal_attributes[(1, "perm")] = list(attributes["perm"])

        return internal_constants, internal_attributes

    def _infer_schema_attributes(
        self, skeleton_match_result: SkeletonMatchResult
    ) -> dict[str, Any]:
        """Infer schema-level attributes from the matched pattern.

        Extracts:
        - transpose_shape: from the first Reshape's shape input (may contain -1)
        - perm: from the Transpose node's perm attribute
        - output_shape: from the second Reshape's shape input (may contain -1)

        Note: -1 values are preserved here for constant constraint validation.
        Resolution of -1 to actual dimensions happens during model generation
        in get_internal_constants_and_attributes of patterns that need it.

        Args:
            skeleton_match_result: The skeleton match result containing matched nodes.

        Returns:
            Dictionary with 'transpose_shape', 'perm', and 'output_shape' attributes.

        Raises:
            PatternMismatchedError: If any required attribute cannot be extracted
                (e.g., shape inputs are not constants).
        """
        attributes: dict[str, Any] = {}
        matcher = skeleton_match_result.matcher
        matched_nodes = skeleton_match_result.matched_nodes

        # Extract transpose_shape from first Reshape's shape input
        reshape1_node = matched_nodes[0]
        if len(reshape1_node.input) <= 1:
            raise PatternMismatchedError("First Reshape node missing shape input")
        shape_input_name = reshape1_node.input[1]
        if shape_input_name not in matcher.tensor_values:
            raise PatternMismatchedError(
                f"First Reshape shape input '{shape_input_name}' is not a constant"
            )
        attributes["transpose_shape"] = tuple(matcher.tensor_values[shape_input_name].tolist())

        # Extract perm from Transpose node's attribute
        transpose_node = matched_nodes[1]
        perm_found = False
        for attr in transpose_node.attribute:
            if attr.name == "perm":
                attributes["perm"] = tuple(attr.ints)
                perm_found = True
                break
        if not perm_found:
            raise PatternMismatchedError("Transpose node missing 'perm' attribute")

        # Extract output_shape from second Reshape's shape input
        reshape2_node = matched_nodes[2]
        if len(reshape2_node.input) <= 1:
            raise PatternMismatchedError("Second Reshape node missing shape input")
        shape_input_name = reshape2_node.input[1]
        if shape_input_name not in matcher.tensor_values:
            raise PatternMismatchedError(
                f"Second Reshape shape input '{shape_input_name}' is not a constant"
            )
        attributes["output_shape"] = tuple(matcher.tensor_values[shape_input_name].tolist())

        return attributes

    def check_skeleton_result(
        self, skeleton_match_result: SkeletonMatchResult
    ) -> "PatternMatchResult | None":
        """Validate and filter the skeleton match result.

        Enforces the OverlyHighDim constraint (intermediate transpose >= 6D) and
        excludes already-merged instances where ``_compute_merged_transpose`` would
        produce no change.  Such subgraphs are already in their optimised form and
        need neither a rewrite nor a report.
        """
        result = super().check_skeleton_result(skeleton_match_result)
        if result is None:
            return None
        transpose_shape = tuple(result.attributes["transpose_shape"])
        if len(transpose_shape) < 6:
            return None
        perm = tuple(result.attributes["perm"])
        merged_shape, merged_perm = _compute_merged_transpose(transpose_shape, perm)
        if merged_shape == transpose_shape and merged_perm == perm:
            return None
        return result

    @property
    def pattern_id(self) -> str:
        """Return pattern ID matching the information rule configuration."""
        return f"SUBGRAPH/{type(self).__name__}"

    def get_schema(self) -> PatternSchema:
        """Return the schema definition for ReshapeTransposeReshape pattern.

        Returns:
            PatternSchema defining the pattern's input/output types and attributes.
        """
        return _RESHAPE_TRANSPOSE_RESHAPE_SCHEMA


class _ReshapeTransposeReshapeInputGeneratorBase(PatternInputGenerator):
    """Base PatternInputGenerator for ReshapeTransposeReshape patterns.

    Subclasses only need to define `pattern` and `registration_name`.
    """

    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """Return finite attribute sets (empty for this pattern)."""
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, Any]]:
        """Generate representative input/attribute combinations for testing.

        The combinations intentionally cover both 3-D and 4-D output shapes,
        which are commonly seen in real RTR subgraphs.
        """
        return [
            {
                "data": InputShapeConstraint((1, 256, 256, 96)),
                "transpose_shape": (1, 32, 8, 32, 8, 96),
                "perm": (0, 1, 3, 2, 4, 5),
                "output_shape": (1024, 8, 8, 96),
            },
            {
                "data": InputShapeConstraint((1, 256, 256, 96)),
                "transpose_shape": (1, 32, 8, 32, 8, 96),
                "perm": (0, 1, 3, 2, 4, 5),
                "output_shape": (1024, 64, 96),
            },
            {
                "data": InputShapeConstraint((1, 256, 256, 96)),
                "transpose_shape": (1, 32, 8, 32, 8, 96),
                "perm": (5, 4, 3, 2, 1, 0),
                "output_shape": (1024, 8, 8, 96),
            },
            {
                "data": InputShapeConstraint((1, 256, 256, 96)),
                "transpose_shape": (1, 32, 8, 32, 8, 96),
                "perm": (5, 4, 3, 2, 1, 0),
                "output_shape": (1, 65536, 96),
            },
            {
                "data": InputShapeConstraint((1, 224, 168, 128)),
                "transpose_shape": (1, 32, 7, 24, 7, 128),
                "perm": (0, 1, 3, 2, 4, 5),
                "output_shape": (768, 49, 128),
            },
            {
                "data": InputShapeConstraint((1, 224, 168, 128)),
                "transpose_shape": (1, 32, 24, 7, 7, 128),
                "perm": (0, 1, 3, 2, 4, 5),
                "output_shape": (-1, 224, 168, 128),
            },
            {
                "data": InputShapeConstraint((1, 192, 49, 256)),
                "transpose_shape": (1, 16, 7, 12, 7, 256),
                "perm": (0, 1, 3, 2, 4, 5),
                "output_shape": (192, 49, 256),
            },
            {
                # PixelShuffle-style RTR variant observed in real models.
                # This case provides merged_transpose_dim=5 coverage.
                "data": InputShapeConstraint((1, 1, 3, 3, 4, 4)),
                "transpose_shape": (1, 1, 3, 3, 4, 4),
                "perm": (0, 1, 4, 2, 5, 3),
                "output_shape": (1, 1, 12, 12),
            },
        ]

    def derive_properties(self, properties: dict) -> dict:
        """Derive additional properties for testing."""
        item = properties.copy()
        transpose_shape = item["attr_transpose_shape"]
        perm = item["attr_perm"]
        item["transpose_dim"] = len(transpose_shape)
        item["output_dim"] = len(item["attr_output_shape"])
        # The trailing channel dimension is a key attention-style layout signal
        # and remains finite under the enumerated test cases.
        item["transpose_last_dim"] = int(transpose_shape[-1])
        merged_shape, _ = _compute_merged_transpose(transpose_shape, perm)
        item["merged_transpose_dim"] = len(merged_shape)
        return item

    def get_infinite_property_names(self) -> list[str]:
        """Return names of properties with infinite possible values."""
        return [
            "attr_transpose_shape",
            "attr_output_shape",
            "attr_perm",
            "data_shape",
        ]


@register_pattern_input_generator
class ReshapeTransposeReshapeOverlyHighDimPatternInputGenerator(
    _ReshapeTransposeReshapeInputGeneratorBase
):
    """PatternInputGenerator for ReshapeTransposeReshapeOverlyHighDim pattern."""

    pattern = ReshapeTransposeReshapeOverlyHighDimPattern()
    registration_name = "ReshapeTransposeReshapeOverlyHighDimPattern"


def _resolve_negative_dims(shape: tuple[int, ...], total_size: int) -> tuple[int, ...]:
    """Resolve -1 dimension in a shape using the total element count.

    ONNX Reshape uses -1 to mean "infer this dimension". This function
    computes the actual dimension value.

    Args:
        shape: Shape tuple that may contain at most one -1.
        total_size: Total number of elements (product of actual shape).

    Returns:
        Shape tuple with -1 resolved to actual dimension value.

    Raises:
        ValueError: If shape contains more than one -1 or division is not exact.
    """
    neg_count = sum(1 for d in shape if d < 0)
    if neg_count == 0:
        return shape
    if neg_count > 1:
        raise ValueError(f"Shape {shape} contains more than one negative dimension")

    # Compute product of known dimensions
    known_product = 1
    neg_idx = -1
    for i, d in enumerate(shape):
        if d < 0:
            neg_idx = i
        else:
            known_product *= d

    if known_product == 0:
        raise ValueError(f"Shape {shape} has zero in known dimensions")
    if total_size % known_product != 0:
        raise ValueError(
            f"Cannot resolve -1 in shape {shape}: total_size={total_size} "
            f"is not divisible by known_product={known_product}"
        )

    inferred_dim = total_size // known_product
    resolved = list(shape)
    resolved[neg_idx] = inferred_dim
    return tuple(resolved)


def _compute_merged_transpose(
    transpose_shape: tuple[int, ...],
    perm: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute merged transpose shape and permutation by combining mergeable axes.

    Consecutive INPUT dimensions can be merged if they map to consecutive OUTPUT
    positions (i.e., they stay together after the transpose).

    Example:
        transpose_shape = (1, 32, 8, 32, 8, 96)
        perm = (0, 1, 3, 2, 4, 5)

        Inverse perm (input dim -> output position): {0:0, 1:1, 2:3, 3:2, 4:4, 5:5}

        Input dims 0,1 map to output positions 0,1 (consecutive) -> can merge
        Input dim 2 maps to output position 3
        Input dim 3 maps to output position 2
        Input dims 4,5 map to output positions 4,5 (consecutive) -> can merge

        Input groups: [[0,1], [2], [3], [4,5]]
        merged_shape = (1*32, 8, 32, 8*96) = (32, 8, 32, 768)

        Merged output groups (by output position order):
        - Output positions 0,1 <- merged input dim 0
        - Output position 2 <- merged input dim 2
        - Output position 3 <- merged input dim 1
        - Output positions 4,5 <- merged input dim 3

        merged_perm = (0, 2, 1, 3)

    Args:
        transpose_shape: Shape for the first Reshape (before Transpose).
        perm: Permutation for the Transpose operation.

    Returns:
        Tuple of (merged_shape, merged_perm).
    """
    n = len(perm)
    if n == 0:
        return transpose_shape, perm

    # Build inverse perm: inv_perm[input_dim] = output_position
    inv_perm = [0] * n
    for out_pos, in_dim in enumerate(perm):
        inv_perm[in_dim] = out_pos

    # Group consecutive INPUT dimensions that map to consecutive OUTPUT positions
    input_groups: list[list[int]] = []
    current_group = [0]

    for in_dim in range(1, n):
        prev_in_dim = in_dim - 1
        # Check if consecutive input dims map to consecutive output positions
        if inv_perm[in_dim] == inv_perm[prev_in_dim] + 1:
            current_group.append(in_dim)
        else:
            input_groups.append(current_group)
            current_group = [in_dim]
    input_groups.append(current_group)

    # Build merged input shape: multiply dimensions within each input group
    merged_shape = []
    for group in input_groups:
        merged_dim = 1
        for in_dim in group:
            merged_dim *= transpose_shape[in_dim]
        merged_shape.append(merged_dim)

    # Map old input dim to new merged input dim
    old_input_to_new: dict[int, int] = {}
    for new_idx, group in enumerate(input_groups):
        for in_dim in group:
            old_input_to_new[in_dim] = new_idx

    # Group consecutive OUTPUT positions that come from the same merged input
    # and build the merged perm
    output_groups: list[list[int]] = []
    current_group = [0]
    for out_pos in range(1, n):
        prev_out_pos = out_pos - 1
        # Check if consecutive output positions come from same merged input
        if old_input_to_new[perm[out_pos]] == old_input_to_new[perm[prev_out_pos]]:
            current_group.append(out_pos)
        else:
            output_groups.append(current_group)
            current_group = [out_pos]
    output_groups.append(current_group)

    # Build merged perm: for each output group, which merged input dim does it come from?
    merged_perm = []
    for group in output_groups:
        # All positions in the group come from the same merged input dim
        orig_input_dim = perm[group[0]]
        merged_input_dim = old_input_to_new[orig_input_dim]
        merged_perm.append(merged_input_dim)

    return tuple(merged_shape), tuple(merged_perm)


class ReshapeTransposeReshapeLowDimPattern(ReshapeTransposeReshapeOverlyHighDimPattern):
    """Target pattern for Reshape -> Transpose -> Reshape after axis-merging optimization.

    This pattern inherits from ReshapeTransposeReshapeOverlyHighDimPattern but generates
    ONNX models with merged axes in the Transpose operation to reduce dimensionality to <= 5D.

    The merging optimization identifies neighboring axes that can be combined without
    affecting the Transpose result. Axes can be merged if they are consecutive in
    both the input permutation positions and values.

    Example:
        Original: reshape to (1, 32, 8, 32, 8, 96) -> Transpose(perm=(0,1,3,2,4,5))
        - Axes 0,1 have perm values 0,1 (consecutive) -> merge to 1*32=32
        - Axes 4,5 have perm values 4,5 (consecutive) -> merge to 8*96=768
        Merged: reshape to (32, 8, 32, 768) -> Transpose(perm=(0,2,1,3))  [4D <= 5D]

    Dimension constraint: intermediate transpose_shape must have <= 5 dimensions.

    This is useful for:
    - Reducing Transpose complexity for better hardware compatibility
    - Enabling pattern rewrites to simpler operations
    - Optimizing attention mechanism implementations

    Note: During model generation (get_onnx_model), the internal constants/attributes
    use the MERGED shapes and perm, while the schema attributes store the ORIGINAL values.
    """

    def check_skeleton_result(
        self, skeleton_match_result: SkeletonMatchResult
    ) -> "PatternMatchResult | None":
        """Validate match and enforce low-dim constraint (intermediate transpose <= 5D)."""
        result = Pattern.check_skeleton_result(self, skeleton_match_result)
        if result is None:
            return None
        transpose_shape = tuple(result.attributes["transpose_shape"])
        if len(transpose_shape) > 5:
            return None
        return result

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict[ONNXDomain, int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants and attributes with merged axes.

        This method computes the MERGED transpose shape and permutation for
        model generation, while the schema attributes contain the ORIGINAL values.

        During model generation (rewriting), -1 dimensions are resolved using
        the actual input tensor size from the `inputs` dict.

        Args:
            inputs: Dictionary mapping input names to numpy array values.
            attributes: Dictionary with 'transpose_shape', 'perm', 'output_shape'.
            is_constant_map: Dict mapping input_name -> is_constant.
            domain_versions: Dict mapping ONNXDomain to opset version.

        Returns:
            Tuple of (internal_constants, internal_attributes) with merged values.
        """
        transpose_shape = tuple(attributes["transpose_shape"])
        perm = tuple(attributes["perm"])
        output_shape = tuple(attributes["output_shape"])

        # Get total size from input tensor to resolve -1 dimensions
        total_size = int(np.prod(inputs["data"].shape))

        # Resolve -1 in transpose_shape and output_shape
        resolved_transpose_shape = _resolve_negative_dims(transpose_shape, total_size)
        resolved_output_shape = _resolve_negative_dims(output_shape, total_size)

        # Compute merged shape and perm with resolved shapes
        merged_shape, merged_perm = _compute_merged_transpose(resolved_transpose_shape, perm)

        internal_constants = [
            (0, 1, np.array(merged_shape, dtype=np.int64)),
            (2, 1, np.array(resolved_output_shape, dtype=np.int64)),
        ]

        internal_attributes: dict[tuple[int, str], Any] = {
            (1, "perm"): list(merged_perm),
        }

        return internal_constants, internal_attributes

    @property
    def pattern_id(self) -> str:
        """Return pattern ID matching the information rule configuration."""
        return f"SUBGRAPH/{type(self).__name__}"

    def get_schema(self) -> PatternSchema:
        """Return the schema definition for ReshapeTransposeReshapeLowDim pattern."""
        return _RESHAPE_TRANSPOSE_RESHAPE_SCHEMA


@register_pattern_input_generator
class ReshapeTransposeReshapeLowDimPatternInputGenerator(
    _ReshapeTransposeReshapeInputGeneratorBase
):
    """PatternInputGenerator for ReshapeTransposeReshapeLowDim pattern."""

    pattern = ReshapeTransposeReshapeLowDimPattern()
    registration_name = "ReshapeTransposeReshapeLowDimPattern"
