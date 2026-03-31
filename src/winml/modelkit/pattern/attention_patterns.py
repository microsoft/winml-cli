# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pattern matching for expanded attention subgraphs and Transpose+Attention equivalents.

This module provides pattern definitions for:
1. ExpandedAttentionPattern: The expanded subgraph pattern representing scaled dot product
   attention with attention mask (7 nodes: 1 Transpose, 2 Mul, 2 MatMul, 1 Add, 1 Softmax)
2. TransposeAttentionPattern: A pattern using Transpose + ai.onnx Attention operator
   to express the same computation (2 nodes: 1 Transpose + 1 Attention)

The ExpandedAttentionPattern models the expanded subgraph with Q, V inputs in BHSD format
and K input in BSHD format. The actual topology is:

    K(BSHD)                          V(BHSD)
       |                                |
    Transpose[0,2,3,1]                  |
       | (BHDS)                         |
       Mul(sqrt_scale)                  |
       |                                |
    Q(BHSD)                             |
       |                                |
       Mul(sqrt_scale)                  |
       |                                |
       ------MatMul------               |
              |                         |
     attn_mask---Add                    |
              |                         |
           Softmax                      |
              |                         |
              --------MatMul-------------
                       |
                       Y(BHSD)

Variants supported:
- Multi-headed Attention (MHA): q_num_heads = kv_num_heads
- Group-query Attention (GQA): q_num_heads > kv_num_heads, q_num_heads % kv_num_heads == 0
- Multi-query Attention (MQA): q_num_heads > kv_num_heads, kv_num_heads = 1
"""

from typing import Any

import numpy as np
from onnx.defs import OpSchema

from winml.modelkit.pattern.base import (
    Pattern,
    PatternInputGenerator,
    PatternMatchResult,
    PatternSchema,
    Skeleton,
    SkeletonMatchResult,
    register_pattern_input_generator,
)
from winml.modelkit.pattern.op_input_gen import InputShapeConstraint

from ..onnx import ONNXDomain


# Type constraints for Attention operator (from opset 24 spec)
_T_TYPES = [
    "tensor(bfloat16)",
    "tensor(double)",
    "tensor(float)",
    "tensor(float16)",
]

_U_TYPES = [
    "tensor(bfloat16)",
    "tensor(bool)",
    "tensor(double)",
    "tensor(float)",
    "tensor(float16)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(int8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(uint8)",
]


# Schema for patterns with Q, V in BHSD format and K in BSHD format
_TRANSPOSE_ATTENTION_SCHEMA = PatternSchema(
    name="TransposeAttentionPattern",
    doc=(
        "Attention pattern with Q, V in BHSD format and K in BSHD format.\n\n"
        "Computes attention using the formula:\n"
        "  Y = Softmax(Q * K^T * scale + attn_mask) * V\n\n"
        "Q and V are in BHSD format (batch, heads, seq, head_size).\n"
        "K is in BSHD format (batch, seq, heads, head_size).\n"
        "Output Y is in BHSD format.\n"
        "This schema is shared by both ExpandedAttentionPattern and TransposeAttentionPattern."
    ),
    type_constraints=[
        OpSchema.TypeConstraintParam(
            type_param_str="T",
            allowed_type_strs=_T_TYPES,
            description="Constrain Q, K, V input types and output to float tensors.",
        ),
        OpSchema.TypeConstraintParam(
            type_param_str="U",
            allowed_type_strs=_U_TYPES,
            description="Constrain attn_mask types to boolean or numeric tensors.",
        ),
    ],
    inputs=[
        OpSchema.FormalParameter(
            name="Q",
            type_str="T",
            description=(
                "Query tensor. 4D tensor with shape "
                "(batch_size, q_num_heads, q_sequence_length, head_size) in BHSD format."
            ),
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        ),
        OpSchema.FormalParameter(
            name="K",
            type_str="T",
            description=(
                "Key tensor. 4D tensor with shape "
                "(batch_size, kv_sequence_length, kv_num_heads, head_size) in BSHD format."
            ),
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        ),
        OpSchema.FormalParameter(
            name="V",
            type_str="T",
            description=(
                "Value tensor. 4D tensor with shape "
                "(batch_size, kv_num_heads, kv_sequence_length, v_head_size) in BHSD format."
            ),
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        ),
        OpSchema.FormalParameter(
            name="attn_mask",
            type_str="U",
            description=(
                "Attention mask. Shape must be broadcastable to "
                "(batch_size, q_num_heads, q_sequence_length, kv_sequence_length). "
                "Float mask added to attention score before softmax."
            ),
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Unknown,
        ),
    ],
    outputs=[
        OpSchema.FormalParameter(
            name="Y",
            type_str="T",
            description=(
                "Output tensor. 4D tensor with shape "
                "(batch_size, q_num_heads, q_sequence_length, v_head_size) in BHSD format."
            ),
            param_option=OpSchema.FormalParameterOption.Single,
            is_homogeneous=True,
            min_arity=1,
            differentiation_category=OpSchema.DifferentiationCategory.Differentiable,
        )
    ],
    attributes={
        "scale": OpSchema.Attribute(
            name="scale",
            description=(
                "Scale factor applied to Q and K before computing attention scores. "
                "The actual multiplication uses sqrt(scale) on both Q and K. "
                "Inferred from the matched pattern's Mul constants."
            ),
            type=OpSchema.AttrType.FLOAT,
            required=True,
        ),
    },
)


class ExpandedAttentionPattern(Pattern):
    """Pattern definition for expanded attention subgraph (with mask).

    This pattern represents the expanded form of scaled dot product attention with mask.
    Q and V are in BHSD format (batch, heads, seq, head_size).
    K is in BSHD format (batch, seq, heads, head_size).
    Output Y is in BHSD format.

    Topology (7 nodes):
        K(BSHD)                          V(BHSD)
           |                                |
        Transpose[0,2,3,1]                  |
           | (BHDS)                         |
           Mul(sqrt_scale)                  |
           |                                |
        Q(BHSD)                             |
           |                                |
           Mul(sqrt_scale)                  |
           |                                |
           ------MatMul------               |
                  |                         |
         attn_mask---Add                    |
                  |                         |
               Softmax                      |
                  |                         |
                  --------MatMul-------------
                           |
                           Y(BHSD)

    Node indices:
    - Node 0 (Transpose): K from BSHD to BHDS with perm=[0, 2, 3, 1]
    - Node 1 (Mul): K * sqrt(scale)
    - Node 2 (Mul): Q * sqrt(scale)
    - Node 3 (MatMul): Q @ K^T (attention scores)
    - Node 4 (Add): scores + attn_mask
    - Node 5 (Softmax): softmax(scores + mask)
    - Node 6 (MatMul): softmax @ V

    This pattern includes attention mask addition.
    """

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for expanded attention pattern with mask.

        Returns:
            Skeleton defining the expanded attention computation graph topology.
        """
        # Expanded attention pattern (with mask) - 7 nodes:
        # K(BSHD) -> Transpose[0,2,3,1] -> Mul(sqrt_scale) ->
        # Q(BHSD) -> Mul(sqrt_scale) -> MatMul -> Add(mask) -> Softmax ->
        # V(BHSD) ------------------------------------------------> MatMul -> Y(BHSD)
        # attn_mask -------------------------------------------->

        # Node indices:
        # 0=Transpose(K), 1=Mul(K), 2=Mul(Q),
        # 3=MatMul(QK), 4=Add, 5=Softmax, 6=MatMul(V)
        node_op_types = [
            "Transpose",  # 0: K transpose BSHD->BHDS
            "Mul",  # 1: K * sqrt_scale
            "Mul",  # 2: Q * sqrt_scale
            "MatMul",  # 3: Q @ K^T
            "Add",  # 4: + attn_mask
            "Softmax",  # 5: softmax
            "MatMul",  # 6: @ V
        ]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        # Edges: (src, src_slot, dst, dst_slot)
        # -1, -2, -3, -4 represent the inputs to the subgraph (Q, K, V, attn_mask)
        edges = [
            (-2, 0, 0, 0),  # K -> Transpose[0] (node 0)
            (0, 0, 1, 0),  # Transpose(K) -> Mul[0] (node 1)
            (-1, 0, 2, 0),  # Q -> Mul[0] (node 2)
            (2, 0, 3, 0),  # Mul(Q) -> MatMul[0] (node 3)
            (1, 0, 3, 1),  # Mul(K) -> MatMul[1] (node 3)
            (3, 0, 4, 0),  # MatMul -> Add[0] (node 4)
            (-4, 0, 4, 1),  # attn_mask -> Add[1] (node 4)
            (4, 0, 5, 0),  # Add -> Softmax[0] (node 5)
            (5, 0, 6, 0),  # Softmax -> MatMul[0] (node 6)
            (-3, 0, 6, 1),  # V -> MatMul[1] (node 6)
        ]

        # Exit node that produces the final output
        exit_nodes = [6]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=exit_nodes,
            n_inputs=4,  # Q, K, V, attn_mask
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict["ONNXDomain", int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants and attributes for expanded attention with mask.

        The expanded attention pattern has:
        - sqrt(scale) constant for Q and K scaling (nodes 1 and 2, slot 1)
        - Transpose perm attribute for K

        Args:
            inputs: Dictionary mapping input names to numpy array values.
            attributes: Dictionary of attribute values for the pattern (scale is required).
            is_constant_map: Dict mapping input_name -> is_constant (bool).

        Returns:
            Tuple of (internal_constants, internal_attributes).
        """
        dtype = inputs["Q"].dtype if "Q" in inputs else np.float32
        # sqrt_scale must be a 1D array with shape (1,) to match typical model constants
        sqrt_scale = np.array([np.sqrt(attributes["scale"])], dtype=dtype)

        internal_constants = [
            (1, 1, sqrt_scale),  # Node 1 (Mul K), slot 1: sqrt(scale)
            (2, 1, sqrt_scale),  # Node 2 (Mul Q), slot 1: sqrt(scale)
        ]

        internal_attributes: dict[tuple[int, str], Any] = {
            (0, "perm"): [0, 2, 3, 1],  # Node 0 (Transpose K): BSHD->BHDS
            (5, "axis"): -1,  # Node 5 (Softmax): axis=-1
        }

        return internal_constants, internal_attributes

    def check_skeleton_result(
        self, skelton_match_result: SkeletonMatchResult
    ) -> PatternMatchResult | None:
        """Check if skeleton match result satisfies expanded attention constraints.

        Validates:
        - sqrt(scale) constants must be equal for Q and K (validated here first)
        - Transpose perms must match expected values (via base class)
        - Scale is inferred and stored in attributes

        Args:
            skelton_match_result: The skeleton match result to validate.

        Returns:
            PatternMatchResult if validation passes, None otherwise.
        """
        matcher = skelton_match_result.matcher
        matched_nodes = skelton_match_result.matched_nodes

        # Validate that Q and K scaling constants are equal
        mul_k_node = matched_nodes[1]
        mul_q_node = matched_nodes[2]

        if len(mul_k_node.input) <= 1 or len(mul_q_node.input) <= 1:
            return None

        scale_k = matcher.tensor_values.get(mul_k_node.input[1])
        scale_q = matcher.tensor_values.get(mul_q_node.input[1])

        # Both scales must be constants and equal
        if scale_k is None or scale_q is None:
            return None
        if not np.allclose(scale_k, scale_q):
            return None

        # Base class validates attribute constraints (internal_constants is empty)
        return super().check_skeleton_result(skelton_match_result)

    def _infer_schema_attributes(self, skelton_match_result: SkeletonMatchResult) -> dict[str, Any]:
        """Infer schema-level attributes from matched expanded attention nodes.

        Extracts scale from the sqrt(scale) constants.

        Args:
            skelton_match_result: The skeleton match result containing matched nodes.

        Returns:
            Dictionary with scale attribute if determinable.
        """
        attributes: dict[str, Any] = {}
        matcher = skelton_match_result.matcher
        matched_nodes = skelton_match_result.matched_nodes

        # Get scale constant from Mul(Q) node (node 2)
        mul_q_node = matched_nodes[2]
        if len(mul_q_node.input) > 1:
            scale_q_name = mul_q_node.input[1]
            sqrt_scale = matcher.tensor_values.get(scale_q_name)
            if sqrt_scale is not None:
                # scale = sqrt_scale^2
                scale = float(sqrt_scale.flatten()[0] ** 2)
                attributes["scale"] = scale

        return attributes

    def get_schema(self) -> PatternSchema:
        """Return the schema definition for expanded attention with mask pattern.

        Returns:
            PatternSchema defining the pattern's input/output types.
        """
        return _TRANSPOSE_ATTENTION_SCHEMA


class TransposeAttentionPattern(Pattern):
    """Pattern definition for Transpose + ai.onnx Attention (with mask).

    This pattern represents attention computation using the ai.onnx Attention operator
    (opset 23+) with a Transpose node to convert K from BSHD to BHSD format.

    Q and V are in BHSD format (batch, heads, seq, head_size).
    K is in BSHD format (batch, seq, heads, head_size).
    Output Y is in BHSD format.

    Topology (2 nodes):
        Q(BHSD) --------------------------->|
        K(BSHD) -> Transpose[0,2,1,3] ------>|-> Attention -> Y(BHSD)
        V(BHSD) --------------------------->|
        attn_mask ------------------------->|

    Node indices:
    - Node 0 (Transpose): K from BSHD to BHSD with perm=[0, 2, 1, 3]
    - Node 1 (Attention): ai.onnx Attention operator

    This pattern includes attention mask input.
    """

    def get_skeleton(self) -> Skeleton:
        """Return the skeleton structure for Transpose+Attention pattern.

        Returns:
            Skeleton defining the Transpose+Attention computation graph topology.
        """
        # Transpose + Attention pattern - 2 nodes:
        # Q(BHSD) --------------------------->|
        # K(BSHD) -> Transpose[0,2,1,3] ------>|-> Attention -> Y(BHSD)
        # V(BHSD) --------------------------->|
        # attn_mask ------------------------->|

        node_op_types = [
            "Transpose",  # 0: K transpose BSHD->BHSD
            "Attention",  # 1: ai.onnx Attention
        ]
        node_domains = [ONNXDomain.AI_ONNX] * len(node_op_types)

        # Edges: (src, src_slot, dst, dst_slot)
        # -1, -2, -3, -4 represent the inputs to the subgraph (Q, K, V, attn_mask)
        edges = [
            (-2, 0, 0, 0),  # K -> Transpose[0] (node 0)
            (-1, 0, 1, 0),  # Q -> Attention[0] (node 1)
            (0, 0, 1, 1),  # Transpose(K) -> Attention[1] (node 1)
            (-3, 0, 1, 2),  # V -> Attention[2] (node 1)
            (-4, 0, 1, 3),  # attn_mask -> Attention[3] (node 1)
        ]

        # Exit node that produces the final output
        exit_nodes = [1]

        return Skeleton(
            node_op_types=node_op_types,
            node_domains=node_domains,
            edges=edges,
            exit_nodes=exit_nodes,
            n_inputs=4,  # Q, K, V, attn_mask
        )

    def get_internal_constants_and_attributes(
        self,
        inputs: dict[str, np.ndarray],
        attributes: dict[str, Any],
        is_constant_map: dict[str, bool],
        domain_versions: dict["ONNXDomain", int],
    ) -> tuple[list[tuple[int, int, np.ndarray]], dict[tuple[int, str], Any]]:
        """Return internal constants and attributes for Transpose+Attention pattern.

        The pattern has no internal constants.
        Transpose perm attribute is constrained, and Attention scale attribute is passed through.

        Args:
            inputs: Dictionary mapping input names to numpy array values.
            attributes: Dictionary of attribute values for the pattern.
            is_constant_map: Dict mapping input_name -> is_constant (bool).

        Returns:
            Tuple of (internal_constants, internal_attributes).
        """
        internal_constants: list[tuple[int, int, np.ndarray]] = []

        # Transpose perm attribute: [0, 2, 1, 3] for BSHD -> BHSD conversion
        internal_attributes: dict[tuple[int, str], Any] = {
            (0, "perm"): [0, 2, 1, 3],  # Node 0 (Transpose K): BSHD->BHSD
        }

        # Forward scale attribute to the Attention node (node 1)
        if "scale" in attributes and attributes["scale"] is not None:
            internal_attributes[(1, "scale")] = attributes["scale"]

        return internal_constants, internal_attributes

    def _infer_schema_attributes(self, skelton_match_result: SkeletonMatchResult) -> dict[str, Any]:
        """Infer schema-level attributes from matched Transpose+Attention nodes.

        Extracts scale attribute from the matched Attention node.

        Args:
            skelton_match_result: The skeleton match result containing matched nodes.

        Returns:
            Dictionary with scale attribute if present.
        """
        attributes: dict[str, Any] = {}
        matched_nodes = skelton_match_result.matched_nodes

        # Get scale from Attention node (node 1)
        attention_node = matched_nodes[1]
        for attr in attention_node.attribute:
            if attr.name == "scale":
                attributes["scale"] = attr.f

        return attributes

    def get_schema(self) -> PatternSchema:
        """Return the schema definition for Transpose+Attention pattern.

        Returns:
            PatternSchema defining the pattern's input/output types.
        """
        return _TRANSPOSE_ATTENTION_SCHEMA


@register_pattern_input_generator
class ExpandedAttentionPatternInputGenerator(PatternInputGenerator):
    """Input generator for ExpandedAttentionPattern.

    Generates test inputs for the expanded attention subgraph pattern
    with attention mask. Q and V are in BHSD format (batch, heads, seq, head_size).
    K is in BSHD format (batch, seq, heads, head_size).
    """

    pattern = ExpandedAttentionPattern()
    registration_name = "ExpandedAttentionPattern"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Returns finite attribute sets for expanded attention with mask pattern.

        The expanded attention pattern has scale as a required attribute,
        but scale is included per-combination in get_input_and_infinite_attribute_combinations
        since it depends on head_size.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputShapeConstraint]]:
        """Returns input combinations for expanded attention with mask pattern testing.

        Provides various 4D input shapes for Q, K, V, and attn_mask tensors.
        Q and V are in BHSD format: (batch, heads, seq, head_size).
        K is in BSHD format: (batch, seq, heads, head_size).
        Scale is included as scale = 1.0 / head_size.
        """
        return [
            # Small attention: batch=1, q_seq=4, kv_seq=6, heads=2, head_size=8
            {
                "Q": InputShapeConstraint((1, 2, 4, 8)),  # BHSD
                "K": InputShapeConstraint((1, 6, 2, 8)),  # BSHD
                "V": InputShapeConstraint((1, 2, 6, 8)),  # BHSD
                "attn_mask": InputShapeConstraint((1, 2, 4, 6)),
                "scale": 1.0 / 8,  # 1.0 / head_size
            },
            # Medium attention: batch=2, seq=8, heads=4, head_size=16
            {
                "Q": InputShapeConstraint((2, 4, 8, 16)),  # BHSD
                "K": InputShapeConstraint((2, 8, 4, 16)),  # BSHD
                "V": InputShapeConstraint((2, 4, 8, 16)),  # BHSD
                "attn_mask": InputShapeConstraint((2, 4, 8, 8)),
                "scale": 1.0 / 16,  # 1.0 / head_size
            },
            # BERT-like: batch=1, seq=512, heads=2, head_size=64
            {
                "Q": InputShapeConstraint((1, 2, 512, 64)),  # BHSD
                "K": InputShapeConstraint((1, 512, 2, 64)),  # BSHD
                "V": InputShapeConstraint((1, 2, 512, 64)),  # BHSD
                "attn_mask": InputShapeConstraint((1, 2, 512, 512)),
                "scale": 1.0 / 64,  # 1.0 / head_size
            },
        ]


@register_pattern_input_generator
class TransposeAttentionPatternInputGenerator(PatternInputGenerator):
    """Input generator for TransposeAttentionPattern.

    Generates test inputs for the Transpose+Attention pattern
    with attention mask. Q and V are in BHSD format (batch, heads, seq, head_size).
    K is in BSHD format (batch, seq, heads, head_size).
    """

    pattern = TransposeAttentionPattern()
    registration_name = "TransposeAttentionPattern"

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Returns finite attribute sets for Transpose+Attention pattern.

        The pattern has scale as an optional attribute.
        """
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, InputShapeConstraint]]:
        """Returns input combinations for Transpose+Attention pattern testing.

        Provides various 4D input shapes for Q, K, V, and attn_mask tensors.
        Q and V are in BHSD format: (batch, heads, seq, head_size).
        K is in BSHD format: (batch, seq, heads, head_size).
        """
        return [
            # Small attention: batch=1, q_seq=4, kv_seq=6, heads=2, head_size=8
            {
                "Q": InputShapeConstraint((1, 2, 4, 8)),  # BHSD
                "K": InputShapeConstraint((1, 6, 2, 8)),  # BSHD
                "V": InputShapeConstraint((1, 2, 6, 8)),  # BHSD
                "attn_mask": InputShapeConstraint((1, 2, 4, 6)),
            },
            # Medium attention: batch=2, seq=8, heads=4, head_size=16
            {
                "Q": InputShapeConstraint((2, 4, 8, 16)),  # BHSD
                "K": InputShapeConstraint((2, 8, 4, 16)),  # BSHD
                "V": InputShapeConstraint((2, 4, 8, 16)),  # BHSD
                "attn_mask": InputShapeConstraint((2, 4, 8, 8)),
            },
        ]
