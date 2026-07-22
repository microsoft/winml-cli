# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generator for ai.onnx Attention operator (opset 23+).

The generated combinations focus on the common 4D attention layout:
Q, K, V in BHSD format and optional attn_mask broadcastable to
(batch, heads, q_seq_len, kv_seq_len).
"""

from .op_input_gen import InputShapeConstraint, OpInputGenerator, register_runtime_checker_op


@register_runtime_checker_op
class AttentionInputGenerator(OpInputGenerator):
    """Input generator for Attention operator.

    Signature (opset 23):
    Attention(Q, K, V, attn_mask?, past_key?, past_value?, ...)

    This generator currently targets 4D Q/K/V combinations that do not require
    explicit q_num_heads/kv_num_heads attributes.
    """

    op_name = "Attention"
    # Keep optional input expansion disabled so we can control optional-mask coverage
    # explicitly and avoid invalid optional combinations.
    expand_optionals = False

    def get_finite_attribute_sets(self) -> dict[str, list]:
        """Attention attribute coverage is handled in input combinations."""
        return {}

    def get_input_and_infinite_attribute_combinations(self) -> list[dict[str, object]]:
        """Return representative 4D input combinations for Attention."""
        return [
            # No mask: Q/K/V only.
            {
                "Q": InputShapeConstraint((1, 2, 4, 8)),
                "K": InputShapeConstraint((1, 2, 6, 8)),
                "V": InputShapeConstraint((1, 2, 6, 8)),
            },
            # With mask (broadcast-exact shape for score tensor).
            {
                "Q": InputShapeConstraint((1, 2, 4, 8)),
                "K": InputShapeConstraint((1, 2, 6, 8)),
                "V": InputShapeConstraint((1, 2, 6, 8)),
                "attn_mask": InputShapeConstraint((1, 2, 4, 6)),
            },
            # Larger shape with explicit scale attribute.
            {
                "Q": InputShapeConstraint((2, 4, 8, 16)),
                "K": InputShapeConstraint((2, 4, 8, 16)),
                "V": InputShapeConstraint((2, 4, 8, 16)),
                "attn_mask": InputShapeConstraint((2, 4, 8, 8)),
                "scale": 1.0 / 16,
            },
        ]