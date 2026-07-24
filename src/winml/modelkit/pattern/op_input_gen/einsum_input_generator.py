# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Input generator for the Einsum operator."""

from typing import Any

from .op_input_gen import (
    InputShapeConstraint,
    OpInputGenerator,
    VariadicInputConstraint,
    register_runtime_checker_op,
)


@register_runtime_checker_op
class EinsumInputGenerator(OpInputGenerator):
    """Input generator for Einsum operator.

    Einsum signature:
    - Inputs: *Inputs (variadic) - List of input tensors
    - Attributes: equation (string, required) - The einsum equation string
    - Output: Output - Result tensor

    Test coverage strategy:
    - Common 2-input equations (matrix multiply, batched matmul, dot product,
      outer product, element-wise multiply with reduction)
    - Single-input equations (transpose, trace, diagonal)
    - Various tensor ranks (1D through 4D)
    """

    op_name = "Einsum"

    def get_finite_attribute_sets(self) -> dict[str, list[Any]]:
        """Return empty dict; equation is specified per combination."""
        return {}

    def get_input_and_infinite_attribute_combinations(
        self,
    ) -> list[dict[str, object]]:
        """Return input combinations for Einsum.

        Each combination specifies the equation and matching input shapes.
        Covers common real-world einsum patterns.
        """
        combinations: list[dict[str, object]] = [
            # === Single-input equations ===
            # Transpose 2D: ij->ji
            {
                "Inputs": VariadicInputConstraint([InputShapeConstraint((3, 4))]),
                "equation": "ij->ji",
            },
            # Diagonal: ii->i
            {
                "Inputs": VariadicInputConstraint([InputShapeConstraint((4, 4))]),
                "equation": "ii->i",
            },
            # Sum all: ij->
            {
                "Inputs": VariadicInputConstraint([InputShapeConstraint((3, 4))]),
                "equation": "ij->",
            },
            # === Two-input equations ===
            # Matrix multiply: ij,jk->ik
            {
                "Inputs": VariadicInputConstraint(
                    [InputShapeConstraint((3, 4)), InputShapeConstraint((4, 5))]
                ),
                "equation": "ij,jk->ik",
            },
            # Dot product: i,i->
            {
                "Inputs": VariadicInputConstraint(
                    [InputShapeConstraint((6,)), InputShapeConstraint((6,))]
                ),
                "equation": "i,i->",
            },
            # Outer product: i,j->ij
            {
                "Inputs": VariadicInputConstraint(
                    [InputShapeConstraint((3,)), InputShapeConstraint((4,))]
                ),
                "equation": "i,j->ij",
            },
            # Element-wise multiply: ij,ij->ij
            {
                "Inputs": VariadicInputConstraint(
                    [InputShapeConstraint((3, 4)), InputShapeConstraint((3, 4))]
                ),
                "equation": "ij,ij->ij",
            },
            # Batched matrix multiply: bij,bjk->bik
            {
                "Inputs": VariadicInputConstraint(
                    [InputShapeConstraint((2, 3, 4)), InputShapeConstraint((2, 4, 5))]
                ),
                "equation": "bij,bjk->bik",
            },
            # Batched dot with ellipsis: ...ij,...jk->...ik
            {
                "Inputs": VariadicInputConstraint(
                    [InputShapeConstraint((2, 3, 4)), InputShapeConstraint((2, 4, 5))]
                ),
                "equation": "...ij,...jk->...ik",
            },
            # Inner product pattern from OWLv2: ...pd,...qd->...pq
            {
                "Inputs": VariadicInputConstraint(
                    [InputShapeConstraint((2, 3, 4)), InputShapeConstraint((2, 5, 4))]
                ),
                "equation": "...pd,...qd->...pq",
            },
            # 4D batched: abij,abjk->abik
            {
                "Inputs": VariadicInputConstraint(
                    [InputShapeConstraint((2, 3, 4, 5)), InputShapeConstraint((2, 3, 5, 6))]
                ),
                "equation": "abij,abjk->abik",
            },
            # Bilinear: ik,jk->ij (shared contraction dim)
            {
                "Inputs": VariadicInputConstraint(
                    [InputShapeConstraint((3, 4)), InputShapeConstraint((5, 4))]
                ),
                "equation": "ik,jk->ij",
            },
        ]

        return combinations

    def derive_properties(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Derive additional properties for Einsum operator testing.

        Args:
            properties: Base properties containing:
                - Inputs_shape: tuple of shapes for each input tensor
                - attr_equation: equation string

        Returns:
            Updated properties with Einsum-specific derived values:
                - num_inputs: number of input tensors
                - Inputs_dim: max rank among input tensors
                - has_ellipsis: whether the equation uses ellipsis notation
                - has_explicit_output: whether the equation has '->' output spec
                - is_full_reduction: whether output is scalar (empty after '->')
                - has_repeated_labels: whether any input has repeated subscripts
                - has_contraction: whether dimensions are contracted (summed out)
                - output_num_labels: number of explicit output labels (output rank)
                - inputs_share_all_labels: whether all inputs use identical label sets
        """
        item = properties.copy()

        inputs_shape = item.get("Inputs_shape")
        if inputs_shape is not None and len(inputs_shape) > 0:
            item["num_inputs"] = len(inputs_shape)
            # Max rank across all inputs
            item["Inputs_dim"] = max(
                (len(s) for s in inputs_shape if s is not None), default=0
            )
        else:
            item["num_inputs"] = 0
            item["Inputs_dim"] = 0

        # Derive semantic properties from the equation string
        equation = item.get("attr_equation")
        if equation is not None:
            item["has_ellipsis"] = "..." in equation
            item["has_explicit_output"] = "->" in equation

            # Parse output portion
            output_part = equation.split("->")[1] if "->" in equation else ""
            output_labels = set(output_part.replace("...", ""))
            item["is_full_reduction"] = (
                "->" in equation and len(output_labels) == 0
            )

            # Check for repeated labels in any single input term (e.g. "ii->i")
            inputs_part = equation.split("->")[0]
            input_terms = inputs_part.split(",")
            item["has_repeated_labels"] = any(
                len(t.replace("...", "")) != len(set(t.replace("...", "")))
                for t in input_terms
            )

            # Check for contraction (input labels absent from output)
            all_input_labels = set(
                "".join(t.replace("...", "") for t in input_terms)
            )
            item["has_contraction"] = len(all_input_labels - output_labels) > 0

            # Output rank (number of explicit output labels, excluding ellipsis)
            # Analogous to Reshape's shape_len — EPs may differ by output dimensionality
            item["output_num_labels"] = len(output_labels)

            # Whether all inputs share the same set of labels (broadcast/element-wise)
            # vs having disjoint/partially-overlapping labels (contraction/outer product)
            input_label_sets = [
                set(t.replace("...", "")) for t in input_terms
            ]
            if len(input_label_sets) >= 2:
                item["inputs_share_all_labels"] = all(
                    s == input_label_sets[0] for s in input_label_sets[1:]
                )
            else:
                item["inputs_share_all_labels"] = True

        return item

    def get_infinite_property_names(self) -> list[str]:
        """Returns names of infinite properties for Einsum operator.

        Input shapes are unbounded. The equation attribute is NOT included here
        so that each distinct equation string participates in exact rule matching.
        This is the conservative approach: rules will not generalize across
        equations, avoiding false positive EP support claims for untested
        equations with different contraction patterns, output permutations,
        or broadcasting semantics.
        """
        return ["Inputs_shape", "Inputs_value", "Inputs_is_constant"]
