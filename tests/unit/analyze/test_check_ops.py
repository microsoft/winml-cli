# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for _compute_case_signature in check_ops."""

import numpy as np

from winml.modelkit.analyze.runtime_checker.check_ops import _compute_case_signature
from winml.modelkit.pattern.op_input_gen import InputValueConstraint


class TestComputeCaseSignature:
    """Tests for _compute_case_signature."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _case(input_constraints: dict) -> dict:
        return {"type_vars": {"T": "FLOAT"}, "input_constraints": input_constraints}

    # ------------------------------------------------------------------
    # Core: compact vs expanded representation
    # ------------------------------------------------------------------

    def test_all_same_value_array_compact_matches_expanded(self) -> None:
        """InputValueConstraint with all-same values serializes to compact form;
        signature must match the equivalent fully-expanded value list (backward compat).

        InputValueConstraint.to_dict() emits:
            {"type": "value", "same_value": 1.0, "same_value_shape": [2, 3], "dtype": "float32"}
        Old code stored the expanded form:
            {"type": "value", "value": [[1.0, ...], ...], "dtype": "float32"}
        Both must hash to the same signature.
        """
        arr = np.ones((2, 3), dtype=np.float32)

        compact_dict = InputValueConstraint(arr).to_dict()
        # Compact form has same_value / same_value_shape keys
        assert "same_value" in compact_dict and "same_value_shape" in compact_dict

        # Manually build what the old code would have stored (expanded form)
        expanded_dict = {
            "type": "value",
            "value": arr.tolist(),
            "dtype": str(arr.dtype),
        }

        compact_case = self._case({"X": compact_dict})
        expanded_case = self._case({"X": expanded_dict})

        assert _compute_case_signature(compact_case, namespace="") == _compute_case_signature(
            expanded_case, namespace=""
        )

    def test_mixed_value_array_uses_expanded_form_stable(self) -> None:
        """InputValueConstraint with non-uniform values always emits the full value list;
        signature is identical to computing it from the raw to_dict() output.
        """
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        full_dict = InputValueConstraint(arr).to_dict()
        assert "value" in full_dict and "same_value" not in full_dict

        case = self._case({"X": full_dict})
        assert _compute_case_signature(case, namespace="") == _compute_case_signature(
            case, namespace=""
        )

    def test_top_level_input_constraints_key_order_does_not_matter(self) -> None:
        """Multiple inputs: swapping their dict insertion order must not change the signature."""
        arr_x = np.zeros((2,), dtype=np.float32)
        arr_y = np.ones((2,), dtype=np.float32)

        case_xy = {
            "type_vars": {"T": "FLOAT"},
            "input_constraints": {
                "X": InputValueConstraint(arr_x).to_dict(),
                "Y": InputValueConstraint(arr_y).to_dict(),
            },
        }
        case_yx = {
            "type_vars": {"T": "FLOAT"},
            "input_constraints": {
                "Y": InputValueConstraint(arr_y).to_dict(),
                "X": InputValueConstraint(arr_x).to_dict(),
            },
        }

        assert _compute_case_signature(case_xy, namespace="") == _compute_case_signature(
            case_yx, namespace=""
        )

    # ------------------------------------------------------------------
    # Basic discriminability
    # ------------------------------------------------------------------

    def test_different_values_produce_different_signatures(self) -> None:
        """Two constraints with different underlying values must not collide."""
        arr_a = np.ones((2, 2), dtype=np.float32)
        arr_b = np.zeros((2, 2), dtype=np.float32)

        case_a = self._case({"X": InputValueConstraint(arr_a).to_dict()})
        case_b = self._case({"X": InputValueConstraint(arr_b).to_dict()})

        assert _compute_case_signature(case_a, namespace="") != _compute_case_signature(
            case_b, namespace=""
        )

    def test_different_shapes_produce_different_signatures(self) -> None:
        """Same fill value but different shapes must produce different signatures."""
        arr_2x3 = np.ones((2, 3), dtype=np.float32)
        arr_3x2 = np.ones((3, 2), dtype=np.float32)

        case_a = self._case({"X": InputValueConstraint(arr_2x3).to_dict()})
        case_b = self._case({"X": InputValueConstraint(arr_3x2).to_dict()})

        assert _compute_case_signature(case_a, namespace="") != _compute_case_signature(
            case_b, namespace=""
        )

    # ------------------------------------------------------------------
    # Namespace / other fields
    # ------------------------------------------------------------------

    def test_namespace_is_included_in_signature(self) -> None:
        """Different namespaces produce different signatures for the same case."""
        case = self._case({"X": InputValueConstraint(np.ones((2,), dtype=np.float32)).to_dict()})
        assert _compute_case_signature(case, namespace="file_a") != _compute_case_signature(
            case, namespace="file_b"
        )

    def test_empty_namespace_omitted_from_signature(self) -> None:
        """An empty namespace string does not appear in the signature."""
        case = {"type_vars": {"T": "FLOAT"}}
        assert "ns:" not in _compute_case_signature(case, namespace="")

    def test_empty_attrs_excluded_from_signature(self) -> None:
        """An empty attrs dict does not affect the signature."""
        base = {"type_vars": {"T": "FLOAT"}}
        with_empty_attrs = {**base, "attrs": {}}
        assert _compute_case_signature(base, namespace="") == _compute_case_signature(
            with_empty_attrs, namespace=""
        )
