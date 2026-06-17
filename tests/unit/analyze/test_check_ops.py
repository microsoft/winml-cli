# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for _compute_case_signature in check_ops."""

import json

import numpy as np

from winml.modelkit.analyze.utils import CheckResultWriter
from winml.modelkit.analyze.utils.avalizble_ep_device_ops.case_index_key_codec import (
    encode_file_name_to_4char_key,
)
from winml.modelkit.analyze.utils.op_utils import compute_case_signature
from winml.modelkit.pattern.op_input_gen import InputValueConstraint, normalize_constraint_dict


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

        assert compute_case_signature(compact_case, namespace="") == compute_case_signature(
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
        assert compute_case_signature(case, namespace="") == compute_case_signature(
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

        assert compute_case_signature(case_xy, namespace="") == compute_case_signature(
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

        assert compute_case_signature(case_a, namespace="") != compute_case_signature(
            case_b, namespace=""
        )

    def test_different_shapes_produce_different_signatures(self) -> None:
        """Same fill value but different shapes must produce different signatures."""
        arr_2x3 = np.ones((2, 3), dtype=np.float32)
        arr_3x2 = np.ones((3, 2), dtype=np.float32)

        case_a = self._case({"X": InputValueConstraint(arr_2x3).to_dict()})
        case_b = self._case({"X": InputValueConstraint(arr_3x2).to_dict()})

        assert compute_case_signature(case_a, namespace="") != compute_case_signature(
            case_b, namespace=""
        )

    # ------------------------------------------------------------------
    # Namespace / other fields
    # ------------------------------------------------------------------

    def test_namespace_is_included_in_signature(self) -> None:
        """Different namespaces produce different signatures for the same case."""
        case = self._case({"X": InputValueConstraint(np.ones((2,), dtype=np.float32)).to_dict()})
        assert compute_case_signature(case, namespace="file_a") != compute_case_signature(
            case, namespace="file_b"
        )

    def test_empty_namespace_omitted_from_signature(self) -> None:
        """An empty namespace string does not appear in the signature."""
        case = {"type_vars": {"T": "FLOAT"}}
        assert "ns:" not in compute_case_signature(case, namespace="")

    def test_empty_attrs_excluded_from_signature(self) -> None:
        """An empty attrs dict does not affect the signature."""
        base = {"type_vars": {"T": "FLOAT"}}
        with_empty_attrs = {**base, "attrs": {}}
        assert compute_case_signature(base, namespace="") == compute_case_signature(
            with_empty_attrs, namespace=""
        )


class TestConstraintRoundTrip:
    """Tests for reversible serialization/deserialization of value constraints."""

    def test_same_value_scalar_bool_roundtrip_preserves_ndarray(self) -> None:
        """Scalar bool arrays should remain 0-D ndarray after normalization."""
        original = np.asarray(True, dtype=np.bool_)
        compact = InputValueConstraint(original).to_dict()

        restored = normalize_constraint_dict(compact)["value"]

        assert isinstance(restored, np.ndarray)
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype
        assert bool(restored.item()) is True

    def test_same_value_dense_array_roundtrip_preserves_dtype_and_shape(self) -> None:
        """Non-scalar arrays should restore with identical dtype and shape."""
        original = np.full((2, 3), 7, dtype=np.int16)
        compact = InputValueConstraint(original).to_dict()

        restored = normalize_constraint_dict(compact)["value"]

        assert isinstance(restored, np.ndarray)
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype
        assert np.array_equal(restored, original)

    def test_scalar_bool_payload_preserves_ndarray(self) -> None:
        """Scalar arrays keep ndarray payload instead of collapsing to Python scalars."""
        scalar = np.asarray(True, dtype=np.bool_)
        compact = InputValueConstraint(scalar).to_dict()

        restored = normalize_constraint_dict(compact)["value"]

        assert isinstance(restored, np.ndarray)
        assert restored.shape == ()
        assert restored.dtype == np.bool_
        assert bool(restored.item()) is True

    def test_dtype_with_scalar_value_keeps_original_payload(self) -> None:
        """Non-compact value payloads keep their original scalar shape."""
        serialized = {
            "type": "value",
            "value": 1,
            "dtype": "int64",
        }

        restored = normalize_constraint_dict(serialized)["value"]

        assert isinstance(restored, int)
        assert restored == 1


class TestReuseExistingResultInputConstraints:
    """Tests that reuse_existing_result upgrades input_constraints to current format."""

    @staticmethod
    def _make_writer(existing_cases: list[dict], tmp_path):
        """Create a CheckResultWriter with pre-loaded existing_signatures from cases."""
        output_file = tmp_path / "Abs_QNNExecutionProvider_NPU_ai.onnx_opset13.json"
        output_file.write_text(json.dumps({"check_results": existing_cases}), encoding="utf-8")
        return CheckResultWriter(output_file, sys_info={}, delta_only=True)

    def test_reuse_upgrades_value_array_to_same_value(self, tmp_path) -> None:
        """When an existing case has the old value-array format, reuse_existing_result
        must replace input_constraints with the compact same_value form from the
        current generator output (the skipped case dict)."""
        arr = np.ones((2, 3), dtype=np.float32)
        compact = InputValueConstraint(arr).to_dict()
        expanded = {"type": "value", "value": arr.tolist(), "dtype": "float32"}

        existing_case = {
            "type_vars": {"T": "FLOAT"},
            "input_constraints": {"X": expanded},
            "check_result": {
                "compile": {"result": {"success": True}},
                "run": {"result": {"success": True}},
            },
        }

        with self._make_writer([existing_case], tmp_path) as writer:
            # The skipped case carries the up-to-date compact form
            skipped_case = {
                "type_vars": {"T": "FLOAT"},
                "input_constraints": {"X": compact},
                "_skipped": True,
            }
            reused = writer.reuse_existing_result(skipped_case)

        assert reused
        saved = writer.results[0]
        assert saved["input_constraints"]["X"] == compact
        assert "same_value" in saved["input_constraints"]["X"]
        assert "value" not in saved["input_constraints"]["X"]

    def test_reuse_preserves_existing_result_fields(self, tmp_path) -> None:
        """Upgrading input_constraints must not discard check_result or other fields."""
        arr = np.ones((2,), dtype=np.float32)
        compact = InputValueConstraint(arr).to_dict()
        expanded = {"type": "value", "value": arr.tolist(), "dtype": "float32"}

        compile_result = {"result": {"success": True}, "stdout": "ok", "stderr": ""}
        run_result = {"result": {"success": True}, "stdout": "ok", "stderr": ""}
        existing_case = {
            "type_vars": {"T": "FLOAT"},
            "input_constraints": {"X": expanded},
            "check_result": {"compile": compile_result, "run": run_result},
        }

        with self._make_writer([existing_case], tmp_path) as writer:
            skipped_case = {
                "type_vars": {"T": "FLOAT"},
                "input_constraints": {"X": compact},
                "_skipped": True,
            }
            writer.reuse_existing_result(skipped_case)

        saved = writer.results[0]
        assert saved["check_result"]["compile"] == compile_result
        assert saved["check_result"]["run"] == run_result

    def test_reuse_already_compact_stays_compact(self, tmp_path) -> None:
        """If the existing case already has same_value format, reuse leaves it intact."""
        arr = np.ones((2, 3), dtype=np.float32)
        compact = InputValueConstraint(arr).to_dict()

        existing_case = {
            "type_vars": {"T": "FLOAT"},
            "input_constraints": {"X": compact},
            "check_result": {
                "compile": {"result": {"success": True}},
                "run": {"result": {"success": True}},
            },
        }

        with self._make_writer([existing_case], tmp_path) as writer:
            skipped_case = {
                "type_vars": {"T": "FLOAT"},
                "input_constraints": {"X": compact},
                "_skipped": True,
            }
            reused = writer.reuse_existing_result(skipped_case)

        assert reused
        saved = writer.results[0]
        assert saved["input_constraints"]["X"] == compact
        assert "same_value" in saved["input_constraints"]["X"]

    def test_reuse_preserves_model_bytes_payload(self, tmp_path) -> None:
        """Reused cases must retain model_bytes_b64 from the current generator payload."""
        arr = np.ones((2,), dtype=np.float32)
        compact = InputValueConstraint(arr).to_dict()

        existing_case = {
            "type_vars": {"T": "FLOAT"},
            "input_constraints": {"X": compact},
            "check_result": {
                "compile": {"result": {"success": True}},
                "run": {"result": {"success": True}},
            },
        }

        with self._make_writer([existing_case], tmp_path) as writer:
            skipped_case = {
                "type_vars": {"T": "FLOAT"},
                "input_constraints": {"X": compact},
                "model_bytes_b64": "test_payload",
                "_skipped": True,
            }
            reused = writer.reuse_existing_result(skipped_case)

        assert reused
        saved = writer.results[0]
        assert saved["model_bytes_b64"] == "test_payload"

    def test_case_index_is_36_chars_and_ep_device_differs_by_first_char(self, tmp_path) -> None:
        """case_index should differ only in the first char across EP/device; last 35 stay equal."""
        case_template = {
            "type_vars": {"T": "FLOAT"},
            "input_constraints": {"X": {"type": "shape", "shape": [1], "min_max": None}},
            "check_result": {
                "compile": {"result": {"success": True}},
                "run": {"result": {"success": True}},
            },
        }

        qnn_output = tmp_path / "Abs_QNNExecutionProvider_NPU_ai.onnx_opset13.json"
        ov_output = tmp_path / "Abs_OpenVINOExecutionProvider_CPU_ai.onnx_opset13.json"

        with CheckResultWriter(qnn_output, sys_info={}) as writer_qnn:
            writer_qnn.append_result(dict(case_template))

        with CheckResultWriter(ov_output, sys_info={}) as writer_ov:
            writer_ov.append_result(dict(case_template))

        qnn_case = writer_qnn.results[0]
        ov_case = writer_ov.results[0]

        qnn_case_index = qnn_case["case_index"]
        ov_case_index = ov_case["case_index"]

        assert len(qnn_case_index) == 36
        assert len(ov_case_index) == 36
        assert qnn_case_index[:4] == encode_file_name_to_4char_key(qnn_output.stem)
        assert ov_case_index[:4] == encode_file_name_to_4char_key(ov_output.stem)
        assert qnn_case_index[0] != ov_case_index[0]
        assert qnn_case_index[1:] == ov_case_index[1:]
        assert "case_index_ignore_ep_device" not in qnn_case
        assert "case_index_ignore_ep_device" not in ov_case
