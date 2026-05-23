# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for per-op rule processing helpers in result_processor."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from winml.modelkit.analyze.runtime_checker.result_processor import (
    PARQUET_STABLE_PANDAS_VERSION,
    _deduplicate_rule_rows,
    _encode_condition_columns_for_parquet,
    _parse_requested_domains,
    _write_parquet_with_stable_pandas_version,
)


class TestParseRequestedDomains:
    """Validate --domains parsing and normalization."""

    def test_parse_requested_domains_filters_unsupported(self):
        result = _parse_requested_domains("ai.onnx,foo,com.microsoft")
        assert result == ["ai.onnx", "com.microsoft"]

    def test_parse_requested_domains_uses_default_when_empty(self):
        result = _parse_requested_domains(" , ")
        assert result == ["ai.onnx", "com.microsoft"]


class TestDeduplicateRuleRows:
    """Validate deduplication and conflict detection for per-op rules."""

    def test_deduplicate_collapses_identical_rules(self):
        df = pd.DataFrame(
            [
                {
                    "T_Add": "FLOAT",
                    "input_dim": 4,
                    "compile_run_success": (True, False),
                    "compile_reason": None,
                    "run_reason": "run_not_supported",
                },
                {
                    "T_Add": "FLOAT",
                    "input_dim": 4,
                    "compile_run_success": (True, False),
                    "compile_reason": None,
                    "run_reason": "run_not_supported",
                },
            ]
        )

        dedup_df, conflict_df = _deduplicate_rule_rows(
            df,
            condition_cols=["T_Add", "input_dim"],
            output_cols=["compile_run_success", "compile_reason", "run_reason"],
        )

        assert conflict_df is None
        assert len(dedup_df) == 1
        assert int(dedup_df.iloc[0]["rule_row_count"]) == 2

    def test_deduplicate_reports_conflicts_for_same_conditions(self):
        df = pd.DataFrame(
            [
                {
                    "case_index": 11,
                    "T_Add": "FLOAT",
                    "input_dim": 4,
                    "compile_run_success": (True, False),
                    "compile_reason": None,
                    "run_reason": "run_not_supported",
                },
                {
                    "case_index": 12,
                    "T_Add": "FLOAT",
                    "input_dim": 4,
                    "compile_run_success": (True, True),
                    "compile_reason": None,
                    "run_reason": None,
                },
            ]
        )

        dedup_df, conflict_df = _deduplicate_rule_rows(
            df,
            condition_cols=["T_Add", "input_dim"],
            output_cols=["compile_run_success", "compile_reason", "run_reason"],
        )

        assert dedup_df.empty
        assert conflict_df is not None
        assert len(conflict_df) == 2
        assert list(conflict_df.columns[:2]) == ["groupid", "case_index"]
        assert set(conflict_df["case_index"].tolist()) == {11, 12}
        assert set(conflict_df["groupid"].tolist()) == {1}

    def test_deduplicate_ignores_reason_differences_when_compare_success_only(self):
        df = pd.DataFrame(
            [
                {
                    "case_index": "a",
                    "T_Add": "FLOAT",
                    "input_dim": 4,
                    "compile_run_success": (False, False),
                    "compile_reason": "compile_error_1",
                    "run_reason": "run_error_1",
                },
                {
                    "case_index": "b",
                    "T_Add": "FLOAT",
                    "input_dim": 4,
                    "compile_run_success": (False, False),
                    "compile_reason": "compile_error_2",
                    "run_reason": "run_error_2",
                },
            ]
        )

        dedup_df, conflict_df = _deduplicate_rule_rows(
            df,
            condition_cols=["T_Add", "input_dim"],
            output_cols=["compile_run_success", "compile_reason", "run_reason"],
            compare_output_cols=["compile_run_success"],
        )

        assert conflict_df is None
        assert len(dedup_df) == 1
        assert int(dedup_df.iloc[0]["rule_row_count"]) == 2
        assert dedup_df.iloc[0]["compile_reason"] == "compile_error_1"
        assert dedup_df.iloc[0]["run_reason"] == "run_error_1"


class TestParquetConditionEncoding:
    """Validate parquet-safe condition encoding for mixed object values."""

    def test_encode_condition_columns_for_parquet_makes_column_homogeneous(self):
        df = pd.DataFrame(
            [
                {
                    "attr_value": (("dataType", 10), ("int32Data", (16384,))),
                    "compile_run_success": (False, True),
                    "compile_reason": "compile_a",
                    "run_reason": None,
                },
                {
                    "attr_value": (("dataType", 2), ("rawData", b"\x00\x01")),
                    "compile_run_success": (False, True),
                    "compile_reason": "compile_b",
                    "run_reason": None,
                },
            ]
        )

        encoded_df = _encode_condition_columns_for_parquet(df, ["attr_value"])

        assert all(isinstance(v, str) for v in encoded_df["attr_value"].tolist())
        # Output columns remain untouched.
        assert encoded_df.iloc[0]["compile_run_success"] == (False, True)


class TestParquetMetadataStability:
    """Validate parquet metadata is pinned for deterministic file diffs."""

    def test_write_parquet_pins_pandas_version(self, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")

        parquet_path = tmp_path / "stable_metadata.parquet"
        df = pd.DataFrame([{"compile_run_success": (True, False), "attr_alpha": "A"}])

        _write_parquet_with_stable_pandas_version(df, parquet_path, compression="snappy")

        metadata = pq.read_metadata(parquet_path).metadata or {}
        pandas_meta_raw = metadata.get(b"pandas")
        assert pandas_meta_raw is not None

        pandas_meta = json.loads(pandas_meta_raw.decode("utf-8"))
        assert pandas_meta["pandas_version"] == PARQUET_STABLE_PANDAS_VERSION

    def test_write_parquet_preserves_metadata_bytes_when_already_pinned(self, tmp_path):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")

        baseline_path = tmp_path / "baseline.parquet"
        rewritten_path = tmp_path / "rewritten.parquet"

        df = pd.DataFrame([{"compile_run_success": (True, False), "attr_alpha": "A"}])

        baseline_table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(baseline_table, baseline_path, compression="snappy")
        baseline_meta = pq.read_metadata(baseline_path).metadata or {}
        baseline_pandas_meta_raw = baseline_meta.get(b"pandas")
        assert baseline_pandas_meta_raw is not None

        _write_parquet_with_stable_pandas_version(df, rewritten_path, compression="snappy")
        rewritten_meta = pq.read_metadata(rewritten_path).metadata or {}
        rewritten_pandas_meta_raw = rewritten_meta.get(b"pandas")
        assert rewritten_pandas_meta_raw is not None

        assert rewritten_pandas_meta_raw == baseline_pandas_meta_raw


