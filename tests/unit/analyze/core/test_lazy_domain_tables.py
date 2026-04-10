# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Unit tests for LazyDomainTables and _LazyNegRules lazy-loading wrappers.

LazyDomainTables tests verify:
- No file I/O before first access (_loaded flag)
- Lazy loading from zip on first access
- Caching on subsequent access
- Raw data cleanup after per-operator loading
- __contains__ and get() methods
- KeyError for non-existent operators
- Graceful handling of missing zip / missing file-in-zip

_LazyNegRules tests verify:
- No file I/O before first access
- Operator-only vs pattern-only filtering
- Error key population when set_error_on_missing=True
- Silent empty dict when set_error_on_missing=False
- __contains__, get(), items() all trigger load
- Value sanitization via _sanitize_domain_neg_rules
"""

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from winml.modelkit.analyze.core.runtime_checker_query import (
    EG_RULE_DEBUG_DETAILS_KEY,
    EG_RULE_ERROR_KEY,
    LazyDomainTables,
    _LazyNegRules,
)


RAW_DATA = {
    "Conv": {
        "attr_kernel_shape": {0: (3, 3), 1: (5, 5)},
        "X_shape": {0: (1, 3, 224, 224), 1: (1, 64, 112, 112)},
        "compile_run_success": {0: (True, True), 1: (True, False)},
    },
    "Add": {
        "A_shape": {0: (1, 3, 224, 224), 1: (1, 64, 112, 112)},
        "B_shape": {0: (1, 3, 224, 224), 1: (1, 64, 112, 112)},
        "compile_run_success": {0: (True, True), 1: (True, True)},
    },
}

FILE_NAME = "tables.json"
COLUMN_FILE_NAME = "table_columns.json"
RAW_COLUMNS = {
    op_name: [col_name for col_name in table_payload if col_name != "compile_run_success"]
    for op_name, table_payload in RAW_DATA.items()
}


@pytest.fixture
def zip_path(tmp_path: Path) -> Path:
    """Create a zip archive containing the sample table JSON."""
    zp = tmp_path / "rules.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr(FILE_NAME, json.dumps(RAW_DATA))
        zf.writestr(COLUMN_FILE_NAME, json.dumps(RAW_COLUMNS))
    return zp


@pytest.fixture
def tables(zip_path: Path) -> LazyDomainTables:
    return LazyDomainTables(zip_path, FILE_NAME, columns_file_name=COLUMN_FILE_NAME)


class TestLazyDomainTablesCore:
    """Test core lazy loading functionality."""

    def test_no_io_before_first_access(self, zip_path: Path):
        """LazyDomainTables must not read the zip during construction."""
        tables = LazyDomainTables(zip_path, FILE_NAME)
        assert tables._loaded is False
        assert tables._raw_data == {}
        assert tables._loaded_tables == {}

    def test_lazy_loading_on_first_access(self, tables: LazyDomainTables):
        """First __getitem__ triggers zip read and returns a DataFrame."""
        conv_df = tables["Conv"]

        assert tables._loaded is True
        assert isinstance(conv_df, pd.DataFrame)
        assert len(conv_df) == 2
        assert "Conv" in tables._loaded_tables
        assert "Conv" not in tables._raw_data  # cleaned up after DataFrame built

    def test_caching_returns_same_object(self, tables: LazyDomainTables):
        """Subsequent access returns the cached DataFrame."""
        first = tables["Conv"]
        second = tables["Conv"]
        assert first is second

    def test_independent_per_operator_loading(self, tables: LazyDomainTables):
        """Each operator is loaded into _loaded_tables and removed from _raw_data individually."""
        _ = tables["Conv"]
        assert "Conv" in tables._loaded_tables
        assert "Conv" not in tables._raw_data
        assert "Add" in tables._raw_data
        assert "Add" not in tables._loaded_tables

        _ = tables["Add"]
        assert "Add" in tables._loaded_tables
        assert tables._raw_data == {}  # all cleaned up


class TestLazyDomainTablesMethods:
    """Test __contains__ and get() methods."""

    def test_contains_triggers_load(self, tables: LazyDomainTables):
        """__contains__ triggers loading and returns correct results."""
        assert tables._loaded is False
        assert "Conv" in tables
        assert tables._loaded is True
        assert "Add" in tables
        assert "Relu" not in tables

    def test_contains_after_partial_load(self, tables: LazyDomainTables):
        """__contains__ works correctly for both loaded and raw-only operators."""
        _ = tables["Conv"]
        assert "Conv" in tables  # in _loaded_tables
        assert "Add" in tables  # still in _raw_data
        assert "Relu" not in tables

    def test_get_existing_operator(self, tables: LazyDomainTables):
        """get() returns a DataFrame for existing operators."""
        df = tables.get("Conv")
        assert isinstance(df, pd.DataFrame)

    def test_get_missing_operator_returns_none(self, tables: LazyDomainTables):
        """get() returns None for non-existent operators."""
        assert tables.get("Relu") is None

    def test_get_custom_default(self, tables: LazyDomainTables):
        """get() returns the custom default for non-existent operators."""
        sentinel = pd.DataFrame()
        assert tables.get("Relu", default=sentinel) is sentinel

    def test_get_caches_result(self, tables: LazyDomainTables):
        """get() returns the cached DataFrame on subsequent calls."""
        first = tables.get("Conv")
        second = tables.get("Conv")
        assert first is second

    def test_get_columns_from_metadata(self, tables: LazyDomainTables):
        """get_columns() returns metadata-backed column order when available."""
        assert tables.get_columns("Conv") == RAW_COLUMNS["Conv"]

    def test_get_columns_after_table_load(self, tables: LazyDomainTables):
        """get_columns() returns DataFrame columns for loaded operators."""
        _ = tables["Conv"]
        assert tables.get_columns("Conv") == tables["Conv"].columns.to_list()

    def test_get_columns_missing_operator(self, tables: LazyDomainTables):
        """get_columns() returns None for unknown operators."""
        assert tables.get_columns("Relu") is None

    def test_get_columns_fallback_without_metadata(self, zip_path: Path):
        """get_columns() falls back to raw table payload if metadata file is absent."""
        tables_without_columns = LazyDomainTables(zip_path, FILE_NAME, "missing_columns.json")
        assert tables_without_columns.get_columns("Add") == [
            "A_shape",
            "B_shape",
        ]


class TestLazyDomainTablesErrors:
    """Test error handling."""

    def test_keyerror_for_non_existent_operator(self, tables: LazyDomainTables):
        """__getitem__ raises KeyError for operators not in the table."""
        with pytest.raises(KeyError, match="Operator 'Relu' not found in tables"):
            _ = tables["Relu"]

    def test_missing_zip_returns_empty(self, tmp_path: Path):
        """When the zip file does not exist, all lookups return empty/False/None."""
        tables = LazyDomainTables(tmp_path / "nonexistent.zip", FILE_NAME)
        assert "Conv" not in tables
        assert tables.get("Conv") is None
        assert tables._loaded is True  # attempted load, found nothing

    def test_file_not_in_zip(self, zip_path: Path):
        """When the named file is absent from the zip, behaves like empty tables."""
        tables = LazyDomainTables(zip_path, "missing_file.json")
        assert "Conv" not in tables
        assert tables.get("Conv") is None
        assert tables._loaded is True


class TestLazyDomainTablesDataFrame:
    """Test DataFrame properties and sanitization."""

    def test_dataframe_columns(self, tables: LazyDomainTables):
        """Loaded DataFrames expose the expected columns."""
        conv_df = tables["Conv"]
        add_df = tables["Add"]

        assert set(conv_df.columns) == {"attr_kernel_shape", "X_shape", "compile_run_success"}
        assert set(add_df.columns) == {"A_shape", "B_shape", "compile_run_success"}

    def test_dataframe_row_counts(self, tables: LazyDomainTables):
        """Loaded DataFrames have the correct number of rows."""
        assert len(tables["Conv"]) == 2
        assert len(tables["Add"]) == 2

    def test_dataframe_sanitization(self, tmp_path: Path):
        """_sanitize_df (make_hashable) is applied during loading."""
        raw = {
            "TestOp": {
                "attr_value": {0: 1.5, 1: 2.5},
                "compile_run_success": {0: (True, True), 1: (True, False)},
            }
        }
        zp = tmp_path / "san.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr(FILE_NAME, json.dumps(raw))

        tables = LazyDomainTables(zp, FILE_NAME)
        df = tables["TestOp"]

        assert isinstance(df, pd.DataFrame)
        assert "attr_value" in df.columns
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Helpers shared by _LazyNegRules tests
# ---------------------------------------------------------------------------


def _make_op_rule(name: str) -> dict:
    """Minimal operator rule entry matching the structure _sanitize_domain_neg_rules expects."""
    return {
        "op_name": name,
        "negative_rules": {"compile": {}, "run": {}},
        "all_failed": {"compile": False, "run": False},
        "total_row_count": 4,
    }


NEG_RULES_FILE = "neg_rules.json"

# Mix of operator and pattern rules
NEG_RULES_RAW = {
    "Conv": _make_op_rule("Conv"),
    "Add": _make_op_rule("Add"),
    "GeluPattern": _make_op_rule("GeluPattern"),  # matched by "Pattern" in key
    "LayerNormPattern": _make_op_rule("LayerNormPattern"),
}

REGISTERED_PATTERNS: set[str] = set()  # rely on "Pattern" suffix detection


@pytest.fixture
def neg_rules_zip(tmp_path: Path) -> Path:
    """Create a zip containing a neg-rules JSON file."""
    zp = tmp_path / "rules.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr(NEG_RULES_FILE, json.dumps(NEG_RULES_RAW))
    return zp


# ---------------------------------------------------------------------------
# _LazyNegRules tests
# ---------------------------------------------------------------------------


class TestLazyNegRulesCore:
    """Test no-IO construction and lazy loading."""

    def test_no_io_before_first_access(self, neg_rules_zip: Path):
        rules = _LazyNegRules(neg_rules_zip, NEG_RULES_FILE, REGISTERED_PATTERNS)
        assert rules._loaded is False
        assert dict.__len__(rules) == 0

    def test_loads_on_contains(self, neg_rules_zip: Path):
        rules = _LazyNegRules(neg_rules_zip, NEG_RULES_FILE, REGISTERED_PATTERNS)
        assert "Conv" in rules
        assert rules._loaded is True

    def test_loads_on_getitem(self, neg_rules_zip: Path):
        rules = _LazyNegRules(neg_rules_zip, NEG_RULES_FILE, REGISTERED_PATTERNS)
        entry = rules["Conv"]
        assert rules._loaded is True
        assert entry["op_name"] == "Conv"

    def test_loads_on_get(self, neg_rules_zip: Path):
        rules = _LazyNegRules(neg_rules_zip, NEG_RULES_FILE, REGISTERED_PATTERNS)
        entry = rules.get("Conv")
        assert rules._loaded is True
        assert entry is not None

    def test_loads_on_items(self, neg_rules_zip: Path):
        rules = _LazyNegRules(neg_rules_zip, NEG_RULES_FILE, REGISTERED_PATTERNS)
        items = list(rules.items())
        assert rules._loaded is True
        assert len(items) > 0

    def test_get_missing_key_returns_none(self, neg_rules_zip: Path):
        rules = _LazyNegRules(neg_rules_zip, NEG_RULES_FILE, REGISTERED_PATTERNS)
        assert rules.get("NonExistent") is None

    def test_missing_key_raises_keyerror(self, neg_rules_zip: Path):
        rules = _LazyNegRules(neg_rules_zip, NEG_RULES_FILE, REGISTERED_PATTERNS)
        with pytest.raises(KeyError):
            _ = rules["NonExistent"]


class TestLazyNegRulesFiltering:
    """Test operator-only vs pattern-only filtering."""

    def test_operator_rules_excludes_patterns(self, neg_rules_zip: Path):
        rules = _LazyNegRules(
            neg_rules_zip, NEG_RULES_FILE, REGISTERED_PATTERNS, patterns_only=False
        )
        keys = set(rules.keys())
        assert "Conv" in keys
        assert "Add" in keys
        assert "GeluPattern" not in keys
        assert "LayerNormPattern" not in keys

    def test_pattern_rules_excludes_operators(self, neg_rules_zip: Path):
        rules = _LazyNegRules(
            neg_rules_zip, NEG_RULES_FILE, REGISTERED_PATTERNS, patterns_only=True
        )
        keys = set(rules.keys())
        assert "GeluPattern" in keys
        assert "LayerNormPattern" in keys
        assert "Conv" not in keys
        assert "Add" not in keys

    def test_registered_pattern_name_also_filtered(self, tmp_path: Path):
        """An op whose name is in registered_patterns is treated as a pattern."""
        raw = {
            "SpecialOp": _make_op_rule("SpecialOp"),
            "NormalOp": _make_op_rule("NormalOp"),
        }
        zp = tmp_path / "r.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr(NEG_RULES_FILE, json.dumps(raw))

        registered = {"SpecialOp"}
        op_rules = _LazyNegRules(zp, NEG_RULES_FILE, registered, patterns_only=False)
        pat_rules = _LazyNegRules(zp, NEG_RULES_FILE, registered, patterns_only=True)

        assert "NormalOp" in op_rules
        assert "SpecialOp" not in op_rules
        assert "SpecialOp" in pat_rules
        assert "NormalOp" not in pat_rules


class TestLazyNegRulesMissing:
    """Test missing zip / missing file behaviour."""

    def test_missing_zip_no_error_flag_is_empty(self, tmp_path: Path):
        rules = _LazyNegRules(tmp_path / "nonexistent.zip", NEG_RULES_FILE, REGISTERED_PATTERNS)
        assert "Conv" not in rules
        assert rules.get("Conv") is None
        assert EG_RULE_ERROR_KEY not in rules

    def test_missing_zip_with_error_flag_sets_error_keys(self, tmp_path: Path):
        rules = _LazyNegRules(
            tmp_path / "nonexistent.zip",
            NEG_RULES_FILE,
            REGISTERED_PATTERNS,
            set_error_on_missing=True,
        )
        assert rules[EG_RULE_ERROR_KEY] == "rules_zip_not_found"
        assert EG_RULE_DEBUG_DETAILS_KEY in rules

    def test_file_not_in_zip_no_error_flag_is_empty(self, neg_rules_zip: Path):
        rules = _LazyNegRules(neg_rules_zip, "absent.json", REGISTERED_PATTERNS)
        assert "Conv" not in rules
        assert EG_RULE_ERROR_KEY not in rules

    def test_file_not_in_zip_with_error_flag_sets_error_keys(self, neg_rules_zip: Path):
        rules = _LazyNegRules(
            neg_rules_zip,
            "absent.json",
            REGISTERED_PATTERNS,
            set_error_on_missing=True,
        )
        assert rules[EG_RULE_ERROR_KEY] == "negative_rule_file_not_found"
        assert EG_RULE_DEBUG_DETAILS_KEY in rules


class TestLazyNegRulesSanitization:
    """Test that _sanitize_domain_neg_rules is applied to loaded values."""

    def test_list_values_are_made_hashable(self, tmp_path: Path):
        """Values in negative_rules entries are passed through make_hashable."""
        raw = {
            "Conv": {
                "op_name": "Conv",
                "negative_rules": {
                    "compile": {"attr_group": [{"value": [1, 2, 3], "row_count": 2}]},
                    "run": {},
                },
                "all_failed": {"compile": False, "run": False},
                "total_row_count": 2,
            }
        }
        zp = tmp_path / "r.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr(NEG_RULES_FILE, json.dumps(raw))

        rules = _LazyNegRules(zp, NEG_RULES_FILE, REGISTERED_PATTERNS)
        value = rules["Conv"]["negative_rules"]["compile"]["attr_group"][0]["value"]
        # make_hashable converts lists to tuples
        assert isinstance(value, tuple)
        assert value == (1, 2, 3)
