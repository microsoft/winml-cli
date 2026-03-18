"""
Unit tests for LazyDomainTables lazy-loading wrapper.

Tests verify:
- Lazy loading on first access
- Caching on subsequent access
- Raw data cleanup after loading
- __contains__ and get() methods
- KeyError for non-existent operators
"""

import pandas as pd
import pytest

from winml.modelkit.analyze.core.runtime_checker_query import LazyDomainTables


@pytest.fixture
def sample_raw_data() -> dict[str, dict]:
    """Create sample raw table data for testing."""
    return {
        "Conv": {
            "attr_kernel_shape": [(3, 3), (5, 5)],
            "X_shape": [(1, 3, 224, 224), (1, 64, 112, 112)],
            "compile_run_success": [(True, True), (True, False)],
        },
        "Add": {
            "A_shape": [(1, 3, 224, 224), (1, 64, 112, 112)],
            "B_shape": [(1, 3, 224, 224), (1, 64, 112, 112)],
            "compile_run_success": [(True, True), (True, True)],
        },
    }


class TestLazyDomainTablesCore:
    """Test core lazy loading functionality."""

    def test_lazy_loading_and_caching(self, sample_raw_data):
        """Test lazy loading on first access, caching, and raw data cleanup."""
        tables = LazyDomainTables(sample_raw_data)

        # Initially: no DataFrames loaded, all data in raw storage
        assert len(tables._loaded_tables) == 0
        assert len(tables._raw_data) == 2

        # First access: triggers loading
        first_access = tables["Conv"]
        assert isinstance(first_access, pd.DataFrame)
        assert len(first_access) == 2
        assert "Conv" in tables._loaded_tables
        assert "Conv" not in tables._raw_data  # Cleaned up

        # Second access: uses cache (same object)
        second_access = tables["Conv"]
        assert first_access is second_access

        # Other operators remain unaffected
        assert "Add" in tables._raw_data
        assert "Add" not in tables._loaded_tables

    def test_multiple_operators_independent_loading(self, sample_raw_data):
        """Test that loading operators independently cleans up raw data progressively."""
        tables = LazyDomainTables(sample_raw_data)

        # Load Conv
        _ = tables["Conv"]
        assert len(tables._loaded_tables) == 1
        assert len(tables._raw_data) == 1

        # Load Add
        _ = tables["Add"]
        assert len(tables._loaded_tables) == 2
        assert len(tables._raw_data) == 0  # All cleaned up


class TestLazyDomainTablesMethods:
    """Test __contains__ and get() methods."""

    def test_contains_operator(self, sample_raw_data):
        """Test __contains__ works for both raw and loaded operators."""
        tables = LazyDomainTables(sample_raw_data)

        # Before loading
        assert "Conv" in tables
        assert "Add" in tables
        assert "Relu" not in tables

        # After loading
        _ = tables["Conv"]
        assert "Conv" in tables  # Still found (in loaded)
        assert "Add" in tables  # Still found (in raw)
        assert "Relu" not in tables

    def test_get_method(self, sample_raw_data):
        """Test get() method with defaults and caching."""
        tables = LazyDomainTables(sample_raw_data)

        # Existing operator
        conv_df = tables.get("Conv")
        assert isinstance(conv_df, pd.DataFrame)

        # Non-existent operator returns None
        assert tables.get("Relu") is None

        # Custom default
        custom_default = pd.DataFrame()
        assert tables.get("Relu", default=custom_default) is custom_default

        # get() also caches
        assert tables.get("Conv") is conv_df


class TestLazyDomainTablesErrors:
    """Test error handling."""

    def test_keyerror_for_non_existent_operator(self, sample_raw_data):
        """Test __getitem__ raises KeyError for non-existent operators."""
        tables = LazyDomainTables(sample_raw_data)

        with pytest.raises(KeyError, match="Operator 'Relu' not found in tables"):
            _ = tables["Relu"]


class TestLazyDomainTablesDataFrame:
    """Test DataFrame properties and sanitization."""

    def test_dataframe_structure(self, sample_raw_data):
        """Test loaded DataFrames have correct structure."""
        tables = LazyDomainTables(sample_raw_data)

        conv_df = tables["Conv"]
        add_df = tables["Add"]

        # Check columns
        assert set(conv_df.columns) == {"attr_kernel_shape", "X_shape", "compile_run_success"}
        assert set(add_df.columns) == {"A_shape", "B_shape", "compile_run_success"}

        # Check row counts
        assert len(conv_df) == 2
        assert len(add_df) == 2

    def test_dataframe_sanitization(self):
        """Test that _sanitize_df is applied during loading."""
        raw_data = {
            "TestOp": {
                "attr_value": [1.5, float("inf"), -float("inf"), float("nan")],
                "compile_run_success": [(True, True), (True, False), (False, False), (True, False)],
            }
        }

        tables = LazyDomainTables(raw_data)
        df = tables["TestOp"]

        assert isinstance(df, pd.DataFrame)
        assert "attr_value" in df.columns
