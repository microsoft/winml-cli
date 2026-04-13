# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for runtime checker query helper functions."""

import pytest

from winml.modelkit.analyze.core.runtime_checker_query import _build_table_filter_conditions
from winml.modelkit.analyze.exceptions import OpOptionalInputSupportError


class TestBuildTableFilterConditions:
    """Test table filter condition extraction for runtime rule lookups."""

    def test_returns_conditions_for_all_present_columns(self):
        """Returns only the requested table columns when all are present."""
        conditions = {
            "input_shape": (1, 3, 224, 224),
            "attr_axis": 1,
            "unused_key": "ignored",
        }

        result = _build_table_filter_conditions(
            conditions=conditions,
            column_names=["input_shape", "attr_axis"],
            infinite_properties=[],
            error_context="op Add",
        )

        assert result == {
            "input_shape": (1, 3, 224, 224),
            "attr_axis": 1,
        }

    def test_raises_when_required_column_is_missing(self):
        """Raises a descriptive error when a required table column is unavailable."""
        conditions = {"input_shape": (1, 3, 224, 224)}

        with pytest.raises(OpOptionalInputSupportError) as exc_info:
            _build_table_filter_conditions(
                conditions=conditions,
                column_names=["input_shape", "attr_axis"],
                infinite_properties=[],
                error_context="op Add",
            )

        assert str(exc_info.value) == (
            "Match key 'attr_axis' not found in conditions for op Add. Available: ['input_shape']"
        )

    def test_skips_columns_listed_in_infinite_properties(self):
        """Columns marked infinite are skipped even when absent from conditions."""
        conditions = {
            "input_shape": (1, 3, 224, 224),
            "attr_axis": 1,
        }

        result = _build_table_filter_conditions(
            conditions=conditions,
            column_names=["input_shape", "attr_axis", "input_value"],
            infinite_properties=["input_value"],
            error_context="op Add",
        )

        assert result == {
            "input_shape": (1, 3, 224, 224),
            "attr_axis": 1,
        }

    def test_returns_empty_dict_for_empty_column_names(self):
        """Returns an empty mapping when no table columns are requested."""
        result = _build_table_filter_conditions(
            conditions={"input_shape": (1, 3, 224, 224)},
            column_names=[],
            infinite_properties=[],
            error_context="op Add",
        )

        assert result == {}
