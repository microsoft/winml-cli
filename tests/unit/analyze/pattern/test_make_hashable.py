# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for make_hashable with numpy.ndarray support."""

from __future__ import annotations

import numpy as np

from winml.modelkit.pattern.utils import make_hashable


class TestMakeHashableNumpyArray:
    """Tests for numpy.ndarray handling in make_hashable."""

    def test_1d_array_becomes_hashable_tuple(self):
        """A 1-D numpy array should be converted to a tuple."""
        arr = np.array([1, 2, 3])
        result = make_hashable(arr)
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)

    def test_array_can_be_used_in_set(self):
        """Converted arrays should be usable as set elements (hashable)."""
        a = make_hashable(np.array([1, 2]))
        b = make_hashable(np.array([3, 4]))
        c = make_hashable(np.array([1, 2]))
        s = {a, b, c}
        assert len(s) == 2

    def test_array_can_be_used_as_dict_key(self):
        """Converted arrays should be usable as dict keys."""
        key = make_hashable(np.array([10, 20]))
        d = {key: "value"}
        assert d[key] == "value"

    def test_float_array_replaces_with_dummy(self):
        """Float elements in arrays should follow replace_float_with_dummy."""
        arr = np.array([1.5, 2.5])
        result = make_hashable(arr, replace_float_with_dummy=True)
        from winml.modelkit.pattern.utils import DUMMY_FLOAT

        assert all(v == DUMMY_FLOAT for v in result)

    def test_float_array_preserves_values_when_no_replace(self):
        """Float elements should be preserved when replace_float_with_dummy=False."""
        arr = np.array([1.5, 2.5])
        result = make_hashable(arr, replace_float_with_dummy=False)
        assert result == (1.5, 2.5)

    def test_nested_list_containing_array(self):
        """A list containing a numpy array should recursively convert."""
        nested = [np.array([1, 2]), "hello"]
        result = make_hashable(nested)
        assert isinstance(result, tuple)
        assert result[0] == (1, 2)
        assert result[1] == "hello"

    def test_empty_array(self):
        """An empty numpy array should become an empty tuple."""
        result = make_hashable(np.array([]))
        assert result == ()
