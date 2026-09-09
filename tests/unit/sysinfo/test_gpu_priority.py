# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared Windows GPU priority semantics."""

from __future__ import annotations

import itertools

import pytest

from winml.modelkit.sysinfo import format_pdh_luid, gpu_priority_key


def test_numeric_preference_is_independent_of_enumeration() -> None:
    ranked = [(format_pdh_luid(str(100 - index)), str(index)) for index in (0, 2, 10)]
    for enumeration in itertools.permutations(ranked):
        assert sorted(enumeration, key=lambda item: gpu_priority_key(*item)) == ranked


@pytest.mark.parametrize("rank", [None, "", "invalid", "-1", -1, 1.5, True, {}, "1.5"])
def test_invalid_ranks_use_luid_after_ranked_devices(rank: object) -> None:
    luids = [format_pdh_luid(str(index)) for index in range(1, 4)]
    enumeration = luids[::-1]
    assert sorted(enumeration, key=lambda luid: gpu_priority_key(luid, rank)) == luids
    assert gpu_priority_key(luids[-1], 10) < gpu_priority_key(luids[0], rank)
    assert gpu_priority_key(luids[-1], rank) < gpu_priority_key(None, rank)


@pytest.mark.parametrize("rank", [0, "0", 10, "10", None])
def test_equal_ranks_use_case_insensitive_luid(rank: object) -> None:
    luids = [format_pdh_luid(str(index)) for index in range(10, 13)]
    enumeration = luids[::-1]
    assert sorted(enumeration, key=lambda luid: gpu_priority_key(luid, rank)) == luids
    assert gpu_priority_key(luids[0], rank) == gpu_priority_key(luids[0].swapcase(), rank)
