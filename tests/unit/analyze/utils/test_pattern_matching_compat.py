# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compatibility coverage for analyzer wildcard helpers."""

from winml.modelkit.analyze.utils import match_pattern_with_wildcards
from winml.modelkit.analyze.utils.pattern_matching import (
    match_type_vars_with_wildcards,
    match_version_with_wildcard,
)


def test_match_pattern_with_wildcards_remains_public() -> None:
    assert match_pattern_with_wildcards(
        {"kernel_shape": "*", "strides": [1, 1]},
        {"kernel_shape": [3, 3], "strides": [1, 1]},
    )
    assert not match_pattern_with_wildcards(
        {"kernel_shape": [3, 3]},
        {"kernel_shape": [5, 5]},
    )


def test_match_type_vars_with_wildcards_keeps_alternative_support() -> None:
    assert match_type_vars_with_wildcards({"T": "float32|float16"}, {"T": "float16"})
    assert not match_type_vars_with_wildcards({"T": "float32"}, {"T": "int8"})


def test_match_version_with_wildcard_keeps_universal_match() -> None:
    assert match_version_with_wildcard("2.3.1", "*")
    assert not match_version_with_wildcard("2.3.1", "2.3.0")
