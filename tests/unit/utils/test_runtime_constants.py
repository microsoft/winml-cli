# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for inference runtime constants."""

from __future__ import annotations

from typing import get_args

from winml.modelkit.utils.constants import RUNTIME_NAMES, RuntimeName


def test_runtime_names_match_runtime_name_literal() -> None:
    assert get_args(RuntimeName) == RUNTIME_NAMES
    assert RUNTIME_NAMES == ("auto", "winml", "winml-genai")
