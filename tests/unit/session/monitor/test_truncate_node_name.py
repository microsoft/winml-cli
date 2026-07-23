# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Left-ellipsis truncation matches mockup spec."""

from winml.modelkit.session.monitor.report import _truncate_node_name


def test_under_max_width_unchanged():
    assert _truncate_node_name("/short", max_width=80) == "/short"


def test_exact_max_width_unchanged():
    name = "x" * 80
    assert _truncate_node_name(name, max_width=80) == name


def test_over_max_width_left_ellipsis():
    name = "/very/long/path/" + "x" * 200
    out = _truncate_node_name(name, max_width=80)
    assert len(out) == 80
    assert out.startswith("…")  # leading ellipsis char
    assert out.endswith("x" * 79)  # right side preserved


def test_max_width_one():
    assert _truncate_node_name("anything", max_width=1) == "…"


def test_max_width_zero_returns_empty():
    assert _truncate_node_name("anything", max_width=0) == ""
