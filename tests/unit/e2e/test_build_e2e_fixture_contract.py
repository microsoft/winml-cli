# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Contract tests for e2e build-test hardware mocking.

The target e2e module imports optional image dependencies at module import
time, so these tests inspect the fixture source without importing it.
"""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / "e2e" / "test_build_e2e.py"


def _fixture_tree() -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_mock_resolve_device":
            return node
    raise AssertionError("_mock_resolve_device fixture not found")


def test_resolve_device_mock_uses_side_effect_not_constant_cpu() -> None:
    fixture = _fixture_tree()
    patch_calls = [
        node
        for node in ast.walk(fixture)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "patch"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "winml.modelkit.session.resolve_device"
    ]
    assert patch_calls

    keyword_names = {keyword.arg for keyword in patch_calls[0].keywords}

    assert "side_effect" in keyword_names
    assert "return_value" not in keyword_names
