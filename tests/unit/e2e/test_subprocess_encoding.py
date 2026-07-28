# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _keyword_value(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def test_text_subprocess_calls_use_utf8_replacement_decoding() -> None:
    test_file = REPO_ROOT / "tests" / "e2e" / "test_perf_e2e.py"
    tree = ast.parse(test_file.read_text(encoding="utf-8"))

    missing: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ):
            continue

        text_value = _keyword_value(node, "text")
        if not (isinstance(text_value, ast.Constant) and text_value.value is True):
            continue

        encoding_value = _keyword_value(node, "encoding")
        errors_value = _keyword_value(node, "errors")
        if not (
            isinstance(encoding_value, ast.Constant)
            and encoding_value.value == "utf-8"
            and isinstance(errors_value, ast.Constant)
            and errors_value.value == "replace"
        ):
            missing.append(node.lineno)

    assert missing == []
