# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression coverage for Transformers 5-compatible public config loading."""

from __future__ import annotations

import ast
from pathlib import Path


_MODELKIT_ROOT = Path(__file__).parents[3] / "src" / "winml" / "modelkit"


class _RawAutoConfigCallFinder(ast.NodeVisitor):
    """Collect direct AutoConfig calls together with their enclosing function."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self.calls: set[tuple[str, str | None]] = set()
        self._functions: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._functions.append(node.name)
        self.generic_visit(node)
        self._functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "from_pretrained"
            and isinstance(func.value, ast.Name)
            and func.value.id == "AutoConfig"
        ):
            self.calls.add(
                (
                    self._path.relative_to(_MODELKIT_ROOT).as_posix(),
                    self._functions[-1] if self._functions else None,
                )
            )
        self.generic_visit(node)


def _find_raw_autoconfig_calls() -> set[tuple[str, str | None]]:
    calls: set[tuple[str, str | None]] = set()
    for path in _MODELKIT_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        finder = _RawAutoConfigCallFinder(path)
        finder.visit(tree)
        calls.update(finder.calls)
    return calls


def test_public_config_loads_route_through_transformers5_compatibility_helper() -> None:
    """Model-type-less remote-code configs must never reach raw AutoConfig loading.

    ``load_hf_config`` is the single compatibility path for those configs. The
    inference metadata attachment is intentionally best-effort and therefore
    remains the only raw call.
    """
    assert _find_raw_autoconfig_calls() == {
        ("inference/engine.py", "_attach_hf_config"),
    }
