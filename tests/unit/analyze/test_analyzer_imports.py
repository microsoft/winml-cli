# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Regression pin for P0-A: analyze/analyzer.py had a dangling
`from ..sysinfo import resolve_device` inside `analyze_from_proto`
after the session refactor removed that export.

Module-level import wouldn't catch it (the import is lazy). We walk the
AST for every relative `from` statement — top-level and nested — and
verify each symbol actually resolves.
"""
from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

import winml.modelkit.analyze.analyzer as analyzer_module


def _package_of(module) -> str:
    return module.__package__ or module.__name__.rsplit(".", 1)[0]


def _resolve_relative(base_package: str, level: int, mod: str | None) -> str:
    parts = base_package.split(".")
    if level > len(parts):
        raise ImportError(f"relative import level {level} escapes package {base_package}")
    root = parts[: len(parts) - (level - 1)] if level > 1 else parts
    tail = [mod] if mod else []
    return ".".join(root + tail)


def _collect_from_imports(source_path: Path) -> list[tuple[int, str, list[str]]]:
    """Return (line, resolved_module, symbols) for every `from X import a,b` in source.

    Includes nested (function-local) imports because those are the ones
    Python's module-load never catches.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    base_package = _package_of(analyzer_module)
    out: list[tuple[int, str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            symbols = [a.name for a in node.names if a.name != "*"]
            if node.level and node.level > 0:
                resolved = _resolve_relative(base_package, node.level, node.module)
            else:
                resolved = node.module or ""
            out.append((node.lineno, resolved, symbols))
    return out


_IMPORTS = _collect_from_imports(Path(analyzer_module.__file__))


@pytest.mark.parametrize(
    "lineno,module_path,symbols",
    [pytest.param(*t, id=f"L{t[0]}:{t[1]}") for t in _IMPORTS],
)
def test_from_imports_resolve(lineno: int, module_path: str, symbols: list[str]) -> None:
    """Every `from X import a, b` in analyzer.py must resolve — top-level or lazy."""
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        pytest.fail(f"analyzer.py:{lineno}: cannot import module '{module_path}': {e}")
    missing = [s for s in symbols if not hasattr(mod, s)]
    assert not missing, (
        f"analyzer.py:{lineno}: `from {module_path} import ...` — "
        f"module has no attribute(s) {missing}"
    )


# Anchor: also assert bare module import still works.
def test_module_imports_cleanly() -> None:
    importlib.import_module("winml.modelkit.analyze.analyzer")


# Silence unused-import warning
_ = pkgutil
