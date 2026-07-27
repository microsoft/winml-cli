# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Architecture regression: no external imports of qnn/_internal.py.

Per spec §8 + PRD NFR-8: the QNN parsing internals are private to
``qnn_monitor.py``. Any other module importing non-``_``-prefixed names
from ``qnn._internal`` violates the v2.4 information-hiding contract.

This scan covers BOTH the source tree and the test tree:

* ``src/`` — strict: any import of ``qnn._internal`` (regardless of
  prefix) is a violation, except in ``qnn_monitor.py`` itself.
* ``tests/`` — relaxed: per CLAUDE.md, test files MAY directly import
  ``_``-prefixed *function* names from ``qnn._internal`` (the
  documented exception for testing private internals). Importing
  non-``_``-prefixed names, OR importing the ``_internal`` module
  itself, remains a violation.

The detector recognises every common shape of the forbidden import:

* ``from X.qnn._internal import Y``           (absolute)
* ``from .qnn._internal import Y``            (relative)
* ``import X.qnn._internal``                  (rare)
* ``import X.qnn._internal as alias``         (rare, with alias)
* ``from .qnn import _internal``              (alias-list, relative)
* ``from X.qnn import _internal``             (alias-list, absolute)
"""

from __future__ import annotations

import ast
import pathlib
import textwrap

import pytest

from winml.modelkit.session.monitor import qnn_monitor


def _is_internal_import(node: ast.AST) -> bool:
    """True if the AST node imports from ``qnn._internal`` in any form.

    Detects:

    * ``from <something>.qnn._internal import Y`` — most common shape;
      module path ends with ``qnn._internal``.
    * ``from .qnn._internal import Y`` — relative variant.
    * ``import <...>.qnn._internal [as alias]`` — module-level import.
    * ``from .qnn import _internal`` / ``from X.qnn import _internal`` —
      alias-list form, where the *module* is ``qnn`` and the imported
      name is ``_internal``. Easy to miss.
    """
    if isinstance(node, ast.ImportFrom) and node.module is not None:
        # Form 1: from <something>.qnn._internal import Y
        if node.module.endswith("qnn._internal") or node.module.endswith(".qnn._internal"):
            return True
        # Form 2: from <something>.qnn import _internal
        #   - relative form: node.module == "qnn" (level >= 1)
        #   - absolute form: node.module ends with ".qnn" or "qnn"
        if (
            node.module == "qnn" or node.module.endswith(".qnn") or node.module.endswith("qnn")
        ) and any(alias.name == "_internal" for alias in node.names):
            return True
    # Form 3: import <...>.qnn._internal [as alias]
    return isinstance(node, ast.Import) and any(
        alias.name.endswith("qnn._internal") for alias in node.names
    )


def _is_test_exception_allowed(node: ast.AST) -> bool:
    """True if a test file may legitimately use this import per CLAUDE.md.

    The CLAUDE.md exception: test files MAY directly import ``_``-prefixed
    *function* names from a private module to test implementation details.

    Strict interpretation: the exception covers ``_``-prefixed function
    imports (e.g. ``from qnn._internal import _aggregate_operators``)
    but does NOT cover importing the ``_internal`` *module* itself
    (e.g. ``from qnn import _internal``). Importing the module reaches
    past the package boundary in a way that prefix-on-the-alias does
    not justify — the spec intent is that the module is private.
    """
    if not isinstance(node, ast.ImportFrom):
        return False
    # If any imported name is the module ``_internal`` itself
    # (alias-list form), no exception applies regardless of file.
    if any(alias.name == "_internal" for alias in node.names):
        return False
    # All imported names must be ``_``-prefixed for the exception.
    return all(alias.name.startswith("_") for alias in node.names)


def test_no_external_imports_of_qnn_internal() -> None:
    """Only ``qnn_monitor.py`` may import non-``_``-prefixed names from ``qnn._internal``.

    Test files may directly import ``_``-prefixed *function* names from
    ``qnn._internal`` (CLAUDE.md exception). Importing non-``_``-prefixed
    names — or importing the ``_internal`` module itself — is a violation
    in any file other than ``qnn_monitor.py``.
    """
    src_root = pathlib.Path(__file__).parents[3] / "src" / "winml" / "modelkit"
    tests_root = pathlib.Path(__file__).parents[2]  # tests/unit/
    assert src_root.is_dir(), f"src/ root not found at {src_root}"
    assert tests_root.is_dir(), f"tests/unit/ root not found at {tests_root}"

    qnn_monitor_path = pathlib.Path(qnn_monitor.__file__).resolve()

    violations: list[str] = []
    for root in (src_root, tests_root):
        is_test_root = root == tests_root
        for py_file in root.rglob("*.py"):
            if py_file.resolve() == qnn_monitor_path:
                continue  # qnn_monitor.py is the sole permitted importer of non-_ names
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not _is_internal_import(node):
                    continue
                # In test files, apply the CLAUDE.md exception for `_`-prefixed
                # *function* imports (but NOT for importing `_internal` itself).
                if is_test_root and _is_test_exception_allowed(node):
                    continue
                violations.append(f"{py_file}:{node.lineno}")

    assert not violations, (
        "External imports of qnn._internal detected. Test files may import "
        "`_`-prefixed function names directly; production code and re-exports "
        "of non-`_`-prefixed names must use only qnn_monitor.py / "
        "the qnn package surface. Violations:\n  " + "\n  ".join(violations)
    )


# --------------------------------------------------------------------------
# Synthetic-AST tests for the detector itself.
#
# These guard against future regressions where a new import shape silently
# slips through the architecture scan.  Each forbidden form here is what the
# scan-walk above is designed to catch (whether or not the scope-walk then
# applies the CLAUDE.md exception is a separate question, tested below).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "from winml.modelkit.session.monitor.qnn._internal import parse_qhas",
        "from .qnn._internal import parse_qhas",
        "import winml.modelkit.session.monitor.qnn._internal",
        "import winml.modelkit.session.monitor.qnn._internal as parser",
        "from .qnn import _internal",  # Gap 2 — alias-list form
        "from winml.modelkit.session.monitor.qnn import _internal",  # absolute alias-list
    ],
)
def test_is_internal_import_detects_each_forbidden_form(source: str) -> None:
    """The detector must catch every forbidden import shape."""
    tree = ast.parse(textwrap.dedent(source))
    nodes = list(ast.walk(tree))
    assert any(_is_internal_import(node) for node in nodes), (
        f"Detector missed import form: {source!r}"
    )


@pytest.mark.parametrize(
    "source",
    [
        # Unrelated imports must NOT be flagged at AST level.
        "from winml.modelkit.session import session",
        "from winml.modelkit.session.monitor import qnn_monitor",
        "from winml.modelkit.session.monitor.qnn import parse_qhas",  # public re-export
        "from winml.modelkit.session.monitor.qnn import parse_qnn_profiling_csv",
    ],
)
def test_is_internal_import_does_not_flag_unrelated_imports(source: str) -> None:
    """The detector must not false-positive on unrelated or public imports."""
    tree = ast.parse(textwrap.dedent(source))
    nodes = list(ast.walk(tree))
    assert not any(_is_internal_import(node) for node in nodes), (
        f"Detector false-positive on {source!r}"
    )


@pytest.mark.parametrize(
    "source",
    [
        "from winml.modelkit.session.monitor.qnn._internal import _aggregate_operators",
        "from .qnn._internal import _split_op_event_id",
        "from winml.modelkit.session.monitor.qnn._internal import _TOKEN_SUFFIX",
    ],
)
def test_underscore_prefixed_imports_are_flagged_at_ast_level(source: str) -> None:
    """`_`-prefixed function imports from qnn._internal ARE qnn._internal imports.

    They must be flagged by the AST detector. The CLAUDE.md test exception
    is applied at the scope-walk level (``test_no_external_imports_of_qnn_internal``),
    not by the detector itself.
    """
    tree = ast.parse(textwrap.dedent(source))
    nodes = list(ast.walk(tree))
    assert any(_is_internal_import(node) for node in nodes), (
        f"Detector should flag {source!r} at AST level "
        "(scope-walk applies the test exception separately)."
    )


@pytest.mark.parametrize(
    ("source", "should_be_allowed"),
    [
        # `_`-prefixed function imports — exception applies.
        (
            "from winml.modelkit.session.monitor.qnn._internal import _aggregate_operators",
            True,
        ),
        (
            "from winml.modelkit.session.monitor.qnn._internal import _split_op_event_id",
            True,
        ),
        # Non-`_`-prefixed imports — exception does NOT apply.
        (
            "from winml.modelkit.session.monitor.qnn._internal import parse_qhas",
            False,
        ),
        # Module-itself import — exception does NOT apply (per strict reading).
        (
            "from winml.modelkit.session.monitor.qnn import _internal",
            False,
        ),
        (
            "from .qnn import _internal",
            False,
        ),
        # Mixed: at least one non-`_`-prefixed name → exception does NOT apply.
        (
            "from winml.modelkit.session.monitor.qnn._internal import _TOKEN_SUFFIX, parse_qhas",
            False,
        ),
    ],
)
def test_test_exception_logic(source: str, should_be_allowed: bool) -> None:
    """Verify the CLAUDE.md test exception applies only to `_`-prefixed function imports."""
    tree = ast.parse(textwrap.dedent(source))
    import_node = next(
        node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    assert _is_test_exception_allowed(import_node) is should_be_allowed
