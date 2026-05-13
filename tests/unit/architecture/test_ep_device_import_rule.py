# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Architecture regression: nobody outside session/ may import directly from ep_device.py.

Per Decision A in docs/plans/2026-05-13-ep-taxonomy-consolidation-plan.md:

* Import shape for source code: ``from ..session import EPDevice, resolve_device, ...``
* Import shape for tests:       ``from winml.modelkit.session import EPDevice, ...``
* Never:                        ``from ..session.ep_device import ...``
                                (drills past the session/ facade)

This scan covers both the source tree and the test tree.

**Allowed** (within-session/ sibling-relative imports):
    ``from .ep_device import ...``   — inside session/ package files
    ``from .ep_device import ...``   — session/__init__.py (this file IS the facade)

**Forbidden** from any file OUTSIDE ``session/``:
    ``from <anything>.session.ep_device import ...``   (absolute form)
    ``from .session.ep_device import ...``             (relative, going into ep_device)
    ``import <anything>.session.ep_device``            (module import)
    ``import <anything>.session.ep_device as alias``   (module import with alias)

Test files that import ``_``-prefixed names (e.g. ``_EP_TO_DEVICE``) directly
for testing implementation details are still forbidden — the architecture rule
applies equally to private symbols.  If a test needs a private symbol it must
be exposed via the session facade or the test must import through the facade.
"""

from __future__ import annotations

import ast
import pathlib
import textwrap

import pytest


def _session_dir() -> pathlib.Path:
    """Absolute path to ``src/winml/modelkit/session/``."""
    src_root = pathlib.Path(__file__).parents[3] / "src" / "winml" / "modelkit"
    return src_root / "session"


def _is_direct_ep_device_import(node: ast.AST) -> bool:
    """True if the AST node directly imports from session/ep_device (forbidden shape).

    Detects all four forbidden patterns:

    1. ``from <...>.session.ep_device import Y``   — absolute ImportFrom
    2. ``from .ep_device import Y``                — relative ImportFrom (relative level≥1,
                                                     module == "ep_device")
    3. ``import <...>.session.ep_device``          — module import
    4. ``import <...>.session.ep_device as alias`` — module import with alias
    """
    if isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        # Absolute form: module path contains ".session.ep_device" or ends in "session.ep_device"
        if "session.ep_device" in mod:
            return True
        # Relative form with level>=1 where module IS "ep_device"
        # This covers ``from .ep_device import X`` (relative to current package).
        # We only flag this when the check is called from outside session/.
        # (The caller skips files inside session/.)
        if node.level >= 1 and mod == "ep_device":
            return True
    if isinstance(node, ast.Import):
        return any("session.ep_device" in alias.name for alias in node.names)
    return False


def _collect_violations(root: pathlib.Path) -> list[str]:
    """Walk *root* and return a list of ``"<file>:<lineno>"`` violation strings."""
    session_path = _session_dir().resolve()
    violations: list[str] = []

    for py_file in root.rglob("*.py"):
        resolved = py_file.resolve()
        # Files inside session/ may use sibling-relative imports (from .ep_device import …).
        if resolved.is_relative_to(session_path):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        violations.extend(
            f"{py_file}:{node.lineno}"
            for node in ast.walk(tree)
            if _is_direct_ep_device_import(node)
        )

    return violations


def test_no_direct_ep_device_imports_in_src() -> None:
    """Source files outside session/ must not import directly from session/ep_device.py."""
    src_root = pathlib.Path(__file__).parents[3] / "src" / "winml" / "modelkit"
    assert src_root.is_dir(), f"src/ root not found at {src_root}"

    violations = _collect_violations(src_root)
    assert not violations, (
        "Direct imports of session/ep_device detected in src/. "
        "Use 'from ..session import <symbol>' through the session/ facade instead. "
        "Violations:\n  " + "\n  ".join(violations)
    )


def test_no_direct_ep_device_imports_in_tests() -> None:
    """Test files must not import directly from session/ep_device.py.

    Use ``from winml.modelkit.session import EPDevice`` (the facade) instead of
    ``from winml.modelkit.session.ep_device import EPDevice``.
    """
    tests_root = pathlib.Path(__file__).parents[2]  # tests/unit/
    assert tests_root.is_dir(), f"tests/unit/ root not found at {tests_root}"

    violations = _collect_violations(tests_root)
    assert not violations, (
        "Direct imports of session/ep_device detected in tests/. "
        "Use 'from winml.modelkit.session import <symbol>' through the facade instead. "
        "Violations:\n  " + "\n  ".join(violations)
    )


# --------------------------------------------------------------------------
# Synthetic-AST tests for the detector itself.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        # Absolute ImportFrom — most common violation shape
        "from winml.modelkit.session.ep_device import EPDevice",
        "from winml.modelkit.session.ep_device import EPDevice, resolve_device",
        "from winml.modelkit.session.ep_device import _EP_TO_DEVICE",
        # Module import forms
        "import winml.modelkit.session.ep_device",
        "import winml.modelkit.session.ep_device as epd",
        # Relative ImportFrom with explicit ep_device module name
        "from .ep_device import EPDevice",
    ],
)
def test_detector_catches_forbidden_forms(source: str) -> None:
    """The detector must flag every forbidden import shape."""
    tree = ast.parse(textwrap.dedent(source))
    nodes = list(ast.walk(tree))
    assert any(_is_direct_ep_device_import(node) for node in nodes), (
        f"Detector missed forbidden import: {source!r}"
    )


@pytest.mark.parametrize(
    "source",
    [
        # Facade import — correct shape
        "from winml.modelkit.session import EPDevice",
        "from winml.modelkit.session import EPDevice, resolve_device, VALID_EPS",
        # Unrelated imports
        "from winml.modelkit.session import WinMLSession",
        "from winml.modelkit.session.session import WinMLSession",
        # Session-package-internal relative import (session/__init__.py style)
        "from .session import EPDevice",
        # Importing session module itself (not ep_device sub-module)
        "import winml.modelkit.session",
        "from winml.modelkit import session",
    ],
)
def test_detector_does_not_flag_allowed_forms(source: str) -> None:
    """The detector must not false-positive on correct/facade import shapes."""
    tree = ast.parse(textwrap.dedent(source))
    nodes = list(ast.walk(tree))
    assert not any(_is_direct_ep_device_import(node) for node in nodes), (
        f"Detector false-positive on allowed import: {source!r}"
    )
