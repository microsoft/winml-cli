# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Architecture regression: nobody outside session/ may import directly from ep_device.py.

Per Decision A in docs/plans/2026-05-13-ep-taxonomy-consolidation-plan.md:

* Import shape for source code: ``from ..session import WinMLEPDevice, resolve_device, ...``
* Import shape for tests:       ``from winml.modelkit.session import WinMLEPDevice, ...``
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

    Use ``from winml.modelkit.session import WinMLEPDevice`` (the facade) instead of
    ``from winml.modelkit.session.ep_device import WinMLEPDevice``.
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
        "from winml.modelkit.session.ep_device import WinMLEPDevice",
        "from winml.modelkit.session.ep_device import WinMLEPDevice, resolve_device",
        "from winml.modelkit.session.ep_device import _EP_TO_DEVICE",
        # Deleted names — sentinels so the detector catches re-additions
        "from winml.modelkit.session.ep_device import _DEVICE_TO_PROVIDER",
        "from winml.modelkit.session.ep_device import _VALID_DEVICES",
        "from winml.modelkit.session.ep_device import _compile_provider",
        "from winml.modelkit.session.ep_device import get_provider_for_device",
        # Module import forms
        "import winml.modelkit.session.ep_device",
        "import winml.modelkit.session.ep_device as epd",
        # Relative ImportFrom with explicit ep_device module name
        "from .ep_device import WinMLEPDevice",
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
        "from winml.modelkit.session import WinMLEPDevice",
        "from winml.modelkit.session import WinMLEPDevice, resolve_device, VALID_EPS",
        # Unrelated imports
        "from winml.modelkit.session import WinMLSession",
        "from winml.modelkit.session.session import WinMLSession",
        # Session-package-internal relative import (session/__init__.py style)
        "from .session import WinMLEPDevice",
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


# --------------------------------------------------------------------------
# N5: Inline EP/device mapping literal detector.
# Catches frozenset/set/list literals whose elements are exclusively
# known EP short names or device strings — a sign that someone is
# re-building catalog data outside session/ep_device.py.
# --------------------------------------------------------------------------

_EP_SHORT_NAMES: frozenset[str] = frozenset(
    {"qnn", "openvino", "vitisai", "migraphx", "dml", "cuda", "tensorrt", "nv_tensorrt_rtx"}
)
_DEVICE_STRINGS: frozenset[str] = frozenset({"npu", "gpu", "cpu"})
# ep_device.py and utils/cli.py are the only authorised homes for EP/device literals.
_INLINE_LITERAL_ALLOWLIST: frozenset[str] = frozenset(
    {"ep_device.py", "cli.py", "conftest.py", "check_ops.py", "check_patterns.py"}
)


def _extract_string_constants(node: ast.AST) -> list[str]:
    """Return string constants from a Set, List, or Tuple literal node."""
    if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return []
    return [
        elt.value
        for elt in node.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    ]


def _is_ep_device_mapping_literal(node: ast.AST) -> bool:
    """True if the node is a set/list/tuple whose members are ALL known EP short
    names or ALL known device strings — i.e., it reconstructs catalog data inline.

    Only flags when every element in the literal is a known name (avoids
    false-positives on mixed-purpose collections).
    """
    strings = _extract_string_constants(node)
    if len(strings) < 2:  # single-element sets are not catalog duplicates
        return False
    as_set = set(strings)
    return as_set.issubset(_EP_SHORT_NAMES) or as_set.issubset(_DEVICE_STRINGS)


def test_no_inline_ep_device_mapping_literals_in_src() -> None:
    """Detect frozenset/set/list/tuple literals that reconstruct EP or device
    catalog data outside session/ep_device.py.

    Only pure-EP-short-name or pure-device-string collections of ≥2 elements
    are flagged. Mixed collections (e.g., ``{"auto", "npu", "gpu", "cpu"}``)
    are NOT flagged because they contain non-catalog strings like "auto".

    Known carve-outs (check_ops.py, check_patterns.py, cli.py, ep_device.py)
    are skipped by the allowlist.
    """
    src_root = pathlib.Path(__file__).parents[3] / "src" / "winml" / "modelkit"
    assert src_root.is_dir(), f"src/ root not found at {src_root}"

    violations: list[str] = []
    for py_file in src_root.rglob("*.py"):
        if py_file.name in _INLINE_LITERAL_ALLOWLIST:
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        violations.extend(
            f"{py_file}:{getattr(node, 'lineno', '?')}"
            for node in ast.walk(tree)
            if _is_ep_device_mapping_literal(node)
        )

    assert not violations, (
        "Inline EP/device mapping literals detected in src/ outside authorised files. "
        "Move them to session/ep_device.py or document as a carve-out. "
        "Violations:\n  " + "\n  ".join(violations)
    )
