# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``winml.modelkit.winml`` deprecation warning emission.

Every public symbol in ``winml.modelkit.winml`` is deprecated in favour of
the session-layer machinery. Each entry point — ``WinML.__init__``, the
module-level ``register_execution_providers`` wrapper, and
``add_ep_for_device`` — must emit a ``DeprecationWarning`` carrying the
migration pointer, and ``stacklevel=2`` must put the user-visible caller
frame on the warning so IDEs / linters point at the caller, not the
shim implementation.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _winml_mod():
    """Lazy import of ``winml.modelkit.winml``.

    Top-level ``import winml.modelkit...`` fails inside this file when run
    in isolation because the test directory ``tests/unit/winml/`` is
    registered by pytest's importlib mode as a package named ``winml``
    that shadows the real distribution. When the broader suite runs,
    earlier tests pre-import ``winml.modelkit`` so the canonical mapping
    in ``sys.modules`` already exists by the time we hit this file. Fall
    back to evicting the shadow and re-resolving when the import fails.
    """
    import sys

    try:
        from winml.modelkit import winml as winml_mod
        return winml_mod
    except ModuleNotFoundError:
        # Evict the shadow test-package binding of ``winml`` so the next
        # import resolves against ``src/winml`` (pinned on ``pythonpath``
        # via pyproject ``tool.pytest.ini_options``).
        for key in [k for k in list(sys.modules) if k == "winml" or k.startswith("winml.")]:
            sys.modules.pop(key, None)
        from winml.modelkit import winml as winml_mod
        return winml_mod


def _reset_winml_singleton() -> None:
    """Force the next ``WinML()`` construction to re-run ``__init__``.

    The singleton fires the deprecation warning only on the first
    ``__init__`` invocation; tests that need a fresh warning must reset
    state before construction.
    """
    _winml_mod()._winml_instance = None


def test_winml_init_emits_deprecation_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """``WinML()`` first construction emits DeprecationWarning(stacklevel=2)."""
    winml_mod = _winml_mod()
    _reset_winml_singleton()
    # Avoid the EP-discovery scan inside __init__ — we only care about the warning.
    monkeypatch.setattr(winml_mod, "discover_all_eps", list)

    with pytest.warns(DeprecationWarning, match="winml.modelkit.winml is deprecated") as record:
        winml_mod.WinML()

    assert len(record) == 1, f"Expected exactly one warning; got {len(record)}"
    msg = str(record[0].message)
    assert "WinMLEPRegistry" in msg or "session/2_coreloop.md" in msg
    # stacklevel=2 → the warning's filename must be this test file
    # (the user-visible caller frame), not winml.py itself.
    assert record[0].filename == __file__


def test_module_level_register_execution_providers_emits_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module-level wrapper emits DeprecationWarning(stacklevel=2)."""
    winml_mod = _winml_mod()
    _reset_winml_singleton()
    monkeypatch.setattr(winml_mod, "discover_all_eps", lambda **_: [])
    # Avoid the inner WinML.__init__ warning interfering — that's a separate
    # call site verified in the test above. We assert at least one matching
    # warning is emitted and that its caller frame is this file.

    fake_ort = SimpleNamespace(
        get_ep_devices=list,
        register_execution_provider_library=lambda *a, **kw: None,
        __name__="onnxruntime",
    )

    with (
        patch.dict(sys.modules, {"onnxruntime": fake_ort}),
        pytest.warns(DeprecationWarning, match="winml.modelkit.winml is deprecated") as record,
    ):
        winml_mod.register_execution_providers(ort=True, ort_genai=False)

    # The wrapper itself emits the warning at stacklevel=2 — verify a
    # warning attributed to this test file (the user-visible caller frame)
    # exists in the record. ``register_execution_providers`` may also fire
    # the inner ``WinML.__init__`` warning at a deeper stack frame; we
    # accept either pointer in the message.
    matching = [
        w for w in record
        if w.filename == __file__
        and (
            "WinMLEPRegistry" in str(w.message)
            or "session/2_coreloop.md" in str(w.message)
        )
    ]
    assert matching, (
        "Expected at least one DeprecationWarning attributed to the test "
        f"caller frame; got filenames: {[w.filename for w in record]}"
    )


def test_add_ep_for_device_emits_deprecation_warning() -> None:
    """``add_ep_for_device`` emits DeprecationWarning(stacklevel=2)."""
    winml_mod = _winml_mod()
    fake_ort = SimpleNamespace(get_ep_devices=list)

    with (
        patch.dict(sys.modules, {"onnxruntime": fake_ort}),
        pytest.warns(DeprecationWarning, match="winml.modelkit.winml is deprecated") as record,
    ):
        # Pass a dummy device_type — the for-loop is a no-op against the
        # empty ``get_ep_devices`` return so the call exits cleanly after
        # the warning fires.
        winml_mod.add_ep_for_device(
            session_options=object(),
            ep_name="QNNExecutionProvider",
            device_type=object(),
        )

    assert len(record) == 1, f"Expected exactly one warning; got {len(record)}"
    msg = str(record[0].message)
    assert "WinMLEPRegistry" in msg or "session/2_coreloop.md" in msg
    assert record[0].filename == __file__
