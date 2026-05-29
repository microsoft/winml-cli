# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``WinML.register_execution_providers`` extra_sources handling.

Specifically covers the C2 fix from the comprehensive review pass:
when a caller invokes ``register_execution_providers(extra_sources=...)``
twice with different overrides, the second call must NOT silently no-op
because the first call cached the EP name in ``self._registered_eps``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from winml.modelkit import ep_path as _ep
from winml.modelkit.ep_path import FilesystemSource
from winml.modelkit.winml import WinML


if TYPE_CHECKING:
    from pathlib import Path


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


@pytest.fixture
def isolated_winml(monkeypatch: pytest.MonkeyPatch) -> WinML:
    """Build a fresh WinML instance with cleared singleton + isolated EP_PATH."""
    # Clear the module-level singleton so each test gets its own instance.
    monkeypatch.setattr("winml.modelkit.winml._winml_instance", None)
    monkeypatch.setattr(_ep, "EP_PATH", [])
    monkeypatch.setattr(_ep, "_get_catalog", lambda: None)
    monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: None)
    monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
    return WinML()


def test_extra_sources_overrides_cached_registration_on_second_call(
    isolated_winml: WinML,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2: extra_sources's "highest precedence" must hold across multiple calls.

    The bug being prevented: first call resolves QNN to path A and caches
    "QNNExecutionProvider" in ``self._registered_eps["onnxruntime"]``.
    Second call with a different extra_source resolving QNN to path B
    used to be silently skipped because the EP name was already in the
    cached list. Fix: bypass the cache when extra_sources is supplied.
    """
    dll_a = _touch(tmp_path / "a" / "qnn.dll")
    dll_b = _touch(tmp_path / "b" / "qnn.dll")
    src_a = FilesystemSource(
        root=tmp_path / "a",
        dll_patterns={"QNNExecutionProvider": "qnn.dll"},
    )
    src_b = FilesystemSource(
        root=tmp_path / "b",
        dll_patterns={"QNNExecutionProvider": "qnn.dll"},
    )

    fake_ort = MagicMock()
    fake_ort.__name__ = "onnxruntime"
    fake_ort.register_execution_provider_library = MagicMock()
    monkeypatch.setitem(
        __import__("sys").modules, "onnxruntime", fake_ort
    )

    # First call: registers QNN with path A.
    isolated_winml.register_execution_providers(
        ort=True, extra_sources=[src_a]
    )
    first_calls = list(fake_ort.register_execution_provider_library.call_args_list)
    assert len(first_calls) == 1
    assert first_calls[0].args == ("QNNExecutionProvider", str(dll_a.resolve()))

    # Second call: with a different extra_source. MUST re-register, not no-op.
    isolated_winml.register_execution_providers(
        ort=True, extra_sources=[src_b]
    )
    all_calls = fake_ort.register_execution_provider_library.call_args_list
    assert len(all_calls) == 2, (
        "Second register_execution_providers(extra_sources=...) call "
        "silently no-op'd despite a different override path. The cache "
        "bypass for extra_sources is broken."
    )
    assert all_calls[1].args == ("QNNExecutionProvider", str(dll_b.resolve()))


def test_no_extra_sources_still_uses_cache(
    isolated_winml: WinML,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without extra_sources, the dedup cache still applies (preserves singleton semantics)."""
    fake_ort = MagicMock()
    fake_ort.__name__ = "onnxruntime"
    fake_ort.register_execution_provider_library = MagicMock()
    monkeypatch.setitem(
        __import__("sys").modules, "onnxruntime", fake_ort
    )

    # Pre-populate the registered cache as if a previous call ran.
    isolated_winml._registered_eps["onnxruntime"] = ["QNNExecutionProvider"]
    isolated_winml._ep_paths = {"QNNExecutionProvider": "/cached/path/qnn.dll"}

    isolated_winml.register_execution_providers(ort=True)
    # Cache says it's already registered → no re-register call.
    assert fake_ort.register_execution_provider_library.call_count == 0
