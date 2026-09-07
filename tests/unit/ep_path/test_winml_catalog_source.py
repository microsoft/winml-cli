# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Detailed unit tests for ``WinMLCatalogSource`` and the ``_get_catalog`` singleton.

These tests inject a fake ``windowsml`` module into ``sys.modules`` to
exercise every branch of ``WinMLCatalogSource.resolve()`` without
requiring the ``windowsml`` package to be installed.

The ONLY mocked surface is the ``windowsml`` Python package itself
— per CLAUDE.md / project test conventions, no mocks for
``importlib.metadata``, ``pathlib``, etc.

Also covers:
    - The default EP source list includes the 5 ``WinMLCatalogSource`` rows
      with the canonical EP names from the design doc.
    - ``_get_catalog()`` disarms the native catalog handle at process exit
      without calling into native release during interpreter shutdown.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest


if TYPE_CHECKING:
    from collections.abc import Iterator

from winml.modelkit import ep_path as _ep
from winml.modelkit.ep_path import (
    WinMLCatalogSource,
    _default_ep_sources,
)


# ---------------------------------------------------------------------------
# Fake windowsml binding helpers.
# ---------------------------------------------------------------------------


class _FakeReadyState:
    """Mimic the ``windowsml`` provider ready-state value."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeReadyOp:
    """Mimic the async op handle returned by ``ensure_ready_async``."""

    def __init__(self) -> None:
        self.get_status_calls = 0
        self.cancel_calls = 0
        self.close_calls = 0

    def get_status(self) -> None:
        self.get_status_calls += 1

    def cancel(self) -> None:
        self.cancel_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _FakeProvider:
    """Mimic a ``windowsml`` execution-provider row."""

    def __init__(
        self,
        name: str,
        ready_state: str,
        library_path: str,
        *,
        ensure_ready_raises: Exception | None = None,
        becomes_ready: bool = True,
        version: str = "",
        package_family_name: str = "",
    ) -> None:
        self.name = name
        self.ready_state = _FakeReadyState(ready_state)
        self.library_path = library_path
        self.ensure_ready_calls = 0
        self._ensure_ready_raises = ensure_ready_raises
        self._becomes_ready = becomes_ready
        self.version = version
        self.package_family_name = package_family_name

    def ensure_ready_async(
        self,
        on_complete: Any = None,
        on_progress: Any = None,
    ) -> _FakeReadyOp:
        # The cold download path: drive a progress callback, (optionally) flip
        # to Ready, then signal completion — mirroring the real windowsml
        # async contract that ``_ensure_provider_ready`` consumes.
        self.ensure_ready_calls += 1
        if self._ensure_ready_raises is not None:
            raise self._ensure_ready_raises
        if on_progress is not None:
            on_progress(1.0)
        if self._becomes_ready:
            self.ready_state = _FakeReadyState("Ready")
        if on_complete is not None:
            on_complete()
        return _FakeReadyOp()


class _FakeCatalog:
    """Mimic ``windowsml.EpCatalog``."""

    def __init__(
        self,
        providers: list[_FakeProvider],
        *,
        find_raises: Exception | None = None,
    ) -> None:
        self._providers = providers
        self._find_raises = find_raises
        self.closed = False
        self._handle: object | None = object()

    def find_all_providers(self) -> list[_FakeProvider]:
        if self._find_raises is not None:
            raise self._find_raises
        return list(self._providers)

    def close(self) -> None:
        self.closed = True


def _install_windowsml_module(
    monkeypatch: pytest.MonkeyPatch,
    catalog: _FakeCatalog | Exception,
) -> None:
    module = types.ModuleType("windowsml")

    class _EpCatalog:
        def __new__(cls) -> _FakeCatalog:
            if isinstance(catalog, Exception):
                raise catalog
            return catalog

    module.EpCatalog = _EpCatalog  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "windowsml", module)


@pytest.fixture
def reset_catalog_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Reset the catalog singleton and warn-once cache around each test.

    The catalog is memoized via ``functools.cache``. Clear it both before
    and after the test so a cached value (or cached ``None`` from a fake
    binding) cannot leak between tests.
    """
    _ep._get_catalog.cache_clear()
    monkeypatch.setattr(_ep, "_winml_catalog_warned_keys", set())
    try:
        yield
    finally:
        _ep._get_catalog.cache_clear()


# ---------------------------------------------------------------------------
# Default EP source list shape.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="WinMLCatalogSource entries are Windows-only")
class TestDefaultEpPathIncludesCatalogEntries:
    """The default EP source list must include the 5 catalog rows from the design doc."""

    def test_five_winml_catalog_entries(self) -> None:
        catalog_entries = [s for s in _default_ep_sources() if isinstance(s, WinMLCatalogSource)]
        assert len(catalog_entries) == 5

    def test_canonical_catalog_names_match_design(self) -> None:
        catalog_names = {
            s.catalog_name for s in _default_ep_sources() if isinstance(s, WinMLCatalogSource)
        }
        # The catalog API returns provider.name as the full canonical EP
        # name (e.g. "QNNExecutionProvider"), so catalog_name in the
        # default source list must match. Verified empirically against the
        # live windowsml 2.0.300 binding on Snapdragon X Elite —
        # find_all_providers() returns provider.name == "QNNExecutionProvider",
        # not the short "QNN" form used by older Microsoft Learn
        # supported-execution-providers tables.
        assert catalog_names == {
            "OpenVINOExecutionProvider",
            "QNNExecutionProvider",
            "VitisAIExecutionProvider",
            "MIGraphXExecutionProvider",
            "NvTensorRTRTXExecutionProvider",
        }

    def test_canonical_ep_names_match_design(self) -> None:
        # Each catalog entry must report exactly one canonical EP name,
        # and those names must match the camelCase canonical keys used
        # in EP_CATALOG.
        ep_names_from_catalog = {
            ep for s in _default_ep_sources() if isinstance(s, WinMLCatalogSource) for ep in s.eps
        }
        assert ep_names_from_catalog == {
            "OpenVINOExecutionProvider",
            "QNNExecutionProvider",
            "VitisAIExecutionProvider",
            "MIGraphXExecutionProvider",
            "NvTensorRTRTXExecutionProvider",
        }

    def test_pypi_sources_precede_catalog_entries(self) -> None:
        # Per the design's "list order is precedence" rule (line 230):
        # PyPI sources are more deterministic than MSIX, so they win.
        from winml.modelkit.ep_path import PyPISource

        sources = _default_ep_sources()
        first_catalog_idx = next(
            i for i, s in enumerate(sources) if isinstance(s, WinMLCatalogSource)
        )
        pypi_indices = [i for i, s in enumerate(sources) if isinstance(s, PyPISource)]
        assert pypi_indices, "default EP source list must include PyPISource rows"
        assert max(pypi_indices) < first_catalog_idx


# ---------------------------------------------------------------------------
# Binding-missing path (DEBUG-once).
# ---------------------------------------------------------------------------


class TestBindingMissing:
    """When the windowsml package is not importable, resolve() yields nothing."""

    def test_yields_nothing_when_binding_missing(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Force the lazy import to fail by mapping the module to ``None``
        # in sys.modules. Python's import machinery treats a ``None`` entry
        # as "module is known to be unimportable" and raises ImportError.
        monkeypatch.setitem(sys.modules, "windowsml", None)
        source = WinMLCatalogSource(catalog_name="VitisAI", eps=("VitisAIExecutionProvider",))
        with caplog.at_level(logging.DEBUG, logger="winml.modelkit.ep_path"):
            assert list(source.resolve()) == []
        # DEBUG-once semantics: the failure was logged at DEBUG level,
        # not WARN.
        debug_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("windowsml package is not installed" in m for m in debug_messages)
        warn_messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warn_messages

    def test_subsequent_resolves_do_not_reattempt_import(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setitem(sys.modules, "windowsml", None)
        s1 = WinMLCatalogSource(catalog_name="VitisAI", eps=("VitisAIExecutionProvider",))
        s2 = WinMLCatalogSource(catalog_name="QNN", eps=("QNNExecutionProvider",))
        with caplog.at_level(logging.DEBUG, logger="winml.modelkit.ep_path"):
            assert list(s1.resolve()) == []
            assert list(s2.resolve()) == []
        # Only one DEBUG line about the missing binding (per-process cache).
        debug_messages = [
            r.getMessage()
            for r in caplog.records
            if "windowsml package is not installed" in r.getMessage()
        ]
        assert len(debug_messages) == 1


# ---------------------------------------------------------------------------
# Successful catalog path with mocked binding.
# ---------------------------------------------------------------------------


class TestWithFakeCatalog:
    """Inject a fake windowsml module and exercise every resolve() branch."""

    def test_yields_for_ready_provider(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        dll = tmp_path / "vitisai.dll"
        dll.write_bytes(b"")
        catalog = _FakeCatalog(
            [
                _FakeProvider(
                    name="VitisAI",
                    ready_state="Ready",
                    library_path=str(dll),
                ),
            ]
        )
        _install_windowsml_module(monkeypatch, catalog)

        source = WinMLCatalogSource(catalog_name="VitisAI", eps=("VitisAIExecutionProvider",))
        results = list(source.resolve())
        assert len(results) == 1
        entry = results[0]
        assert entry.ep_name == "VitisAIExecutionProvider"
        assert entry.dll_path == Path(str(dll))
        # OQ-2 deferral: WinMLCatalogSource currently yields version=None.
        assert entry.version is None

    def test_provider_name_mismatch_skipped(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        dll = tmp_path / "qnn.dll"
        dll.write_bytes(b"")
        catalog = _FakeCatalog(
            [
                _FakeProvider(
                    name="QNN",
                    ready_state="Ready",
                    library_path=str(dll),
                ),
            ]
        )
        _install_windowsml_module(monkeypatch, catalog)

        source = WinMLCatalogSource(catalog_name="VitisAI", eps=("VitisAIExecutionProvider",))
        # No provider with name "VitisAI" -> nothing yielded.
        assert list(source.resolve()) == []

    def test_not_present_provider_skipped(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        catalog = _FakeCatalog(
            [
                _FakeProvider(
                    name="MIGraphX",
                    ready_state="NotPresent",
                    library_path="",
                ),
            ]
        )
        _install_windowsml_module(monkeypatch, catalog)

        source = WinMLCatalogSource(catalog_name="MIGraphX", eps=("MIGraphXExecutionProvider",))
        # NotPresent providers are skipped by default (auto_download=False).
        assert list(source.resolve()) == []

    def test_empty_library_path_skipped(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        catalog = _FakeCatalog(
            [
                _FakeProvider(
                    name="QNN",
                    ready_state="Ready",
                    library_path="",
                ),
            ]
        )
        _install_windowsml_module(monkeypatch, catalog)

        source = WinMLCatalogSource(catalog_name="QNN", eps=("QNNExecutionProvider",))
        assert list(source.resolve()) == []

    def test_ensure_ready_leaves_not_ready_warns_and_skips(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Provider whose download completes but never flips to Ready.
        class _StuckProvider:
            name = "VitisAI"
            ready_state = _FakeReadyState("NotReady")
            library_path = str(tmp_path / "v.dll")
            version = ""
            package_family_name = ""

            def ensure_ready_async(
                self, on_complete: Any = None, on_progress: Any = None
            ) -> _FakeReadyOp:
                # Completes the async op but intentionally does NOT update
                # ready_state, so the caller must warn-and-skip.
                if on_progress is not None:
                    on_progress(1.0)
                if on_complete is not None:
                    on_complete()
                return _FakeReadyOp()

        catalog = _FakeCatalog([_StuckProvider()])  # type: ignore[list-item]
        _install_windowsml_module(monkeypatch, catalog)

        source = WinMLCatalogSource(catalog_name="VitisAI", eps=("VitisAIExecutionProvider",))
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.ep_path"):
            assert list(source.resolve()) == []
        warn_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("ensure_ready left provider in state" in m for m in warn_messages), warn_messages

    def test_ensure_ready_raises_warns_and_continues(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # First provider raises during ensure_ready, second is already Ready
        # — the walk must continue past the first failure.
        good_dll = tmp_path / "good.dll"
        good_dll.write_bytes(b"")
        catalog = _FakeCatalog(
            [
                _FakeProvider(
                    name="OpenVINO",
                    ready_state="NotReady",
                    library_path="ignored",
                    ensure_ready_raises=RuntimeError("fake hardware missing"),
                ),
                _FakeProvider(
                    name="OpenVINO",
                    ready_state="Ready",
                    library_path=str(good_dll),
                ),
            ]
        )
        _install_windowsml_module(monkeypatch, catalog)

        source = WinMLCatalogSource(catalog_name="OpenVINO", eps=("OpenVINOExecutionProvider",))
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.ep_path"):
            results = list(source.resolve())
        # The good provider should still yield.
        assert len(results) == 1
        assert results[0].ep_name == "OpenVINOExecutionProvider"
        assert results[0].dll_path == Path(str(good_dll))
        warn_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("ensure_ready raised" in m for m in warn_messages), warn_messages

    def test_find_all_providers_raises_yields_nothing(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        catalog = _FakeCatalog(
            [],
            find_raises=RuntimeError("catalog query failed"),
        )
        _install_windowsml_module(monkeypatch, catalog)

        source = WinMLCatalogSource(catalog_name="QNN", eps=("QNNExecutionProvider",))
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.ep_path"):
            assert list(source.resolve()) == []
        warn_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("find_all_providers" in m for m in warn_messages)

    def test_epcatalog_constructor_raises_yields_nothing(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _install_windowsml_module(monkeypatch, RuntimeError("EpCatalog() failed"))
        source = WinMLCatalogSource(catalog_name="QNN", eps=("QNNExecutionProvider",))
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.ep_path"):
            assert list(source.resolve()) == []
        warn_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("EpCatalog()" in m for m in warn_messages)


# ---------------------------------------------------------------------------
# Readiness and lifecycle expectations (async progress-driven download API).
# ---------------------------------------------------------------------------


def test_not_ready_provider_is_prepared_and_yielded(
    reset_catalog_singleton: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dll = tmp_path / "qnn.dll"
    dll.write_bytes(b"")
    provider = _FakeProvider("QNNExecutionProvider", "NotReady", str(dll))
    _install_windowsml_module(monkeypatch, _FakeCatalog([provider]))

    entries = list(
        WinMLCatalogSource(
            catalog_name="QNNExecutionProvider",
            eps=("QNNExecutionProvider",),
        ).resolve()
    )

    assert provider.ensure_ready_calls == 1
    assert [entry.dll_path for entry in entries] == [dll]


def test_ready_provider_is_not_prepared_again(
    reset_catalog_singleton: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dll = tmp_path / "qnn.dll"
    dll.write_bytes(b"")
    provider = _FakeProvider("QNNExecutionProvider", "Ready", str(dll))
    _install_windowsml_module(monkeypatch, _FakeCatalog([provider]))

    entries = list(
        WinMLCatalogSource(
            catalog_name="QNNExecutionProvider",
            eps=("QNNExecutionProvider",),
        ).resolve()
    )

    assert provider.ensure_ready_calls == 0
    assert len(entries) == 1


def test_not_present_provider_is_not_downloaded_by_default(
    reset_catalog_singleton: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dll = tmp_path / "qnn.dll"
    dll.write_bytes(b"")
    provider = _FakeProvider("QNNExecutionProvider", "NotPresent", str(dll))
    catalog = _FakeCatalog([provider])
    _install_windowsml_module(monkeypatch, catalog)

    default_source = WinMLCatalogSource(
        catalog_name="QNNExecutionProvider",
        eps=("QNNExecutionProvider",),
    )
    assert list(default_source.resolve()) == []
    assert provider.ensure_ready_calls == 0


def test_not_present_provider_downloads_with_opt_in(
    reset_catalog_singleton: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dll = tmp_path / "qnn.dll"
    dll.write_bytes(b"")
    provider = _FakeProvider("QNNExecutionProvider", "NotPresent", str(dll))
    _install_windowsml_module(monkeypatch, _FakeCatalog([provider]))

    download_source = WinMLCatalogSource(
        catalog_name="QNNExecutionProvider",
        eps=("QNNExecutionProvider",),
        auto_download=True,
    )
    assert len(list(download_source.resolve())) == 1
    assert provider.ensure_ready_calls == 1


def test_explicit_catalog_resolution_downloads_requested_provider(
    reset_catalog_singleton: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dll = tmp_path / "qnn.dll"
    dll.write_bytes(b"")
    provider = _FakeProvider("QNNExecutionProvider", "NotPresent", str(dll))
    _install_windowsml_module(monkeypatch, _FakeCatalog([provider]))

    entries = _ep._resolve_requested_winml_catalog_ep("QNNExecutionProvider")

    assert provider.ensure_ready_calls == 1
    assert [entry.dll_path for entry in entries] == [dll]
    assert all(
        isinstance(entry.source, WinMLCatalogSource) and entry.source.auto_download
        for entry in entries
    )


# ---------------------------------------------------------------------------
# _is_ready helper.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (_FakeReadyState("Ready"), True),
        (_FakeReadyState("READY"), True),
        (_FakeReadyState("NotReady"), False),
        (_FakeReadyState("NOT_READY"), False),
        (_FakeReadyState("NotPresent"), False),
        (None, False),
    ],
)
def test_is_ready(value: Any, expected: bool) -> None:
    assert WinMLCatalogSource._is_ready(value) is expected


# ---------------------------------------------------------------------------
# catalog lifecycle.
# ---------------------------------------------------------------------------


class TestCatalogLifecycle:
    """The catalog singleton avoids process-exit native release calls."""

    def test_get_catalog_registers_atexit_handle_disarm_without_close(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        registered: list[Any] = []

        def fake_register(func: Any, *args: Any, **kwargs: Any) -> Any:
            registered.append((func, args, kwargs))
            return func

        monkeypatch.setattr(atexit, "register", fake_register)

        # Install a working fake module.
        dll = tmp_path / "x.dll"
        dll.write_bytes(b"")
        catalog = _FakeCatalog(
            [
                _FakeProvider(
                    name="VitisAI",
                    ready_state="Ready",
                    library_path=str(dll),
                ),
            ]
        )
        _install_windowsml_module(monkeypatch, catalog)

        # First call — initializes and registers.
        c1 = _ep._get_catalog()
        # Subsequent calls — return cached singleton, no re-register.
        c2 = _ep._get_catalog()
        c3 = _ep._get_catalog()
        assert c1 is not None
        assert c2 is c1
        assert c3 is c1
        assert len(registered) == 1

        cleanup, args, kwargs = registered[0]
        assert args == (catalog,)
        assert kwargs == {}
        cleanup(*args, **kwargs)

        assert catalog._handle is None
        assert catalog.closed is False
