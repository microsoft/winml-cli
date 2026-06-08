# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Detailed unit tests for ``WinMLCatalogSource`` and the ``_get_catalog`` singleton.

These tests inject a fake WinAppSDK ML Python binding into ``sys.modules``
to exercise every branch of ``WinMLCatalogSource.resolve()`` without
requiring the optional ``winml-catalog`` extra to be installed.

The ONLY mocked surface is the WinAppSDK Python binding itself
(``winui3.microsoft.windows.ai.machinelearning`` and
``winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap``)
— per CLAUDE.md / project test conventions, no mocks for ``importlib.metadata``,
``pathlib``, etc.

Also covers:
    - The default EP source list includes the 5 ``WinMLCatalogSource`` rows
      with the canonical EP names from the design doc.
    - ``atexit`` cleanup is registered exactly once across many
      ``_get_catalog()`` calls.
"""

from __future__ import annotations

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
# Fake WinAppSDK binding helpers.
# ---------------------------------------------------------------------------


class _FakeReadyState:
    """Mimic the WinAppSDK ML ``ProviderReadyState`` enum."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeStatus:
    """Mimic the WinAppSDK ML ``EnsureReadyResult.status`` enum."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeAsyncOp:
    """Mimic the WinAppSDK ML async-op object returned by ensure_ready_async."""

    def __init__(self, status: str, *, raises: Exception | None = None) -> None:
        self._status = status
        self._raises = raises

    def get(self) -> Any:
        if self._raises is not None:
            raise self._raises
        return types.SimpleNamespace(status=_FakeStatus(self._status))


class _FakeProvider:
    """Mimic the WinAppSDK ML ``ExecutionProvider`` row."""

    def __init__(
        self,
        name: str,
        ready_state: str,
        library_path: str,
        *,
        status: str = "Success",
        ensure_ready_raises: Exception | None = None,
    ) -> None:
        self.name = name
        self.ready_state = _FakeReadyState(ready_state)
        self.library_path = library_path
        self._status = status
        self._ensure_ready_raises = ensure_ready_raises

    def ensure_ready_async(self) -> _FakeAsyncOp:
        return _FakeAsyncOp(self._status, raises=self._ensure_ready_raises)


class _FakeCatalog:
    """Mimic the WinAppSDK ML ``ExecutionProviderCatalog``."""

    def __init__(self, providers: list[_FakeProvider]) -> None:
        self._providers = providers

    def find_all_providers(self) -> list[_FakeProvider]:
        return list(self._providers)


def _build_fake_binding(catalog: _FakeCatalog | Exception) -> dict[str, types.ModuleType]:
    """Build the minimal module shape the lazy import needs.

    Returns a dict suitable for ``monkeypatch.setitem(sys.modules, ...)``.
    """
    # winui3.microsoft.windows.ai.machinelearning module:
    ml = types.ModuleType("winui3.microsoft.windows.ai.machinelearning")

    class _Catalog:
        @staticmethod
        def get_default() -> Any:
            if isinstance(catalog, Exception):
                raise catalog
            return catalog

    ml.ExecutionProviderCatalog = _Catalog  # type: ignore[attr-defined]

    # winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap:
    boot = types.ModuleType(
        "winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap"
    )

    class _Options:
        NONE = "NONE"
        ON_ERROR_DEBUG_BREAK = "ON_ERROR_DEBUG_BREAK"
        ON_ERROR_DEBUG_BREAK_IF_DEBUGGER_ATTACHED = "ON_ERROR_DEBUG_BREAK_IF_DEBUGGER_ATTACHED"
        ON_ERROR_FAIL_FAST = "ON_ERROR_FAIL_FAST"
        ON_NO_MATCH_SHOW_UI = "ON_NO_MATCH_SHOW_UI"
        ON_PACKAGE_IDENTITY_NOOP = "ON_PACKAGE_IDENTITY_NOOP"

    class _Handle:
        entered = 0
        exited = 0

        def __enter__(self) -> _Handle:
            type(self).entered += 1
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            type(self).exited += 1

    def _initialize(options: Any = None) -> _Handle:
        return _Handle()

    boot.InitializeOptions = _Options  # type: ignore[attr-defined]
    boot.initialize = _initialize  # type: ignore[attr-defined]
    boot._Handle = _Handle  # type: ignore[attr-defined]

    # Parent placeholder modules so ``import winui3.microsoft...`` can
    # walk the package chain.
    parents: list[tuple[str, types.ModuleType]] = []
    name = "winui3.microsoft.windows.ai.machinelearning"
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        if parent_name not in sys.modules:
            parents.append((parent_name, types.ModuleType(parent_name)))

    boot_name = (
        "winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap"
    )
    parts = boot_name.split(".")
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        if parent_name not in sys.modules:
            parents.append((parent_name, types.ModuleType(parent_name)))

    out: dict[str, types.ModuleType] = dict(parents)
    out["winui3.microsoft.windows.ai.machinelearning"] = ml
    out[boot_name] = boot
    return out


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
        # live WinAppSDK ML 2.0.1 binding on Snapdragon X Elite —
        # find_all_providers() returns provider.name == "QNNExecutionProvider",
        # not the short "QNN" form used by older Microsoft Learn
        # supported-execution-providers tables.
        assert catalog_names == {
            "OpenVINOExecutionProvider",
            "QNNExecutionProvider",
            "VitisAIExecutionProvider",
            "MIGraphXExecutionProvider",
            "NvTensorRtRtxExecutionProvider",
        }

    def test_canonical_ep_names_match_design(self) -> None:
        # Each catalog entry must report exactly one canonical EP name,
        # and those names must match the camelCase canonical keys used
        # in EP_CATALOG.
        ep_names_from_catalog = {
            ep
            for s in _default_ep_sources()
            if isinstance(s, WinMLCatalogSource)
            for ep in s.eps
        }
        assert ep_names_from_catalog == {
            "OpenVINOExecutionProvider",
            "QNNExecutionProvider",
            "VitisAIExecutionProvider",
            "MIGraphXExecutionProvider",
            "NvTensorRtRtxExecutionProvider",
        }

    def test_pypi_sources_precede_catalog_entries(self) -> None:
        # Per the design's "list order is precedence" rule (line 230):
        # PyPI sources are more deterministic than MSIX, so they win.
        from winml.modelkit.ep_path import PyPISource

        sources = _default_ep_sources()
        first_catalog_idx = next(
            i for i, s in enumerate(sources) if isinstance(s, WinMLCatalogSource)
        )
        pypi_indices = [
            i for i, s in enumerate(sources) if isinstance(s, PyPISource)
        ]
        assert pypi_indices, "default EP source list must include PyPISource rows"
        assert max(pypi_indices) < first_catalog_idx


# ---------------------------------------------------------------------------
# Binding-missing path (DEBUG-once).
# ---------------------------------------------------------------------------


class TestBindingMissing:
    """When the WinAppSDK ML Python binding is not importable, resolve() yields nothing."""

    def test_yields_nothing_when_binding_missing(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Force the lazy import to fail by mapping the binding module to
        # ``None`` in sys.modules. Python's import machinery treats a
        # ``None`` entry in sys.modules as "module is known to be
        # unimportable" and raises ImportError on import.
        monkeypatch.setitem(
            sys.modules, "winui3.microsoft.windows.ai.machinelearning", None
        )
        source = WinMLCatalogSource(
            catalog_name="VitisAI", eps=("VitisAIExecutionProvider",)
        )
        with caplog.at_level(logging.DEBUG, logger="winml.modelkit.ep_path"):
            assert list(source.resolve()) == []
        # DEBUG-once semantics: the failure was logged at DEBUG level,
        # not WARN.
        debug_messages = [
            r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG
        ]
        assert any(
            "WinAppSDK ML Python binding not installed" in m for m in debug_messages
        )
        warn_messages = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert not warn_messages

    def test_subsequent_resolves_do_not_reattempt_import(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setitem(
            sys.modules, "winui3.microsoft.windows.ai.machinelearning", None
        )
        s1 = WinMLCatalogSource(catalog_name="VitisAI", eps=("VitisAIExecutionProvider",))
        s2 = WinMLCatalogSource(catalog_name="QNN", eps=("QNNExecutionProvider",))
        with caplog.at_level(logging.DEBUG, logger="winml.modelkit.ep_path"):
            assert list(s1.resolve()) == []
            assert list(s2.resolve()) == []
        # Only one DEBUG line about the missing binding (per-process cache).
        debug_messages = [
            r.getMessage()
            for r in caplog.records
            if "WinAppSDK ML Python binding not installed" in r.getMessage()
        ]
        assert len(debug_messages) == 1


# ---------------------------------------------------------------------------
# Successful catalog path with mocked binding.
# ---------------------------------------------------------------------------


class TestWithFakeCatalog:
    """Inject a fake WinAppSDK ML binding and exercise every resolve() branch."""

    def _install_binding(
        self, monkeypatch: pytest.MonkeyPatch, catalog: _FakeCatalog | Exception
    ) -> None:
        for name, mod in _build_fake_binding(catalog).items():
            monkeypatch.setitem(sys.modules, name, mod)

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
                    status="Success",
                ),
            ]
        )
        self._install_binding(monkeypatch, catalog)

        source = WinMLCatalogSource(
            catalog_name="VitisAI", eps=("VitisAIExecutionProvider",)
        )
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
        self._install_binding(monkeypatch, catalog)

        source = WinMLCatalogSource(
            catalog_name="VitisAI", eps=("VitisAIExecutionProvider",)
        )
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
        self._install_binding(monkeypatch, catalog)

        source = WinMLCatalogSource(
            catalog_name="MIGraphX", eps=("MIGraphXExecutionProvider",)
        )
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
                    status="Success",
                ),
            ]
        )
        self._install_binding(monkeypatch, catalog)

        source = WinMLCatalogSource(
            catalog_name="QNN", eps=("QNNExecutionProvider",)
        )
        assert list(source.resolve()) == []

    def test_non_success_status_warns_and_skips(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        catalog = _FakeCatalog(
            [
                _FakeProvider(
                    name="VitisAI",
                    ready_state="NotReady",
                    library_path=str(tmp_path / "v.dll"),
                    status="Failed",
                ),
            ]
        )
        self._install_binding(monkeypatch, catalog)

        source = WinMLCatalogSource(
            catalog_name="VitisAI", eps=("VitisAIExecutionProvider",)
        )
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.ep_path"):
            assert list(source.resolve()) == []
        warn_messages = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any(
            "non-Success status" in m for m in warn_messages
        ), warn_messages

    def test_ensure_ready_raises_warns_and_continues(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # First provider raises, second is good — the walk must continue.
        good_dll = tmp_path / "good.dll"
        good_dll.write_bytes(b"")
        catalog = _FakeCatalog(
            [
                _FakeProvider(
                    name="OpenVINO",
                    ready_state="Ready",
                    library_path="ignored",
                    ensure_ready_raises=RuntimeError("fake hardware missing"),
                ),
                _FakeProvider(
                    name="OpenVINO",
                    ready_state="Ready",
                    library_path=str(good_dll),
                    status="Success",
                ),
            ]
        )
        self._install_binding(monkeypatch, catalog)

        source = WinMLCatalogSource(
            catalog_name="OpenVINO", eps=("OpenVINOExecutionProvider",)
        )
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.ep_path"):
            results = list(source.resolve())
        # The good provider should still yield.
        assert len(results) == 1
        assert results[0].ep_name == "OpenVINOExecutionProvider"
        assert results[0].dll_path == Path(str(good_dll))

    def test_find_all_providers_raises_yields_nothing(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class _BadCatalog:
            def find_all_providers(self) -> list[Any]:
                raise RuntimeError("catalog query failed")

        # _build_fake_binding takes a real catalog or an exception; here
        # we want a catalog object whose method raises, so install
        # manually.
        for name, mod in _build_fake_binding(_FakeCatalog([])).items():
            monkeypatch.setitem(sys.modules, name, mod)
        ml = sys.modules["winui3.microsoft.windows.ai.machinelearning"]

        class _Catalog2:
            @staticmethod
            def get_default() -> Any:
                return _BadCatalog()

        monkeypatch.setattr(ml, "ExecutionProviderCatalog", _Catalog2)

        source = WinMLCatalogSource(catalog_name="QNN", eps=("QNNExecutionProvider",))
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.ep_path"):
            assert list(source.resolve()) == []
        warn_messages = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("find_all_providers" in m for m in warn_messages)

    def test_get_default_raises_yields_nothing(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        self._install_binding(monkeypatch, RuntimeError("get_default boom"))
        source = WinMLCatalogSource(catalog_name="QNN", eps=("QNNExecutionProvider",))
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.ep_path"):
            assert list(source.resolve()) == []
        warn_messages = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert any("get_default" in m for m in warn_messages)


# ---------------------------------------------------------------------------
# atexit cleanup.
# ---------------------------------------------------------------------------


class TestAtexitCleanup:
    """The bootstrap handle is registered for cleanup exactly once."""

    def test_atexit_registered_once_across_multiple_calls(
        self,
        reset_catalog_singleton: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Track atexit.register calls within ep_path.
        registered: list[Any] = []

        def fake_register(func: Any, *args: Any, **kwargs: Any) -> Any:
            registered.append((func, args, kwargs))
            return func

        monkeypatch.setattr(_ep.atexit, "register", fake_register)

        # Install a working fake binding.
        dll = tmp_path / "x.dll"
        dll.write_bytes(b"")
        catalog = _FakeCatalog(
            [
                _FakeProvider(
                    name="VitisAI",
                    ready_state="Ready",
                    library_path=str(dll),
                    status="Success",
                ),
            ]
        )
        for name, mod in _build_fake_binding(catalog).items():
            monkeypatch.setitem(sys.modules, name, mod)

        # First call — initializes and registers.
        c1 = _ep._get_catalog()
        # Subsequent calls — return cached singleton, no re-register.
        c2 = _ep._get_catalog()
        c3 = _ep._get_catalog()
        assert c1 is not None
        assert c2 is c1
        assert c3 is c1
        # Exactly one atexit registration.
        cleanup_callbacks = [
            r for r in registered if r[0] is _ep._release_winml_handle
        ]
        assert len(cleanup_callbacks) == 1

    def test_release_handle_swallows_exceptions(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Cleanup must not propagate exceptions during interpreter shutdown.
        class _BoomHandle:
            def __exit__(self, *args: Any) -> None:
                raise RuntimeError("cleanup failure")

        # Should not raise.
        _ep._release_winml_handle(_BoomHandle())
