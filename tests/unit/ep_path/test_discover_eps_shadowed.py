# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``discover_all_eps()`` and the default ``discover_eps()`` shape.

The default ``discover_eps()`` shape (one (path, source) per EP) is
also covered by ``test_ep_path.py``. This file focuses on
``discover_all_eps()`` and the ``ResolvedEp`` ordering rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from pathlib import Path

from winml.modelkit import ep_path as _ep
from winml.modelkit.ep_path import (
    EpSource,
    FilesystemSource,
    ResolvedEp,
    discover_all_eps,
    discover_eps,
)


@pytest.fixture(autouse=True)
def _isolate_default_ep_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace _default_ep_sources and skip live catalog/MSIX scans for every test here."""
    monkeypatch.setattr(_ep, "_default_ep_sources", list)
    monkeypatch.setattr(_ep, "_get_catalog", lambda: None)
    monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: None)
    monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _filesystem_source_for(root: Path, ep: str, dll_name: str) -> FilesystemSource:
    return FilesystemSource(root=root, dll_patterns={ep: dll_name})


# ---------------------------------------------------------------------------
# discover_all_eps() semantics.
# ---------------------------------------------------------------------------


class TestDiscoverAllEpsFormerReturnShadowed:
    """``discover_all_eps()`` returns dict[str, list[ResolvedEp]]."""

    def test_single_source_per_ep_one_primary(self, tmp_path: Path) -> None:
        dll = _touch(tmp_path / "qnn.dll")
        src = _filesystem_source_for(tmp_path, "QNNExecutionProvider", dll.name)
        result = discover_all_eps(extra_sources=[src])

        assert isinstance(result, dict)
        assert "QNNExecutionProvider" in result
        entries = result["QNNExecutionProvider"]
        assert len(entries) == 1
        assert isinstance(entries[0], ResolvedEp)
        assert entries[0].status == "primary"

    def test_two_sources_one_ep_yields_primary_plus_shadowed(
        self, tmp_path: Path
    ) -> None:
        # Both sources resolve QNN; first wins (primary), second is shadowed.
        dll_a = _touch(tmp_path / "a" / "qnn.dll")
        dll_b = _touch(tmp_path / "b" / "qnn.dll")
        src_a = _filesystem_source_for(tmp_path / "a", "QNNExecutionProvider", "qnn.dll")
        src_b = _filesystem_source_for(tmp_path / "b", "QNNExecutionProvider", "qnn.dll")

        result = discover_all_eps(extra_sources=[src_a, src_b])
        entries = result["QNNExecutionProvider"]
        assert len(entries) == 2
        assert entries[0].status == "primary"
        assert entries[0].dll_path == dll_a.resolve()
        assert entries[1].status == "shadowed"
        assert entries[1].dll_path == dll_b.resolve()

    def test_three_sources_one_primary_two_shadowed(self, tmp_path: Path) -> None:
        for sub in ("a", "b", "c"):
            _touch(tmp_path / sub / "qnn.dll")
        srcs: list[EpSource] = [
            _filesystem_source_for(tmp_path / sub, "QNNExecutionProvider", "qnn.dll")
            for sub in ("a", "b", "c")
        ]
        result = discover_all_eps(extra_sources=srcs)
        entries = result["QNNExecutionProvider"]
        assert len(entries) == 3
        assert [e.status for e in entries] == ["primary", "shadowed", "shadowed"]

    def test_multiple_eps_grouped_correctly(self, tmp_path: Path) -> None:
        _touch(tmp_path / "qnn.dll")
        _touch(tmp_path / "ov.dll")
        srcs: list[EpSource] = [
            _filesystem_source_for(tmp_path, "QNNExecutionProvider", "qnn.dll"),
            _filesystem_source_for(tmp_path, "OpenVINOExecutionProvider", "ov.dll"),
        ]
        result = discover_all_eps(extra_sources=srcs)
        assert set(result.keys()) == {"QNNExecutionProvider", "OpenVINOExecutionProvider"}
        for entries in result.values():
            assert len(entries) == 1
            assert entries[0].status == "primary"

    def test_empty_inputs_yield_empty_dict(self) -> None:
        result = discover_all_eps()
        assert result == {}

    def test_extra_sources_takes_precedence_over_ep_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _touch(tmp_path / "extra" / "qnn.dll")
        _touch(tmp_path / "default" / "qnn.dll")
        extra = _filesystem_source_for(
            tmp_path / "extra", "QNNExecutionProvider", "qnn.dll"
        )
        default = _filesystem_source_for(
            tmp_path / "default", "QNNExecutionProvider", "qnn.dll"
        )
        monkeypatch.setattr(_ep, "_default_ep_sources", lambda: [default])

        result = discover_all_eps(extra_sources=[extra])
        entries = result["QNNExecutionProvider"]
        assert len(entries) == 2
        assert "extra" in str(entries[0].dll_path)  # primary
        assert "default" in str(entries[1].dll_path)  # shadowed

    # -----------------------------------------------------------------------
    # extra_sources_after — the load-bearing kwarg for `winml sys --list-ep`.
    # MUST appear AFTER the default EP source list precedence-wise so
    # injected MSIX entries don't artificially override the user's normal
    # precedence. Coverage added per review C-4.
    # -----------------------------------------------------------------------

    def test_extra_sources_after_appears_after_ep_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _touch(tmp_path / "default" / "qnn.dll")
        _touch(tmp_path / "after" / "qnn.dll")
        default = _filesystem_source_for(
            tmp_path / "default", "QNNExecutionProvider", "qnn.dll"
        )
        after = _filesystem_source_for(
            tmp_path / "after", "QNNExecutionProvider", "qnn.dll"
        )
        monkeypatch.setattr(_ep, "_default_ep_sources", lambda: [default])

        result = discover_all_eps(extra_sources_after=[after])
        entries = result["QNNExecutionProvider"]
        assert len(entries) == 2
        # Default source list wins primary; extra_sources_after lands shadowed.
        assert "default" in str(entries[0].dll_path)
        assert entries[0].status == "primary"
        assert "after" in str(entries[1].dll_path)
        assert entries[1].status == "shadowed"

    def test_extra_sources_after_does_not_promote_to_primary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # When BOTH extra_sources (prepended) AND extra_sources_after
        # (appended) provide the same EP, precedence order is:
        #   extra_sources -> default EP sources -> extra_sources_after.
        _touch(tmp_path / "before" / "qnn.dll")
        _touch(tmp_path / "default" / "qnn.dll")
        _touch(tmp_path / "after" / "qnn.dll")
        before = _filesystem_source_for(
            tmp_path / "before", "QNNExecutionProvider", "qnn.dll"
        )
        default = _filesystem_source_for(
            tmp_path / "default", "QNNExecutionProvider", "qnn.dll"
        )
        after = _filesystem_source_for(
            tmp_path / "after", "QNNExecutionProvider", "qnn.dll"
        )
        monkeypatch.setattr(_ep, "_default_ep_sources", lambda: [default])

        result = discover_all_eps(
            extra_sources=[before],
            extra_sources_after=[after],
        )
        entries = result["QNNExecutionProvider"]
        assert len(entries) == 3
        statuses = [e.status for e in entries]
        assert statuses == ["primary", "shadowed", "shadowed"]
        assert "before" in str(entries[0].dll_path)
        assert "default" in str(entries[1].dll_path)
        assert "after" in str(entries[2].dll_path)

    def test_extra_sources_after_alone_yields_primary_when_ep_path_empty(
        self,
        tmp_path: Path,
    ) -> None:
        # Autouse fixture sets _default_ep_sources=[]; only extra_sources_after
        # provides an EP. That EP becomes primary by default (no other source
        # competes).
        _touch(tmp_path / "qnn.dll")
        only = _filesystem_source_for(tmp_path, "QNNExecutionProvider", "qnn.dll")
        result = discover_all_eps(extra_sources_after=[only])
        entries = result["QNNExecutionProvider"]
        assert len(entries) == 1
        assert entries[0].status == "primary"

    # NOTE: an earlier draft asserted that two sources spelling the same EP
    # with different casing (NVIDIA's PascalCase vs canonical camelCase)
    # would collapse into one bucket. That alias-normalization layer was
    # intentionally removed — sources are now matched by exact EP-name
    # string. The "test_canonicalization_collapses_aliases" check used to
    # live here; deleted because it asserted behavior the design rejects.


# ---------------------------------------------------------------------------
# Default shape: discover_eps() returns (path, source) per EP.
# ---------------------------------------------------------------------------


class TestDefaultShape:
    """``discover_eps()`` (default) returns dict[str, (path, source)]."""

    def test_default_returns_legacy_shape(self, tmp_path: Path) -> None:
        dll = _touch(tmp_path / "qnn.dll")
        src = _filesystem_source_for(tmp_path, "QNNExecutionProvider", dll.name)
        result = discover_eps(extra_sources=[src])
        # Legacy: one entry per name, value is (path, source) tuple.
        assert isinstance(result, dict)
        path, source = result["QNNExecutionProvider"]
        assert path == dll.resolve()
        assert source is src

    def test_default_drops_shadowed_entries(self, tmp_path: Path) -> None:
        _touch(tmp_path / "a" / "qnn.dll")
        _touch(tmp_path / "b" / "qnn.dll")
        srcs = [
            _filesystem_source_for(tmp_path / "a", "QNNExecutionProvider", "qnn.dll"),
            _filesystem_source_for(tmp_path / "b", "QNNExecutionProvider", "qnn.dll"),
        ]
        result = discover_eps(extra_sources=srcs)
        # One entry only; the b/ source is shadowed and not in legacy shape.
        path, source = result["QNNExecutionProvider"]
        assert "a" in str(path)
        assert source is srcs[0]


# ---------------------------------------------------------------------------
# discover_all_eps() — new dedicated function; additional coverage.
# ---------------------------------------------------------------------------


class TestDiscoverAllEps:
    """``discover_all_eps()`` returns dict[str, list[ResolvedEp]] with primary + shadowed."""

    def test_discover_all_eps_returns_full_shape(self, tmp_path: Path) -> None:
        """discover_all_eps() returns dict[str, list[ResolvedEp]]."""
        dll_a = _touch(tmp_path / "a" / "qnn.dll")
        dll_b = _touch(tmp_path / "b" / "qnn.dll")
        src_a = _filesystem_source_for(tmp_path / "a", "QNNExecutionProvider", "qnn.dll")
        src_b = _filesystem_source_for(tmp_path / "b", "QNNExecutionProvider", "qnn.dll")

        result = discover_all_eps(extra_sources=[src_a, src_b])

        assert isinstance(result, dict)
        assert "QNNExecutionProvider" in result
        entries = result["QNNExecutionProvider"]
        assert isinstance(entries, list)
        assert all(isinstance(r, ResolvedEp) for r in entries)
        assert len(entries) == 2
        assert entries[0].status == "primary"
        assert entries[0].dll_path == dll_a.resolve()
        assert entries[1].status == "shadowed"
        assert entries[1].dll_path == dll_b.resolve()

    def test_discover_all_eps_empty_yields_empty_dict(self) -> None:
        result = discover_all_eps()
        assert result == {}

    def test_discover_all_eps_single_source_per_ep(self, tmp_path: Path) -> None:
        dll = _touch(tmp_path / "qnn.dll")
        src = _filesystem_source_for(tmp_path, "QNNExecutionProvider", dll.name)
        result = discover_all_eps(extra_sources=[src])
        entries = result["QNNExecutionProvider"]
        assert len(entries) == 1
        assert entries[0].status == "primary"

    def test_discover_all_eps_extra_sources_after(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _touch(tmp_path / "default" / "qnn.dll")
        _touch(tmp_path / "after" / "qnn.dll")
        default = _filesystem_source_for(tmp_path / "default", "QNNExecutionProvider", "qnn.dll")
        after = _filesystem_source_for(tmp_path / "after", "QNNExecutionProvider", "qnn.dll")
        monkeypatch.setattr(_ep, "_default_ep_sources", lambda: [default])

        result = discover_all_eps(extra_sources_after=[after])
        entries = result["QNNExecutionProvider"]
        assert len(entries) == 2
        assert entries[0].status == "primary"
        assert "default" in str(entries[0].dll_path)
        assert entries[1].status == "shadowed"
        assert "after" in str(entries[1].dll_path)
