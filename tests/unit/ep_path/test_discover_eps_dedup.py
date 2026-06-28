# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``discover_all_eps()`` (ep_name, dll_path) dedup.

Two distinct ``EPSource`` instances may legitimately resolve to the SAME
on-disk DLL — most commonly when the WinAppSDK ``ExecutionProviderCatalog``
and the WinRT ``PackageManager`` MSIX enumerator both inspect the same
installed package. The discovery layer must collapse such duplicates so
``winml sys --list-ep`` shows ONE entry per (EP, canonical DLL path),
keeping the first occurrence in precedence order so the higher-precedence
source's attribution wins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from winml.modelkit import ep_path as _ep
from winml.modelkit.ep_path import (
    DirectorySource,
    EPEntry,
    discover_all_eps,
)


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_default_ep_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace _default_ep_sources and skip live catalog/MSIX scans."""
    monkeypatch.setattr(_ep, "_default_ep_sources", list)
    monkeypatch.setattr(_ep, "_get_catalog", lambda: None)
    monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: None)
    monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _filesystem_source_for(root: Path, ep: str, dll_name: str) -> DirectorySource:
    return DirectorySource(root=root, dll_patterns={ep: dll_name})


def _entries_for(result: list[EPEntry], ep_name: str) -> list[EPEntry]:
    return [e for e in result if e.ep_name == ep_name]


class TestDiscoverAllEpsDedupSamePath:
    """Two different sources resolving to the same DLL must collapse to one row."""

    def test_two_sources_same_path_collapse(self, tmp_path: Path) -> None:
        """Two DirectorySource instances yielding the SAME EPEntry path → one row.

        Models the WinMLCatalogSource + MSIXPackageSource overlap on the
        Intel OpenVINO MSIX package, where both sources legitimately point
        at the same installed DLL.
        """
        dll = _touch(tmp_path / "openvino_plugin.dll")
        # Two distinct sources, both pointing at the same DLL.
        src_a = _filesystem_source_for(
            tmp_path, "OpenVINOExecutionProvider", "openvino_plugin.dll"
        )
        src_b = _filesystem_source_for(
            tmp_path, "OpenVINOExecutionProvider", "openvino_plugin.dll"
        )

        result = discover_all_eps(extra_sources=[src_a, src_b])
        entries = _entries_for(result, "OpenVINOExecutionProvider")

        assert len(entries) == 1, (
            f"Expected one entry after dedup; got {len(entries)}:\n"
            + "\n".join(f"  {e.source!r} -> {e.dll_path}" for e in entries)
        )
        assert entries[0].dll_path == dll.resolve()
        # First-occurrence-wins: src_a's attribution is preserved.
        assert entries[0].source is src_a
        assert entries[0].status == "primary"

    def test_three_sources_same_path_collapse(self, tmp_path: Path) -> None:
        """Three sources all pointing at the same DLL → one row."""
        _touch(tmp_path / "openvino_plugin.dll")
        srcs = [
            _filesystem_source_for(tmp_path, "OpenVINOExecutionProvider", "openvino_plugin.dll")
            for _ in range(3)
        ]
        result = discover_all_eps(extra_sources=srcs)
        entries = _entries_for(result, "OpenVINOExecutionProvider")

        assert len(entries) == 1
        # First-occurrence-wins.
        assert entries[0].source is srcs[0]
        assert entries[0].status == "primary"

    def test_dedup_preserves_distinct_paths(self, tmp_path: Path) -> None:
        """Different DLL paths must NOT be deduped — both surface as shadowed/primary."""
        _touch(tmp_path / "a" / "openvino_plugin.dll")
        _touch(tmp_path / "b" / "openvino_plugin.dll")
        src_a = _filesystem_source_for(
            tmp_path / "a", "OpenVINOExecutionProvider", "openvino_plugin.dll"
        )
        src_b = _filesystem_source_for(
            tmp_path / "b", "OpenVINOExecutionProvider", "openvino_plugin.dll"
        )

        result = discover_all_eps(extra_sources=[src_a, src_b])
        entries = _entries_for(result, "OpenVINOExecutionProvider")

        assert len(entries) == 2
        assert entries[0].status == "primary"
        assert entries[1].status == "shadowed"

    def test_dedup_across_extra_sources_and_extra_sources_after(
        self, tmp_path: Path
    ) -> None:
        """Same DLL via extra_sources AND extra_sources_after → one row.

        Mirrors the bug shape: discover_all_eps walks the default list
        (Catalog) and extra_sources_after (MSIX list_msix_eps result),
        which both surface the same Microsoft-published OpenVINO DLL.
        """
        _touch(tmp_path / "openvino_plugin.dll")
        src_default = _filesystem_source_for(
            tmp_path, "OpenVINOExecutionProvider", "openvino_plugin.dll"
        )
        src_after = _filesystem_source_for(
            tmp_path, "OpenVINOExecutionProvider", "openvino_plugin.dll"
        )

        result = discover_all_eps(
            extra_sources=[src_default],
            extra_sources_after=[src_after],
        )
        entries = _entries_for(result, "OpenVINOExecutionProvider")

        assert len(entries) == 1
        # Higher-precedence source (extra_sources) wins attribution.
        assert entries[0].source is src_default
        assert entries[0].status == "primary"

    def test_dedup_does_not_drop_other_eps(self, tmp_path: Path) -> None:
        """Dedup of one EP's duplicate must not affect a different EP's entries."""
        _touch(tmp_path / "openvino_plugin.dll")
        _touch(tmp_path / "qnn.dll")
        # Two duplicate OpenVINO sources, one QNN source.
        src_ov_a = _filesystem_source_for(
            tmp_path, "OpenVINOExecutionProvider", "openvino_plugin.dll"
        )
        src_ov_b = _filesystem_source_for(
            tmp_path, "OpenVINOExecutionProvider", "openvino_plugin.dll"
        )
        src_qnn = _filesystem_source_for(tmp_path, "QNNExecutionProvider", "qnn.dll")

        result = discover_all_eps(extra_sources=[src_ov_a, src_ov_b, src_qnn])

        assert len(_entries_for(result, "OpenVINOExecutionProvider")) == 1
        assert len(_entries_for(result, "QNNExecutionProvider")) == 1

    def test_dedup_does_not_collapse_different_eps_same_path(
        self, tmp_path: Path
    ) -> None:
        """Two DIFFERENT ep_names sharing one DLL must both survive dedup.

        ``discover_all_eps`` dedups on ``(ep_name, canonical dll_path)``;
        a single DLL exposing two distinct EPs (commit ``043aec01`` open
        question — e.g., a hypothetical ``OpenVINOExecutionProvider`` vs
        ``OpenVINOExecutionProvider.AUTO`` sharing one plugin DLL) must
        produce two entries, not one.
        """
        _touch(tmp_path / "multi-ep.dll")
        src_a = _filesystem_source_for(
            tmp_path, "OpenVINOExecutionProvider", "multi-ep.dll"
        )
        src_b = _filesystem_source_for(
            tmp_path, "OpenVINOExecutionProvider.AUTO", "multi-ep.dll"
        )

        result = discover_all_eps(extra_sources=[src_a, src_b])
        # Both entries land in the flat result with the same dll_path.
        ep_a = _entries_for(result, "OpenVINOExecutionProvider")
        ep_b = _entries_for(result, "OpenVINOExecutionProvider.AUTO")

        assert len(ep_a) == 1, f"Expected one OpenVINO entry; got {len(ep_a)}"
        assert len(ep_b) == 1, f"Expected one OpenVINO.AUTO entry; got {len(ep_b)}"
        # Both attribute to the same DLL — dedup must NOT collapse across ep_names.
        assert ep_a[0].dll_path == ep_b[0].dll_path


class TestDiscoverAllEpsBuiltinSourcePassthrough:
    """F-09: BuiltinSource entries with sentinel Path("") must survive the
    is_file() guard in discover_all_eps so a future caller passing a
    BuiltinSource via extra_sources doesn't silently lose the entry.
    """

    def test_builtin_source_entry_survives_is_file_guard(self) -> None:
        from pathlib import Path

        from winml.modelkit.ep_path import BuiltinSource, EPSource

        class _YieldingBuiltinSource(EPSource):
            """BuiltinSource whose resolve() yields a sentinel-pathed entry.

            Mirrors a hypothetical future refactor where BuiltinSource
            participates in discover_all_eps rather than being synthesized
            directly into WinMLEPRegistry._discovered.
            """

            def resolve(self):
                yield EPEntry(
                    ep_name="CPUExecutionProvider",
                    dll_path=Path(),
                    source=BuiltinSource(eps=("CPUExecutionProvider",)),
                )

            def iter_eps(self):
                return ("CPUExecutionProvider",)

        result = discover_all_eps(extra_sources=[_YieldingBuiltinSource()])
        cpu = _entries_for(result, "CPUExecutionProvider")
        assert len(cpu) == 1, (
            f"BuiltinSource entry was dropped by is_file() guard "
            f"(got {len(cpu)} entries; expected 1)."
        )
        assert isinstance(cpu[0].source, BuiltinSource)
        assert cpu[0].dll_path == Path()

    def test_filesystem_source_with_missing_dll_still_dropped(
        self, tmp_path: Path
    ) -> None:
        """Negative control: filesystem-backed entries with nonexistent
        DLLs are still filtered out — the is_file() guard's original
        purpose (catch typos in DirectorySource patterns) is preserved.
        """
        # Don't create the DLL on disk; DirectorySource discovers nothing.
        src = _filesystem_source_for(tmp_path, "MissingEP", "missing.dll")
        result = discover_all_eps(extra_sources=[src])
        assert _entries_for(result, "MissingEP") == []
