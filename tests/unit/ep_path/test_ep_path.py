# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the ep_path EP discovery module.

Covers:
    - PyPISource.resolve(): present/missing distribution.
    - DirectorySource.resolve(): env-var gating, required marker,
      glob patterns, multiple EPs in one root.
    - WinMLCatalogSource.resolve(): graceful no-yield when the
      WinAppSDK ML Python binding is not installed.
    - WINMLCLI_EP_PATH env-var override parsing.
    - _winners(discover_all_eps()): first-hit-wins precedence, extra_sources
      override, dedup, error tolerance.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from winml.modelkit.ep_path import (
    EP_CATALOG,
    DirectorySource,
    EPEntry,
    EPSource,
    MSIXPackageSource,
    NuGetSource,
    PyPISource,
    WinMLCatalogSource,
    _default_ep_sources,
    _get_detected_vendors,
    _parse_winmlcli_ep_path,
    _resolve_arch_key,
    discover_all_eps,
)
from winml.modelkit.sysinfo import CPU


def _distribution_installed(name: str) -> bool:
    """True when a PyPI distribution is importable in this environment."""
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


# The two PyPISource "resolves an installed distribution" tests below need the
# optional onnxruntime-ep-openvino wheel (the ``[openvino]`` extra). CI installs
# it via ``uv sync --all-extras`` (.github/workflows/modelkit-ci.yml); a base
# local ``uv sync`` does not. Gate them so a base install SKIPS cleanly rather
# than failing on a missing optional EP package — consistent with the
# EP-availability skip allowance in CLAUDE.md and the existing precedent in
# tests/unit/commands/test_cli.py.
_OPENVINO_EP_INSTALLED = _distribution_installed("onnxruntime-ep-openvino")

_requires_openvino_ep = pytest.mark.skipif(
    not _OPENVINO_EP_INSTALLED,
    reason="onnxruntime-ep-openvino not installed; run `uv sync --all-extras`",
)


def _winners(entries: list[EPEntry]) -> dict[str, tuple[Path, EPSource]]:
    """Mirror legacy ``discover_eps`` precedence-winner-only shape from a
    flat ``discover_all_eps`` result. Kept local to tests; production code
    uses ``discover_all_eps`` directly."""
    return {e.ep_name: (e.dll_path, e.source) for e in entries if e.status == "primary"}


# ---------------------------------------------------------------------------
# File-scoped autouse: prevent any test in this file from loading the live
# wasdk binding via ``_get_catalog``. None of the tests here need it; without
# this gate, tests that call ``_winners(discover_all_eps())`` (which walks the default
# EP source list including WinMLCatalogSource entries) would lazy-load the binding
# on machines with the [winml-catalog] extra installed and the OS-level
# Windows App Runtime present, polluting the module-level catalog singleton
# state for downstream fake-binding tests in test_winml_catalog_source.py.
#
# Tests in test_winml_catalog_source.py do not see this fixture (it is
# defined at file scope in test_ep_path.py, not in conftest.py), so they
# retain access to the real ``_get_catalog`` implementation needed to
# exercise their fake-binding-via-sys.modules injection path.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _skip_live_catalog_in_ep_path_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``_get_catalog`` to return None so the default EP source list stays inert."""
    from winml.modelkit import ep_path as _ep

    monkeypatch.setattr(_ep, "_get_catalog", lambda: None)


@pytest.fixture(autouse=True)
def _reset_resolve_arch_key_cache() -> None:
    """Clear ``_resolve_arch_key``'s ``lru_cache`` before AND after every
    test in this file.

    The cache is correct for production (host/process arch can't change
    within a process), but it's process-global — a test that fakes
    ``os.name``, ``CPU.get_all``, or ``platform.machine`` to simulate a
    different host would otherwise leak its cached result into whichever
    test runs next, independent of whether that later test does any
    mocking at all.
    """
    _resolve_arch_key.cache_clear()
    yield
    _resolve_arch_key.cache_clear()


# ---------------------------------------------------------------------------
# Module-level public API.
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Confirm the public module surface is intact."""

    def test_ep_catalog_has_five_plugin_eps(self) -> None:
        plugin_eps = {ep for ep in EP_CATALOG.all_eps() if EP_CATALOG.dll_name_for(ep)}
        assert plugin_eps == {
            "OpenVINOExecutionProvider",
            "QNNExecutionProvider",
            "VitisAIExecutionProvider",
            "MIGraphXExecutionProvider",
            "NvTensorRTRTXExecutionProvider",
        }

    def test_ep_catalog_uses_canonical_casing_for_nvidia(self) -> None:
        assert EP_CATALOG.dll_name_for("NvTensorRTRTXExecutionProvider") is not None
        assert EP_CATALOG.dll_name_for("NvTensorRtRtxExecutionProvider") is None

    def test_ep_path_is_a_list(self) -> None:
        sources = _default_ep_sources()
        assert isinstance(sources, list)
        for entry in sources:
            assert isinstance(
                entry,
                (
                    PyPISource,
                    NuGetSource,
                    DirectorySource,
                    WinMLCatalogSource,
                    MSIXPackageSource,
                ),
            )

    def test_ep_source_subclasses_inherit_from_abc(self) -> None:
        # EPSource is the abstract base class for all source kinds.
        assert PyPISource is not None
        assert NuGetSource is not None
        assert DirectorySource is not None
        assert WinMLCatalogSource is not None
        # Every concrete source kind must subclass the ABC.
        for cls in (PyPISource, NuGetSource, DirectorySource, WinMLCatalogSource):
            assert issubclass(cls, EPSource)


# ---------------------------------------------------------------------------
# Arch-key resolution (process arch x host arch -> one of 3 reachable keys).
# ---------------------------------------------------------------------------


class TestResolveArchKey:
    """_resolve_arch_key reports one of 3 physically-reachable combinations.

    The 4th (native-ARM64 process on genuine x64 hardware) is impossible —
    no Windows emulation path runs an ARM64-only executable on x64 silicon —
    so it's intentionally absent from both the function and these tests.
    """

    def test_returns_one_of_three_known_keys_on_this_host(self) -> None:
        # Smoke test against the real host — must not raise.
        assert _resolve_arch_key() in ("x64_native", "x64_on_arm64", "arm64_native")

    @pytest.mark.parametrize(
        ("host_is_arm64", "machine", "expected_key"),
        [
            (False, "AMD64", "x64_native"),  # x64 process, x64 host, no emulation
            (True, "AMD64", "x64_on_arm64"),  # x64 process, ARM64 host — the bug scenario
            (True, "ARM64", "arm64_native"),  # native ARM64 process, ARM64 host
        ],
    )
    def test_all_three_reachable_combinations(
        self,
        host_is_arm64: bool,
        machine: str,
        expected_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force all 3 branches so coverage is host-independent (mirrors the
        # prior review S-6 requirement for the function this replaced).
        #
        # Patches CPU.get_all (host truth — WMI's Win32_Processor.Architecture,
        # correct under x64-on-ARM64 emulation) and platform.machine (process
        # truth — correct as-is, per _resolve_arch_key's docstring)
        # independently.
        arch = CPU.Architecture.ARM64 if host_is_arm64 else CPU.Architecture.x64
        monkeypatch.setattr("winml.modelkit.ep_path.os.name", "nt")
        monkeypatch.setattr(CPU, "get_all", lambda: [SimpleNamespace(architecture=arch)])
        monkeypatch.setattr("platform.machine", lambda: machine)
        assert _resolve_arch_key() == expected_key

    def test_false_on_non_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # CPU.get_all is never even consulted off Windows — force it to
        # raise, to prove the os.name guard short-circuits before the WMI
        # query, not just that the end result happens to match.
        def _boom() -> list[CPU]:
            raise AssertionError("CPU.get_all should not be called on non-Windows")

        monkeypatch.setattr("winml.modelkit.ep_path.os.name", "posix")
        monkeypatch.setattr(CPU, "get_all", _boom)
        assert _resolve_arch_key() == "x64_native"

    @pytest.mark.parametrize(
        ("machine", "expected_key"),
        [
            ("AMD64", "x64_on_arm64"),
            ("x86_64", "x64_on_arm64"),
            ("ARM64", "arm64_native"),
            ("aarch64", "arm64_native"),
        ],
    )
    def test_recognizes_process_arch_spellings_under_arm64_host(
        self, machine: str, expected_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only reachable/meaningful when the host is ARM64 — that's the
        # branch that checks platform.machine()'s spelling.
        monkeypatch.setattr("winml.modelkit.ep_path.os.name", "nt")
        monkeypatch.setattr(
            CPU, "get_all", lambda: [SimpleNamespace(architecture=CPU.Architecture.ARM64)]
        )
        monkeypatch.setattr("platform.machine", lambda: machine)
        assert _resolve_arch_key() == expected_key

    def test_returns_x64_native_when_cpu_query_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # CPU.get_all shells out to PowerShell; any failure there (missing
        # binary, WMI unavailable, malformed JSON) must degrade to the safe
        # default rather than propagate and break EP discovery.
        def _boom() -> list[CPU]:
            raise OSError("powershell not found")

        monkeypatch.setattr("winml.modelkit.ep_path.os.name", "nt")
        monkeypatch.setattr(CPU, "get_all", _boom)
        assert _resolve_arch_key() == "x64_native"

    def test_returns_x64_native_when_no_cpus_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("winml.modelkit.ep_path.os.name", "nt")
        monkeypatch.setattr(CPU, "get_all", list)
        assert _resolve_arch_key() == "x64_native"


class TestQnnArchFolderMapProductionData:
    """The QNN entry's arch_folder_map, read from _default_ep_sources() itself.

    Prior to the arch_folder_map refactor, the exact function object under
    test (_qnn_arch_resolver) was the one wired into production, so a wrong
    mapping would fail the branch tests directly. Now the mapping is a plain
    dict literal in _default_ep_sources() — nothing forces it to stay in
    sync with what's tested unless a test reads THAT literal specifically
    (not a hand-typed copy). A silently wrong mapping here (e.g. swapped
    amd64/arm64ec) doesn't raise or fail loudly — it just makes wmk pick the
    wrong DLL, which per this module's own docs hangs the QNN HTP EP
    indefinitely on some SDK versions and runs ~1000x slower on others.
    """

    @staticmethod
    def _qnn_source() -> PyPISource:
        (qnn_source,) = (
            s
            for s in _default_ep_sources()
            if isinstance(s, PyPISource) and s.distribution == "onnxruntime-qnn"
        )
        return qnn_source

    def test_arch_folder_map_has_exact_expected_values(self) -> None:
        assert self._qnn_source().arch_folder_map == {
            "x64_native": "amd64",
            "x64_on_arm64": "arm64ec",
            "arm64_native": "arm64ec",
        }

    def test_arch_folder_map_covers_every_reachable_key(self) -> None:
        # If _resolve_arch_key() ever gains a 4th reachable value, this
        # fails loudly instead of silently falling through to "no known
        # DLL layout for this machine" for real users on that combination.
        reachable_keys = {"x64_native", "x64_on_arm64", "arm64_native"}
        assert set(self._qnn_source().arch_folder_map) == reachable_keys


# ---------------------------------------------------------------------------
# PyPISource.
# ---------------------------------------------------------------------------


class TestPyPISource:
    """PyPISource resolves via importlib.metadata against the live env."""

    @_requires_openvino_ep
    def test_resolves_installed_distribution(self) -> None:
        # ``onnxruntime-ep-openvino`` is in pyproject.toml deps and
        # installed in the venv used to run the test suite.
        source = PyPISource(
            distribution="onnxruntime-ep-openvino",
            relative_dll=("onnxruntime_ep_openvino/onnxruntime_providers_openvino_plugin.dll"),
            eps=("OpenVINOExecutionProvider",),
        )
        results = list(source.resolve())
        assert len(results) == 1
        entry = results[0]
        assert isinstance(entry, EPEntry)
        assert entry.ep_name == "OpenVINOExecutionProvider"
        assert entry.dll_path.is_file(), f"Expected {entry.dll_path} to exist"
        assert entry.dll_path.name == "onnxruntime_providers_openvino_plugin.dll"
        # PyPISource plumbs the distribution version into EPEntry.
        assert entry.version is not None

    def test_yields_nothing_for_missing_distribution(self) -> None:
        source = PyPISource(
            distribution="this-distribution-does-not-exist-xyz",
            relative_dll="ignored.dll",
            eps=("FakeEP",),
        )
        assert list(source.resolve()) == []

    def test_arch_folder_map_is_invoked(self) -> None:
        # The QNN entry uses an arch_folder_map; verify it's actually
        # consulted by checking the resolved path includes a known arch
        # directory, never the unsubstituted token.
        source = PyPISource(
            distribution="onnxruntime-qnn",
            relative_dll="onnxruntime_qnn/libs/{arch}/onnxruntime_providers_qnn.dll",
            eps=("QNNExecutionProvider",),
            arch_folder_map={
                "x64_native": "amd64",
                "x64_on_arm64": "arm64ec",
                "arm64_native": "arm64ec",
            },
        )
        results = list(source.resolve())
        # Whether the file exists depends on the host arch + wheel
        # contents. Either we got a valid path, or we got nothing
        # (the arch's libs subdir was missing). What we DO require:
        # if a path is yielded, it must NOT contain the unsubstituted
        # token.
        for entry in results:
            assert "{arch}" not in str(entry.dll_path)

    def test_arch_folder_map_missing_key_yields_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A map that doesn't cover the current arch_key is "no known build
        # for this machine" — not an error, just nothing yielded.
        monkeypatch.setattr("winml.modelkit.ep_path._resolve_arch_key", lambda: "x64_native")
        source = PyPISource(
            distribution="onnxruntime-ep-openvino",
            relative_dll="onnxruntime_ep_openvino/{arch}/onnxruntime_providers_openvino_plugin.dll",
            eps=("OpenVINOExecutionProvider",),
            arch_folder_map={"arm64_native": "arm64"},  # no "x64_native" entry
        )
        assert list(source.resolve()) == []

    @_requires_openvino_ep
    def test_none_arch_folder_map_uses_relative_dll_as_is(self) -> None:
        # Default (arch_folder_map=None) behavior — e.g. OpenVINO's real
        # entry today — must be completely unaffected by this mechanism.
        source = PyPISource(
            distribution="onnxruntime-ep-openvino",
            relative_dll=("onnxruntime_ep_openvino/onnxruntime_providers_openvino_plugin.dll"),
            eps=("OpenVINOExecutionProvider",),
        )
        results = list(source.resolve())
        assert len(results) == 1
        assert results[0].dll_path.name == "onnxruntime_providers_openvino_plugin.dll"

    def test_yields_nothing_when_dll_missing_from_distribution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = PyPISource(
            distribution="onnxruntime-ep-openvino",
            relative_dll="onnxruntime_ep_openvino/this_file_does_not_exist.dll",
            eps=("OpenVINOExecutionProvider",),
        )
        assert list(source.resolve()) == []


# ---------------------------------------------------------------------------
# DirectorySource.
# ---------------------------------------------------------------------------


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


class TestFilesystemSource:
    """DirectorySource scans a directory for plugin DLLs."""

    def test_resolves_single_dll_in_root(self, tmp_path: Path) -> None:
        dll = _touch(tmp_path / "onnxruntime_providers_vitisai.dll")
        source = DirectorySource(
            root=tmp_path,
            dll_patterns={"VitisAIExecutionProvider": dll.name},
        )
        results = list(source.resolve())
        assert len(results) == 1
        entry = results[0]
        assert isinstance(entry, EPEntry)
        assert entry.ep_name == "VitisAIExecutionProvider"
        assert entry.dll_path == dll.resolve()
        # DirectorySource has no version concept.
        assert entry.version is None

    def test_yields_nothing_when_root_missing(self, tmp_path: Path) -> None:
        source = DirectorySource(
            root=tmp_path / "does-not-exist",
            dll_patterns={"VitisAIExecutionProvider": "any.dll"},
        )
        assert list(source.resolve()) == []

    def test_env_var_unset_yields_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FAKE_INSTALLATION_PATH", raising=False)
        source = DirectorySource(
            root=Path("deployment"),
            env_var="FAKE_INSTALLATION_PATH",
            dll_patterns={"VitisAIExecutionProvider": "vitisai.dll"},
        )
        assert list(source.resolve()) == []

    def test_env_var_resolves_relative_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Mimic a Ryzen AI install layout:
        # %FAKE_INSTALLATION_PATH%/deployment/onnxruntime_providers_vitisai.dll
        deployment = tmp_path / "deployment"
        dll = _touch(deployment / "onnxruntime_providers_vitisai.dll")
        marker = _touch(deployment / "onnxruntime_providers_shared.dll")
        monkeypatch.setenv("FAKE_INSTALLATION_PATH", str(tmp_path))

        source = DirectorySource(
            root=Path("deployment"),
            env_var="FAKE_INSTALLATION_PATH",
            dll_patterns={"VitisAIExecutionProvider": dll.name},
            required_marker=marker.name,
        )
        results = list(source.resolve())
        assert len(results) == 1
        assert results[0].ep_name == "VitisAIExecutionProvider"
        assert results[0].dll_path == dll.resolve()

    def test_required_marker_missing_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deployment = tmp_path / "deployment"
        _touch(deployment / "onnxruntime_providers_vitisai.dll")
        monkeypatch.setenv("FAKE_INSTALLATION_PATH", str(tmp_path))

        source = DirectorySource(
            root=Path("deployment"),
            env_var="FAKE_INSTALLATION_PATH",
            dll_patterns={"VitisAIExecutionProvider": "onnxruntime_providers_vitisai.dll"},
            required_marker="onnxruntime_providers_shared.dll",
        )
        assert list(source.resolve()) == []

    def test_multiple_eps_in_one_root(self, tmp_path: Path) -> None:
        dll_a = _touch(tmp_path / "onnxruntime_providers_openvino_plugin.dll")
        dll_b = _touch(tmp_path / "onnxruntime_providers_qnn.dll")
        source = DirectorySource(
            root=tmp_path,
            dll_patterns={
                "OpenVINOExecutionProvider": dll_a.name,
                "QNNExecutionProvider": dll_b.name,
            },
        )
        results = {entry.ep_name: entry.dll_path for entry in source.resolve()}
        assert results == {
            "OpenVINOExecutionProvider": dll_a.resolve(),
            "QNNExecutionProvider": dll_b.resolve(),
        }

    def test_glob_pattern_matches(self, tmp_path: Path) -> None:
        dll = _touch(tmp_path / "subdir" / "onnxruntime_providers_qnn.dll")
        source = DirectorySource(
            root=tmp_path,
            dll_patterns={"QNNExecutionProvider": "*/onnxruntime_providers_qnn.dll"},
        )
        results = list(source.resolve())
        assert len(results) == 1
        assert results[0].ep_name == "QNNExecutionProvider"
        assert results[0].dll_path == dll.resolve()


# ---------------------------------------------------------------------------
# WinMLCatalogSource stub.
# ---------------------------------------------------------------------------


class TestWinMLCatalogSourceBindingMissing:
    """When the optional WinAppSDK ML binding is absent, resolve() yields nothing.

    Detailed behavior (mocked binding shape, atexit registration, etc.)
    is covered in ``test_winml_catalog_source.py``.
    """

    def test_resolve_yields_nothing_without_binding(self) -> None:
        # The autouse fixture ``_skip_live_catalog_in_ep_path_tests``
        # forces ``_get_catalog`` to return ``None``, simulating the
        # binding-missing case. resolve() must yield nothing silently.
        source = WinMLCatalogSource(catalog_name="VitisAI", eps=("VitisAIExecutionProvider",))
        # ``resolve()`` is a generator; we have to iterate to trigger.
        assert list(source.resolve()) == []


# ---------------------------------------------------------------------------
# WINMLCLI_EP_PATH env var override.
# ---------------------------------------------------------------------------


class TestWinmlEpPathOverride:
    """Parsing the WINMLCLI_EP_PATH env var into DirectorySource entries."""

    def test_unset_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
        assert _parse_winmlcli_ep_path() == []

    def test_empty_string_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WINMLCLI_EP_PATH", "")
        assert _parse_winmlcli_ep_path() == []

    def test_single_entry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WINMLCLI_EP_PATH", str(tmp_path))
        sources = _parse_winmlcli_ep_path()
        # Per the C1 fix, the parser emits ONE DirectorySource per
        # (root, ep, dll_filename) combination — so a single entry with
        # five EPs (each with exactly one .dll filename) yields five sources.
        # Every known EP must be covered at least once.
        assert all(isinstance(s, DirectorySource) for s in sources)
        assert all(s.root == tmp_path for s in sources)
        covered_eps = {
            ep for s in sources if isinstance(s, DirectorySource) for ep in s.dll_patterns
        }
        assert covered_eps == {ep for ep in EP_CATALOG.all_eps() if EP_CATALOG.dll_name_for(ep)}

    def test_emits_source_per_dll_filename_windows_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # This is a Windows-only project. EP_CATALOG entries contain
        # only .dll filenames (Linux .so entries were dropped in Task 6).
        # The parser emits one DirectorySource per (root, ep, dll_filename)
        # tuple — all EPs should resolve correctly with their single DLL name.
        monkeypatch.setenv("WINMLCLI_EP_PATH", str(tmp_path))
        sources = _parse_winmlcli_ep_path()
        ov_dlls = [
            next(iter(s.dll_patterns.values()))
            for s in sources
            if isinstance(s, DirectorySource) and "OpenVINOExecutionProvider" in s.dll_patterns
        ]
        assert "onnxruntime_providers_openvino_plugin.dll" in ov_dlls
        # .so filenames were dropped — Windows-only project
        assert not any(dll.endswith(".so") for dll in ov_dlls)

    def test_multi_entry_separator(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        # os.pathsep is ; on Windows and : elsewhere — same as PATH.
        import os

        monkeypatch.setenv("WINMLCLI_EP_PATH", f"{a}{os.pathsep}{b}")
        sources = _parse_winmlcli_ep_path()
        roots = {s.root for s in sources if isinstance(s, DirectorySource)}
        assert a in roots
        assert b in roots

    def test_winml_ep_path_finds_dll(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # End-to-end: drop a synthetic plugin DLL in a directory, set
        # WINMLCLI_EP_PATH to that directory, confirm discover_eps finds it.
        dll = _touch(tmp_path / "onnxruntime_providers_vitisai.dll")
        monkeypatch.setenv("WINMLCLI_EP_PATH", str(tmp_path))
        # Skip env-var-gated DirectorySource so the env-var path is the
        # only producer of a VitisAI hit. (The autouse
        # _skip_live_catalog_in_ep_path_tests fixture handles the catalog
        # source side.)
        monkeypatch.delenv("RYZEN_AI_INSTALLATION_PATH", raising=False)
        resolved = _winners(discover_all_eps())
        assert "VitisAIExecutionProvider" in resolved
        path, _src = resolved["VitisAIExecutionProvider"]
        assert path == dll.resolve()

    def test_winmlcli_ep_path_warns_on_nonexistent_directory(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """WINMLCLI_EP_PATH entries pointing at non-directories must log a WARN and skip."""
        import logging

        from winml.modelkit.ep_path import _parse_winmlcli_ep_path

        monkeypatch.setenv("WINMLCLI_EP_PATH", "/this/path/does/not/exist")
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.ep_path"):
            sources = _parse_winmlcli_ep_path()

        assert sources == [], f"Expected empty list (nonexistent dir), got {sources}"
        assert any("not a directory" in record.message for record in caplog.records), (
            f"Expected WARN about non-directory; got: {[r.message for r in caplog.records]}"
        )

    def test_winmlcli_ep_path_accepts_existing_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WINMLCLI_EP_PATH with a valid directory yields one DirectorySource per known EP DLL."""
        from winml.modelkit.ep_path import _parse_winmlcli_ep_path

        monkeypatch.setenv("WINMLCLI_EP_PATH", str(tmp_path))
        sources = _parse_winmlcli_ep_path()

        assert len(sources) > 0, "Expected at least one DirectorySource"
        # All sources must point at the configured directory.
        for s in sources:
            assert tmp_path in s.root.parents or s.root == tmp_path


# ---------------------------------------------------------------------------
# discover_eps precedence + dedup + error tolerance.
# ---------------------------------------------------------------------------


class TestDiscoverEps:
    """The walk algorithm: first-hit-wins, dedup, no-fail-on-source-error."""

    def test_extra_sources_override_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Build an extra source that "claims" OpenVINOExecutionProvider
        # with a synthetic DLL. It should beat the PyPI-resolved real one.
        fake_dll = _touch(tmp_path / "fake_openvino.dll")
        extra = DirectorySource(
            root=tmp_path,
            dll_patterns={"OpenVINOExecutionProvider": fake_dll.name},
        )
        monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
        resolved = _winners(discover_all_eps(extra_sources=[extra]))
        assert "OpenVINOExecutionProvider" in resolved
        path, source = resolved["OpenVINOExecutionProvider"]
        assert path == fake_dll.resolve()
        assert source is extra

    def test_first_hit_wins_among_extra_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = _touch(tmp_path / "a" / "vitisai.dll")
        b = _touch(tmp_path / "b" / "vitisai.dll")
        first = DirectorySource(
            root=a.parent,
            dll_patterns={"VitisAIExecutionProvider": a.name},
        )
        second = DirectorySource(
            root=b.parent,
            dll_patterns={"VitisAIExecutionProvider": b.name},
        )
        monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
        resolved = _winners(discover_all_eps(extra_sources=[first, second]))
        assert resolved["VitisAIExecutionProvider"][0] == a.resolve()

    def test_winml_catalog_source_does_not_abort_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On a CI machine without the WinAppSDK binding, resolve() on a
        # WinMLCatalogSource yields nothing and discover_eps continues
        # to subsequent sources. ``_get_catalog`` is already mocked to
        # None by the file-scoped autouse fixture
        # ``_skip_live_catalog_in_ep_path_tests``.
        fake_dll = _touch(tmp_path / "fake_qnn.dll")
        catalog = WinMLCatalogSource(catalog_name="QNN", eps=("QNNExecutionProvider",))
        good = DirectorySource(
            root=tmp_path,
            dll_patterns={"QNNExecutionProvider": fake_dll.name},
        )
        monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
        resolved = _winners(discover_all_eps(extra_sources=[catalog, good]))
        assert resolved["QNNExecutionProvider"][0] == fake_dll.resolve()

    def test_resolve_returns_path_and_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_dll = _touch(tmp_path / "fake.dll")
        src = DirectorySource(
            root=tmp_path,
            dll_patterns={"QNNExecutionProvider": fake_dll.name},
        )
        monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
        resolved = _winners(discover_all_eps(extra_sources=[src]))
        assert resolved["QNNExecutionProvider"] == (fake_dll.resolve(), src)

    def test_discover_eps_returns_tuple_shape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_winners(discover_all_eps()) (no flag) returns dict[str, (Path, EPSource)]."""
        fake_dll = _touch(tmp_path / "fake.dll")
        src = DirectorySource(
            root=tmp_path,
            dll_patterns={"QNNExecutionProvider": fake_dll.name},
        )
        monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
        result = _winners(discover_all_eps(extra_sources=[src]))
        for value in result.values():
            assert isinstance(value, tuple)
            assert len(value) == 2


# ---------------------------------------------------------------------------
# NuGetSource.
# ---------------------------------------------------------------------------


class TestNuGetSource:
    """NuGetSource resolves plugin DLLs from the NuGet global-packages cache."""

    def test_nuget_source_resolves_highest_stable_over_prerelease(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NuGetSource picks the highest stable version, even when prereleases sort later."""
        import winml.modelkit.ep_path as mod

        root = tmp_path / "packages" / "foo"
        for ver in ("1.0.0", "1.0.0-rc.10", "1.0.0-rc.2", "0.9.0"):
            (root / ver / "native").mkdir(parents=True)
            (root / ver / "native" / "lib.dll").write_bytes(b"")

        monkeypatch.setattr(mod, "_nuget_packages_root", lambda: tmp_path / "packages")

        src = NuGetSource(
            distribution="Foo",
            relative_dll="native/lib.dll",
            eps=("FooExecutionProvider",),
        )
        results = list(src.resolve())

        assert len(results) == 1
        entry = results[0]
        assert entry.ep_name == "FooExecutionProvider"
        path = entry.dll_path
        assert "1.0.0" in str(path)
        path_str = str(path)
        assert "1.0.0-rc" not in path_str and "1.0.0-" not in path_str

    def test_nuget_source_compares_prerelease_with_semver_numeric_ordering(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When only prereleases exist, -rc.10 must beat -rc.2 (numeric, not lex).

        This test exposes the bug in the hand-rolled parser: when sorting
        prerelease versions by their string suffix, "rc.2" > "rc.10"
        lexicographically. The SemVer 2.0 spec requires numeric ordering
        within prerelease identifiers.
        """
        import winml.modelkit.ep_path as mod

        root = tmp_path / "packages" / "foo"
        # Create rc.2 first so it's sorted first by iterdir(). Then the
        # old code will pick rc.2 (preserved by stable sort on equal keys),
        # while the new code will pick rc.10 (correct SemVer ordering).
        for ver in ("1.0.0-rc.2", "1.0.0-rc.10"):
            (root / ver / "native").mkdir(parents=True)
            (root / ver / "native" / "lib.dll").write_bytes(b"")

        monkeypatch.setattr(mod, "_nuget_packages_root", lambda: tmp_path / "packages")

        src = NuGetSource(
            distribution="Foo",
            relative_dll="native/lib.dll",
            eps=("FooExecutionProvider",),
        )
        results = list(src.resolve())

        assert len(results) == 1
        path = results[0].dll_path
        # The version picked should be rc.10 (semantically newer).
        path_str = str(path)
        assert "1.0.0-rc.10" in path_str


# ---------------------------------------------------------------------------
# Vendor detection error handling.
# ---------------------------------------------------------------------------


class TestEPCatalog:
    """EPCatalog: forward/inverse lookups, bundled-EP handling, immutability."""

    def test_ep_catalog_dll_name_for_known_ep(self) -> None:
        """EP_CATALOG.dll_name_for returns the DLL filename for known plugin EPs."""
        from winml.modelkit.ep_path import EP_CATALOG

        assert EP_CATALOG.dll_name_for("OpenVINOExecutionProvider") == (
            "onnxruntime_providers_openvino_plugin.dll"
        )
        assert EP_CATALOG.dll_name_for("QNNExecutionProvider") == "onnxruntime_providers_qnn.dll"

    def test_ep_catalog_dll_name_for_bundled_ep_returns_none(self) -> None:
        """Bundled EPs (CPU/DML/Azure) have no DLL filename — `dll_name_for` returns None."""
        from winml.modelkit.ep_path import EP_CATALOG

        assert EP_CATALOG.dll_name_for("CPUExecutionProvider") is None
        assert EP_CATALOG.dll_name_for("DmlExecutionProvider") is None

    def test_ep_catalog_ep_for_dll_inverse_lookup(self) -> None:
        """EP_CATALOG.ep_for_dll resolves a DLL filename back to its EP name."""
        from winml.modelkit.ep_path import EP_CATALOG

        assert (
            EP_CATALOG.ep_for_dll("onnxruntime_providers_openvino_plugin.dll")
            == "OpenVINOExecutionProvider"
        )
        assert EP_CATALOG.ep_for_dll("nonexistent.dll") is None

    def test_ep_catalog_is_frozen(self) -> None:
        """EPCatalog must be truly immutable — both attribute rebinding and
        underlying dict mutation must fail at runtime."""
        import pytest

        from winml.modelkit.ep_path import EP_CATALOG

        # Attribute rebind raises
        with pytest.raises(AttributeError):
            EP_CATALOG._by_name = {}  # type: ignore[misc]

        # Direct dict mutation raises (MappingProxyType protects it)
        with pytest.raises(TypeError):
            EP_CATALOG._by_name["FakeExecutionProvider"] = None  # type: ignore[index]
        with pytest.raises(TypeError):
            EP_CATALOG._by_dll["fake.dll"] = "FakeExecutionProvider"  # type: ignore[index]

        # Confirm the lookups still work after the failed mutations
        assert EP_CATALOG.dll_name_for("QNNExecutionProvider") == "onnxruntime_providers_qnn.dll"


class TestGetDetectedVendorsErrorHandling:
    """Vendor detection must raise on failure, not silently return empty."""

    def test_raises_on_hardware_import_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When sysinfo.hardware import fails, _get_detected_vendors must raise."""
        import sys

        from winml.modelkit import ep_path

        ep_path._get_detected_vendors.cache_clear()

        # Hide the sysinfo.hardware module so the import inside _get_detected_vendors fails.
        monkeypatch.setitem(sys.modules, "winml.modelkit.sysinfo.hardware", None)

        with pytest.raises(RuntimeError, match="Hardware detection unavailable"):
            _get_detected_vendors()

    def test_raises_when_gpu_get_all_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When GPU.get_all() raises, propagate as RuntimeError (no silent empty)."""
        from winml.modelkit import ep_path
        from winml.modelkit.sysinfo.hardware import GPU

        ep_path._get_detected_vendors.cache_clear()

        def _broken(*_a, **_kw):
            raise RuntimeError("WMI handle invalid")

        monkeypatch.setattr(GPU, "get_all", _broken)
        with pytest.raises(RuntimeError, match=r"GPU\.get_all\(\) failed"):
            _get_detected_vendors()
