# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the ep_path EP discovery module.

Covers:
    - PyPiSource.resolve(): present/missing distribution.
    - FilesystemSource.resolve(): env-var gating, required marker,
      glob patterns, multiple EPs in one root.
    - WinMlCatalogSource.resolve(): graceful no-yield when the
      WinAppSDK ML Python binding is not installed.
    - WINMLCLI_EP_PATH env-var override parsing.
    - discover_eps(): first-hit-wins precedence, extra_sources
      override, dedup, error tolerance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from winml.modelkit.ep_path import (
    EP_DLL_NAMES,
    EpSource,
    FilesystemSource,
    NuGetSource,
    PyPiSource,
    WinMlCatalogSource,
    _default_ep_sources,
    _get_detected_vendors,
    _parse_winmlcli_ep_path,
    _qnn_arch_resolver,
    discover_eps,
)


# ---------------------------------------------------------------------------
# File-scoped autouse: prevent any test in this file from loading the live
# wasdk binding via ``_get_catalog``. None of the tests here need it; without
# this gate, tests that call ``discover_eps()`` (which walks the default
# EP source list including WinMlCatalogSource entries) would lazy-load the binding
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


# ---------------------------------------------------------------------------
# Module-level public API.
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Confirm the public module surface is intact."""

    def test_ep_dll_names_has_five_eps(self) -> None:
        assert set(EP_DLL_NAMES) == {
            "OpenVINOExecutionProvider",
            "QNNExecutionProvider",
            "VitisAIExecutionProvider",
            "MIGraphXExecutionProvider",
            "NvTensorRtRtxExecutionProvider",
        }

    def test_ep_dll_names_uses_camelcase_for_nvidia(self) -> None:
        # The canonical key follows MS Learn (camelCase). NVIDIA's
        # PascalCase 'NvTensorRTRTX...' is the alias, not the canonical.
        assert "NvTensorRtRtxExecutionProvider" in EP_DLL_NAMES
        assert "NvTensorRTRTXExecutionProvider" not in EP_DLL_NAMES

    def test_ep_path_is_a_list(self) -> None:
        sources = _default_ep_sources()
        assert isinstance(sources, list)
        for entry in sources:
            assert isinstance(
                entry,
                (PyPiSource, NuGetSource, FilesystemSource, WinMlCatalogSource),
            )

    def test_ep_source_subclasses_inherit_from_abc(self) -> None:
        # EpSource is the abstract base class for all source kinds.
        assert PyPiSource is not None
        assert NuGetSource is not None
        assert FilesystemSource is not None
        assert WinMlCatalogSource is not None
        # Every concrete source kind must subclass the ABC.
        for cls in (PyPiSource, NuGetSource, FilesystemSource, WinMlCatalogSource):
            assert issubclass(cls, EpSource)


# ---------------------------------------------------------------------------
# QNN arch resolver.
# ---------------------------------------------------------------------------


class TestQnnArchResolver:
    """The arch resolver substitutes ``{arch}`` per host architecture."""

    def test_substitutes_arch_token(self) -> None:
        out = _qnn_arch_resolver("libs/{arch}/foo.dll")
        # On all hosts, the result must be one of the two known values.
        assert out in ("libs/amd64/foo.dll", "libs/arm64ec/foo.dll")

    def test_passthrough_when_no_token(self) -> None:
        # ``str.format`` with no placeholders returns the same string.
        assert _qnn_arch_resolver("foo.dll") == "foo.dll"

    @pytest.mark.parametrize(
        ("machine", "expected_arch"),
        [
            ("AMD64", "amd64"),     # x64 native
            ("x86_64", "amd64"),    # POSIX x64 spelling
            ("ARM64", "arm64ec"),   # Snapdragon native
            ("aarch64", "arm64ec"), # POSIX arm64 spelling
        ],
    )
    def test_arch_branches_force_both_paths(
        self,
        machine: str,
        expected_arch: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force both arch branches so test coverage is host-independent
        # (review S-6). On x86_64 CI, the arm64ec branch would otherwise
        # never execute.
        monkeypatch.setattr("platform.machine", lambda: machine)
        out = _qnn_arch_resolver("libs/{arch}/foo.dll")
        assert out == f"libs/{expected_arch}/foo.dll"


# ---------------------------------------------------------------------------
# PyPiSource.
# ---------------------------------------------------------------------------


class TestPyPiSource:
    """PyPiSource resolves via importlib.metadata against the live env."""

    def test_resolves_installed_distribution(self) -> None:
        # ``onnxruntime-ep-openvino`` is in pyproject.toml deps and
        # installed in the venv used to run the test suite.
        source = PyPiSource(
            distribution="onnxruntime-ep-openvino",
            relative_dll=(
                "onnxruntime_ep_openvino/onnxruntime_providers_openvino_plugin.dll"
            ),
            eps=("OpenVINOExecutionProvider",),
        )
        results = list(source.resolve())
        assert len(results) == 1
        ep_name, path = results[0]
        assert ep_name == "OpenVINOExecutionProvider"
        assert path.is_file(), f"Expected {path} to exist"
        assert path.name == "onnxruntime_providers_openvino_plugin.dll"

    def test_yields_nothing_for_missing_distribution(self) -> None:
        source = PyPiSource(
            distribution="this-distribution-does-not-exist-xyz",
            relative_dll="ignored.dll",
            eps=("FakeEP",),
        )
        assert list(source.resolve()) == []

    def test_arch_resolver_is_invoked(self) -> None:
        # The QNN entry uses an arch_resolver; verify it's actually
        # called by checking the resolved path includes a known arch
        # directory.
        source = PyPiSource(
            distribution="onnxruntime-qnn",
            relative_dll="onnxruntime_qnn/libs/{arch}/onnxruntime_providers_qnn.dll",
            eps=("QNNExecutionProvider",),
            arch_resolver=_qnn_arch_resolver,
        )
        results = list(source.resolve())
        # Whether the file exists depends on the host arch + wheel
        # contents. Either we got a valid path, or we got nothing
        # (the arch's libs subdir was missing). What we DO require:
        # if a path is yielded, it must NOT contain the unsubstituted
        # token.
        for _ep, path in results:
            assert "{arch}" not in str(path)

    def test_yields_nothing_when_dll_missing_from_distribution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = PyPiSource(
            distribution="onnxruntime-ep-openvino",
            relative_dll="onnxruntime_ep_openvino/this_file_does_not_exist.dll",
            eps=("OpenVINOExecutionProvider",),
        )
        assert list(source.resolve()) == []


# ---------------------------------------------------------------------------
# FilesystemSource.
# ---------------------------------------------------------------------------


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


class TestFilesystemSource:
    """FilesystemSource scans a directory for plugin DLLs."""

    def test_resolves_single_dll_in_root(self, tmp_path: Path) -> None:
        dll = _touch(tmp_path / "onnxruntime_providers_vitisai.dll")
        source = FilesystemSource(
            root=tmp_path,
            dll_patterns={"VitisAIExecutionProvider": dll.name},
        )
        results = list(source.resolve())
        assert len(results) == 1
        ep_name, path = results[0]
        assert ep_name == "VitisAIExecutionProvider"
        assert path == dll.resolve()

    def test_yields_nothing_when_root_missing(self, tmp_path: Path) -> None:
        source = FilesystemSource(
            root=tmp_path / "does-not-exist",
            dll_patterns={"VitisAIExecutionProvider": "any.dll"},
        )
        assert list(source.resolve()) == []

    def test_env_var_unset_yields_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FAKE_INSTALLATION_PATH", raising=False)
        source = FilesystemSource(
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

        source = FilesystemSource(
            root=Path("deployment"),
            env_var="FAKE_INSTALLATION_PATH",
            dll_patterns={"VitisAIExecutionProvider": dll.name},
            required_marker=marker.name,
        )
        results = list(source.resolve())
        assert results == [("VitisAIExecutionProvider", dll.resolve())]

    def test_required_marker_missing_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deployment = tmp_path / "deployment"
        _touch(deployment / "onnxruntime_providers_vitisai.dll")
        monkeypatch.setenv("FAKE_INSTALLATION_PATH", str(tmp_path))

        source = FilesystemSource(
            root=Path("deployment"),
            env_var="FAKE_INSTALLATION_PATH",
            dll_patterns={"VitisAIExecutionProvider": "onnxruntime_providers_vitisai.dll"},
            required_marker="onnxruntime_providers_shared.dll",
        )
        assert list(source.resolve()) == []

    def test_multiple_eps_in_one_root(self, tmp_path: Path) -> None:
        dll_a = _touch(tmp_path / "onnxruntime_providers_openvino_plugin.dll")
        dll_b = _touch(tmp_path / "onnxruntime_providers_qnn.dll")
        source = FilesystemSource(
            root=tmp_path,
            dll_patterns={
                "OpenVINOExecutionProvider": dll_a.name,
                "QNNExecutionProvider": dll_b.name,
            },
        )
        results = dict(source.resolve())
        assert results == {
            "OpenVINOExecutionProvider": dll_a.resolve(),
            "QNNExecutionProvider": dll_b.resolve(),
        }

    def test_glob_pattern_matches(self, tmp_path: Path) -> None:
        dll = _touch(tmp_path / "subdir" / "onnxruntime_providers_qnn.dll")
        source = FilesystemSource(
            root=tmp_path,
            dll_patterns={"QNNExecutionProvider": "*/onnxruntime_providers_qnn.dll"},
        )
        results = list(source.resolve())
        assert results == [("QNNExecutionProvider", dll.resolve())]


# ---------------------------------------------------------------------------
# WinMlCatalogSource stub.
# ---------------------------------------------------------------------------


class TestWinMlCatalogSourceBindingMissing:
    """When the optional WinAppSDK ML binding is absent, resolve() yields nothing.

    Detailed behavior (mocked binding shape, atexit registration, etc.)
    is covered in ``test_winml_catalog_source.py``.
    """

    def test_resolve_yields_nothing_without_binding(self) -> None:
        # The autouse fixture ``_skip_live_catalog_in_ep_path_tests``
        # forces ``_get_catalog`` to return ``None``, simulating the
        # binding-missing case. resolve() must yield nothing silently.
        source = WinMlCatalogSource(
            catalog_name="VitisAI", eps=("VitisAIExecutionProvider",)
        )
        # ``resolve()`` is a generator; we have to iterate to trigger.
        assert list(source.resolve()) == []


# ---------------------------------------------------------------------------
# WINMLCLI_EP_PATH env var override.
# ---------------------------------------------------------------------------


class TestWinmlEpPathOverride:
    """Parsing the WINMLCLI_EP_PATH env var into FilesystemSource entries."""

    def test_unset_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
        assert _parse_winmlcli_ep_path() == []

    def test_empty_string_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WINMLCLI_EP_PATH", "")
        assert _parse_winmlcli_ep_path() == []

    def test_single_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WINMLCLI_EP_PATH", str(tmp_path))
        sources = _parse_winmlcli_ep_path()
        # Per the C1 fix, the parser emits ONE FilesystemSource per
        # (root, ep, dll_filename) combination — so a single entry with
        # five EPs (some with both .dll and .so filenames) yields more
        # than five sources. Every known EP must be covered at least once.
        assert all(isinstance(s, FilesystemSource) for s in sources)
        assert all(s.root == tmp_path for s in sources)
        covered_eps = {
            ep
            for s in sources
            if isinstance(s, FilesystemSource)
            for ep in s.dll_patterns
        }
        assert covered_eps == set(EP_DLL_NAMES.keys())

    def test_emits_source_per_dll_filename_for_cross_platform(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # C1: EP_DLL_NAMES["OpenVINOExecutionProvider"] has both the
        # Windows .dll filename and the Linux .so filename. The parser
        # must emit a FilesystemSource for EACH so a Linux user with
        # WINMLCLI_EP_PATH set finds .so files too.
        monkeypatch.setenv("WINMLCLI_EP_PATH", str(tmp_path))
        sources = _parse_winmlcli_ep_path()
        ov_dlls = [
            next(iter(s.dll_patterns.values()))
            for s in sources
            if isinstance(s, FilesystemSource)
            and "OpenVINOExecutionProvider" in s.dll_patterns
        ]
        assert "onnxruntime_providers_openvino_plugin.dll" in ov_dlls
        assert "libonnxruntime_providers_openvino_plugin.so" in ov_dlls

    def test_multi_entry_separator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        # os.pathsep is ; on Windows and : elsewhere — same as PATH.
        import os

        monkeypatch.setenv("WINMLCLI_EP_PATH", f"{a}{os.pathsep}{b}")
        sources = _parse_winmlcli_ep_path()
        roots = {s.root for s in sources if isinstance(s, FilesystemSource)}
        assert a in roots
        assert b in roots

    def test_winml_ep_path_finds_dll(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end: drop a synthetic plugin DLL in a directory, set
        # WINMLCLI_EP_PATH to that directory, confirm discover_eps finds it.
        dll = _touch(tmp_path / "onnxruntime_providers_vitisai.dll")
        monkeypatch.setenv("WINMLCLI_EP_PATH", str(tmp_path))
        # Skip env-var-gated FilesystemSource so the env-var path is the
        # only producer of a VitisAI hit. (The autouse
        # _skip_live_catalog_in_ep_path_tests fixture handles the catalog
        # source side.)
        monkeypatch.delenv("RYZEN_AI_INSTALLATION_PATH", raising=False)
        resolved = discover_eps()
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
        """WINMLCLI_EP_PATH with a valid directory yields one FilesystemSource per known EP DLL."""
        from winml.modelkit.ep_path import _parse_winmlcli_ep_path

        monkeypatch.setenv("WINMLCLI_EP_PATH", str(tmp_path))
        sources = _parse_winmlcli_ep_path()

        assert len(sources) > 0, "Expected at least one FilesystemSource"
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
        extra = FilesystemSource(
            root=tmp_path,
            dll_patterns={"OpenVINOExecutionProvider": fake_dll.name},
        )
        monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
        resolved = discover_eps(extra_sources=[extra])
        assert "OpenVINOExecutionProvider" in resolved
        path, source = resolved["OpenVINOExecutionProvider"]
        assert path == fake_dll.resolve()
        assert source is extra

    def test_first_hit_wins_among_extra_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = _touch(tmp_path / "a" / "vitisai.dll")
        b = _touch(tmp_path / "b" / "vitisai.dll")
        first = FilesystemSource(
            root=a.parent,
            dll_patterns={"VitisAIExecutionProvider": a.name},
        )
        second = FilesystemSource(
            root=b.parent,
            dll_patterns={"VitisAIExecutionProvider": b.name},
        )
        monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
        resolved = discover_eps(extra_sources=[first, second])
        assert resolved["VitisAIExecutionProvider"][0] == a.resolve()

    def test_winml_catalog_source_does_not_abort_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On a CI machine without the WinAppSDK binding, resolve() on a
        # WinMlCatalogSource yields nothing and discover_eps continues
        # to subsequent sources. ``_get_catalog`` is already mocked to
        # None by the file-scoped autouse fixture
        # ``_skip_live_catalog_in_ep_path_tests``.
        fake_dll = _touch(tmp_path / "fake_qnn.dll")
        catalog = WinMlCatalogSource(catalog_name="QNN", eps=("QNNExecutionProvider",))
        good = FilesystemSource(
            root=tmp_path,
            dll_patterns={"QNNExecutionProvider": fake_dll.name},
        )
        monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
        resolved = discover_eps(extra_sources=[catalog, good])
        assert resolved["QNNExecutionProvider"][0] == fake_dll.resolve()

    def test_resolve_returns_path_and_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_dll = _touch(tmp_path / "fake.dll")
        src = FilesystemSource(
            root=tmp_path,
            dll_patterns={"QNNExecutionProvider": fake_dll.name},
        )
        monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
        resolved = discover_eps(extra_sources=[src])
        assert resolved["QNNExecutionProvider"] == (fake_dll.resolve(), src)

    def test_discover_eps_returns_tuple_shape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """discover_eps() (no flag) returns dict[str, (Path, EpSource)]."""
        fake_dll = _touch(tmp_path / "fake.dll")
        src = FilesystemSource(
            root=tmp_path,
            dll_patterns={"QNNExecutionProvider": fake_dll.name},
        )
        monkeypatch.delenv("WINMLCLI_EP_PATH", raising=False)
        result = discover_eps(extra_sources=[src])
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
        ep_name, path = results[0]
        assert ep_name == "FooExecutionProvider"
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
        _, path = results[0]
        # The version picked should be rc.10 (semantically newer).
        path_str = str(path)
        assert "1.0.0-rc.10" in path_str


# ---------------------------------------------------------------------------
# Vendor detection error handling.
# ---------------------------------------------------------------------------


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

