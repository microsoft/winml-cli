# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for ``MSIXPackageSource`` and ``list_msix_eps``.

The WinRT ``PackageManager`` is mocked via ``_get_pkg_manager`` so the
tests are hermetic and run on any platform. Synthetic ``Package`` shapes
mimic the real ``winrt.windows.management.deployment.Package`` interface
with the fields actually consumed by the implementation
(``id.family_name``, ``id.full_name``, ``id.version.{major,minor,build,
revision}``, ``installed_path``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from winml.modelkit import ep_path as _ep
from winml.modelkit.ep_path import MSIXPackageSource, list_msix_eps


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fake PackageManager helpers.
# ---------------------------------------------------------------------------


class _FakeVersion:
    def __init__(self, major: int, minor: int, build: int, revision: int) -> None:
        self.major = major
        self.minor = minor
        self.build = build
        self.revision = revision


class _FakePackageId:
    def __init__(self, family_name: str, version: _FakeVersion) -> None:
        self.family_name = family_name
        self.version = version
        # full_name is informational; resolve() does not use it.
        head, _, tail = family_name.partition("_")
        ver_str = (
            f"{version.major}.{version.minor}.{version.build}.{version.revision}"
        )
        self.full_name = f"{head}_{ver_str}_arm64__{tail}"


class _FakePackage:
    def __init__(
        self,
        family_name: str,
        version: tuple[int, int, int, int],
        installed_path: Path,
    ) -> None:
        self.id = _FakePackageId(family_name, _FakeVersion(*version))
        self.installed_path = str(installed_path)


class _FakeManager:
    """Mimics the bits of ``PackageManager`` we use."""

    def __init__(self, packages: list[_FakePackage] | Exception) -> None:
        self._packages = packages

    def find_packages_by_user_security_id(self, sid: str) -> list[_FakePackage]:
        if isinstance(self._packages, Exception):
            raise self._packages
        # Empty SID = current user (matching real WinRT behavior).
        assert sid == "", f"expected empty SID for current user, got {sid!r}"
        return list(self._packages)


@pytest.fixture
def reset_pkg_manager_cache() -> None:
    """Reset the @functools.cache singleton between tests."""
    _ep._get_pkg_manager.cache_clear()


def _make_qnn_package(
    tmp_path: Path,
    family_short: str,
    version: tuple[int, int, int, int],
    *,
    create_dll: bool = True,
    dll_relative: str = "ExecutionProvider/onnxruntime_providers_qnn.dll",
) -> _FakePackage:
    """Build a fake QNN MSIX package on disk under tmp_path.

    Uses the real on-disk layout (``ExecutionProvider/<dll>``) so
    ``MSIXPackageSource.resolve()`` sees a real DLL file. Returns the
    fake Package with installed_path pointing at the layout.
    """
    family_name = f"MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.{family_short}_8wekyb3d8bbwe"
    install_root = tmp_path / family_name / f"v{'.'.join(str(p) for p in version)}"
    install_root.mkdir(parents=True, exist_ok=True)
    if create_dll:
        # POSIX path joined via Path / str works on both Windows and POSIX
        # without explicit separator translation. The invariant being
        # tested: MSIXPackageSource.relative_dll is POSIX-style; resolve()
        # rejects backslash inputs. Don't mask that with a manual replace.
        dll = install_root / dll_relative
        dll.parent.mkdir(parents=True, exist_ok=True)
        dll.write_bytes(b"")
    return _FakePackage(family_name, version, install_root)


# ---------------------------------------------------------------------------
# MSIXPackageSource.resolve()
# ---------------------------------------------------------------------------


class TestMSIXPackageSourceResolve:
    """Selection rules for ``MSIXPackageSource.resolve``."""

    def test_single_match_yields_dll(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        pkg = _make_qnn_package(tmp_path, "1.8", (1, 8, 30, 0))
        monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: _FakeManager([pkg]))

        src = MSIXPackageSource(
            family_name_prefix="MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.1.8_",
            relative_dll="ExecutionProvider/onnxruntime_providers_qnn.dll",
            eps=("QNNExecutionProvider",),
        )
        results = list(src.resolve())
        assert len(results) == 1
        entry = results[0]
        assert entry.ep_name == "QNNExecutionProvider"
        assert entry.dll_path.name == "onnxruntime_providers_qnn.dll"
        assert entry.dll_path.is_file()
        # MSIXPackageSource plumbs the matched package's Package.Id.Version
        # into EPEntry as "M.m.b.r".
        assert entry.version == "1.8.30.0"

    def test_multiple_versions_picks_highest(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Two builds within the same family: pick the highest.
        pkg_old = _make_qnn_package(tmp_path, "2", (2, 2400, 1, 0))
        pkg_new = _make_qnn_package(tmp_path, "2", (2, 2420, 44, 0))
        monkeypatch.setattr(
            _ep, "_get_pkg_manager", lambda: _FakeManager([pkg_old, pkg_new])
        )

        src = MSIXPackageSource(
            family_name_prefix="MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.2_",
            relative_dll="ExecutionProvider/onnxruntime_providers_qnn.dll",
            eps=("QNNExecutionProvider",),
        )
        results = list(src.resolve())
        assert len(results) == 1
        dll_path = results[0].dll_path
        assert "v2.2420.44.0" in str(dll_path)

    def test_version_pin_selects_exact(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        pkg_v1 = _make_qnn_package(tmp_path, "1.8", (1, 8, 30, 0))
        pkg_v2 = _make_qnn_package(tmp_path, "2", (2, 2420, 44, 0))
        monkeypatch.setattr(
            _ep, "_get_pkg_manager", lambda: _FakeManager([pkg_v1, pkg_v2])
        )

        src = MSIXPackageSource(
            family_name_prefix="MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.",  # spans both
            relative_dll="ExecutionProvider/onnxruntime_providers_qnn.dll",
            eps=("QNNExecutionProvider",),
            version="1.8.30.0",
        )
        results = list(src.resolve())
        assert len(results) == 1
        dll_path = results[0].dll_path
        assert "v1.8.30.0" in str(dll_path)

    def test_version_pin_no_match_yields_nothing(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        pkg = _make_qnn_package(tmp_path, "1.8", (1, 8, 30, 0))
        monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: _FakeManager([pkg]))

        src = MSIXPackageSource(
            family_name_prefix="MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.1.8_",
            relative_dll="ExecutionProvider/onnxruntime_providers_qnn.dll",
            eps=("QNNExecutionProvider",),
            version="9.9.9.9",
        )
        assert list(src.resolve()) == []

    def test_prefix_no_match_yields_nothing(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        pkg = _make_qnn_package(tmp_path, "1.8", (1, 8, 30, 0))
        monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: _FakeManager([pkg]))

        src = MSIXPackageSource(
            family_name_prefix="MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.",
            relative_dll="ignored",
            eps=("OpenVINOExecutionProvider",),
        )
        assert list(src.resolve()) == []

    def test_dll_missing_yields_nothing(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pkg = _make_qnn_package(
            tmp_path, "1.8", (1, 8, 30, 0), create_dll=False
        )
        monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: _FakeManager([pkg]))

        src = MSIXPackageSource(
            family_name_prefix="MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.1.8_",
            relative_dll="ExecutionProvider/onnxruntime_providers_qnn.dll",
            eps=("QNNExecutionProvider",),
        )
        import logging
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.ep_path"):
            assert list(src.resolve()) == []
        assert any("DLL missing" in r.message for r in caplog.records)

    def test_binding_unavailable_yields_nothing(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: None)
        src = MSIXPackageSource(
            family_name_prefix="MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.1.8_",
            relative_dll="ignored",
            eps=("QNNExecutionProvider",),
        )
        assert list(src.resolve()) == []

    def test_backslash_relative_dll_raises(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # POSIX-style invariant: resolve() must reject backslash separators
        # so a hand-constructed source fails loudly instead of silently
        # returning empty on Linux.
        pkg = _make_qnn_package(tmp_path, "1.8", (1, 8, 30, 0))
        monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: _FakeManager([pkg]))
        src = MSIXPackageSource(
            family_name_prefix="MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.1.8_",
            relative_dll="ExecutionProvider\\onnxruntime_providers_qnn.dll",
            eps=("QNNExecutionProvider",),
        )
        with pytest.raises(ValueError, match="POSIX-style"):
            list(src.resolve())

    def test_find_packages_raises_yields_nothing(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            _ep,
            "_get_pkg_manager",
            lambda: _FakeManager(RuntimeError("WinRT failure")),
        )
        src = MSIXPackageSource(
            family_name_prefix="...QNN.EP.1.8_",
            relative_dll="ignored",
            eps=("QNNExecutionProvider",),
        )
        import logging
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.ep_path"):
            assert list(src.resolve()) == []
        assert any("find_packages_by_user_security_id" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# list_msix_eps()
# ---------------------------------------------------------------------------


class TestListMsixEps:
    """``list_msix_eps`` returns one fully-pinned MSIXPackageSource per match."""

    def test_returns_one_per_family_version(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        pkg_v1 = _make_qnn_package(tmp_path, "1.8", (1, 8, 30, 0))
        pkg_v2 = _make_qnn_package(tmp_path, "2", (2, 2420, 44, 0))
        monkeypatch.setattr(
            _ep, "_get_pkg_manager", lambda: _FakeManager([pkg_v1, pkg_v2])
        )

        results = list_msix_eps()
        assert len(results) == 2
        # Sorted by family then version → 1.8 first then 2 (lexical sort
        # of family_name strings).
        assert "QNN.EP.1.8" in results[0].family_name_prefix
        assert results[0].version == "1.8.30.0"
        assert "QNN.EP.2" in results[1].family_name_prefix
        assert results[1].version == "2.2420.44.0"

    def test_each_result_is_round_trip_resolvable(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        pkg = _make_qnn_package(tmp_path, "1.8", (1, 8, 30, 0))
        monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: _FakeManager([pkg]))

        results = list_msix_eps()
        assert len(results) == 1
        # The returned MSIXPackageSource must resolve back to the same DLL.
        listed = results[0]
        resolved = list(listed.resolve())
        assert len(resolved) == 1
        entry = resolved[0]
        assert entry.ep_name == "QNNExecutionProvider"
        assert entry.dll_path.is_file()

    def test_auto_detects_ep_name_from_dll_filename(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Synthesize an OpenVINO MSIX with the openvino plugin DLL name.
        family = "MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_8wekyb3d8bbwe"
        install_root = tmp_path / family
        ep_dir = install_root / "ExecutionProvider"
        ep_dir.mkdir(parents=True)
        (ep_dir / "onnxruntime_providers_openvino_plugin.dll").write_bytes(b"")
        pkg = _FakePackage(family, (1, 8, 69, 0), install_root)
        monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: _FakeManager([pkg]))

        results = list_msix_eps()
        assert len(results) == 1
        assert results[0].eps == ("OpenVINOExecutionProvider",)

    def test_skips_packages_with_no_recognizable_dll(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Package matches the prefix but contains no onnxruntime_providers_*.dll.
        family = "MicrosoftCorporationII.WinML.Random.Junk.EP.1.0_8wekyb3d8bbwe"
        install_root = tmp_path / family
        install_root.mkdir(parents=True)
        (install_root / "some_other.dll").write_bytes(b"")
        pkg = _FakePackage(family, (1, 0, 0, 0), install_root)
        monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: _FakeManager([pkg]))

        assert list_msix_eps() == []

    def test_filters_by_prefix(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        qnn = _make_qnn_package(tmp_path, "2", (2, 2420, 44, 0))
        # A non-WinML package that should not match the default prefix.
        other_pkg = _FakePackage(
            "Microsoft.UnrelatedApp_8wekyb3d8bbwe",
            (1, 0, 0, 0),
            tmp_path / "unrelated",
        )
        monkeypatch.setattr(
            _ep, "_get_pkg_manager", lambda: _FakeManager([qnn, other_pkg])
        )

        results = list_msix_eps()  # default prefix
        assert len(results) == 1
        assert "QNN.EP.2" in results[0].family_name_prefix

    def test_default_matches_windows_workload_channel(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # The OEM/Windows-Workloads channel publishes EP MSIXes under a
        # different family root than the public WinML EP catalog channel.
        # On Lunar Lake hardware the Intel OpenVINO EP arrives as
        # ``WindowsWorkload.EP.Intel.OpenVINO.1.8_*`` rather than
        # ``MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.*``. The default
        # prefix list must catch both so ``winml sys --list-ep`` surfaces
        # the OEM-provisioned package alongside any catalog-managed one.
        family_oem = (
            "WindowsWorkload.EP.Intel.OpenVINO.1.8_8wekyb3d8bbwe"
        )
        oem_root = tmp_path / family_oem
        ep_dir_oem = oem_root / "ExecutionProvider"
        ep_dir_oem.mkdir(parents=True)
        (ep_dir_oem / "onnxruntime_providers_openvino_plugin.dll").write_bytes(b"")
        oem_pkg = _FakePackage(family_oem, (1, 8, 61, 0), oem_root)

        family_catalog = (
            "MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_8wekyb3d8bbwe"
        )
        catalog_root = tmp_path / family_catalog
        ep_dir_cat = catalog_root / "ExecutionProvider"
        ep_dir_cat.mkdir(parents=True)
        (ep_dir_cat / "onnxruntime_providers_openvino_plugin.dll").write_bytes(b"")
        catalog_pkg = _FakePackage(family_catalog, (1, 8, 69, 0), catalog_root)

        monkeypatch.setattr(
            _ep, "_get_pkg_manager", lambda: _FakeManager([oem_pkg, catalog_pkg])
        )

        results = list_msix_eps()  # default prefixes
        family_names = {r.family_name_prefix for r in results}
        assert family_oem in family_names, (
            f"WindowsWorkload-channel package not matched. Got: {family_names}"
        )
        assert family_catalog in family_names, (
            f"MicrosoftCorporationII-channel package not matched. Got: {family_names}"
        )
        assert len(results) == 2

    def test_binding_unavailable_returns_empty(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(_ep, "_get_pkg_manager", lambda: None)
        assert list_msix_eps() == []

    def test_find_packages_raises_returns_empty(
        self,
        reset_pkg_manager_cache: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            _ep,
            "_get_pkg_manager",
            lambda: _FakeManager(RuntimeError("WinRT failure")),
        )
        import logging
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.ep_path"):
            assert list_msix_eps() == []
        assert any("find_packages_by_user_security_id" in r.message for r in caplog.records)
