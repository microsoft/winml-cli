# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for ``NuGetSource``.

The on-disk NuGet cache layout (``~/.nuget/packages/<id>/<version>/...``)
is mocked via ``tmp_path`` + ``monkeypatch.setenv("USERPROFILE", ...)``
so the tests run hermetically without a real NuGet install. The
``relative_dll`` paths mimic the real package layout
(``runtimes/<rid>/native/<dll>``) verified against
``Intel.ML.OnnxRuntime.EP.OpenVINO`` 1.4.0 and
``Qualcomm.ML.OnnxRuntime.QNN`` 2.1.0 in Phase 1 research.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from winml.modelkit.ep_path import NuGetSource


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _make_cache_pkg(
    home: Path,
    distribution: str,
    version: str,
    relative_dll: str,
    *,
    create_dll: bool = True,
) -> Path:
    """Build a fake NuGet cache entry under ``home/.nuget/packages/<id>/<ver>/``.

    Returns the absolute path to the DLL (whether or not it was created).
    """
    pkg_dir = (
        home
        / ".nuget"
        / "packages"
        / distribution.lower()
        / version
    )
    pkg_dir.mkdir(parents=True, exist_ok=True)
    dll_path = pkg_dir / relative_dll
    if create_dll:
        dll_path.parent.mkdir(parents=True, exist_ok=True)
        dll_path.write_bytes(b"")
    return dll_path


_OPENVINO_REL = "runtimes/win-x64/native/onnxruntime_providers_openvino_plugin.dll"


# ---------------------------------------------------------------------------
# NuGetSource.resolve() — selection rules.
# ---------------------------------------------------------------------------


class TestNuGetSource:
    """Selection rules for ``NuGetSource.resolve``."""

    def test_single_version_yields_dll(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        dll = _make_cache_pkg(
            tmp_path,
            "Intel.ML.OnnxRuntime.EP.OpenVINO",
            "1.4.0",
            _OPENVINO_REL,
        )

        src = NuGetSource(
            distribution="Intel.ML.OnnxRuntime.EP.OpenVINO",
            relative_dll=_OPENVINO_REL,
            eps=("OpenVINOExecutionProvider",),
        )
        results = list(src.resolve())
        assert len(results) == 1
        entry = results[0]
        assert entry.ep_name == "OpenVINOExecutionProvider"
        assert entry.dll_path == dll.resolve()
        assert entry.dll_path.is_file()
        # NuGetSource plumbs the cache-subdir version into EPEntry.
        assert entry.version == "1.4.0"

    def test_multiple_versions_picks_highest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        for ver in ("1.0.0", "1.4.0", "1.2.0"):
            _make_cache_pkg(
                tmp_path,
                "Intel.ML.OnnxRuntime.EP.OpenVINO",
                ver,
                _OPENVINO_REL,
            )

        src = NuGetSource(
            distribution="Intel.ML.OnnxRuntime.EP.OpenVINO",
            relative_dll=_OPENVINO_REL,
            eps=("OpenVINOExecutionProvider",),
        )
        results = list(src.resolve())
        assert len(results) == 1
        dll_path = results[0].dll_path
        # Path components include the version-string subdir; assert on it.
        assert "1.4.0" in dll_path.parts

    def test_prerelease_skipped_when_stable_exists(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        for ver in ("1.4.0", "1.5.0-beta"):
            _make_cache_pkg(
                tmp_path,
                "Intel.ML.OnnxRuntime.EP.OpenVINO",
                ver,
                _OPENVINO_REL,
            )

        src = NuGetSource(
            distribution="Intel.ML.OnnxRuntime.EP.OpenVINO",
            relative_dll=_OPENVINO_REL,
            eps=("OpenVINOExecutionProvider",),
        )
        results = list(src.resolve())
        assert len(results) == 1
        dll_path = results[0].dll_path
        assert "1.4.0" in dll_path.parts
        assert "1.5.0-beta" not in dll_path.parts

    def test_prerelease_used_when_only_option(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        _make_cache_pkg(
            tmp_path,
            "Intel.ML.OnnxRuntime.EP.OpenVINO",
            "1.5.0-beta",
            _OPENVINO_REL,
        )

        src = NuGetSource(
            distribution="Intel.ML.OnnxRuntime.EP.OpenVINO",
            relative_dll=_OPENVINO_REL,
            eps=("OpenVINOExecutionProvider",),
        )
        results = list(src.resolve())
        assert len(results) == 1
        dll_path = results[0].dll_path
        assert "1.5.0-beta" in dll_path.parts

    def test_package_absent_yields_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        # No package dir created at all.

        src = NuGetSource(
            distribution="Intel.ML.OnnxRuntime.EP.OpenVINO",
            relative_dll=_OPENVINO_REL,
            eps=("OpenVINOExecutionProvider",),
        )
        assert list(src.resolve()) == []

    def test_dll_missing_yields_nothing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        # Version dir present but DLL not at expected path.
        _make_cache_pkg(
            tmp_path,
            "Intel.ML.OnnxRuntime.EP.OpenVINO",
            "1.4.0",
            _OPENVINO_REL,
            create_dll=False,
        )

        src = NuGetSource(
            distribution="Intel.ML.OnnxRuntime.EP.OpenVINO",
            relative_dll=_OPENVINO_REL,
            eps=("OpenVINOExecutionProvider",),
        )
        assert list(src.resolve()) == []

    def test_arch_resolver_substitutes_into_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        # Real on-disk DLL lives under win-arm64 (matching QNN).
        rel_real = "runtimes/win-arm64/native/onnxruntime_providers_qnn.dll"
        _make_cache_pkg(
            tmp_path,
            "Qualcomm.ML.OnnxRuntime.QNN",
            "2.1.0",
            rel_real,
        )
        rel_template = "runtimes/win-{rid}/native/onnxruntime_providers_qnn.dll"

        src = NuGetSource(
            distribution="Qualcomm.ML.OnnxRuntime.QNN",
            relative_dll=rel_template,
            eps=("QNNExecutionProvider",),
            arch_resolver=lambda t: t.format(rid="arm64"),
        )
        results = list(src.resolve())
        assert len(results) == 1
        entry = results[0]
        assert entry.ep_name == "QNNExecutionProvider"
        assert entry.dll_path.name == "onnxruntime_providers_qnn.dll"
        assert "win-arm64" in entry.dll_path.parts

    def test_iter_eps_returns_declared_eps(
        self,
    ) -> None:
        src = NuGetSource(
            distribution="Intel.ML.OnnxRuntime.EP.OpenVINO",
            relative_dll=_OPENVINO_REL,
            eps=("OpenVINOExecutionProvider",),
        )
        assert tuple(src.iter_eps()) == ("OpenVINOExecutionProvider",)

    def test_is_compatible_matches_hardware(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Hit both branches by mocking _get_detected_vendors.
        from winml.modelkit import ep_path as _ep

        _ep._get_detected_vendors.cache_clear()
        monkeypatch.setattr(
            _ep,
            "_get_detected_vendors",
            lambda: frozenset({"Intel(R) Corporation"}),
        )
        ov_src = NuGetSource(
            distribution="Intel.ML.OnnxRuntime.EP.OpenVINO",
            relative_dll=_OPENVINO_REL,
            eps=("OpenVINOExecutionProvider",),
        )
        qnn_src = NuGetSource(
            distribution="Qualcomm.ML.OnnxRuntime.QNN",
            relative_dll="runtimes/win-arm64/native/onnxruntime_providers_qnn.dll",
            eps=("QNNExecutionProvider",),
        )
        # Intel host: OpenVINO compatible, QNN not.
        assert ov_src.is_compatible() is True
        assert qnn_src.is_compatible() is False
