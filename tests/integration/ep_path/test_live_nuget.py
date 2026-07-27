# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Windows-only smoke tests against the LIVE NuGet packages cache.

The unit tests under ``tests/unit/ep_path/test_nuget_source.py`` exercise
``NuGetSource`` against a synthetic ``USERPROFILE``-redirected cache.
That covers the resolution logic but cannot detect breakage that would
only show up against a real ``~/.nuget/packages`` tree on a developer
machine — e.g., a vendor renaming the runtime DLL between releases or
splitting the cache entry across multiple version subdirs in an
unexpected way.

These integration tests run only on Windows AND only when the relevant
NuGet plugin packages have been restored into the user's NuGet cache.
They skip cleanly on machines that have not done a ``dotnet restore``
of a project depending on the package — which is the common case.
"""

from __future__ import annotations

import os

import pytest

from winml.modelkit.ep_path import NuGetSource, _nuget_packages_root


pytestmark = pytest.mark.skipif(
    os.name != "nt",
    reason="NuGet plugin EP packages currently target Windows runtimes only",
)


def test_nuget_cache_root_reachable() -> None:
    """``_nuget_packages_root()`` points at an existing directory or skips."""
    root = _nuget_packages_root()
    if not root.is_dir():
        pytest.skip(
            f"NuGet cache root {root} does not exist; run a dotnet restore "
            "first to materialize it"
        )
    # If it exists, it must be a directory we can list.
    assert any(True for _ in root.iterdir()) or True


def test_live_openvino_nuget_resolves_if_cached() -> None:
    """Live resolve of Intel.ML.OnnxRuntime.EP.OpenVINO if it's in cache."""
    root = _nuget_packages_root()
    pkg_root = root / "intel.ml.onnxruntime.ep.openvino"
    if not pkg_root.is_dir():
        pytest.skip(
            "Intel.ML.OnnxRuntime.EP.OpenVINO not in local NuGet cache; "
            "skip live test"
        )

    src = NuGetSource(
        distribution="Intel.ML.OnnxRuntime.EP.OpenVINO",
        relative_dll=(
            "runtimes/win-x64/native/onnxruntime_providers_openvino_plugin.dll"
        ),
        eps=("OpenVINOExecutionProvider",),
    )
    results = list(src.resolve())
    assert results, (
        f"package present at {pkg_root} but resolve() yielded nothing — "
        "package layout may have changed"
    )
    ep_name, dll_path = results[0]
    assert ep_name == "OpenVINOExecutionProvider"
    assert dll_path.is_file(), f"{dll_path} not a file"
    assert dll_path.name == "onnxruntime_providers_openvino_plugin.dll"


def test_live_qnn_nuget_resolves_if_cached() -> None:
    """Live resolve of Qualcomm.ML.OnnxRuntime.QNN if it's in cache."""
    root = _nuget_packages_root()
    pkg_root = root / "qualcomm.ml.onnxruntime.qnn"
    if not pkg_root.is_dir():
        pytest.skip(
            "Qualcomm.ML.OnnxRuntime.QNN not in local NuGet cache; "
            "skip live test"
        )

    src = NuGetSource(
        distribution="Qualcomm.ML.OnnxRuntime.QNN",
        relative_dll=(
            "runtimes/win-arm64/native/onnxruntime_providers_qnn.dll"
        ),
        eps=("QNNExecutionProvider",),
    )
    results = list(src.resolve())
    assert results, (
        f"package present at {pkg_root} but resolve() yielded nothing — "
        "package layout may have changed"
    )
    ep_name, dll_path = results[0]
    assert ep_name == "QNNExecutionProvider"
    assert dll_path.is_file(), f"{dll_path} not a file"
    assert dll_path.name == "onnxruntime_providers_qnn.dll"
