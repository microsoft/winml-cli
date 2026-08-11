# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``_entry_source_tag`` dispatch in ep_registry.

``_entry_source_tag`` maps an :class:`EPEntry`'s ``.source`` (an
:class:`EPSource` subclass instance) to one of the canonical tag strings
used by :meth:`WinMLEPRegistry.auto_device` to filter cached discovery
entries by source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from winml.modelkit.ep_path import (
    BuiltinSource,
    DirectorySource,
    EPEntry,
    MSIXPackageSource,
    NuGetSource,
    PyPISource,
    WinMLCatalogSource,
)
from winml.modelkit.session.ep_registry import _entry_source_tag


_FAKE_DLL = Path("C:/fake/plugin.dll")


def _entry(source: object) -> EPEntry:
    """Build an EPEntry whose source field is the value under test."""
    return EPEntry(
        ep_name="OpenVINOExecutionProvider",
        dll_path=_FAKE_DLL,
        source=source,  # type: ignore[arg-type]
    )


def _pypi() -> PyPISource:
    return PyPISource(
        distribution="onnxruntime-ep-openvino",
        relative_dll="onnxruntime_ep_openvino/plugin.dll",
        eps=("OpenVINOExecutionProvider",),
    )


def _nuget() -> NuGetSource:
    return NuGetSource(
        distribution="Intel.ML.OnnxRuntime.EP.OpenVINO",
        relative_dll="runtimes/win-x64/native/plugin.dll",
        eps=("OpenVINOExecutionProvider",),
    )


def _winml_catalog() -> WinMLCatalogSource:
    return WinMLCatalogSource(
        catalog_name="OpenVINOExecutionProvider",
        eps=("OpenVINOExecutionProvider",),
    )


def _directory() -> DirectorySource:
    return DirectorySource(
        root=Path("C:/fake/dir"),
        dll_patterns={"OpenVINOExecutionProvider": "plugin.dll"},
    )


def _msix_ms_channel() -> MSIXPackageSource:
    return MSIXPackageSource(
        family_name_prefix="MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.",
        relative_dll="ExecutionProvider/plugin.dll",
        eps=("OpenVINOExecutionProvider",),
    )


def _msix_workload() -> MSIXPackageSource:
    return MSIXPackageSource(
        family_name_prefix="WindowsWorkload.EP.Intel.OpenVINO.",
        relative_dll="ExecutionProvider/plugin.dll",
        eps=("OpenVINOExecutionProvider",),
    )


def _builtin() -> BuiltinSource:
    return BuiltinSource(eps=("CPUExecutionProvider",))


@pytest.mark.parametrize(
    ("source_factory", "expected_tag"),
    [
        (_pypi, "pypi"),
        (_nuget, "nuget"),
        (_winml_catalog, "winml-catalog"),
        (_directory, "directory"),
        (_msix_ms_channel, "msix"),
        (_msix_workload, "msix"),
        (_builtin, "bundled"),
    ],
    ids=[
        "pypi",
        "nuget",
        "winml-catalog",
        "directory",
        "msix-ms-channel",
        "msix-workload-channel",
        "bundled",
    ],
)
def test_entry_source_tag_dispatch(source_factory: object, expected_tag: str) -> None:
    """Each EPSource subclass maps to its canonical tag string."""
    entry = _entry(source_factory())  # type: ignore[operator]
    assert _entry_source_tag(entry) == expected_tag


def test_msix_tag_is_channel_agnostic() -> None:
    """MSIXPackageSource always maps to ``"msix"`` regardless of channel.

    Both the ``MicrosoftCorporationII.*`` and ``WindowsWorkload.*``
    family-name prefixes collapse to the single ``"msix"`` tag — the
    per-channel distinction no longer surfaces in the source vocabulary.
    """
    workload = MSIXPackageSource(
        family_name_prefix="WindowsWorkload.EP.Intel.OpenVINO.1.8_",
        relative_dll="ExecutionProvider/plugin.dll",
        eps=("OpenVINOExecutionProvider",),
    )
    ms_channel = MSIXPackageSource(
        family_name_prefix="MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.",
        relative_dll="ExecutionProvider/plugin.dll",
        eps=("QNNExecutionProvider",),
    )
    assert _entry_source_tag(_entry(workload)) == "msix"
    assert _entry_source_tag(_entry(ms_channel)) == "msix"


def test_entry_source_tag_unknown_fallback() -> None:
    """A non-EPSource ``source`` value yields the ``"unknown"`` fallback."""
    # None — caller hand-rolled an EPEntry with no source.
    assert _entry_source_tag(_entry(None)) == "unknown"
    # Arbitrary non-EPSource value — same fallback.
    assert _entry_source_tag(_entry(object())) == "unknown"
