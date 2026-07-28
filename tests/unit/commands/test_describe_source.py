# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``commands.sys._describe_source`` per-source descriptor.

``_describe_source`` builds the JSON-friendly per-source dict surfaced by
``winml sys --list-ep --format json``. T-11 wires it through
:func:`session.ep_registry._entry_source_tag` so the canonical short tag
(``"pypi"``, ``"bundled"``, ``"msix"`` …) lives in one place instead of
being re-derived ad-hoc by future renderers from the ``source_kind``
class name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from winml.modelkit.commands.sys import _describe_source
from winml.modelkit.ep_path import (
    BuiltinSource,
    DirectorySource,
    EPEntry,
    MSIXPackageSource,
    NuGetSource,
    PyPISource,
    WinMLCatalogSource,
)


_FAKE_DLL = Path("C:/fake/plugin.dll")


def _entry(source: object, version: str | None = None) -> EPEntry:
    return EPEntry(
        ep_name="OpenVINOExecutionProvider",
        dll_path=_FAKE_DLL,
        source=source,  # type: ignore[arg-type]
        version=version,
    )


def _pypi() -> PyPISource:
    return PyPISource(
        distribution="onnxruntime-ep-openvino",
        relative_dll="onnxruntime_ep_openvino/plugin.dll",
        eps=("OpenVINOExecutionProvider",),
    )


def _nuget() -> NuGetSource:
    return NuGetSource(
        distribution="Microsoft.ML.OnnxRuntime.QNN",
        relative_dll="runtimes/win-arm64/native/QnnEp.dll",
        eps=("QNNExecutionProvider",),
    )


def _winml_catalog() -> WinMLCatalogSource:
    return WinMLCatalogSource(
        catalog_name="OpenVINO",
        eps=("OpenVINOExecutionProvider",),
    )


def _directory() -> DirectorySource:
    return DirectorySource(
        root=Path(r"C:/fake/ep-dir"),
        dll_patterns={"OpenVINOExecutionProvider": "openvino_ep.dll"},
        env_var="WINMLCLI_EP_PATH",
    )


def _msix_ms_channel() -> MSIXPackageSource:
    return MSIXPackageSource(
        family_name_prefix="MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.",
        relative_dll="ExecutionProvider/plugin.dll",
        eps=("QNNExecutionProvider",),
    )


def _msix_workload() -> MSIXPackageSource:
    return MSIXPackageSource(
        family_name_prefix="WindowsWorkload.EP.Intel.OpenVINO.1.8_",
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
)
def test_describe_source_emits_canonical_source_tag(
    source_factory: object,
    expected_tag: str,
) -> None:
    """``_describe_source`` carries the canonical short tag for every kind.

    The dedup contract introduced by T-11: ``source_tag`` is derived by
    :func:`_entry_source_tag`, so adding a new ``EPSource`` subclass only
    requires updating the tag table once.
    """
    desc = _describe_source(_entry(source_factory()))  # type: ignore[operator]
    assert desc["source_tag"] == expected_tag


def test_describe_source_keeps_source_kind_class_name() -> None:
    """``source_kind`` remains the EPSource subclass name (verbose form).

    The class-name field is the legacy identifier; ``source_tag`` is the
    canonical short form. Both surface on the JSON contract so existing
    consumers continue to work.
    """
    desc = _describe_source(_entry(_builtin()))
    assert desc["source_kind"] == "BuiltinSource"
    assert desc["source_tag"] == "bundled"
