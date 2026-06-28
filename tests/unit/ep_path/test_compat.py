# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for EP hardware-compatibility helpers in ``ep_path``.

Covers:
    - ``EP_CATALOG`` vendor requirement contents.
    - ``_get_detected_vendors()`` aggregation across GPU/NPU.
    - ``EP_CATALOG.is_compatible()`` matching rules.
    - ``is_compatible()`` method on every EPSource subclass.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from winml.modelkit.ep_path import (
    EP_CATALOG,
    DirectorySource,
    MSIXPackageSource,
    PyPISource,
    WinMLCatalogSource,
    _get_detected_vendors,
)


@pytest.fixture
def reset_vendor_cache() -> None:
    """Clear the vendor-detection cache before each test."""
    _get_detected_vendors.cache_clear()


# ---------------------------------------------------------------------------
# EP_CATALOG vendor requirement entries.
# ---------------------------------------------------------------------------


class TestVendorRequirementTable:
    """Sanity checks on the EP_CATALOG vendor requirement entries."""

    def test_required_keys_present(self) -> None:
        catalog_eps = set(EP_CATALOG.all_eps())
        for ep in (
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "VitisAIExecutionProvider",
            "MIGraphXExecutionProvider",
            "NvTensorRtRtxExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
            "AzureExecutionProvider",
        ):
            assert ep in catalog_eps

    def test_unconstrained_eps_have_empty_set(self) -> None:
        # CPU/DML/Azure work everywhere — empty requirement, never marked
        # incompatible.
        from winml.modelkit.ep_path import EPCatalog

        for ep in ("CPUExecutionProvider", "DmlExecutionProvider", "AzureExecutionProvider"):
            # Access the internal entry to inspect vendor_requirements directly.
            entry: EPCatalog.Row = EP_CATALOG._by_name[ep]
            assert entry.vendor_requirements == frozenset()

    def test_vendor_constrained_eps_have_nonempty_set(self) -> None:
        from winml.modelkit.ep_path import EPCatalog

        for ep in (
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "VitisAIExecutionProvider",
            "MIGraphXExecutionProvider",
            "NvTensorRtRtxExecutionProvider",
        ):
            entry: EPCatalog.Row = EP_CATALOG._by_name[ep]
            assert entry.vendor_requirements, f"{ep} must declare ≥1 vendor"


# ---------------------------------------------------------------------------
# EP_CATALOG.is_compatible() matching rules.
# ---------------------------------------------------------------------------


class TestEpIsCompatible:
    """Match rules for ``EP_CATALOG.is_compatible``."""

    def test_empty_requirement_always_compatible(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset(),
        )
        assert EP_CATALOG.is_compatible("CPUExecutionProvider") is True
        assert EP_CATALOG.is_compatible("DmlExecutionProvider") is True
        assert EP_CATALOG.is_compatible("AzureExecutionProvider") is True

    def test_unknown_ep_defaults_compatible(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Forward-compat: an EP we have not yet added to the table is
        # treated as compatible (rather than incompatible) so a new EP
        # is not silently hidden in `--list-ep`.
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset(),
        )
        assert EP_CATALOG.is_compatible("FutureEpNotInTable") is True

    def test_qualcomm_substring_match(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset({"Qualcomm Technologies, Inc."}),
        )
        assert EP_CATALOG.is_compatible("QNNExecutionProvider") is True
        assert EP_CATALOG.is_compatible("OpenVINOExecutionProvider") is False
        assert EP_CATALOG.is_compatible("NvTensorRtRtxExecutionProvider") is False

    def test_intel_substring_match_case_insensitive(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Vendor strings vary: "Intel(R) Corporation", "Intel Corp", "Intel"
        # — substring lowercase match accepts any.
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset({"intel(r) corporation"}),
        )
        assert EP_CATALOG.is_compatible("OpenVINOExecutionProvider") is True

    def test_amd_matches_both_vitisai_and_migraphx(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset({"AMD Radeon Graphics"}),
        )
        assert EP_CATALOG.is_compatible("VitisAIExecutionProvider") is True
        assert EP_CATALOG.is_compatible("MIGraphXExecutionProvider") is True

    def test_no_vendor_detected_means_constrained_eps_incompatible(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset(),
        )
        assert EP_CATALOG.is_compatible("QNNExecutionProvider") is False
        assert EP_CATALOG.is_compatible("OpenVINOExecutionProvider") is False
        # CPU still passes (empty requirement).
        assert EP_CATALOG.is_compatible("CPUExecutionProvider") is True

    def test_partial_match_within_long_vendor_string(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Some Windows reports include the device name itself
        # (e.g., "Snapdragon(R) X Elite - Qualcomm(R) Hexagon(TM) NPU").
        # Substring matching handles this.
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset({"Snapdragon(R) X Elite - Qualcomm(R) Hexagon(TM) NPU"}),
        )
        assert EP_CATALOG.is_compatible("QNNExecutionProvider") is True


# ---------------------------------------------------------------------------
# _get_detected_vendors() aggregation.
# ---------------------------------------------------------------------------


class TestGetDetectedVendors:
    """``_get_detected_vendors`` aggregates GPU.manufacturer/name + NPU.manufacturer/name."""

    def test_aggregates_gpu_and_npu(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gpu = MagicMock()
        gpu.manufacturer = "NVIDIA Corporation"
        gpu.name = "NVIDIA RTX 4090"
        npu = MagicMock()
        npu.manufacturer = "Intel Corporation"
        npu.name = "Intel AI Boost"

        monkeypatch.setattr(
            "winml.modelkit.sysinfo.hardware.GPU.get_all", lambda: [gpu]
        )
        monkeypatch.setattr(
            "winml.modelkit.sysinfo.hardware.NPU.get_all", lambda: [npu]
        )

        result = _get_detected_vendors()
        assert "NVIDIA Corporation" in result
        assert "NVIDIA RTX 4090" in result
        assert "Intel Corporation" in result
        assert "Intel AI Boost" in result

    def test_handles_missing_attribute(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If a hardware object lacks manufacturer or name, that field is
        # silently skipped — we do not raise.
        gpu = MagicMock(spec=["manufacturer"])
        gpu.manufacturer = "AMD"

        monkeypatch.setattr(
            "winml.modelkit.sysinfo.hardware.GPU.get_all", lambda: [gpu]
        )
        monkeypatch.setattr(
            "winml.modelkit.sysinfo.hardware.NPU.get_all", list
        )

        result = _get_detected_vendors()
        assert result == frozenset({"AMD"})

    def test_get_all_failure_raises(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If GPU.get_all raises (WMI failure), the whole detection fails with RuntimeError.
        # (Old behavior was to swallow the error and continue; new behavior is to raise
        # so functools.cache doesn't pin a false "no hardware" result.)
        npu = MagicMock()
        npu.manufacturer = "Qualcomm"
        npu.name = "Qualcomm Hexagon"

        def raise_wmi() -> list:
            raise RuntimeError("WMI down")

        monkeypatch.setattr(
            "winml.modelkit.sysinfo.hardware.GPU.get_all", raise_wmi
        )
        monkeypatch.setattr(
            "winml.modelkit.sysinfo.hardware.NPU.get_all", lambda: [npu]
        )

        with pytest.raises(RuntimeError, match=r"GPU\.get_all"):
            _get_detected_vendors()

    def test_no_hardware_returns_empty(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "winml.modelkit.sysinfo.hardware.GPU.get_all", list
        )
        monkeypatch.setattr(
            "winml.modelkit.sysinfo.hardware.NPU.get_all", list
        )
        assert _get_detected_vendors() == frozenset()


# ---------------------------------------------------------------------------
# is_compatible() on each EPSource subclass.
# ---------------------------------------------------------------------------


class TestSourceIsCompatible:
    """``is_compatible()`` delegates to the central rule for every source kind."""

    def test_pypi_source_compatible(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset({"Qualcomm Inc"}),
        )
        src = PyPISource(
            distribution="onnxruntime-qnn",
            relative_dll="ignored",
            eps=("QNNExecutionProvider",),
        )
        assert src.is_compatible() is True

    def test_pypi_source_incompatible(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # OpenVINO PyPI on a Snapdragon-only box.
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset({"Qualcomm Inc"}),
        )
        src = PyPISource(
            distribution="onnxruntime-ep-openvino",
            relative_dll="ignored",
            eps=("OpenVINOExecutionProvider",),
        )
        assert src.is_compatible() is False

    def test_filesystem_source_uses_dll_patterns_keys(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset({"AMD"}),
        )
        src = DirectorySource(
            root=Path("ignored"),
            dll_patterns={"VitisAIExecutionProvider": "vitisai.dll"},
        )
        assert src.is_compatible() is True

    def test_winml_catalog_source_compatible(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset({"NVIDIA Corp"}),
        )
        src = WinMLCatalogSource(
            catalog_name="NvTensorRtRtxExecutionProvider",
            eps=("NvTensorRtRtxExecutionProvider",),
        )
        assert src.is_compatible() is True

    def test_msix_package_source_incompatible(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset({"Qualcomm"}),
        )
        src = MSIXPackageSource(
            family_name_prefix="...OpenVINO.EP._...",
            relative_dll="ignored",
            eps=("OpenVINOExecutionProvider",),
        )
        assert src.is_compatible() is False

    def test_multi_ep_source_all_must_match(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An EPSource that provides multiple EPs is compatible iff ALL
        # of them are. (Mostly theoretical — current sources provide one
        # EP — but the contract should be strict.)
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset({"AMD"}),
        )
        # AMD-only box: VitisAI ok, but QNN and OpenVINO not.
        src = DirectorySource(
            root=Path("ignored"),
            dll_patterns={
                "VitisAIExecutionProvider": "vitisai.dll",
                "QNNExecutionProvider": "qnn.dll",
            },
        )
        assert src.is_compatible() is False


# ---------------------------------------------------------------------------
# iter_eps() — direct coverage of the abstract method (review S-4).
# ---------------------------------------------------------------------------


class TestIterEps:
    """``iter_eps()`` returns the canonical EP names a source declares."""

    def test_pypi_source_iter_eps(self) -> None:
        src = PyPISource(
            distribution="onnxruntime-qnn",
            relative_dll="ignored",
            eps=("QNNExecutionProvider",),
        )
        assert list(src.iter_eps()) == ["QNNExecutionProvider"]

    def test_filesystem_source_iter_eps(self) -> None:
        src = DirectorySource(
            root=Path("ignored"),
            dll_patterns={
                "VitisAIExecutionProvider": "vitisai.dll",
                "QNNExecutionProvider": "qnn.dll",
            },
        )
        # iter_eps returns the dll_patterns keys (insertion order).
        assert list(src.iter_eps()) == [
            "VitisAIExecutionProvider",
            "QNNExecutionProvider",
        ]

    def test_winml_catalog_source_iter_eps(self) -> None:
        src = WinMLCatalogSource(
            catalog_name="QNNExecutionProvider",
            eps=("QNNExecutionProvider",),
        )
        assert list(src.iter_eps()) == ["QNNExecutionProvider"]

    def test_msix_package_source_iter_eps(self) -> None:
        src = MSIXPackageSource(
            family_name_prefix="MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.1.8_",
            relative_dll="ExecutionProvider/onnxruntime_providers_qnn.dll",
            eps=("QNNExecutionProvider",),
        )
        assert list(src.iter_eps()) == ["QNNExecutionProvider"]

    def test_iter_eps_drives_is_compatible(
        self, reset_vendor_cache: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # is_compatible iterates iter_eps; a multi-EP source where one EP
        # is incompatible -> overall False. Confirms iter_eps is the
        # actual driver (not a hardcoded path through self.eps).
        monkeypatch.setattr(
            "winml.modelkit.ep_path._get_detected_vendors",
            lambda: frozenset({"AMD"}),
        )
        ok_src = DirectorySource(
            root=Path("ignored"),
            dll_patterns={"VitisAIExecutionProvider": "v.dll"},
        )
        bad_src = DirectorySource(
            root=Path("ignored"),
            dll_patterns={
                "VitisAIExecutionProvider": "v.dll",
                "QNNExecutionProvider": "q.dll",
            },
        )
        assert ok_src.is_compatible() is True
        assert bad_src.is_compatible() is False
