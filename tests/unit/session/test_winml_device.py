# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the WinMLDevice vendor-normalized adapter (Batch B/C).

WinMLDevice wraps :class:`ort.OrtEpDevice` and exposes a stable, vendor-agnostic
view of EP + device metadata. Per-EP dispatch is keyed on ``self._ort.ep_name``
and ``self.device_type`` — all tests here drive those branches via mocked
OrtEpDevice handles.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from winml.modelkit.session import WinMLDevice, wrap_ort_device
from winml.modelkit.session.ep_device import _format_bytes


def make_fake_ort_ep_device(
    *,
    ep_name: str,
    device_type: str,
    ep_metadata: dict[str, str] | None = None,
    device_metadata: dict[str, str] | None = None,
    device_vendor: str = "Intel",
    ep_vendor: str = "Microsoft",
) -> MagicMock:
    """Construct a MagicMock OrtEpDevice with controllable per-EP metadata.

    Module-level helper so each test can craft the exact (ep_name, device_type,
    metadata) shape it needs. Mirrors the conftest pattern but stays scoped to
    this file (smaller blast radius).
    """
    d = MagicMock()
    d.ep_name = ep_name
    d.device.type.name = device_type
    d.ep_metadata = dict(ep_metadata or {})
    d.device.metadata = dict(device_metadata or {})
    d.device.vendor = device_vendor
    d.ep_vendor = ep_vendor
    return d


class TestWrapOrtDeviceFactory:
    """wrap_ort_device(handle) constructs a WinMLDevice."""

    def test_wrap_returns_winml_device(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider", device_type="NPU"
        )
        wd = wrap_ort_device(handle)
        assert isinstance(wd, WinMLDevice)

    def test_wrap_preserves_handle_identity(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider", device_type="NPU"
        )
        wd = wrap_ort_device(handle)
        # _ort is the implementation-internal handle reference — verified
        # indirectly via the public properties below.
        assert wd._ort is handle


class TestCommonProperties:
    """ep_name / device_type / hardware_name / vendor / ep_vendor / library_path."""

    def test_ep_name_passes_through(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider", device_type="NPU"
        )
        assert wrap_ort_device(handle).ep_name == "OpenVINOExecutionProvider"

    def test_device_type_is_upper(self) -> None:
        """device_type is forced uppercase — ORT may return mixed case in future."""
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider", device_type="npu"
        )
        assert wrap_ort_device(handle).device_type == "NPU"

    def test_hardware_name_prefers_ep_metadata(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"FULL_DEVICE_NAME": "Intel(R) AI Boost"},
            device_metadata={"Description": "Generic NPU"},
        )
        assert wrap_ort_device(handle).hardware_name == "Intel(R) AI Boost"

    def test_hardware_name_falls_back_to_device_metadata(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            device_metadata={"Description": "Intel Arc"},
        )
        assert wrap_ort_device(handle).hardware_name == "Intel Arc"

    def test_hardware_name_unknown_fallback(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider", device_type="NPU"
        )
        assert wrap_ort_device(handle).hardware_name == "<unknown>"

    def test_vendor_passes_through(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            device_vendor="Intel",
        )
        assert wrap_ort_device(handle).vendor == "Intel"

    def test_ep_vendor_passes_through(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_vendor="Microsoft",
        )
        assert wrap_ort_device(handle).ep_vendor == "Microsoft"

    def test_library_path_present(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"library_path": "C:/plugins/openvino_ep.dll"},
        )
        assert wrap_ort_device(handle).library_path == "C:/plugins/openvino_ep.dll"

    def test_library_path_missing_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider", device_type="NPU"
        )
        assert wrap_ort_device(handle).library_path is None


class TestMemoryBytes:
    """memory_bytes is per-EP / per-device — dispatch table lives in winml_device."""

    def test_openvino_npu_reads_npu_total_mem(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DEVICE_TOTAL_MEM_SIZE": "8589934592"},  # 8 GiB
        )
        assert wrap_ort_device(handle).memory_bytes == 8589934592

    def test_openvino_gpu_reads_gpu_total_mem(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={"GPU_DEVICE_TOTAL_MEM_SIZE": "4294967296"},  # 4 GiB
        )
        assert wrap_ort_device(handle).memory_bytes == 4294967296

    def test_openvino_npu_missing_key_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
        )
        assert wrap_ort_device(handle).memory_bytes is None

    def test_openvino_npu_bad_int_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DEVICE_TOTAL_MEM_SIZE": "not-a-number"},
        )
        assert wrap_ort_device(handle).memory_bytes is None

    def test_dml_parses_mb_string(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="DmlExecutionProvider",
            device_type="GPU",
            device_metadata={"DxgiVideoMemory": "128 MB"},
        )
        assert wrap_ort_device(handle).memory_bytes == 128 * 1024**2

    def test_dml_parses_gb_string(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="DmlExecutionProvider",
            device_type="GPU",
            device_metadata={"DxgiVideoMemory": "4 GB"},
        )
        assert wrap_ort_device(handle).memory_bytes == 4 * 1024**3

    def test_dml_missing_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="DmlExecutionProvider",
            device_type="GPU",
        )
        assert wrap_ort_device(handle).memory_bytes is None

    def test_dml_unknown_unit_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="DmlExecutionProvider",
            device_type="GPU",
            device_metadata={"DxgiVideoMemory": "4 PB"},  # PB not in multiplier table
        )
        assert wrap_ort_device(handle).memory_bytes is None

    def test_unknown_ep_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="UnknownEP",
            device_type="NPU",
            ep_metadata={"NPU_DEVICE_TOTAL_MEM_SIZE": "100"},
        )
        assert wrap_ort_device(handle).memory_bytes is None

    def test_openvino_cpu_returns_none(self) -> None:
        """OpenVINO CPU has no memory key in the dispatch table."""
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="CPU",
            ep_metadata={"NPU_DEVICE_TOTAL_MEM_SIZE": "100"},
        )
        assert wrap_ort_device(handle).memory_bytes is None


class TestArchitecture:
    """architecture strips 'arch=' prefix for OpenVINO; None for unknown EPs."""

    def test_openvino_with_arch_prefix(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={"DEVICE_ARCHITECTURE": "GPU: vendor=0x8086 arch=v20.4.4"},
        )
        assert wrap_ort_device(handle).architecture == "v20.4.4"

    def test_openvino_passthrough_no_prefix(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="CPU",
            ep_metadata={"DEVICE_ARCHITECTURE": "intel64"},
        )
        assert wrap_ort_device(handle).architecture == "intel64"

    def test_openvino_missing_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider", device_type="NPU"
        )
        assert wrap_ort_device(handle).architecture is None

    def test_unknown_ep_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="QNNExecutionProvider",
            device_type="NPU",
            ep_metadata={"DEVICE_ARCHITECTURE": "anything"},
        )
        assert wrap_ort_device(handle).architecture is None


class TestCapabilities:
    """capabilities normalizes OPTIMIZATION_CAPABILITIES tokens for OpenVINO."""

    def test_openvino_normalizes_tokens(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={
                "OPTIMIZATION_CAPABILITIES": "FP32 FP16 GPU_HW_MATMUL GPU_USM_MEMORY"
            },
        )
        caps = wrap_ort_device(handle).capabilities
        # Order preserved, rewrites applied
        assert caps == ("FP32", "FP16", "MatMul", "USM")

    def test_openvino_drops_export_import(self) -> None:
        """EXPORT_IMPORT maps to empty string and is filtered out."""
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={"OPTIMIZATION_CAPABILITIES": "FP16 EXPORT_IMPORT"},
        )
        assert wrap_ort_device(handle).capabilities == ("FP16",)

    def test_openvino_missing_returns_empty(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider", device_type="GPU"
        )
        assert wrap_ort_device(handle).capabilities == ()

    def test_unknown_ep_returns_empty(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="QNNExecutionProvider",
            device_type="NPU",
            ep_metadata={"OPTIMIZATION_CAPABILITIES": "FP32"},
        )
        assert wrap_ort_device(handle).capabilities == ()


class TestDriverAndCompilerVersion:
    """driver_version / compiler_version for OpenVINO NPU only."""

    def test_openvino_npu_driver_version(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DRIVER_VERSION": "32.0.100.4023"},
        )
        assert wrap_ort_device(handle).driver_version == "32.0.100.4023"

    def test_openvino_npu_compiler_version(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_COMPILER_VERSION": "5.13.0"},
        )
        assert wrap_ort_device(handle).compiler_version == "5.13.0"

    def test_openvino_gpu_driver_returns_none(self) -> None:
        """Only NPU exposes driver/compiler version — GPU returns None."""
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={"NPU_DRIVER_VERSION": "ignored"},
        )
        assert wrap_ort_device(handle).driver_version is None

    def test_openvino_gpu_compiler_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={"NPU_COMPILER_VERSION": "ignored"},
        )
        assert wrap_ort_device(handle).compiler_version is None

    def test_unknown_ep_driver_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="QNNExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DRIVER_VERSION": "anything"},
        )
        assert wrap_ort_device(handle).driver_version is None


class TestAvailableMetadata:
    """available_metadata() returns the raw ep_metadata dict."""

    def test_returns_full_metadata(self) -> None:
        meta = {"FULL_DEVICE_NAME": "Intel AI Boost", "library_path": "fake.dll"}
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata=meta,
        )
        wd = wrap_ort_device(handle)
        result = wd.available_metadata()
        assert result["FULL_DEVICE_NAME"] == "Intel AI Boost"
        assert result["library_path"] == "fake.dll"

    def test_returns_empty_when_no_metadata(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
        )
        assert dict(wrap_ort_device(handle).available_metadata()) == {}


class TestFacts:
    """facts() returns a tuple of pipe-join-ready strings."""

    def test_openvino_npu_facts_include_memory_driver_compiler(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={
                "NPU_DEVICE_TOTAL_MEM_SIZE": str(1024**3),  # 1 GiB
                "NPU_DRIVER_VERSION": "32.0.100",
                "NPU_COMPILER_VERSION": "5.13.0",
            },
        )
        facts = wrap_ort_device(handle).facts()
        assert any("Memory: 1.0 GiB" in f for f in facts)
        assert any("Driver: 32.0.100" in f for f in facts)
        assert any("Compiler: 5.13.0" in f for f in facts)

    def test_openvino_gpu_facts_include_architecture_and_capabilities(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={
                "GPU_DEVICE_TOTAL_MEM_SIZE": str(2 * 1024**3),  # 2 GiB
                "DEVICE_ARCHITECTURE": "GPU: vendor=0x8086 arch=v20.4.4",
                "OPTIMIZATION_CAPABILITIES": "FP32 FP16",
            },
        )
        facts = wrap_ort_device(handle).facts()
        joined = " | ".join(facts)
        assert "Memory: 2.0 GiB" in joined
        assert "Architecture: v20.4.4" in joined
        assert "Capabilities: FP32, FP16" in joined

    def test_facts_returns_tuple_of_strings(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider", device_type="NPU"
        )
        facts = wrap_ort_device(handle).facts()
        assert isinstance(facts, tuple)
        assert all(isinstance(f, str) for f in facts)

    def test_facts_pipe_join_ready(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DEVICE_TOTAL_MEM_SIZE": str(1024**3)},
        )
        # Caller documented usage: '  |  '.join(facts())
        joined = "  |  ".join(wrap_ort_device(handle).facts())
        assert "Memory:" in joined

    def test_unknown_ep_facts_is_empty(self) -> None:
        """An EP with no per-EP dispatch contributes no facts."""
        handle = make_fake_ort_ep_device(
            ep_name="UnknownEP", device_type="NPU"
        )
        assert wrap_ort_device(handle).facts() == ()


class TestFormatBytesHelper:
    """_format_bytes helper exercised indirectly via memory_bytes + facts."""

    def test_format_gib(self) -> None:
        assert _format_bytes(1024**3) == "1.0 GiB"

    def test_format_mib(self) -> None:
        assert _format_bytes(2 * 1024**2) == "2.0 MiB"

    def test_format_kib(self) -> None:
        assert _format_bytes(4 * 1024) == "4.0 KiB"

    def test_format_bytes(self) -> None:
        assert _format_bytes(512) == "512 B"

    def test_format_via_facts(self) -> None:
        """Sanity: facts() ties memory_bytes to _format_bytes correctly."""
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DEVICE_TOTAL_MEM_SIZE": str(8 * 1024**3)},
        )
        facts = wrap_ort_device(handle).facts()
        assert any("Memory: 8.0 GiB" in f for f in facts)
