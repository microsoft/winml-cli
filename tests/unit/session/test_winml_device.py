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

from winml.modelkit.session import WinMLDevice, _format_bytes


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
    """WinMLDevice(handle) constructs a WinMLDevice."""

    def test_wrap_returns_winml_device(self) -> None:
        handle = make_fake_ort_ep_device(ep_name="OpenVINOExecutionProvider", device_type="NPU")
        wd = WinMLDevice(handle)
        assert isinstance(wd, WinMLDevice)

    def test_wrap_preserves_handle_identity(self) -> None:
        handle = make_fake_ort_ep_device(ep_name="OpenVINOExecutionProvider", device_type="NPU")
        wd = WinMLDevice(handle)
        # _ort is the implementation-internal handle reference — verified
        # indirectly via the public properties below.
        assert wd._ort is handle


class TestCommonProperties:
    """ep_name / device_type / hardware_name / vendor / ep_vendor / library_path."""

    def test_ep_name_passes_through(self) -> None:
        handle = make_fake_ort_ep_device(ep_name="OpenVINOExecutionProvider", device_type="NPU")
        assert WinMLDevice(handle).ep_name == "OpenVINOExecutionProvider"

    def test_device_type_is_upper(self) -> None:
        """device_type is forced uppercase — ORT may return mixed case in future."""
        handle = make_fake_ort_ep_device(ep_name="OpenVINOExecutionProvider", device_type="npu")
        assert WinMLDevice(handle).device_type == "NPU"

    def test_hardware_name_prefers_ep_metadata(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"FULL_DEVICE_NAME": "Intel(R) AI Boost"},
            device_metadata={"Description": "Generic NPU"},
        )
        assert WinMLDevice(handle).hardware_name == "Intel(R) AI Boost"

    def test_hardware_name_falls_back_to_device_metadata(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            device_metadata={"Description": "Intel Arc"},
        )
        assert WinMLDevice(handle).hardware_name == "Intel Arc"

    def test_hardware_name_unknown_fallback(self) -> None:
        handle = make_fake_ort_ep_device(ep_name="OpenVINOExecutionProvider", device_type="NPU")
        assert WinMLDevice(handle).hardware_name == "<unknown>"

    def test_vendor_passes_through(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            device_vendor="Intel",
        )
        assert WinMLDevice(handle).vendor == "Intel"

    def test_ep_vendor_passes_through(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_vendor="Microsoft",
        )
        assert WinMLDevice(handle).ep_vendor == "Microsoft"

    def test_library_path_present(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"library_path": "C:/plugins/openvino_ep.dll"},
        )
        assert WinMLDevice(handle).library_path == "C:/plugins/openvino_ep.dll"

    def test_library_path_missing_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(ep_name="OpenVINOExecutionProvider", device_type="NPU")
        assert WinMLDevice(handle).library_path is None


class TestMemoryBytes:
    """memory_bytes is per-EP / per-device — dispatch table lives in winml_device."""

    def test_openvino_npu_reads_npu_total_mem(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DEVICE_TOTAL_MEM_SIZE": "8589934592"},  # 8 GiB
        )
        assert WinMLDevice(handle).memory_bytes == 8589934592

    def test_openvino_gpu_reads_gpu_total_mem(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={"GPU_DEVICE_TOTAL_MEM_SIZE": "4294967296"},  # 4 GiB
        )
        assert WinMLDevice(handle).memory_bytes == 4294967296

    def test_openvino_npu_missing_key_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
        )
        assert WinMLDevice(handle).memory_bytes is None

    def test_openvino_npu_bad_int_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DEVICE_TOTAL_MEM_SIZE": "not-a-number"},
        )
        assert WinMLDevice(handle).memory_bytes is None

    def test_dml_parses_mb_string(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="DmlExecutionProvider",
            device_type="GPU",
            device_metadata={"DxgiVideoMemory": "128 MB"},
        )
        assert WinMLDevice(handle).memory_bytes == 128 * 1024**2

    def test_dml_parses_gb_string(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="DmlExecutionProvider",
            device_type="GPU",
            device_metadata={"DxgiVideoMemory": "4 GB"},
        )
        assert WinMLDevice(handle).memory_bytes == 4 * 1024**3

    def test_dml_missing_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="DmlExecutionProvider",
            device_type="GPU",
        )
        assert WinMLDevice(handle).memory_bytes is None

    def test_dml_unknown_unit_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="DmlExecutionProvider",
            device_type="GPU",
            device_metadata={"DxgiVideoMemory": "4 PB"},  # PB not in multiplier table
        )
        assert WinMLDevice(handle).memory_bytes is None

    def test_unknown_ep_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="UnknownEP",
            device_type="NPU",
            ep_metadata={"NPU_DEVICE_TOTAL_MEM_SIZE": "100"},
        )
        assert WinMLDevice(handle).memory_bytes is None

    def test_openvino_cpu_returns_none(self) -> None:
        """OpenVINO CPU has no memory key in the dispatch table."""
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="CPU",
            ep_metadata={"NPU_DEVICE_TOTAL_MEM_SIZE": "100"},
        )
        assert WinMLDevice(handle).memory_bytes is None


class TestArchitecture:
    """architecture strips 'arch=' prefix for OpenVINO; None for unknown EPs."""

    def test_openvino_with_arch_prefix(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={"DEVICE_ARCHITECTURE": "GPU: vendor=0x8086 arch=v20.4.4"},
        )
        assert WinMLDevice(handle).architecture == "v20.4.4"

    def test_openvino_passthrough_no_prefix(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="CPU",
            ep_metadata={"DEVICE_ARCHITECTURE": "intel64"},
        )
        assert WinMLDevice(handle).architecture == "intel64"

    def test_openvino_missing_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(ep_name="OpenVINOExecutionProvider", device_type="NPU")
        assert WinMLDevice(handle).architecture is None

    def test_unknown_ep_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="QNNExecutionProvider",
            device_type="NPU",
            ep_metadata={"DEVICE_ARCHITECTURE": "anything"},
        )
        assert WinMLDevice(handle).architecture is None


class TestCapabilities:
    """capabilities normalizes OPTIMIZATION_CAPABILITIES tokens for OpenVINO."""

    def test_openvino_normalizes_tokens(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={"OPTIMIZATION_CAPABILITIES": "FP32 FP16 GPU_HW_MATMUL GPU_USM_MEMORY"},
        )
        caps = WinMLDevice(handle).capabilities
        # Order preserved, rewrites applied
        assert caps == ("FP32", "FP16", "MatMul", "USM")

    def test_openvino_drops_export_import(self) -> None:
        """EXPORT_IMPORT maps to empty string and is filtered out."""
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={"OPTIMIZATION_CAPABILITIES": "FP16 EXPORT_IMPORT"},
        )
        assert WinMLDevice(handle).capabilities == ("FP16",)

    def test_openvino_missing_returns_empty(self) -> None:
        handle = make_fake_ort_ep_device(ep_name="OpenVINOExecutionProvider", device_type="GPU")
        assert WinMLDevice(handle).capabilities == ()

    def test_unknown_ep_returns_empty(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="QNNExecutionProvider",
            device_type="NPU",
            ep_metadata={"OPTIMIZATION_CAPABILITIES": "FP32"},
        )
        assert WinMLDevice(handle).capabilities == ()


class TestDriverAndCompilerVersion:
    """driver_version / compiler_version for OpenVINO NPU only."""

    def test_openvino_npu_driver_version(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DRIVER_VERSION": "32.0.100.4023"},
        )
        assert WinMLDevice(handle).driver_version == "32.0.100.4023"

    def test_openvino_npu_compiler_version(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_COMPILER_VERSION": "5.13.0"},
        )
        assert WinMLDevice(handle).compiler_version == "5.13.0"

    def test_openvino_gpu_driver_returns_none(self) -> None:
        """Only NPU exposes driver/compiler version — GPU returns None."""
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={"NPU_DRIVER_VERSION": "ignored"},
        )
        assert WinMLDevice(handle).driver_version is None

    def test_openvino_gpu_compiler_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={"NPU_COMPILER_VERSION": "ignored"},
        )
        assert WinMLDevice(handle).compiler_version is None

    def test_unknown_ep_driver_returns_none(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="QNNExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DRIVER_VERSION": "anything"},
        )
        assert WinMLDevice(handle).driver_version is None


class TestAvailableMetadata:
    """available_metadata() returns the raw ep_metadata dict."""

    def test_returns_full_metadata(self) -> None:
        meta = {"FULL_DEVICE_NAME": "Intel AI Boost", "library_path": "fake.dll"}
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata=meta,
        )
        wd = WinMLDevice(handle)
        result = wd.available_metadata()
        assert result["FULL_DEVICE_NAME"] == "Intel AI Boost"
        assert result["library_path"] == "fake.dll"

    def test_returns_empty_when_no_metadata(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
        )
        assert dict(WinMLDevice(handle).available_metadata()) == {}


class TestDeviceFacts:
    """device_facts() returns Architecture + Driver only (device-intrinsic).

    Per ``docs/design/session/4_winml_device.md`` §4 + §4.1, these are the
    facts that surface in the *Available Devices* section of
    ``winml sys --list-ep`` — values keyed off the underlying silicon
    and kernel driver, invariant across the EPs that bind to the device.
    Memory and Capabilities live on :meth:`ep_facts` instead.
    """

    def test_openvino_npu_device_facts_include_driver(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DRIVER_VERSION": "32.0.100"},
        )
        facts = WinMLDevice(handle).device_facts()
        assert any("Driver: 32.0.100" in f for f in facts)

    def test_openvino_gpu_device_facts_include_architecture(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={"DEVICE_ARCHITECTURE": "GPU: vendor=0x8086 arch=v20.4.4"},
        )
        facts = WinMLDevice(handle).device_facts()
        assert any("Architecture: v20.4.4" in f for f in facts)

    def test_device_facts_returns_architecture_and_driver_only(self) -> None:
        """Memory + Capabilities + Compiler must NOT appear in device_facts."""
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={
                "NPU_DEVICE_TOTAL_MEM_SIZE": str(1024**3),  # 1 GiB
                "NPU_DRIVER_VERSION": "32.0.100",
                "NPU_COMPILER_VERSION": "5.13.0",
                "DEVICE_ARCHITECTURE": "uarch4000",
                "OPTIMIZATION_CAPABILITIES": "FP32 FP16",
            },
        )
        facts = WinMLDevice(handle).device_facts()
        joined = " | ".join(facts)
        assert "Architecture:" in joined
        assert "Driver:" in joined
        # The EP-mediated facts must NOT leak into the device section.
        assert "Memory:" not in joined
        assert "Capabilities:" not in joined
        assert "Compiler:" not in joined

    def test_device_facts_empty_when_no_metadata(self) -> None:
        """A WinMLDevice with no architecture/driver returns empty tuple."""
        handle = make_fake_ort_ep_device(ep_name="OpenVINOExecutionProvider", device_type="NPU")
        assert WinMLDevice(handle).device_facts() == ()

    def test_device_facts_returns_tuple_of_strings(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DRIVER_VERSION": "32.0.100"},
        )
        facts = WinMLDevice(handle).device_facts()
        assert isinstance(facts, tuple)
        assert all(isinstance(f, str) for f in facts)

    def test_unknown_ep_device_facts_is_empty(self) -> None:
        """An EP with no per-EP dispatch contributes no device_facts."""
        handle = make_fake_ort_ep_device(ep_name="UnknownEP", device_type="NPU")
        assert WinMLDevice(handle).device_facts() == ()


class TestEpFacts:
    """ep_facts() returns Memory + Capabilities only (EP-mediated).

    Per ``docs/design/session/4_winml_device.md`` §4 + §4.1, these are the
    facts that surface in the per-source EP rows of
    ``winml sys --list-ep`` — values keyed off this specific EP runtime's
    view of the device. Architecture/Driver live on :meth:`device_facts`
    instead; ``compiler_version`` is deferred to ``--verbose``.
    """

    def test_openvino_npu_ep_facts_include_memory(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DEVICE_TOTAL_MEM_SIZE": str(1024**3)},  # 1 GiB
        )
        facts = WinMLDevice(handle).ep_facts()
        assert any("Memory: 1.0 GB" in f for f in facts)

    def test_openvino_gpu_ep_facts_include_memory_and_capabilities(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="GPU",
            ep_metadata={
                "GPU_DEVICE_TOTAL_MEM_SIZE": str(2 * 1024**3),  # 2 GiB
                "OPTIMIZATION_CAPABILITIES": "FP32 FP16",
            },
        )
        facts = WinMLDevice(handle).ep_facts()
        joined = " | ".join(facts)
        assert "Memory: 2.0 GB" in joined
        assert "Capabilities: FP32, FP16" in joined

    def test_ep_facts_returns_memory_and_capabilities_only(self) -> None:
        """Architecture + Driver + Compiler must NOT appear in ep_facts."""
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={
                "NPU_DEVICE_TOTAL_MEM_SIZE": str(1024**3),
                "NPU_DRIVER_VERSION": "32.0.100",
                "NPU_COMPILER_VERSION": "5.13.0",
                "DEVICE_ARCHITECTURE": "uarch4000",
            },
        )
        facts = WinMLDevice(handle).ep_facts()
        joined = " | ".join(facts)
        assert "Memory:" in joined
        # The device-intrinsic facts must NOT leak into the EP section.
        assert "Architecture:" not in joined
        assert "Driver:" not in joined
        # compiler_version is deferred to --verbose.
        assert "Compiler:" not in joined

    def test_ep_facts_empty_when_no_metadata(self) -> None:
        """A WinMLDevice with no memory/capabilities returns empty tuple."""
        handle = make_fake_ort_ep_device(ep_name="OpenVINOExecutionProvider", device_type="NPU")
        assert WinMLDevice(handle).ep_facts() == ()

    def test_ep_facts_returns_tuple_of_strings(self) -> None:
        handle = make_fake_ort_ep_device(ep_name="OpenVINOExecutionProvider", device_type="NPU")
        facts = WinMLDevice(handle).ep_facts()
        assert isinstance(facts, tuple)
        assert all(isinstance(f, str) for f in facts)

    def test_ep_facts_pipe_join_ready(self) -> None:
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DEVICE_TOTAL_MEM_SIZE": str(1024**3)},
        )
        # Caller documented usage: '  |  '.join(ep_facts())
        joined = "  |  ".join(WinMLDevice(handle).ep_facts())
        assert "Memory:" in joined

    def test_unknown_ep_ep_facts_is_empty(self) -> None:
        """An EP with no per-EP dispatch contributes no ep_facts."""
        handle = make_fake_ort_ep_device(ep_name="UnknownEP", device_type="NPU")
        assert WinMLDevice(handle).ep_facts() == ()


class TestFormatBytesHelper:
    """_format_bytes helper exercised indirectly via memory_bytes + facts.

    T-14: the helper is now sourced from ``session.monitor.report`` so the
    codebase has a single byte-formatter. The renderer uses ``GB/MB/KB``
    labels (1024-based math); the IEC ``GiB/MiB/KiB`` labels previously
    emitted here are gone with the local copy.
    """

    def test_format_gb(self) -> None:
        assert _format_bytes(1024**3) == "1.0 GB"

    def test_format_mb(self) -> None:
        assert _format_bytes(2 * 1024**2) == "2.0 MB"

    def test_format_kb(self) -> None:
        assert _format_bytes(4 * 1024) == "4.0 KB"

    def test_format_bytes(self) -> None:
        assert _format_bytes(512) == "512 B"

    def test_format_via_ep_facts(self) -> None:
        """Sanity: ep_facts() ties memory_bytes to _format_bytes correctly."""
        handle = make_fake_ort_ep_device(
            ep_name="OpenVINOExecutionProvider",
            device_type="NPU",
            ep_metadata={"NPU_DEVICE_TOTAL_MEM_SIZE": str(8 * 1024**3)},
        )
        facts = WinMLDevice(handle).ep_facts()
        assert any("Memory: 8.0 GB" in f for f in facts)

    def test_format_bytes_is_single_source_of_truth(self) -> None:
        """``ep_device._format_bytes`` IS ``monitor.report._format_bytes``.

        Pins the T-14 dedup contract: there is exactly one ``_format_bytes``
        implementation, owned by ``session.monitor.report``. ``ep_device``
        re-exports it via a module-level import so both consumers stay in
        sync.
        """
        from winml.modelkit.session import ep_device as _ep_device
        from winml.modelkit.session.monitor import report as _report

        assert _ep_device._format_bytes is _report._format_bytes
