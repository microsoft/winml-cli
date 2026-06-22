# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.sysinfo.hardware module.

Focuses on the architecture-agnostic, branchy logic that the e2e perf tests
(``tests/e2e/test_perf_e2e.py``) structurally cannot cover:

  * ``get_vendor_id_device_id_from_pnp_id`` — a pure parser with Intel/AMD/
    NVIDIA hex paths plus two Qualcomm byte-packing quirks and three error
    paths. e2e only ever exercises whichever vendor the runner happens to
    have, so the QCOM branches are essentially never hit there.
  * ``CPU/GPU/NPU/RAM`` field mapping, the GPU PCI/ACPI software-adapter
    filter, the CPU architecture-enum fallback, and the ``to_dict`` key
    contract consumed by ``winml sys`` JSON output.

All of these run off constructed ``CimInstance`` / ``PnpDevice`` objects, so
they execute deterministically on any host (no real hardware required).
"""

from __future__ import annotations

import pytest

from winml.modelkit.sysinfo.hardware import CPU, GPU, NPU, RAM, get_vendor_id_device_id_from_pnp_id
from winml.modelkit.sysinfo.helper import CimInstance, PnpDevice


def _pnp_device(pnp_id: str, device: dict | None = None, props: dict | None = None) -> PnpDevice:
    """Build a ``PnpDevice`` without spawning the per-device PowerShell query.

    Passing ``prefetched_properties`` (even empty) short-circuits the
    Get-PnpDeviceProperty subprocess in ``PnpDevice.__init__``. ``props`` is a
    flat ``{KeyName: Data}`` mapping rendered into the list-of-dicts shape that
    Get-PnpDeviceProperty returns.
    """
    obj = {"PNPDeviceID": pnp_id, **(device or {})}
    prefetched = [{"KeyName": k, "Data": v} for k, v in (props or {}).items()]
    return PnpDevice(obj, prefetched_properties=prefetched)


class TestGetVendorIdDeviceIdFromPnpId:
    """Tests for the PNP-ID → (vendor_id, device_id) parser."""

    @pytest.mark.parametrize(
        ("pnp_id", "vendor_id", "device_id"),
        [
            # NVIDIA discrete GPU (PCI hex path).
            ("PCI\\VEN_10DE&DEV_2204&SUBSYS_00000000&REV_A1", 0x10DE, 0x2204),
            # AMD GPU.
            ("PCI\\VEN_1002&DEV_73BF", 0x1002, 0x73BF),
            # Intel iGPU; hex device-ID digits may be upper or lower case.
            ("PCI\\VEN_8086&DEV_9a49", 0x8086, 0x9A49),
        ],
    )
    def test_pci_hex_path(self, pnp_id: str, vendor_id: int, device_id: int) -> None:
        assert get_vendor_id_device_id_from_pnp_id(pnp_id) == (vendor_id, device_id)

    def test_qualcomm_acpi_quirk(self) -> None:
        """``ACPI\\QCOM####`` packs the 8-char segment as two little-endian uint32s.

        Expected values derived independently of the implementation:
        "QCOM" -> int.from_bytes(b"QCOM", "little") == 0x4D4F4351,
        "0C40" -> int.from_bytes(b"0C40", "little") == 0x30344330.
        """
        vendor_id, device_id = get_vendor_id_device_id_from_pnp_id("ACPI\\QCOM0C40\\3&11583659&0")
        assert (vendor_id, device_id) == (0x4D4F4351, 0x30344330)

    def test_qualcomm_acpi_quirk_invalid_length_raises(self) -> None:
        """A QCOM segment that isn't exactly 8 chars is rejected, not silently parsed."""
        with pytest.raises(ValueError, match="Invalid Qualcomm NPU PNPDeviceID"):
            get_vendor_id_device_id_from_pnp_id("ACPI\\QCOM123\\3&11583659&0")

    def test_qualcomm_ven_quirk(self) -> None:
        """``VEN_QCOM`` is byte-packed (ASCII->uint32) rather than parsed as hex.

        "QCOM" -> 0x4D4F4351, "5C40" -> int.from_bytes(b"5C40", "little") == 0x30344335.
        Note "5C40" would also be valid hex (0x5C40); the QCOM branch must win.
        """
        vendor_id, device_id = get_vendor_id_device_id_from_pnp_id("PCI\\VEN_QCOM&DEV_5C40")
        assert (vendor_id, device_id) == (0x4D4F4351, 0x30344335)

    def test_missing_vendor_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid PNPDeviceID"):
            get_vendor_id_device_id_from_pnp_id("PCI\\DEV_2204")

    def test_missing_device_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid PNPDeviceID"):
            get_vendor_id_device_id_from_pnp_id("PCI\\VEN_10DE")


class TestCPU:
    """Tests for CPU field mapping, architecture enum, and to_dict contract."""

    def test_fields_and_to_dict(self) -> None:
        cpu = CPU(
            CimInstance(
                {
                    "Name": "Intel(R) Core(TM) i9",
                    "Manufacturer": "GenuineIntel",
                    "NumberOfCores": 8,
                    "NumberOfLogicalProcessors": 16,
                    "Architecture": 9,  # x64
                }
            )
        )
        assert cpu.name == "Intel(R) Core(TM) i9"
        assert cpu.manufacturer == "GenuineIntel"
        assert cpu.core_count == 8
        assert cpu.thread_count == 16
        assert cpu.architecture is CPU.Architecture.x64
        assert cpu.to_dict() == {
            "name": "Intel(R) Core(TM) i9",
            "manufacturer": "GenuineIntel",
            "coreCount": 8,
            "threadCount": 16,
            "architecture": "x64",
        }

    def test_arm64_architecture(self) -> None:
        cpu = CPU(CimInstance({"Architecture": 12}))
        assert cpu.architecture is CPU.Architecture.ARM64
        assert cpu.to_dict()["architecture"] == "ARM64"

    def test_unknown_architecture_falls_back(self) -> None:
        """Missing Architecture defaults to -1, which maps to UNKNOWN rather than raising."""
        cpu = CPU(CimInstance({"Name": "Mystery CPU"}))
        assert cpu.architecture is CPU.Architecture.UNKNOWN

    def test_missing_fields_default(self) -> None:
        """Absent CIM properties fall back to empty/zero, not exceptions."""
        cpu = CPU(CimInstance({}))
        assert cpu.name == ""
        assert cpu.manufacturer == ""
        assert cpu.core_count == 0
        assert cpu.thread_count == 0
        assert cpu.architecture is CPU.Architecture.UNKNOWN


class TestGPU:
    """Tests for GPU field mapping, VRAM conversion, and the software-adapter filter."""

    def _gpu_cim(self, pnp_id: str = "PCI\\VEN_10DE&DEV_2204", **overrides: object) -> CimInstance:
        obj = {
            "Name": "NVIDIA GeForce RTX 3080",
            "AdapterCompatibility": "NVIDIA",
            "DriverVersion": "31.0.15.3179",
            "AdapterRAM": 8 * 1024 * 1024,  # 8 MiB
            "PNPDeviceID": pnp_id,
        }
        obj.update(overrides)
        return CimInstance(obj)

    def test_fields_and_to_dict(self) -> None:
        gpu = GPU(self._gpu_cim())
        assert gpu.name == "NVIDIA GeForce RTX 3080"
        assert gpu.manufacturer == "NVIDIA"
        assert gpu.driver_version == "31.0.15.3179"
        assert gpu.vram_mib == 8
        assert gpu.vendor_id == 0x10DE
        assert gpu.device_id == 0x2204
        assert gpu.to_dict() == {
            "name": "NVIDIA GeForce RTX 3080",
            "manufacturer": "NVIDIA",
            "driverVersion": "31.0.15.3179",
            "vramMib": 8,
            "vendorId": 0x10DE,
            "deviceId": 0x2204,
        }

    def test_vram_truncates_to_mib(self) -> None:
        """Sub-MiB AdapterRAM truncates toward zero (int division), not rounds up."""
        gpu = GPU(self._gpu_cim(AdapterRAM=1024 * 1024 + 512 * 1024))  # 1.5 MiB
        assert gpu.vram_mib == 1

    def test_get_all_filters_non_pci_acpi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Software adapters (e.g. RDP) lack a PCI/ACPI PNP ID and must be dropped."""
        instances = [
            self._gpu_cim(pnp_id="PCI\\VEN_10DE&DEV_2204"),
            self._gpu_cim(
                pnp_id="SWD\\RDPUDD\\RDPUDD_INDIRECTDISPLAY", Name="Remote Desktop Adapter"
            ),
            self._gpu_cim(pnp_id="ACPI\\QCOM0C40\\3&11583659&0", Name="Qualcomm Adreno"),
        ]
        monkeypatch.setattr(
            CimInstance, "get_by_class_name", staticmethod(lambda *a, **k: instances)
        )

        gpus = GPU.get_all()

        names = [g.name for g in gpus]
        assert "Remote Desktop Adapter" not in names
        assert names == ["NVIDIA GeForce RTX 3080", "Qualcomm Adreno"]


class TestNPU:
    """Tests for NPU field mapping off a PnpDevice (incl. the extra-property driver version)."""

    def test_fields_and_to_dict(self) -> None:
        npu = NPU(
            _pnp_device(
                "PCI\\VEN_QCOM&DEV_5C40",
                device={"Name": "Snapdragon X Elite - Hexagon NPU", "Manufacturer": "Qualcomm"},
                props={"DEVPKEY_Device_DriverVersion": "30.0.110.0"},
            )
        )
        assert npu.name == "Snapdragon X Elite - Hexagon NPU"
        assert npu.manufacturer == "Qualcomm"
        assert npu.driver_version == "30.0.110.0"
        assert npu.vendor_id == 0x4D4F4351  # "QCOM" byte-packed
        assert npu.device_id == 0x30344335  # "5C40" byte-packed
        assert npu.to_dict() == {
            "name": "Snapdragon X Elite - Hexagon NPU",
            "manufacturer": "Qualcomm",
            "driverVersion": "30.0.110.0",
            "vendorId": 0x4D4F4351,
            "deviceId": 0x30344335,
        }


class TestRAM:
    """Tests for RAM capacity conversion and to_dict contract."""

    def test_fields_and_to_dict(self) -> None:
        ram = RAM(
            CimInstance(
                {
                    "Capacity": 16 * 1024 * 1024 * 1024,  # 16 GiB in bytes
                    "Speed": 4800,
                    "Manufacturer": "Samsung",
                }
            )
        )
        assert ram.capacity_mib == 16 * 1024
        assert ram.speed_mt == 4800
        assert ram.manufacturer == "Samsung"
        assert ram.to_dict() == {
            "capacityMib": 16 * 1024,
            "speedMt": 4800,
            "manufacturer": "Samsung",
        }

    def test_missing_fields_default_to_zero(self) -> None:
        ram = RAM(CimInstance({}))
        assert ram.capacity_mib == 0
        assert ram.speed_mt == 0
        assert ram.manufacturer == ""
