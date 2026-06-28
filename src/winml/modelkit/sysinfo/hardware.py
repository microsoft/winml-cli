# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from enum import Enum

from .helper import CimInstance, PnpDevice


logger = logging.getLogger(__name__)


def get_available_devices() -> list[str]:
    """Prioritized list of device categories present on this host (via WMI).

    Priority: NPU > GPU > CPU. Always includes "cpu" as fallback.
    Returns category strings ("npu", "gpu", "cpu"), not hardware instances —
    use ``NPU.get_all()`` / ``GPU.get_all()`` for instance-level inventory.
    """
    devices: list[str] = []

    try:
        if NPU.get_all():
            devices.append("npu")
    except Exception:
        logger.debug("NPU detection failed or unavailable")

    try:
        if GPU.get_all():
            devices.append("gpu")
    except Exception:
        logger.debug("GPU detection failed or unavailable")

    devices.append("cpu")  # CPU always available
    return devices


def get_vendor_id_device_id_from_pnp_id(pnp_id: str) -> tuple[int, int]:
    """Extract vendor ID and device ID from PNP device ID string."""
    # Qualcomm NPU id quirk 1
    if pnp_id.startswith("ACPI\\QCOM"):
        id_segment = pnp_id.split("\\")[1]
        if len(id_segment) != 8:
            raise ValueError(f"Invalid Qualcomm NPU PNPDeviceID format: {pnp_id}")
        # first four chars memcpy to a uint32_t
        vendor_id = int.from_bytes(id_segment[0:4].encode("ascii"), byteorder="little")
        # last four chars memcpy to a uint32_t
        device_id = int.from_bytes(id_segment[4:].encode("ascii"), byteorder="little")
        return vendor_id, device_id

    vendor_id_str_groups = re.search(r"VEN_([0-9A-Za-z]+)", pnp_id)
    if vendor_id_str_groups is None:
        raise ValueError(f"Invalid PNPDeviceID format: {pnp_id}")
    vendor_id_str = vendor_id_str_groups.group(1)
    device_id_str_groups = re.search(r"DEV_([0-9A-Za-z]+)", pnp_id)
    if device_id_str_groups is None:
        raise ValueError(f"Invalid PNPDeviceID format: {pnp_id}")
    device_id_str = device_id_str_groups.group(1)

    # Qualcomm NPU id quirk 2
    if vendor_id_str == "QCOM":
        # vendor id chars memcpy to uint32_t
        vendor_id = int.from_bytes(vendor_id_str.encode("ascii"), byteorder="little")
        # device id chars memcpy to uint32_t
        device_id = int.from_bytes(device_id_str.encode("ascii"), byteorder="little")
        return vendor_id, device_id

    vendor_id = int(vendor_id_str, 16)
    device_id = int(device_id_str, 16)
    return vendor_id, device_id


class CPU:
    """Represents CPU information from Windows WMI."""

    # reference https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-processor
    class Architecture(Enum):
        """CPU architecture types."""

        UNKNOWN = -1
        x86 = 0  # pylint: disable=invalid-name
        MIPS = 1
        Alpha = 2  # pylint: disable=invalid-name
        PowerPC = 3  # pylint: disable=invalid-name
        ARM = 5
        ia64 = 6  # pylint: disable=invalid-name
        x64 = 9  # pylint: disable=invalid-name
        ARM64 = 12

    @staticmethod
    def get_all() -> list["CPU"]:
        """Get all CPUs in the system."""
        cim_instances = CimInstance.get_by_class_name("Win32_Processor")
        return [CPU(cim_instance) for cim_instance in cim_instances]

    def __init__(self, cim_instance: CimInstance) -> None:
        self._name = cim_instance.try_get_property("Name", str, "")
        self._manufacturer = cim_instance.try_get_property("Manufacturer", str, "")
        self._core_count = cim_instance.try_get_property("NumberOfCores", int, 0)
        self._thread_count = cim_instance.try_get_property("NumberOfLogicalProcessors", int, 0)
        architecture = cim_instance.try_get_property("Architecture", int, -1)
        self._architecture = CPU.Architecture(architecture)

    @property
    def name(self) -> str:
        """CPU name."""
        return self._name

    @property
    def manufacturer(self) -> str:
        """CPU manufacturer."""
        return self._manufacturer

    @property
    def core_count(self) -> int:
        """Number of physical cores."""
        return self._core_count

    @property
    def thread_count(self) -> int:
        """Number of logical processors."""
        return self._thread_count

    @property
    def architecture(self) -> Architecture:
        """CPU architecture."""
        return self._architecture

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self._name,
            "manufacturer": self._manufacturer,
            "coreCount": self._core_count,
            "threadCount": self._thread_count,
            "architecture": self._architecture.name,
        }


class GPU:
    """Represents GPU information from Windows WMI."""

    @staticmethod
    def get_all() -> list["GPU"]:
        """Get all hardware GPUs in the system."""
        cim_instances = CimInstance.get_by_class_name("Win32_VideoController")
        results = []
        for cim_instance in cim_instances:
            # Assumption: Hardware GPUs are PCI or ACPI devices.
            # This is to filter out software GPUs like rdp adapters.
            pnp_id = cim_instance.get_property("PNPDeviceID", str)
            if pnp_id.startswith(("PCI\\", "ACPI\\")):
                results.append(GPU(cim_instance))
        return results

    def __init__(self, cim_instance: CimInstance) -> None:
        self._name = cim_instance.try_get_property("Name", str, "")
        self._manufacturer = cim_instance.try_get_property("AdapterCompatibility", str, "")
        self._driver_version = cim_instance.get_property("DriverVersion", str)
        self._vram_mib = int(cim_instance.try_get_property("AdapterRAM", int, 0) / (1024 * 1024))
        pnp_id = cim_instance.get_property("PNPDeviceID", str)
        self._vendor_id, self._device_id = get_vendor_id_device_id_from_pnp_id(pnp_id)

    @property
    def name(self) -> str:
        """GPU name."""
        return self._name

    @property
    def manufacturer(self) -> str:
        """GPU manufacturer."""
        return self._manufacturer

    @property
    def driver_version(self) -> str:
        """GPU driver version."""
        return self._driver_version

    @property
    def vram_mib(self) -> int:
        """GPU VRAM in MiB."""
        return self._vram_mib

    @property
    def vendor_id(self) -> int:
        """GPU vendor ID."""
        return self._vendor_id

    @property
    def device_id(self) -> int:
        """GPU device ID."""
        return self._device_id

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self._name,
            "manufacturer": self._manufacturer,
            "driverVersion": self._driver_version,
            "vramMib": self._vram_mib,
            "vendorId": self._vendor_id,
            "deviceId": self._device_id,
        }


class NPU:
    """Represents NPU (Neural Processing Unit) information."""

    @staticmethod
    def get_all() -> list["NPU"]:
        """Get all NPUs in the system."""
        pnp_devices = PnpDevice.get_by_class_name("ComputeAccelerator")
        return [NPU(pnp_device) for pnp_device in pnp_devices]

    def __init__(self, pnp_device: PnpDevice) -> None:
        self._name = pnp_device.try_get_property("Name", str, "")
        self._manufacturer = pnp_device.try_get_property("Manufacturer", str, "")
        self._driver_version = pnp_device.get_extra_property("DEVPKEY_Device_DriverVersion", str)
        pnp_id = pnp_device.get_property("PNPDeviceID", str)
        self._vendor_id, self._device_id = get_vendor_id_device_id_from_pnp_id(pnp_id)

    @property
    def name(self) -> str:
        """NPU name."""
        return self._name

    @property
    def manufacturer(self) -> str:
        """NPU manufacturer."""
        return self._manufacturer

    @property
    def driver_version(self) -> str:
        """NPU driver version."""
        return self._driver_version

    @property
    def vendor_id(self) -> int:
        """NPU vendor ID."""
        return self._vendor_id

    @property
    def device_id(self) -> int:
        """NPU device ID."""
        return self._device_id

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self._name,
            "manufacturer": self._manufacturer,
            "driverVersion": self._driver_version,
            "vendorId": self._vendor_id,
            "deviceId": self._device_id,
        }


class RAM:
    """Represents RAM module information."""

    @staticmethod
    def get_all() -> list["RAM"]:
        """Get all RAM modules in the system."""
        cim_instances = CimInstance.get_by_class_name("Win32_PhysicalMemory")
        return [RAM(cim_instance) for cim_instance in cim_instances]

    def __init__(self, cim_instance: CimInstance) -> None:
        self._capacity_mib = int(cim_instance.try_get_property("Capacity", int, 0) / (1024 * 1024))
        self._speed_mt = cim_instance.try_get_property("Speed", int, 0)
        self._manufacturer = cim_instance.try_get_property("Manufacturer", str, "")

    @property
    def capacity_mib(self) -> int:
        """RAM capacity in MiB."""
        return self._capacity_mib

    @property
    def speed_mt(self) -> int:
        """RAM speed in MT/s."""
        return self._speed_mt

    @property
    def manufacturer(self) -> str:
        """RAM manufacturer."""
        return self._manufacturer

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "capacityMib": self._capacity_mib,
            "speedMt": self._speed_mt,
            "manufacturer": self._manufacturer,
        }
