# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Read-only power configuration probes for the system report."""

from __future__ import annotations

import ctypes
import logging
import math
import re
import shutil
import subprocess
import sys
from typing import Any, ClassVar


logger = logging.getLogger(__name__)


def _get_active_power_scheme() -> dict[str, str] | None:
    """Read the active Windows scheme without relying on localized labels."""
    if sys.platform != "win32":
        return None
    executable = shutil.which("powercfg")
    if executable is None:
        logger.warning("Cannot read the active power scheme: powercfg is unavailable")
        return None
    try:
        output = subprocess.check_output(  # noqa: S603 - Resolved tool path and fixed arguments
            [executable, "/getactivescheme"],
            text=True,
            errors="replace",
            stderr=subprocess.PIPE,
            timeout=10,
        )
        match = re.search(
            r"([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})\s*\((.*)\)",
            output,
        )
        if match is None:
            raise ValueError("powercfg returned no active scheme GUID and name")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning("Failed to read the active power scheme: %s", exc)
        return None
    return {"guid": match[1].lower(), "name": match[2]}


def _parse_nvidia_power_limits(output: str) -> dict[str, dict[str, float]]:
    """Extract current GPU/module limits, keyed by canonical PCI bus ID."""
    result: dict[str, dict[str, float]] = {}
    bus_id: str | None = None
    section: str | None = None
    for line in output.splitlines():
        text = line.strip()
        gpu = re.fullmatch(
            r"GPU ([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-9a-fA-F]+)", text
        )
        if gpu:
            domain, bus, device, function = (int(part, 16) for part in gpu.groups())
            bus_id = f"{domain:08x}:{bus:02x}:{device:02x}.{function:x}"
            result[bus_id] = {}
            section = None
        elif line.startswith("    ") and not line.startswith("        "):
            section = {
                "GPU Power Readings": "gpu_current_power_limit_w",
                "Module Power Readings": "module_current_power_limit_w",
            }.get(text)
        elif bus_id is not None and section is not None:
            label, separator, value = text.partition(":")
            if separator and label.strip() == "Current Power Limit":
                value = value.strip()
                if value == "N/A":
                    continue
                try:
                    if not value.endswith(" W"):
                        raise ValueError("expected watts")
                    watts = float(value[:-2])
                    if not math.isfinite(watts) or watts < 0:
                        raise ValueError("expected a finite nonnegative power limit")
                except ValueError:
                    logger.warning("Invalid NVIDIA current power limit for %s: %s", bus_id, value)
                    continue
                result[bus_id][section] = watts
    if not result:
        logger.warning("NVIDIA power query returned no GPU PCI identities")
    return result


def _get_nvidia_power_limits() -> dict[str, dict[str, float]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        logger.debug("NVIDIA power limits are unavailable: nvidia-smi was not found")
        return {}
    try:
        output = subprocess.check_output(  # noqa: S603 - Resolved tool path and fixed arguments
            [executable, "-q", "-d", "POWER"],
            text=True,
            errors="replace",
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Failed to read NVIDIA power limits: %s", exc)
        return {}
    return _parse_nvidia_power_limits(output)


class _LUID(ctypes.Structure):
    _fields_: ClassVar = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]


class _OpenAdapterFromLUID(ctypes.Structure):
    _fields_: ClassVar = [("luid", _LUID), ("handle", ctypes.c_uint32)]


class _AdapterAddress(ctypes.Structure):
    _fields_: ClassVar = [
        ("bus", ctypes.c_uint32),
        ("device", ctypes.c_uint32),
        ("function", ctypes.c_uint32),
    ]


class _QueryAdapterInfo(ctypes.Structure):
    _fields_: ClassVar = [
        ("handle", ctypes.c_uint32),
        ("query_type", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


class _CloseAdapter(ctypes.Structure):
    _fields_: ClassVar = [("handle", ctypes.c_uint32)]


def _get_gpu_pci_bus_id(luid: str) -> str | None:
    """Bridge a DXCore LUID to its PCI address, including identical GPUs."""
    if sys.platform != "win32":
        return None
    try:
        high, low = (int(part, 16) for part in luid.split("_"))
        gdi32 = ctypes.WinDLL("gdi32")
        open_adapter = gdi32.D3DKMTOpenAdapterFromLuid
        query_adapter = gdi32.D3DKMTQueryAdapterInfo
        close_adapter = gdi32.D3DKMTCloseAdapter
        for function, structure in (
            (open_adapter, _OpenAdapterFromLUID),
            (query_adapter, _QueryAdapterInfo),
            (close_adapter, _CloseAdapter),
        ):
            function.argtypes = [ctypes.POINTER(structure)]
            function.restype = ctypes.c_int32
        adapter = _OpenAdapterFromLUID(_LUID(low, high), 0)
        status = open_adapter(ctypes.byref(adapter))
        if status < 0:
            raise OSError(f"D3DKMTOpenAdapterFromLuid returned 0x{status & 0xFFFFFFFF:08x}")
        try:
            address = _AdapterAddress()
            # KMTQAITYPE_ADAPTERADDRESS = 6 (d3dkmthk.h).
            query = _QueryAdapterInfo(
                adapter.handle, 6, ctypes.addressof(address), ctypes.sizeof(address)
            )
            status = query_adapter(ctypes.byref(query))
            if status < 0:
                raise OSError(f"D3DKMTQueryAdapterInfo returned 0x{status & 0xFFFFFFFF:08x}")
            # Windows' adapter address API exposes no PCI domain; do not
            # match readings from other domains by dropping that identity.
            return f"00000000:{address.bus:02x}:{address.device:02x}.{address.function:x}"
        finally:
            status = close_adapter(ctypes.byref(_CloseAdapter(adapter.handle)))
            if status < 0:
                logger.warning(
                    "D3DKMTCloseAdapter failed for %s: 0x%08x", luid, status & 0xFFFFFFFF
                )
    except (OSError, AttributeError, ValueError) as exc:
        logger.warning("Cannot resolve GPU PCI identity for LUID %s: %s", luid, exc)
        return None


def _enrich_nvidia_power_limits(devices: list[dict[str, Any]]) -> None:
    """Join power limits by PCI identity, never by GPU name or list order."""
    if not devices:
        return
    limits = _get_nvidia_power_limits()
    if not limits:
        return
    for device in devices:
        luid = device["details"].get("luid")
        if not luid:
            logger.warning(
                "Cannot match NVIDIA power limits for %s: no device LUID", device["name"]
            )
            continue
        bus_id = _get_gpu_pci_bus_id(luid)
        if bus_id is not None:
            if bus_id in limits:
                device["details"].update(limits[bus_id])
            else:
                logger.warning("No NVIDIA power readings match GPU LUID %s (PCI %s)", luid, bus_id)
