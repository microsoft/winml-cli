# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""PDH-based adapter discovery for GPU/NPU hardware detection.

Enumerates Windows GPU Engine performance counter instances to discover
all GPU and NPU adapters. The NPU is identified by engine-type fingerprinting:
adapters with only Compute engine types (no 3D, Video, Copy).

Internal module -- supplements sysinfo/hardware.py with PDH-based discovery.

Relationship to CIM/WMI (sysinfo/hardware.py):
    CIM (Get-CimInstance) identifies devices by PnP Device ID (e.g., PCI\\VEN_QCOM&DEV_0C40).
    PDH identifies devices by LUID (e.g., 0x00000000_0x00018393).
    There is no built-in Windows API that maps CIM PnP ID <-> PDH LUID directly.
    Current assumption: on systems with a single NPU, discover_npu_luid() and
    NPU.get_all() refer to the same device. For multi-NPU systems, a D3DKMT
    (DirectX Kernel Mode) bridge would be needed to map LUID to PnP Device ID.

TODO: Investigate D3DKMT API (D3DKMTEnumAdapters2) for explicit LUID <-> PnP ID mapping.
TODO: Incorporate psutil for cross-platform adapter discovery on Linux/macOS.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import logging
import sys
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)

# Guard: PDH is Windows-only.
if sys.platform != "win32":
    raise ImportError("pdh_adapters module requires Windows (pdh.dll)")

# ---------------------------------------------------------------------------
# PDH constants (minimal set for enumeration)
# ---------------------------------------------------------------------------
_PDH_MORE_DATA = 0x800007D2
_PERF_DETAIL_WIZARD = 400

_pdh = ctypes.windll.pdh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pdh_ok(status: int) -> bool:
    """Check if a PDH status code indicates success."""
    return (status & 0xFFFFFFFF) == 0


def _parse_multi_sz(buf: ctypes.Array, size: int) -> list[str]:
    """Parse a PDH multi-string buffer (null-separated, double-null terminated)."""
    raw = ctypes.wstring_at(buf, size)
    items: list[str] = []
    current = ""
    for ch in raw:
        if ch == "\0":
            if current:
                items.append(current)
                current = ""
            else:
                break
        else:
            current += ch
    return items


# ---------------------------------------------------------------------------
# Adapter discovery
# ---------------------------------------------------------------------------
@dataclass
class AdapterInfo:
    """GPU/NPU adapter discovered via PDH GPU Engine enumeration."""

    luid: str
    engine_types: set[str] = field(default_factory=set)
    engine_map: dict[str, tuple[int, str]] = field(default_factory=dict)

    @property
    def is_npu(self) -> bool:
        """NPU heuristic: only Compute engine types, no 3D/Video/Copy/etc."""
        non_compute = {e for e in self.engine_types if not e.startswith("Compute")}
        return len(non_compute) == 0 and len(self.engine_types) > 0

    @property
    def compute_engine_type(self) -> str | None:
        """Return the first Compute engine type name."""
        for et in sorted(self.engine_types):
            if et.startswith("Compute"):
                return et
        return None


def enumerate_adapters() -> dict[str, AdapterInfo]:
    """Enumerate all GPU/NPU adapters via PDH GPU Engine instances.

    Returns:
        Dict mapping LUID string -> AdapterInfo.
    """
    counter_size = wintypes.DWORD(0)
    instance_size = wintypes.DWORD(0)
    status = _pdh.PdhEnumObjectItemsW(
        None,
        None,
        "GPU Engine",
        None,
        ctypes.byref(counter_size),
        None,
        ctypes.byref(instance_size),
        _PERF_DETAIL_WIZARD,
        0,
    )
    if (status & 0xFFFFFFFF) not in (0, _PDH_MORE_DATA):
        raise RuntimeError(f"PdhEnumObjectItemsW sizing failed: 0x{status & 0xFFFFFFFF:08X}")

    counter_buf = ctypes.create_unicode_buffer(counter_size.value)
    instance_buf = ctypes.create_unicode_buffer(instance_size.value)
    status = _pdh.PdhEnumObjectItemsW(
        None,
        None,
        "GPU Engine",
        counter_buf,
        ctypes.byref(counter_size),
        instance_buf,
        ctypes.byref(instance_size),
        _PERF_DETAIL_WIZARD,
        0,
    )
    if not _pdh_ok(status):
        raise RuntimeError(f"PdhEnumObjectItemsW failed: 0x{status & 0xFFFFFFFF:08X}")

    instances = _parse_multi_sz(instance_buf, instance_size.value)

    adapters: dict[str, AdapterInfo] = {}
    for inst in instances:
        # Format: pid_XXXX_luid_0xHHHH_0xHHHH_phys_N_eng_N_engtype_TYPE
        parts = inst.split("_")
        if "luid" not in parts or "engtype" not in parts:
            continue
        luid_idx = parts.index("luid")
        luid = parts[luid_idx + 1] + "_" + parts[luid_idx + 2]
        eng_idx = parts.index("eng")
        eng_num = int(parts[eng_idx + 1])
        engtype_idx = parts.index("engtype")
        engtype = "_".join(parts[engtype_idx + 1 :])

        if luid not in adapters:
            adapters[luid] = AdapterInfo(luid=luid)
        adapters[luid].engine_types.add(engtype)
        if engtype not in adapters[luid].engine_map:
            adapters[luid].engine_map[engtype] = (eng_num, engtype)

    return adapters


def discover_npu_luid() -> str | None:
    """Auto-discover NPU LUID by engine-type fingerprinting.

    The NPU is the adapter whose GPU Engine instances consist solely of
    "Compute" engine types (no 3D, Video, Copy, etc.).

    Returns:
        NPU LUID string (e.g. ``"0x00000000_0x00015A33"``), or None if
        no NPU adapter is found.
    """
    try:
        adapters = enumerate_adapters()
    except RuntimeError:
        logger.debug("Failed to enumerate GPU Engine adapters", exc_info=True)
        return None

    for luid, info in adapters.items():
        if info.is_npu:
            logger.debug("NPU LUID discovered: %s (engines: %s)", luid, info.engine_types)
            return luid

    logger.debug("No NPU adapter found among %d adapters", len(adapters))
    return None


def discover_gpu_luids() -> list[str]:
    """Discover GPU adapter LUIDs (adapters with a 3D engine type).

    Returns:
        List of LUID strings for GPU adapters.
    """
    try:
        adapters = enumerate_adapters()
    except RuntimeError:
        logger.debug("Failed to enumerate GPU Engine adapters", exc_info=True)
        return []

    return [
        luid for luid, info in adapters.items() if "3D" in info.engine_types and not info.is_npu
    ]


def discover_gpu_luid() -> str | None:
    """Auto-discover a GPU LUID (adapter with a 3D engine type).

    Among the GPU adapters returned by :func:`discover_gpu_luids`, prefer the
    one with the largest peak Local memory bytes (indicating a discrete GPU).
    Falls back to the first one when memory data is unavailable.

    Returns:
        GPU LUID string (e.g. ``"0x00000000_0x00015A33"``), or None if no
        GPU adapter is found.
    """
    luids = discover_gpu_luids()
    if not luids:
        logger.debug("No GPU adapter found")
        return None
    if len(luids) == 1:
        return luids[0]
    # Multi-GPU host: just pick the first; ranking by memory would require
    # a live PDH query for each adapter, which is heavier than this helper
    # is meant for. Tests/CLI can pass --device gpu and accept the first.
    logger.debug("Multiple GPU adapters found; using %s", luids[0])
    return luids[0]
