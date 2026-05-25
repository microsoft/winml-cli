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
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..utils.constants import EPName

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
        """NPU heuristic: only Compute/Neural engine types, no 3D/Video/Copy/etc."""
        non_compute = {e for e in self.engine_types if not e.startswith("Compute") and not e.startswith("Neural")}
        return len(non_compute) == 0 and len(self.engine_types) > 0


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

    On hosts with multiple GPUs, returns the first LUID :func:`discover_gpu_luids`
    yields. Callers that need a specific adapter (e.g. the discrete one on a
    hybrid laptop) should pair ``--device gpu`` with ``--ep`` so that
    :func:`resolve_adapter_luid` consults ORT's autoEP metadata instead.

    Returns:
        GPU LUID string (e.g. ``"0x00000000_0x00015A33"``), or None if no
        GPU adapter is found.
    """
    luids = discover_gpu_luids()
    if not luids:
        logger.debug("No GPU adapter found")
        return None
    if len(luids) > 1:
        logger.debug("Multiple GPU adapters found; using %s", luids[0])
    return luids[0]


def _format_pdh_luid(decimal_luid: str) -> str:
    """Format a decimal LUID string as PDH ``"0xHHHHHHHH_0xHHHHHHHH"``.

    ORT's autoEP exposes ``OrtHardwareDevice.metadata["LUID"]`` as a base-10
    integer string covering the full 64 bits. PDH counter paths split that
    into high/low 32-bit halves rendered in upper-case hex. This helper does
    that conversion (and only that — callers must ensure the input parses
    as an integer).
    """
    v = int(decimal_luid)
    return f"0x{(v >> 32) & 0xFFFFFFFF:08X}_0x{v & 0xFFFFFFFF:08X}"


def resolve_adapter_luid(
    device_kind: str,
    ep_name: EPName | None = None,
) -> str | None:
    """Resolve the adapter LUID that ORT will bind an EP to.

    Asks ``onnxruntime.get_ep_devices()`` first: ORT's autoEP API publishes
    ``OrtHardwareDevice.metadata["LUID"]`` for every EP-device pair, which is
    authoritative for *which* GPU/NPU the inference session ends up using.
    Falls back to the PDH-only heuristic helpers when ORT can't be imported,
    when no matching ep_device is registered, or when the matched device has
    no LUID metadata (some EPs / older ORT builds).

    Args:
        device_kind: ``"npu"`` or ``"gpu"`` (case-insensitive).
        ep_name: Full ORT EP name (e.g. ``"QNNExecutionProvider"``) to
            disambiguate when several EPs cover the same device type. Pass
            ``None`` to match the first ep_device of the requested type.

    Returns:
        PDH-formatted LUID (``"0xHHHHHHHH_0xHHHHHHHH"``) or None if neither
        ORT nor the PDH fallback found a matching adapter.
    """
    kind = (device_kind or "").lower()
    if kind not in ("npu", "gpu"):
        return None

    luid = _resolve_via_ort(kind, ep_name)
    if luid is not None:
        return luid

    # Fallback: PDH-only heuristic discovery.
    if kind == "npu":
        return discover_npu_luid()
    return discover_gpu_luid()


def _resolve_via_ort(kind: str, ep_name: EPName | None) -> str | None:
    r"""Look up the adapter LUID through ORT's autoEP registry.

    Returns None silently on any failure so callers can fall back. ORT-resolved
    LUIDs are sanity-checked against :func:`enumerate_adapters` because some
    EPs publish LUIDs that PDH doesn't expose under ``\GPU Engine`` (e.g.
    drivers that register the adapter only for autoEP); using such a LUID
    would later raise inside :func:`build_adapter_query`.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return None

    try:
        ep_devices = ort.get_ep_devices()
    except Exception:
        logger.debug("ort.get_ep_devices() failed", exc_info=True)
        return None

    target_type = getattr(ort.OrtHardwareDeviceType, "NPU" if kind == "npu" else "GPU", None)
    if target_type is None:
        return None

    # Lazy PDH enumeration: only probe once, only if ORT yields a candidate.
    # None = not yet probed; {} = enumeration failed (skip validation).
    pdh_known: dict[str, AdapterInfo] | None = None

    for ep_dev in ep_devices:
        try:
            if ep_name and ep_dev.ep_name != ep_name:
                continue
            if ep_dev.device.type != target_type:
                continue
            metadata = ep_dev.device.metadata
            decimal_luid = dict(metadata).get("LUID") if metadata is not None else None
        except Exception:
            logger.debug("Skipping malformed ep_device", exc_info=True)
            continue
        if not decimal_luid:
            continue
        try:
            formatted = _format_pdh_luid(decimal_luid)
        except (TypeError, ValueError):
            logger.debug("Unparseable LUID metadata: %r", decimal_luid)
            continue
        if pdh_known is None:
            try:
                pdh_known = enumerate_adapters()
            except Exception:
                logger.debug("enumerate_adapters() failed; skipping LUID validation", exc_info=True)
                pdh_known = {}
        if pdh_known and formatted not in pdh_known:
            logger.debug(
                "ORT-resolved %s LUID %s not in PDH enumeration; skipping",
                kind.upper(),
                formatted,
            )
            continue
        logger.debug(
            "Resolved %s LUID via ORT (ep=%s, ep_name=%s): %s",
            kind.upper(),
            ep_dev.ep_name,
            ep_name,
            formatted,
        )
        return formatted

    return None
