# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Windows PDH (Performance Data Helper) ctypes wrapper for GPU/NPU monitoring.

Provides zero-dependency, in-process access to Windows performance counters
via ctypes calls to pdh.dll. Used to monitor NPU utilization, memory, and
running time during ONNX Runtime inference.

Adapter discovery (NPU/GPU enumeration) lives in
``modelkit.sysinfo.pdh_adapters`` and is imported here for use by
``build_adapter_query``, ``build_npu_query``, and ``PdhPoller``.

Internal module -- not part of the public API.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import logging
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar


if TYPE_CHECKING:
    from ...utils.constants import EPName

logger = logging.getLogger(__name__)

# Guard: PDH is Windows-only.
if sys.platform != "win32":
    raise ImportError("_pdh module requires Windows (pdh.dll)")

# Device discovery lives in sysinfo; import here for use by
# build_adapter_query / build_npu_query / PdhPoller.
from ...sysinfo.pdh_adapters import (  # noqa: E402
    discover_gpu_luid,
    discover_npu_luid,
    enumerate_adapters,
    resolve_adapter_luid,
)


# Device kind for hardware monitoring. "auto" probes NPU first, then GPU.
_DEVICE_KINDS = ("npu", "gpu", "cpu", "auto")


# ---------------------------------------------------------------------------
# PDH constants
# ---------------------------------------------------------------------------
_PDH_FMT_DOUBLE = 0x00000200
_PDH_FMT_LARGE = 0x00000400

_pdh = ctypes.windll.pdh


# ---------------------------------------------------------------------------
# PDH structure definitions
# ---------------------------------------------------------------------------
class _PdhFmtDouble(ctypes.Structure):
    _fields_: ClassVar = [
        ("CStatus", wintypes.DWORD),
        ("doubleValue", ctypes.c_double),
    ]


class _PdhFmtLarge(ctypes.Structure):
    _fields_: ClassVar = [
        ("CStatus", wintypes.DWORD),
        ("_padding", wintypes.DWORD),
        ("largeValue", ctypes.c_longlong),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pdh_ok(status: int) -> bool:
    """Check if a PDH status code indicates success."""
    return (status & 0xFFFFFFFF) == 0


# ---------------------------------------------------------------------------
# PDH Query
# ---------------------------------------------------------------------------
@dataclass
class _CounterEntry:
    """Internal: a registered PDH counter."""

    name: str
    path: str
    handle: wintypes.HANDLE = field(default_factory=wintypes.HANDLE)
    fmt: int = _PDH_FMT_LARGE
    registered: bool = False


class PdhQuery:
    r"""Manages a PDH query with named counters.

    Usage::

        q = PdhQuery()
        q.open()
        q.add_counter("util", r"\\GPU Engine(...)\\Utilization Percentage",
                       fmt="double")
        q.prime()           # first collect (needed for rate counters)
        values = q.collect()  # retries until valid; {"util": 42.5}
        q.close()
    """

    def __init__(self) -> None:
        self._query = wintypes.HANDLE()
        self._counters: list[_CounterEntry] = []
        self._opened = False

    def open(self) -> None:
        """Open the PDH query."""
        status = _pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query))
        if not _pdh_ok(status):
            raise RuntimeError(f"PdhOpenQueryW failed: 0x{status & 0xFFFFFFFF:08X}")
        self._opened = True

    def add_counter(
        self,
        name: str,
        path: str,
        *,
        fmt: str = "large",
    ) -> bool:
        """Register a counter. Returns True if registration succeeded.

        Args:
            name: Logical name for this counter.
            path: Full English PDH counter path.
            fmt: ``"double"`` for percentages, ``"large"`` for bytes/ns.
        """
        pdh_fmt = _PDH_FMT_DOUBLE if fmt == "double" else _PDH_FMT_LARGE
        entry = _CounterEntry(name=name, path=path, fmt=pdh_fmt)
        status = _pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(entry.handle))
        entry.registered = _pdh_ok(status)
        if not entry.registered:
            logger.debug("Counter '%s' registration failed: 0x%08X", name, status & 0xFFFFFFFF)
        self._counters.append(entry)
        return entry.registered

    def prime(self) -> None:
        """Perform an initial collect (required for rate-based counters).

        Rate counters like Utilization Percentage need two consecutive
        collects to compute a rate. Call this once before the first
        meaningful collect.
        """
        _pdh.PdhCollectQueryData(self._query)

    def _collect_once(self) -> dict[str, float | int | None]:
        """Single-shot PDH query. May return ``None`` for rate counters."""
        _pdh.PdhCollectQueryData(self._query)

        values: dict[str, float | int | None] = {}
        ct = wintypes.DWORD()

        for entry in self._counters:
            if not entry.registered:
                values[entry.name] = None
                continue

            if entry.fmt == _PDH_FMT_DOUBLE:
                val = _PdhFmtDouble()
                s = _pdh.PdhGetFormattedCounterValue(
                    entry.handle, _PDH_FMT_DOUBLE, ctypes.byref(ct), ctypes.byref(val)
                )
                values[entry.name] = (
                    val.doubleValue if _pdh_ok(s) and _pdh_ok(val.CStatus) else None
                )
            else:
                val = _PdhFmtLarge()
                s = _pdh.PdhGetFormattedCounterValue(
                    entry.handle, _PDH_FMT_LARGE, ctypes.byref(ct), ctypes.byref(val)
                )
                values[entry.name] = val.largeValue if _pdh_ok(s) and _pdh_ok(val.CStatus) else None

        return values

    def collect(
        self,
        *,
        timeout: float = 1.0,
        interval: float = 0.1,
    ) -> dict[str, float | int | None]:
        """Collect current values for all registered counters.

        Rate-based PDH counters (e.g. ``% Processor Time``) can transiently
        return ``None`` right after :meth:`prime` on busy systems.  This method
        retries at *interval* seconds until every registered counter yields a
        non-None value or *timeout* is exceeded.

        Args:
            timeout: Maximum seconds to wait for valid data (default 1.0).
            interval: Seconds between retries (default 0.1).

        Returns:
            Dict mapping counter name -> value.  Values may still be ``None``
            if *timeout* is exceeded.
        """
        deadline = time.monotonic() + timeout
        while True:
            values = self._collect_once()
            if all(v is not None for v in values.values()):
                return values
            if time.monotonic() >= deadline:
                return values
            time.sleep(interval)

    def close(self) -> None:
        """Close the PDH query and release resources."""
        if self._opened:
            _pdh.PdhCloseQuery(self._query)
            self._opened = False

    @property
    def counter_names(self) -> list[str]:
        """Names of all registered counters."""
        return [c.name for c in self._counters]


def build_adapter_query(
    luid: str,
    engine_type: str = "Compute",
    *,
    pid: int | None = None,
) -> PdhQuery:
    """Build a PdhQuery for any GPU/NPU adapter.

    Generic builder that works for NPU (Compute engine), GPU (3D engine),
    or any other adapter type.

    Args:
        luid: Adapter LUID string (e.g. ``"0x00000000_0x00015A33"``).
        engine_type: Engine type to monitor (e.g. ``"Compute"`` for NPU,
            ``"3D"`` for GPU). Must match an engine type on the adapter.
        pid: Process ID to monitor. Defaults to current process.

    Returns:
        An opened PdhQuery with utilization, running time, and memory
        counters registered.

    Raises:
        ValueError: If the LUID or engine type is not found.
    """
    if pid is None:
        pid = os.getpid()

    adapters = enumerate_adapters()
    adapter_info = adapters.get(luid)
    if adapter_info is None:
        raise ValueError(f"LUID {luid} not found in adapter enumeration")

    # Find the matching engine type
    matched_engine = None
    for et in sorted(adapter_info.engine_types):
        if et.startswith(engine_type):
            matched_engine = et
            break
    if matched_engine is None:
        raise ValueError(
            f"Engine type '{engine_type}' not found on adapter {luid}. "
            f"Available: {sorted(adapter_info.engine_types)}"
        )
    eng_num = adapter_info.engine_map[matched_engine][0]

    query = PdhQuery()
    query.open()

    # Per-process engine counters
    query.add_counter(
        "utilization_pct",
        rf"\GPU Engine(pid_{pid}_luid_{luid}"
        rf"_phys_0_eng_{eng_num}_engtype_{matched_engine})\Utilization Percentage",
        fmt="double",
    )
    query.add_counter(
        "running_time_ns",
        rf"\GPU Engine(pid_{pid}_luid_{luid}"
        rf"_phys_0_eng_{eng_num}_engtype_{matched_engine})\Running Time",
        fmt="large",
    )

    # Per-process memory
    query.add_counter(
        "memory_local_bytes",
        rf"\GPU Process Memory(pid_{pid}_luid_{luid}_phys_0)\Local Usage",
        fmt="large",
    )
    query.add_counter(
        "memory_shared_bytes",
        rf"\GPU Process Memory(pid_{pid}_luid_{luid}_phys_0)\Shared Usage",
        fmt="large",
    )

    return query


def build_npu_query(npu_luid: str, pid: int | None = None) -> PdhQuery:
    """Convenience wrapper: build a query for the NPU (Compute engine).

    Args:
        npu_luid: NPU LUID string.
        pid: Process ID to monitor. Defaults to current process.

    Returns:
        An opened PdhQuery configured for NPU monitoring.
    """
    return build_adapter_query(npu_luid, engine_type="Compute", pid=pid)


def build_gpu_query(gpu_luid: str, pid: int | None = None) -> PdhQuery:
    """Convenience wrapper: build a query for the GPU (3D engine).

    Args:
        gpu_luid: GPU LUID string.
        pid: Process ID to monitor. Defaults to current process.

    Returns:
        An opened PdhQuery configured for GPU monitoring.
    """
    return build_adapter_query(gpu_luid, engine_type="3D", pid=pid)


# ---------------------------------------------------------------------------
# PdhPoller — reusable background polling component
# ---------------------------------------------------------------------------
class PdhPoller:
    """Reusable background PDH polling component.

    Monitors CPU, RAM, and optionally NPU/GPU metrics via Windows PDH counters.
    Handles: discover NPU/GPU LUID, register counters, background thread,
    sample collection, cleanup.

    The ``device`` argument selects which adapter to monitor:

    * ``"npu"`` - discover and poll the NPU (Compute engine).
    * ``"gpu"`` - discover and poll the GPU (3D engine).
    * ``"cpu"`` - skip adapter polling; collect only CPU/RAM.
    * ``"auto"`` - probe NPU first, fall back to GPU, then CPU/RAM only.

    # TODO: Incorporate psutil for cross-platform CPU/RAM monitoring.
    # PDH is Windows-only; psutil would enable monitoring on Linux/macOS.
    # See: https://github.com/giampaolo/psutil

    Usage::

        poller = PdhPoller(poll_interval_ms=200, device="gpu")
        poller.start()
        # ... run inference ...
        poller.stop()
        print(poller.mean_utilization_pct)
    """

    def __init__(
        self,
        poll_interval_ms: int = 200,
        device: str = "auto",
        ep_name: EPName | None = None,
    ) -> None:
        device_norm = (device or "auto").lower()
        if device_norm not in _DEVICE_KINDS:
            raise ValueError(f"Unknown device {device!r}; expected one of {_DEVICE_KINDS}")
        self._poll_interval_s = poll_interval_ms / 1000.0
        self._requested_device = device_norm
        # Full ORT EP name (e.g. "QNNExecutionProvider") to disambiguate when
        # multiple EPs cover the same device type during LUID resolution.
        self._ep_name = ep_name
        self._device_kind: str | None = None  # resolved at start(): "npu" | "gpu" | None
        self._query: PdhQuery | None = None
        self._adapter_luid: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._util_samples: list[float] = []
        self._memory_local_bytes: list[int] = []
        self._memory_shared_bytes: list[int] = []
        self._cpu_samples: list[float] = []
        self._ram_used_bytes: list[int] = []
        self._running_time_start_ns: int | None = None
        self._running_time_end_ns: int | None = None

    def start(self) -> None:
        """Resolve target device, register PDH counters, start background thread.

        Monitors CPU and RAM always. Adapter utilization and memory are added
        when the requested NPU/GPU adapter is discovered via ORT (preferred)
        or PDH fingerprinting (fallback).
        """
        try:
            self._adapter_luid, self._device_kind = self._resolve_adapter(
                self._requested_device, self._ep_name
            )

            # Try to build the per-adapter query. If the resolved LUID is
            # missing from PDH enumeration or the engine type isn't present
            # build_adapter_query raises ValueError; we degrade to CPU/RAM
            # rather than failing the whole monitor.
            self._query = None
            if self._adapter_luid is not None and self._device_kind in ("npu", "gpu"):
                try:
                    if self._device_kind == "npu":
                        self._query = build_npu_query(self._adapter_luid)
                    else:
                        self._query = build_gpu_query(self._adapter_luid)
                except ValueError as exc:
                    logger.info(
                        "Adapter query unavailable for %s LUID %s (%s); monitoring CPU/RAM only",
                        self._device_kind.upper(),
                        self._adapter_luid,
                        exc,
                    )
                    self._adapter_luid = None
                    self._device_kind = None

            if self._query is None:
                if self._requested_device in ("npu", "gpu"):
                    logger.info(
                        "%s not found via PDH; monitoring CPU/RAM only",
                        self._requested_device.upper(),
                    )
                else:
                    logger.info("No NPU/GPU found via PDH; monitoring CPU/RAM only")
                self._query = PdhQuery()
                self._query.open()

            # System-wide counters (always available)
            self._query.add_counter(
                "cpu_pct",
                r"\Processor(_Total)\% Processor Time",
                fmt="double",
            )
            self._query.add_counter(
                "ram_committed_bytes",
                r"\Memory\Committed Bytes",
                fmt="large",
            )

            self._query.prime()

            initial = self._query.collect(interval=0.05)
            rt = initial.get("running_time_ns")
            if rt is not None:
                self._running_time_start_ns = rt

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._poll_loop,
                name="PdhPoller",
                daemon=True,
            )
            self._thread.start()
            logger.debug("PdhPoller started (interval=%.1fms)", self._poll_interval_s * 1000)

        except (ImportError, RuntimeError) as exc:
            logger.warning("PDH monitoring unavailable: %s", exc)

    def stop(self) -> None:
        """Stop polling thread, capture final running_time, close query."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("PdhPoller thread did not stop within timeout")
            self._thread = None

        if self._query is not None:
            try:
                final = self._query._collect_once()
                rt = final.get("running_time_ns")
                if rt is not None:
                    self._running_time_end_ns = rt
            except Exception:
                pass
            self._query.close()
            self._query = None

        logger.debug(
            "PdhPoller stopped: %d util samples, %d local mem, %d shared mem, %d cpu samples",
            len(self._util_samples),
            len(self._memory_local_bytes),
            len(self._memory_shared_bytes),
            len(self._cpu_samples),
        )

    def _poll_loop(self) -> None:
        """Background thread: poll PDH counters at fixed interval."""
        while not self._stop_event.is_set():
            try:
                values = self._query._collect_once()
                util = values.get("utilization_pct")
                mem_local = values.get("memory_local_bytes")
                mem_shared = values.get("memory_shared_bytes")
                cpu = values.get("cpu_pct")
                ram = values.get("ram_committed_bytes")
                with self._lock:
                    if util is not None:
                        self._util_samples.append(util)
                    if mem_local is not None:
                        self._memory_local_bytes.append(mem_local)
                    if mem_shared is not None:
                        self._memory_shared_bytes.append(mem_shared)
                    if cpu is not None:
                        self._cpu_samples.append(cpu)
                    if ram is not None:
                        self._ram_used_bytes.append(ram)
            except Exception:
                logger.debug("PdhPoller poll error", exc_info=True)
            self._stop_event.wait(self._poll_interval_s)

    @staticmethod
    def _resolve_adapter(
        requested: str, ep_name: EPName | None = None
    ) -> tuple[str | None, str | None]:
        """Return (luid, kind) for the requested device.

        Uses :func:`resolve_adapter_luid` so that — when ORT's autoEP API
        publishes ``OrtHardwareDevice.metadata["LUID"]`` — the monitor tracks
        the same adapter the inference session binds to. Falls back to PDH
        fingerprinting when ORT data is unavailable.

        ``kind`` is ``"npu"`` or ``"gpu"`` when an adapter is found; both
        elements are ``None`` when the requested device is ``"cpu"`` or no
        adapter could be discovered.
        """
        if requested == "cpu":
            return None, None
        if requested == "npu":
            luid = resolve_adapter_luid("npu", ep_name=ep_name)
            return luid, ("npu" if luid else None)
        if requested == "gpu":
            luid = resolve_adapter_luid("gpu", ep_name=ep_name)
            return luid, ("gpu" if luid else None)
        # auto: prefer NPU, then GPU. EP name (if any) only narrows within
        # the matching device type, so the same hint is reused for both.
        luid = resolve_adapter_luid("npu", ep_name=ep_name)
        if luid is not None:
            return luid, "npu"
        luid = resolve_adapter_luid("gpu", ep_name=ep_name)
        if luid is not None:
            return luid, "gpu"
        return None, None

    @property
    def adapter_luid(self) -> str | None:
        """LUID string for the monitored adapter (NPU or GPU), or None."""
        return self._adapter_luid

    @property
    def device_kind(self) -> str | None:
        """Resolved adapter kind: ``"npu"``, ``"gpu"``, or None when only CPU/RAM."""
        return self._device_kind

    @property
    def mean_utilization_pct(self) -> float:
        """Mean adapter (NPU/GPU) utilization % during polling period."""
        with self._lock:
            valid = [s for s in self._util_samples if s is not None]
        if not valid:
            return 0.0
        return statistics.mean(valid)

    @property
    def peak_utilization_pct(self) -> float:
        """Peak adapter (NPU/GPU) utilization % during polling period."""
        with self._lock:
            valid = [s for s in self._util_samples if s is not None]
        if not valid:
            return 0.0
        return max(valid)

    @property
    def peak_memory_local_mb(self) -> float:
        """Peak dedicated device memory in MB during polling period."""
        with self._lock:
            valid = [s for s in self._memory_local_bytes if s is not None]
        if not valid:
            return 0.0
        return max(valid) / (1024 * 1024)

    @property
    def peak_memory_shared_mb(self) -> float:
        """Peak shared system memory used by device in MB during polling period."""
        with self._lock:
            valid = [s for s in self._memory_shared_bytes if s is not None]
        if not valid:
            return 0.0
        return max(valid) / (1024 * 1024)

    @property
    def peak_memory_mb(self) -> float:
        """Peak device memory (local preferred, shared fallback) in MB."""
        local = self.peak_memory_local_mb
        return local if local > 0 else self.peak_memory_shared_mb

    @property
    def utilization_samples(self) -> list[float]:
        """All utilization % samples (time series copy)."""
        with self._lock:
            return self._util_samples.copy()

    @property
    def memory_samples_mb(self) -> list[float]:
        """All local memory samples in MB (time series copy)."""
        with self._lock:
            return [b / (1024 * 1024) if b is not None else 0.0 for b in self._memory_local_bytes]

    @property
    def is_active(self) -> bool:
        """Whether the poller is actively collecting data."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def utilization_sample_count(self) -> int:
        """Number of utilization samples collected."""
        with self._lock:
            return len(self._util_samples)

    @property
    def memory_sample_count(self) -> int:
        """Number of local memory samples collected."""
        with self._lock:
            return len(self._memory_local_bytes)

    @property
    def cpu_samples(self) -> list[float]:
        """All CPU utilization % samples (time series copy)."""
        with self._lock:
            return self._cpu_samples.copy()

    @property
    def mean_cpu_pct(self) -> float:
        """Mean CPU utilization % during polling period."""
        with self._lock:
            valid = [s for s in self._cpu_samples if s is not None]
        if not valid:
            return 0.0
        return statistics.mean(valid)

    @property
    def peak_cpu_pct(self) -> float:
        """Peak CPU utilization % during polling period."""
        with self._lock:
            valid = [s for s in self._cpu_samples if s is not None]
        if not valid:
            return 0.0
        return max(valid)

    @property
    def ram_used_mb(self) -> float:
        """Latest committed RAM in MB."""
        with self._lock:
            if not self._ram_used_bytes:
                return 0.0
            return self._ram_used_bytes[-1] / (1024 * 1024)

    @property
    def peak_ram_used_mb(self) -> float:
        """Peak committed RAM in MB during polling period."""
        with self._lock:
            valid = [s for s in self._ram_used_bytes if s is not None]
        if not valid:
            return 0.0
        return max(valid) / (1024 * 1024)

    @property
    def cpu_sample_count(self) -> int:
        """Number of CPU samples collected."""
        with self._lock:
            return len(self._cpu_samples)

    @property
    def running_time_delta_ns(self) -> int:
        """Total adapter (NPU/GPU) running time delta in nanoseconds."""
        if self._running_time_start_ns is None or self._running_time_end_ns is None:
            return 0
        return max(0, self._running_time_end_ns - self._running_time_start_ns)

    @staticmethod
    def is_npu_available() -> bool:
        """Whether PDH can discover an NPU on this system."""
        return discover_npu_luid() is not None

    @staticmethod
    def is_gpu_available() -> bool:
        """Whether PDH can discover a GPU on this system."""
        return discover_gpu_luid() is not None
