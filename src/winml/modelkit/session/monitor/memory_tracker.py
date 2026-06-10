# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""Process memory tracking for perf benchmarking.

Provides lightweight, zero-dependency process memory snapshots via Windows
``GetProcessMemoryInfo`` (ctypes). Used by ``winml perf --memory`` to measure
memory consumption at each benchmark phase.

For device (NPU/GPU) memory, a single-shot PDH query is used to read
``\GPU Process Memory\Local Usage`` and ``\GPU Process Memory\Shared Usage``.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


if sys.platform == "win32":
    import ctypes.wintypes as wintypes


logger = logging.getLogger(__name__)


# =============================================================================
# Process Memory via GetProcessMemoryInfo (Windows)
# =============================================================================

_MB = 1024 * 1024


if sys.platform == "win32":

    class _ProcessMemoryCountersEx(ctypes.Structure):
        """PROCESS_MEMORY_COUNTERS_EX structure from psapi.h."""

        _fields_: ClassVar = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]


def _get_process_memory() -> tuple[float, float, float, float]:
    """Get current process memory via K32GetProcessMemoryInfo.

    Uses kernel32.K32GetProcessMemoryInfo (Windows 7+) which supports
    PROCESS_MEMORY_COUNTERS_EX natively.

    Returns:
        (working_set_mb, peak_working_set_mb, private_bytes_mb, peak_private_bytes_mb)
    """
    if sys.platform != "win32":
        return _get_process_memory_linux()

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel32.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]

    handle = kernel32.GetCurrentProcess()
    counters = _ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)

    success = kernel32.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not success:
        err = ctypes.get_last_error()
        logger.warning("K32GetProcessMemoryInfo failed (error=%d), returning zeros", err)
        return (0.0, 0.0, 0.0, 0.0)

    return (
        counters.WorkingSetSize / _MB,
        counters.PeakWorkingSetSize / _MB,
        counters.PrivateUsage / _MB,
        counters.PeakPagefileUsage / _MB,
    )


def _get_process_memory_linux() -> tuple[float, float, float, float]:
    """Fallback for Linux: read /proc/self/status."""
    try:
        with Path("/proc/self/status").open() as f:
            content = f.read()

        values: dict[str, float] = {}
        for line in content.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].rstrip(":") in (
                "VmRSS",
                "VmPeak",
                "VmSize",
            ):
                values[parts[0].rstrip(":")] = float(parts[1]) / 1024  # kB -> MB

        rss = values.get("VmRSS", 0.0)
        peak = values.get("VmPeak", 0.0)
        return (rss, peak, rss, peak)
    except OSError:
        return (0.0, 0.0, 0.0, 0.0)


# =============================================================================
# Device Memory via single-shot PDH query
# =============================================================================


def _get_device_memory_mb(luid: str | None) -> tuple[float, float]:
    """Single-shot PDH query for device memory (local, shared) in MB.

    Args:
        luid: Adapter LUID string. If None, returns (0, 0).

    Returns:
        (local_mb, shared_mb)
    """
    if luid is None or sys.platform != "win32":
        return (0.0, 0.0)

    try:
        from ._pdh import PdhQuery

        pid = os.getpid()
        query = PdhQuery()
        query.open()

        local_ok = query.add_counter(
            "local",
            rf"\GPU Process Memory(pid_{pid}_luid_{luid}_phys_0)\Local Usage",
            fmt="large",
        )
        shared_ok = query.add_counter(
            "shared",
            rf"\GPU Process Memory(pid_{pid}_luid_{luid}_phys_0)\Shared Usage",
            fmt="large",
        )

        if not local_ok and not shared_ok:
            query.close()
            return (0.0, 0.0)

        # PDH large counters don't need priming (not rate-based)
        query.prime()
        values = query.collect()
        query.close()

        local_bytes = values.get("local") or 0
        shared_bytes = values.get("shared") or 0
        return (local_bytes / _MB, shared_bytes / _MB)
    except Exception:
        logger.debug("Device memory query failed", exc_info=True)
        return (0.0, 0.0)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MemorySnapshot:
    """A point-in-time memory measurement."""

    working_set_mb: float = 0.0
    peak_working_set_mb: float = 0.0
    private_bytes_mb: float = 0.0
    peak_private_bytes_mb: float = 0.0
    device_local_mb: float = 0.0
    device_shared_mb: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """JSON-serializable dictionary."""
        return {
            "working_set_mb": round(self.working_set_mb, 2),
            "peak_working_set_mb": round(self.peak_working_set_mb, 2),
            "private_bytes_mb": round(self.private_bytes_mb, 2),
            "peak_private_bytes_mb": round(self.peak_private_bytes_mb, 2),
            "device_local_mb": round(self.device_local_mb, 2),
            "device_shared_mb": round(self.device_shared_mb, 2),
        }


@dataclass
class MemoryProfile:
    """Memory measurements across benchmark phases."""

    baseline: MemorySnapshot
    post_load: MemorySnapshot
    post_compile: MemorySnapshot
    post_inference: MemorySnapshot

    @property
    def load_delta_mb(self) -> float:
        """Working set increase from model loading."""
        return self.post_load.working_set_mb - self.baseline.working_set_mb

    @property
    def compile_delta_mb(self) -> float:
        """Working set increase from session compilation."""
        return self.post_compile.working_set_mb - self.post_load.working_set_mb

    @property
    def inference_delta_mb(self) -> float:
        """Working set increase during inference."""
        return self.post_inference.working_set_mb - self.post_compile.working_set_mb

    @property
    def total_delta_mb(self) -> float:
        """Total working set increase from baseline."""
        return self.post_inference.working_set_mb - self.baseline.working_set_mb

    @property
    def peak_working_set_mb(self) -> float:
        """Peak working set across all phases (from OS counter)."""
        return self.post_inference.peak_working_set_mb

    @property
    def peak_device_local_mb(self) -> float:
        """Peak device local memory across all phases."""
        return max(
            self.baseline.device_local_mb,
            self.post_load.device_local_mb,
            self.post_compile.device_local_mb,
            self.post_inference.device_local_mb,
        )

    @property
    def peak_device_shared_mb(self) -> float:
        """Peak device shared memory across all phases."""
        return max(
            self.baseline.device_shared_mb,
            self.post_load.device_shared_mb,
            self.post_compile.device_shared_mb,
            self.post_inference.device_shared_mb,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dictionary."""
        return {
            "baseline": self.baseline.to_dict(),
            "post_load": self.post_load.to_dict(),
            "post_compile": self.post_compile.to_dict(),
            "post_inference": self.post_inference.to_dict(),
            "peak_working_set_mb": round(self.peak_working_set_mb, 2),
            "peak_device_local_mb": round(self.peak_device_local_mb, 2),
            "peak_device_shared_mb": round(self.peak_device_shared_mb, 2),
            "total_delta_working_set_mb": round(self.total_delta_mb, 2),
        }


# =============================================================================
# MemoryTracker
# =============================================================================


class MemoryTracker:
    """Lightweight memory tracker that takes snapshots at phase boundaries.

    Usage::

        tracker = MemoryTracker()
        tracker.snapshot_baseline()
        # ... load model ...
        tracker.snapshot_post_load()
        # ... compile ...
        tracker.snapshot_post_compile(adapter_luid="0x...")
        # ... run benchmark ...
        tracker.snapshot_post_inference(adapter_luid="0x...")
        profile = tracker.profile()
    """

    def __init__(self) -> None:
        self._baseline: MemorySnapshot | None = None
        self._post_load: MemorySnapshot | None = None
        self._post_compile: MemorySnapshot | None = None
        self._post_inference: MemorySnapshot | None = None

    def _take_snapshot(self, adapter_luid: str | None = None) -> MemorySnapshot:
        """Take a point-in-time memory snapshot."""
        ws, peak_ws, priv, peak_priv = _get_process_memory()
        dev_local, dev_shared = _get_device_memory_mb(adapter_luid)
        return MemorySnapshot(
            working_set_mb=ws,
            peak_working_set_mb=peak_ws,
            private_bytes_mb=priv,
            peak_private_bytes_mb=peak_priv,
            device_local_mb=dev_local,
            device_shared_mb=dev_shared,
        )

    def snapshot_baseline(self) -> None:
        """Capture baseline memory before model loading."""
        self._baseline = self._take_snapshot()

    def snapshot_post_load(self) -> None:
        """Capture memory after model loading."""
        self._post_load = self._take_snapshot()

    def snapshot_post_compile(self, adapter_luid: str | None = None) -> None:
        """Capture memory after session compilation.

        Args:
            adapter_luid: Adapter LUID for device memory query.
                Available after compile resolves the EP.
        """
        self._post_compile = self._take_snapshot(adapter_luid)

    def snapshot_post_inference(self, adapter_luid: str | None = None) -> None:
        """Capture memory after benchmark completion.

        Args:
            adapter_luid: Adapter LUID for device memory query.
        """
        self._post_inference = self._take_snapshot(adapter_luid)

    def profile(self) -> MemoryProfile | None:
        """Build a complete MemoryProfile from collected snapshots.

        Returns None if any phase snapshot is missing.
        """
        if any(
            s is None
            for s in (self._baseline, self._post_load, self._post_compile, self._post_inference)
        ):
            logger.warning("Incomplete memory snapshots, cannot build profile")
            return None

        assert self._baseline is not None
        assert self._post_load is not None
        assert self._post_compile is not None
        assert self._post_inference is not None

        return MemoryProfile(
            baseline=self._baseline,
            post_load=self._post_load,
            post_compile=self._post_compile,
            post_inference=self._post_inference,
        )
