# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Process memory tracking for perf benchmarking.

Measures RSS (Resident Set Size) at benchmark phase boundaries to compute
memory deltas for model loading, compilation, and inference. Uses the same
approach as standalone memory measurement scripts: psutil for process memory
with a ctypes fallback on Windows.

The tracker excludes one-time EP initialization costs (DLL loading) by
taking the baseline *after* the EP registry is warmed up.
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
# Memory Measurement
# =============================================================================

_MB = 1024 * 1024


def _get_memory_mb() -> dict[str, float]:
    """Return current RSS and peak working set in MB for this process.

    Tries psutil first (cross-platform), falls back to ctypes on Windows
    or /proc/self/status on Linux.
    """
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        info = proc.memory_info()
        return {
            "rss_mb": info.rss / _MB,
            "peak_wset_mb": getattr(info, "peak_wset", info.rss) / _MB,
        }
    except ImportError:
        pass

    # Fallback: platform-specific
    if sys.platform == "win32":
        return _get_memory_mb_win32()
    return _get_memory_mb_linux()


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


def _get_memory_mb_win32() -> dict[str, float]:
    """Fallback for Windows: ctypes K32GetProcessMemoryInfo."""
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
        logger.warning("K32GetProcessMemoryInfo failed, returning zeros")
        return {"rss_mb": 0.0, "peak_wset_mb": 0.0}

    return {
        "rss_mb": counters.WorkingSetSize / _MB,
        "peak_wset_mb": counters.PeakWorkingSetSize / _MB,
    }


def _get_memory_mb_linux() -> dict[str, float]:
    """Fallback for Linux: read /proc/self/status."""
    try:
        with Path("/proc/self/status").open() as f:
            content = f.read()

        values: dict[str, float] = {}
        for line in content.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].rstrip(":") in ("VmRSS", "VmPeak"):
                values[parts[0].rstrip(":")] = float(parts[1]) / 1024  # kB -> MB

        rss = values.get("VmRSS", 0.0)
        peak = values.get("VmPeak", 0.0)
        return {"rss_mb": rss, "peak_wset_mb": peak}
    except OSError:
        return {"rss_mb": 0.0, "peak_wset_mb": 0.0}


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

    rss_mb: float = 0.0
    peak_wset_mb: float = 0.0
    device_local_mb: float = 0.0
    device_shared_mb: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """JSON-serializable dictionary."""
        return {
            "rss_mb": round(self.rss_mb, 2),
            "peak_wset_mb": round(self.peak_wset_mb, 2),
            "device_local_mb": round(self.device_local_mb, 2),
            "device_shared_mb": round(self.device_shared_mb, 2),
        }


@dataclass
class MemoryProfile:
    """Memory measurements across benchmark phases.

    Mirrors the structure used in standalone memory measurement scripts:
    baseline → after_compile → after_warmup, with computed deltas.
    """

    baseline: MemorySnapshot
    post_compile: MemorySnapshot
    post_inference: MemorySnapshot

    @property
    def model_load_delta_mb(self) -> float:
        """RSS increase from model loading + compilation."""
        return self.post_compile.rss_mb - self.baseline.rss_mb

    @property
    def inference_alloc_delta_mb(self) -> float:
        """RSS increase from inference (warmup + benchmark)."""
        return self.post_inference.rss_mb - self.post_compile.rss_mb

    @property
    def total_delta_mb(self) -> float:
        """Total RSS increase from baseline."""
        return self.post_inference.rss_mb - self.baseline.rss_mb

    @property
    def peak_wset_mb(self) -> float:
        """Peak working set (from OS counter at end of benchmark)."""
        return self.post_inference.peak_wset_mb

    @property
    def peak_delta_mb(self) -> float:
        """Peak working set increase from baseline."""
        return self.post_inference.peak_wset_mb - self.baseline.peak_wset_mb

    @property
    def peak_device_local_mb(self) -> float:
        """Peak device local memory across all phases."""
        return max(
            self.baseline.device_local_mb,
            self.post_compile.device_local_mb,
            self.post_inference.device_local_mb,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dictionary."""
        return {
            "rss_baseline_mb": round(self.baseline.rss_mb, 2),
            "rss_after_compile_mb": round(self.post_compile.rss_mb, 2),
            "rss_after_inference_mb": round(self.post_inference.rss_mb, 2),
            "model_load_delta_mb": round(self.model_load_delta_mb, 2),
            "inference_alloc_delta_mb": round(self.inference_alloc_delta_mb, 2),
            "total_delta_mb": round(self.total_delta_mb, 2),
            "peak_working_set_mb": round(self.peak_wset_mb, 2),
            "peak_delta_mb": round(self.peak_delta_mb, 2),
            "device_local_mb": round(self.peak_device_local_mb, 2),
        }


# =============================================================================
# MemoryTracker
# =============================================================================


class MemoryTracker:
    """Lightweight memory tracker that takes snapshots at phase boundaries.

    Follows the same measurement approach as standalone memory scripts:
    - Baseline is taken *after* EP initialization (excludes DLL loading)
    - Snapshots after compile and after inference warmup
    - Deltas show model load cost and inference allocation cost

    Usage::

        tracker = MemoryTracker()
        tracker.snapshot_baseline()
        # ... load model + compile ...
        tracker.snapshot_post_compile(adapter_luid="0x...")
        # ... run benchmark ...
        tracker.snapshot_post_inference(adapter_luid="0x...")
        profile = tracker.profile()
    """

    def __init__(self) -> None:
        self._baseline: MemorySnapshot | None = None
        self._post_compile: MemorySnapshot | None = None
        self._post_inference: MemorySnapshot | None = None

    def _take_snapshot(self, adapter_luid: str | None = None) -> MemorySnapshot:
        """Take a point-in-time memory snapshot."""
        mem = _get_memory_mb()
        dev_local, dev_shared = _get_device_memory_mb(adapter_luid)
        return MemorySnapshot(
            rss_mb=mem["rss_mb"],
            peak_wset_mb=mem["peak_wset_mb"],
            device_local_mb=dev_local,
            device_shared_mb=dev_shared,
        )

    def snapshot_baseline(self) -> None:
        """Capture baseline memory.

        Should be called *after* EP registry initialization so that one-time
        DLL loading costs are excluded from model measurements.
        """
        self._baseline = self._take_snapshot()

    def snapshot_post_compile(self, adapter_luid: str | None = None) -> None:
        """Capture memory after model load + session compilation.

        Args:
            adapter_luid: Adapter LUID for device memory query.
                Available after compile resolves the EP.
        """
        self._post_compile = self._take_snapshot(adapter_luid)

    def snapshot_post_inference(self, adapter_luid: str | None = None) -> None:
        """Capture memory after inference (warmup + benchmark).

        Args:
            adapter_luid: Adapter LUID for device memory query.
        """
        self._post_inference = self._take_snapshot(adapter_luid)

    def profile(self) -> MemoryProfile | None:
        """Build a complete MemoryProfile from collected snapshots.

        Returns None if any phase snapshot is missing.
        """
        if any(s is None for s in (self._baseline, self._post_compile, self._post_inference)):
            logger.warning("Incomplete memory snapshots, cannot build profile")
            return None

        assert self._baseline is not None
        assert self._post_compile is not None
        assert self._post_inference is not None

        return MemoryProfile(
            baseline=self._baseline,
            post_compile=self._post_compile,
            post_inference=self._post_inference,
        )
