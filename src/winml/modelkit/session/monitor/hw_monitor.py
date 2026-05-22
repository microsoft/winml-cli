# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""HWMonitor - System-wide hardware monitor via Windows PDH counters.

Monitors CPU utilization, system RAM, and NPU/GPU utilization and memory
for any adapter that registers as a Windows GPU Engine device.
Works independently of the EPMonitor hierarchy.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from ._pdh import PdhPoller


if TYPE_CHECKING:
    from typing import Self


class HWMonitor:
    """System-wide hardware monitor via Windows PDH counters.

    Monitors CPU, RAM, and NPU/GPU utilization. Works for any NPU
    that registers as a Windows GPU Engine adapter with Compute-only
    engine types (Qualcomm, AMD, Intel).

    Independent of the EPMonitor hierarchy — provides system-wide
    resource visibility rather than EP-specific proof-of-execution.

    Example::

        with HWMonitor() as hw:
            # ... run inference ...
            pass

        print(hw.mean_utilization_pct)  # NPU %
        print(hw.mean_cpu_pct)          # CPU %
        print(hw.ram_used_mb)           # RAM MB
    """

    def __init__(self, poll_interval_ms: int = 200) -> None:
        """Initialize the monitor.

        Args:
            poll_interval_ms: PDH polling interval in milliseconds.
        """
        self._pdh = PdhPoller(poll_interval_ms)

    def __enter__(self) -> Self:
        """Start PDH background polling."""
        self._pdh.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Stop PDH polling and finalize metrics."""
        self._pdh.stop()

    # --- NPU metrics ---

    @property
    def mean_utilization_pct(self) -> float:
        """Mean NPU utilization % during monitoring period."""
        return self._pdh.mean_utilization_pct

    @property
    def peak_utilization_pct(self) -> float:
        """Peak NPU utilization % during monitoring period."""
        return self._pdh.peak_utilization_pct

    @property
    def peak_memory_mb(self) -> float:
        """Peak device memory (local preferred, shared fallback) in MB."""
        return self._pdh.peak_memory_mb

    @property
    def peak_memory_local_mb(self) -> float:
        """Peak dedicated device memory in MB."""
        return self._pdh.peak_memory_local_mb

    @property
    def peak_memory_shared_mb(self) -> float:
        """Peak shared system memory used by device in MB."""
        return self._pdh.peak_memory_shared_mb

    # --- CPU metrics ---

    @property
    def mean_cpu_pct(self) -> float:
        """Mean CPU utilization % during monitoring period."""
        return self._pdh.mean_cpu_pct

    @property
    def peak_cpu_pct(self) -> float:
        """Peak CPU utilization % during monitoring period."""
        return self._pdh.peak_cpu_pct

    # --- RAM metrics ---

    @property
    def ram_used_mb(self) -> float:
        """Latest committed RAM in MB."""
        return self._pdh.ram_used_mb

    @property
    def peak_ram_used_mb(self) -> float:
        """Peak committed RAM in MB during monitoring period."""
        return self._pdh.peak_ram_used_mb

    # --- Availability ---

    @classmethod
    def is_available(cls) -> bool:
        """Whether this monitor can work on the current system.

        Always available on Windows (CPU/RAM always monitorable).
        NPU metrics are added when an NPU adapter is discovered.
        """
        return sys.platform == "win32"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary of all collected metrics."""
        return {
            "monitor": "HWMonitor",
            "npu_luid": self._pdh.npu_luid,
            "cpu": {
                "mean_pct": round(self._pdh.mean_cpu_pct, 2),
                "peak_pct": round(self._pdh.peak_cpu_pct, 2),
                "sample_count": self._pdh.cpu_sample_count,
            },
            "ram": {
                "used_mb": round(self._pdh.ram_used_mb, 2),
                "peak_mb": round(self._pdh.peak_ram_used_mb, 2),
            },
            "npu": {
                "mean_pct": round(self._pdh.mean_utilization_pct, 2),
                "peak_pct": round(self._pdh.peak_utilization_pct, 2),
                "sample_count": self._pdh.utilization_sample_count,
            },
            "device_memory": {
                "local_peak_mb": round(self._pdh.peak_memory_local_mb, 2),
                "shared_peak_mb": round(self._pdh.peak_memory_shared_mb, 2),
            },
            "running_time_ns": self._pdh.running_time_delta_ns,
        }

    # --- Chart-compatible properties ---

    @property
    def utilization_samples(self) -> list[float]:
        """NPU utilization % samples (time series)."""
        return self._pdh.utilization_samples

    @property
    def cpu_samples(self) -> list[float]:
        """CPU utilization % samples (time series)."""
        return self._pdh.cpu_samples

    @property
    def memory_samples_mb(self) -> list[float]:
        """NPU memory samples in MB (time series)."""
        return self._pdh.memory_samples_mb
