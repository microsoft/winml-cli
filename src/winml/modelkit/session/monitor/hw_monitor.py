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
    from typing_extensions import Self


class HWMonitor:
    """System-wide hardware monitor via Windows PDH counters.

    Monitors CPU, RAM, and the requested adapter's (NPU or GPU) utilization.
    Works for any NPU that registers as a Windows GPU Engine adapter with
    Compute-only engine types (Qualcomm, AMD, Intel) and for GPUs with a
    3D engine.

    Independent of the EPMonitor hierarchy — provides system-wide
    resource visibility rather than EP-specific proof-of-execution.

    Example::

        with HWMonitor(device="gpu") as hw:
            # ... run inference ...
            pass

        print(hw.mean_utilization_pct)  # GPU %
        print(hw.mean_cpu_pct)          # CPU %
        print(hw.ram_used_mb)           # RAM MB
    """

    def __init__(
        self,
        poll_interval_ms: int = 200,
        device: str = "auto",
        ep_name: str | None = None,
    ) -> None:
        """Initialize the monitor.

        Args:
            poll_interval_ms: PDH polling interval in milliseconds.
            device: Which adapter to monitor. ``"npu"`` polls the NPU
                (Compute engine), ``"gpu"`` polls the GPU (3D engine),
                ``"cpu"`` skips adapter polling (CPU/RAM only), and
                ``"auto"`` probes NPU first then GPU.
            ep_name: Full ORT EP name (e.g. ``"QNNExecutionProvider"``).
                When provided, the monitor uses ``OrtHardwareDevice``
                metadata to resolve the same LUID the inference session
                will bind to — useful on hybrid systems where multiple
                adapters share a device type.
        """
        self._pdh = PdhPoller(poll_interval_ms, device=device, ep_name=ep_name)

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

    # --- Adapter (NPU/GPU) metrics ---

    @property
    def device_kind(self) -> str | None:
        """Resolved adapter kind: ``"npu"``, ``"gpu"``, or None."""
        return self._pdh.device_kind

    @property
    def mean_utilization_pct(self) -> float:
        """Mean adapter (NPU/GPU) utilization % during monitoring period."""
        return self._pdh.mean_utilization_pct

    @property
    def peak_utilization_pct(self) -> float:
        """Peak adapter (NPU/GPU) utilization % during monitoring period."""
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
        """JSON-serializable summary of all collected metrics.

        Emits an adapter block keyed by the resolved device kind (``"npu"``
        or ``"gpu"``). The ``"npu"`` key is always present for backward
        compatibility — when only a GPU is being monitored, it contains
        zeros while the live data is in the ``"gpu"`` block.
        """
        kind = self._pdh.device_kind  # "npu", "gpu", or None
        adapter_block = {
            "mean_pct": round(self._pdh.mean_utilization_pct, 2),
            "peak_pct": round(self._pdh.peak_utilization_pct, 2),
            "sample_count": self._pdh.utilization_sample_count,
        }
        zero_block = {"mean_pct": 0.0, "peak_pct": 0.0, "sample_count": 0}
        result: dict[str, Any] = {
            "monitor": "HWMonitor",
            "device_kind": kind,
            "adapter_luid": self._pdh.adapter_luid,
            # Back-compat alias for callers that read "npu_luid" directly.
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
            # "npu" key is always present; populated when device_kind=="npu".
            "npu": adapter_block if kind == "npu" else zero_block,
            "device_memory": {
                "local_peak_mb": round(self._pdh.peak_memory_local_mb, 2),
                "shared_peak_mb": round(self._pdh.peak_memory_shared_mb, 2),
            },
            "running_time_ns": self._pdh.running_time_delta_ns,
        }
        if kind == "gpu":
            result["gpu"] = adapter_block
        return result

    # --- Chart-compatible properties ---

    @property
    def utilization_samples(self) -> list[float]:
        """Adapter (NPU/GPU) utilization % samples (time series)."""
        return self._pdh.utilization_samples

    @property
    def cpu_samples(self) -> list[float]:
        """CPU utilization % samples (time series)."""
        return self._pdh.cpu_samples

    @property
    def memory_samples_mb(self) -> list[float]:
        """Adapter memory samples in MB (time series)."""
        return self._pdh.memory_samples_mb
