# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""QNNMonitor - Placeholder for future Qualcomm QNN-specific NPU monitoring.

For real-time NPU utilization monitoring with QNN EP, use HWMonitor
(universal PDH-based). This module is reserved for future Qualcomm-specific
telemetry such as QAIRT profiling via qnn-profile-viewer.exe (device
execution time, queue wait, per-op traces).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .ep_monitor import EPMonitor


if TYPE_CHECKING:
    from typing_extensions import Self


class QNNMonitor(EPMonitor):
    """Placeholder for future Qualcomm QNN-specific NPU monitoring.

    For real-time NPU utilization monitoring with QNN EP,
    use ``HWMonitor`` (universal PDH-based).

    Future: Will wrap QAIRT profiling via ``qnn-profile-viewer.exe``
    for Qualcomm-specific metrics (device execution time, queue wait,
    per-op traces).
    """

    def __enter__(self) -> Self:
        """No-op: no Qualcomm-specific monitoring yet."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """No-op: no cleanup needed."""

    @classmethod
    def is_available(cls) -> bool:
        """No Qualcomm-specific telemetry available yet."""
        return False

    def to_dict(self) -> dict[str, Any]:
        """Stub dict indicating not-implemented status."""
        return {"ep": "QNN", "device": "NPU", "status": "not_implemented"}
