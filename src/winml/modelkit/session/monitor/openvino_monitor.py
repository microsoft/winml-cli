# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""OpenVinoMonitor - Placeholder for future Intel OpenVINO-specific NPU monitoring.

For real-time NPU utilization monitoring with OpenVINO EP, use HWMonitor
(universal PDH-based). This module is reserved for future Intel-specific
telemetry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .ep_monitor import WinMLEPMonitor


if TYPE_CHECKING:
    from typing import Self


class OpenVinoMonitor(WinMLEPMonitor):
    """Placeholder for future Intel OpenVINO-specific NPU monitoring.

    For real-time NPU utilization monitoring with OpenVINO EP,
    use ``HWMonitor`` (universal PDH-based).
    """

    def __enter__(self) -> Self:
        """No-op: no Intel-specific monitoring yet."""
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
        """No Intel-specific telemetry available yet."""
        return False

    def to_dict(self) -> dict[str, Any]:
        """Stub dict indicating not-implemented status."""
        return {"ep": "OpenVINO", "device": "NPU", "status": "not_implemented"}
