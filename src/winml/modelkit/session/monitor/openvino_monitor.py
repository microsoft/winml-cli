# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Placeholder OpenVINO monitor retained for public compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .ep_monitor import EPMonitor


if TYPE_CHECKING:
    from typing import Self


class OpenVinoMonitor(EPMonitor):
    """Placeholder for future Intel OpenVINO-specific NPU monitoring."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        return None

    @classmethod
    def is_available(cls) -> bool:
        """Return ``False`` until a real OpenVINO-specific monitor exists."""
        return False

    def to_dict(self) -> dict[str, Any]:
        """Return the legacy placeholder payload expected by downstream callers."""
        return {"ep": "OpenVINO", "device": "NPU", "status": "not_implemented"}
