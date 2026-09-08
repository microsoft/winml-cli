# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""GPU ordering shared by runtime selection and native system inventory."""

from __future__ import annotations


def gpu_priority_key(
    luid: str | None, high_performance_index: object = None
) -> tuple[bool, int, bool, str]:
    """Sort by Windows high-performance preference, then adapter LUID.

    ORT publishes the numeric DXGI high-performance enumeration position as
    ``DxgiHighPerformanceIndex`` in hardware metadata (zero is best). Missing
    or invalid positions sort after ranked GPUs. Canonical, fixed-width PDH
    LUIDs break ties independently of enumeration order; absent LUIDs sort last.
    """
    index = None
    if isinstance(high_performance_index, (str, int)) and not isinstance(
        high_performance_index, bool
    ):
        try:
            parsed = int(high_performance_index)
            if parsed >= 0:
                index = parsed
        except ValueError:
            # Invalid optional ranking metadata leaves this GPU unranked; use LUID ordering.
            pass
    return (index is None, index if index is not None else 0, not luid, (luid or "").casefold())
