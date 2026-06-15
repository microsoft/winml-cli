# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Process memory helpers for perf benchmarking."""

from __future__ import annotations

import logging
import os
import sys

import psutil


logger = logging.getLogger(__name__)

_MB = 1024 * 1024


def get_rss_mb() -> float:
    """Return current RSS in MB for this process."""
    return psutil.Process(os.getpid()).memory_info().rss / _MB


def get_device_memory_mb(luid: str | None) -> float:
    """Single-shot PDH query for device local memory in MB."""
    if luid is None or sys.platform != "win32":
        return 0.0
    try:
        from ._pdh import PdhQuery

        pid = os.getpid()
        query = PdhQuery()
        query.open()
        ok = query.add_counter(
            "local",
            rf"\GPU Process Memory(pid_{pid}_luid_{luid}_phys_0)\Local Usage",
            fmt="large",
        )
        if not ok:
            query.close()
            return 0.0
        query.prime()
        values = query.collect()
        query.close()
        return (values.get("local") or 0) / _MB
    except Exception:
        logger.debug("Device memory query failed", exc_info=True)
        return 0.0
