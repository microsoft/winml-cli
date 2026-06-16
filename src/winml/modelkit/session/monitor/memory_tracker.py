# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Process memory helper for perf benchmarking."""

from __future__ import annotations

import logging
import os
import sys

import psutil


logger = logging.getLogger(__name__)


def get_rss_mb() -> float:
    """Return current RSS in MB for this process."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def get_vram_mb(adapter_luid: str | None) -> float:
    """Return current VRAM usage (local + shared) in MB via PDH.

    Returns 0.0 on non-Windows, if no adapter_luid is provided, or on failure.
    """
    if sys.platform != "win32" or not adapter_luid:
        return 0.0

    try:
        from ._pdh import PdhQuery

        pid = os.getpid()
        q = PdhQuery()
        q.open()
        q.add_counter(
            "local",
            rf"\GPU Process Memory(pid_{pid}_luid_{adapter_luid}_phys_0)\Local Usage",
        )
        q.add_counter(
            "shared",
            rf"\GPU Process Memory(pid_{pid}_luid_{adapter_luid}_phys_0)\Shared Usage",
        )
        # Memory counters are absolute (not rate-based), single collect suffices.
        values = q.collect()
        q.close()
        local = values.get("local") or 0
        shared = values.get("shared") or 0
        return (local + shared) / (1024 * 1024)
    except Exception:
        logger.debug("VRAM query failed", exc_info=True)
        return 0.0
