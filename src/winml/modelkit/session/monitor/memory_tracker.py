# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Process memory helper for perf benchmarking."""

from __future__ import annotations

import os

import psutil


def get_rss_mb() -> float:
    """Return current RSS in MB for this process."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
