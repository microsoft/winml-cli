# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Backward-compatibility shim. Moved to session/monitor/qnn/csv_parser.py."""

from __future__ import annotations

from ...session.monitor.qnn.csv_parser import *  # noqa: F403
from ...session.monitor.qnn.csv_parser import parse_qnn_profiling_csv


__all__ = ["parse_qnn_profiling_csv"]
