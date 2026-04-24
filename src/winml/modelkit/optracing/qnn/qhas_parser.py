# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Backward-compatibility shim. Moved to session/monitor/qnn/qhas_parser.py."""

from __future__ import annotations

from ...session.monitor.qnn.qhas_parser import *  # noqa: F403
from ...session.monitor.qnn.qhas_parser import parse_qhas


__all__ = ["parse_qhas"]
