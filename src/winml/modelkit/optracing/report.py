# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Backward-compatibility shim.

Report helpers moved to ``winml.modelkit.session.monitor.report``.
"""

from __future__ import annotations

from ..session.monitor.report import (
    display_op_trace_report,
    write_op_trace_json,
)


__all__ = ["display_op_trace_report", "write_op_trace_json"]
