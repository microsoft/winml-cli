# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Backward-compatibility shim.

``OpTraceResult`` and ``OperatorMetrics`` moved to
``winml.modelkit.session.monitor.op_metrics``. This shim keeps old imports
working during the op-tracing refactor; removed once all callers are updated.
"""

from __future__ import annotations

from ..session.monitor.op_metrics import OperatorMetrics, OpTraceResult


__all__ = ["OpTraceResult", "OperatorMetrics"]
