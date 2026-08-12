# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pin the default top_n value to match the mockup spec (top-K = 5).

The mockup canon (`docs/design/perf/console_mockup.py`) declares
``OP_TRACING_TOP_K_DEFAULT = 5``. The production renderer must agree —
otherwise the headline ``wmk perf -m <model> --op-tracing basic`` invocation
shows 15 rows (the prior default) when the spec wants 5.
"""

from __future__ import annotations

import inspect

from winml.modelkit.session.monitor.report import display_op_trace_report


def test_default_top_n_is_five():
    """display_op_trace_report's top_n default must be 5 (mockup spec)."""
    sig = inspect.signature(display_op_trace_report)
    assert sig.parameters["top_n"].default == 5
