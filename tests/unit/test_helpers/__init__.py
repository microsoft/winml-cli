# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared helpers for unit tests.

Guardrail:
        This package is helper-only; do not add ``test_*.py`` modules here.
"""

from .gguf_fakes import RUNTIME_CALLBACK, FakeLogApi, make_timing
from .stable_node_keys import stable_test_node_keys
from .task_fakes import (
    FakeTaskResult,
    FakeTaskSession,
    FakeTaskStream,
    TaskOptions11,
    TaskOptions12,
    hold_cyclic_interface,
)


__all__ = [
    "RUNTIME_CALLBACK",
    "FakeLogApi",
    "FakeTaskResult",
    "FakeTaskSession",
    "FakeTaskStream",
    "TaskOptions11",
    "TaskOptions12",
    "hold_cyclic_interface",
    "make_timing",
    "stable_test_node_keys",
]
