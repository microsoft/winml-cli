# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test QNN EP detection for op-tracing."""

from winml.modelkit.optracing import is_qnn_profiling_available


def test_is_qnn_profiling_available_returns_bool():
    result = is_qnn_profiling_available()
    assert isinstance(result, bool)
