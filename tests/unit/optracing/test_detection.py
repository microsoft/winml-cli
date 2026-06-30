# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test op-tracing support detection for the resolved EP/device/level."""

import pytest

from winml.modelkit.optracing import is_profiling_available


def test_is_profiling_available_returns_bool():
    result = is_profiling_available("QNNExecutionProvider", "npu", "basic")
    assert isinstance(result, bool)


def test_supported_combination():
    assert is_profiling_available("QNNExecutionProvider", "npu", "basic") is True


def test_supported_cpu_combination():
    assert is_profiling_available("CPUExecutionProvider", "cpu", "basic") is True


def test_cpu_alias_is_normalized():
    assert is_profiling_available("cpu", "cpu", "basic") is True


def test_qnn_alias_is_normalized():
    # The benchmark may carry the user's EP alias verbatim; it must still match.
    assert is_profiling_available("qnn", "npu", "basic") is True


def test_device_is_case_insensitive():
    assert is_profiling_available("QNNExecutionProvider", "NPU", "basic") is True


def test_level_is_case_insensitive():
    assert is_profiling_available("QNNExecutionProvider", "npu", "BASIC") is True


@pytest.mark.parametrize(
    ("ep", "device", "level"),
    [
        ("CPUExecutionProvider", "npu", "basic"),  # CPU EP only supported on cpu device
        ("CPUExecutionProvider", "cpu", "detail"),  # unsupported level for CPU
        ("QNNExecutionProvider", "gpu", "basic"),  # wrong device
        ("QNNExecutionProvider", "npu", "detail"),  # unsupported level
        (None, "npu", "basic"),  # no EP
        ("QNNExecutionProvider", None, "basic"),  # no device
        ("QNNExecutionProvider", "npu", None),  # no level
    ],
)
def test_unsupported_combinations(ep, device, level):
    assert is_profiling_available(ep, device, level) is False
