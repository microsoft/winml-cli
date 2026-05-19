# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared fixtures for config tests.

Mocks ``resolve_device`` and ``resolve_eps`` to avoid slow EP discovery
in CI and to keep ``compile_provider`` resolution deterministic regardless
of which EPs the test host has installed (e.g., OpenVINO would otherwise
out-rank QNN/DML/CPU under the dynamic resolution in ``resolve_precision``).
"""

from unittest.mock import patch

import pytest


_DEVICE_TO_EPS = {
    "npu": ["QNNExecutionProvider"],
    "gpu": ["DmlExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


@pytest.fixture(autouse=True)
def mock_resolve_device():
    """Mock device + EP resolution globally for all config tests.

    - ``resolve_device``: stubbed so EP discovery via WinML doesn't slow CI.
    - ``resolve_eps``: returns a canonical single-EP list per device so
      ``resolve_precision`` produces deterministic ``compile_provider``
      values (QNN for npu, DML for gpu, CPU→None for cpu) independent of
      what ORT/WinML advertises on the host.
    """
    with (
        patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("npu", ["npu", "gpu", "cpu"]),
        ),
        patch(
            "winml.modelkit.config.precision.resolve_eps",
            side_effect=lambda device: list(_DEVICE_TO_EPS.get(device, [])),
        ),
    ):
        yield
