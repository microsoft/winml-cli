# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared fixtures for config tests.

Mocks resolve_device to avoid slow EP discovery in CI.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_resolve_device():
    """Mock resolve_device globally for all config tests.

    EP discovery via WinML can be very slow on CI runners without hardware.
    This fixture ensures config tests run fast by returning a fixed device.
    """
    with patch(
        "winml.modelkit.sysinfo.resolve_device",
        return_value=("npu", ["npu", "gpu", "cpu"]),
    ):
        yield
