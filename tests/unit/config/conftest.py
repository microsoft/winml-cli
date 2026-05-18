# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared fixtures for config tests.

Mocks device resolution to avoid slow EP discovery in CI.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_resolve_device():
    """Mock auto_detect_device / get_available_devices globally for all config tests.

    EP discovery via WinML can be very slow on CI runners without hardware.
    This fixture ensures config tests run fast by returning a fixed device.
    """
    with (
        patch(
            "winml.modelkit.session.auto_detect_device",
            return_value="npu",
        ),
        patch(
            "winml.modelkit.sysinfo.hardware.get_available_devices",
            return_value=["npu", "gpu", "cpu"],
        ),
    ):
        yield
