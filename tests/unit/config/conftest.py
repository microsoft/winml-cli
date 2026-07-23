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

    Also pretends every catalog EP is registered so that tests asserting
    static-catalog deduction (e.g. ``device="npu"`` → ``compile_provider="qnn"``)
    remain host-independent after the registration-aware contract in
    docs/design/session/3_design_ep.md §6.4. Tests that want to exercise
    a specific subset of available EPs override this with a local
    ``patch.object(WinMLEPRegistry, "available_eps", return_value=...)``.
    """
    from winml.modelkit.ep_path import EPCatalog
    from winml.modelkit.session import EP_DEVICE_SPECS
    from winml.modelkit.session.ep_registry import WinMLEPRegistry

    all_eps = frozenset(s.ep for s in EP_DEVICE_SPECS)
    with (
        patch(
            "winml.modelkit.session.auto_detect_device",
            return_value="npu",
        ),
        patch(
            "winml.modelkit.sysinfo.hardware.get_available_devices",
            return_value=["npu", "gpu", "cpu"],
        ),
        patch.object(
            WinMLEPRegistry, "available_eps", return_value=all_eps,
        ),
        patch.object(
            EPCatalog, "is_compatible", return_value=True,
        ),
    ):
        yield
