# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared fixtures for compiler tests.

Mocks ``resolve_device`` and ``resolve_eps`` at their ``commands/compile.py``
import site so the compile CLI's auto-EP resolution is deterministic across
hosts (e.g., OpenVINO would otherwise out-rank QNN/DML/CPU on a dev box).
"""

from unittest.mock import MagicMock, patch

import pytest


_DEVICE_TO_EPS = {
    "npu": ["QNNExecutionProvider"],
    "gpu": ["DmlExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


@pytest.fixture(autouse=True)
def mock_compile_resolution():
    """Mock device + EP resolution for tests under ``tests/unit/compiler/``.

    ``WinMLEPRegistry.is_ep_available`` is also stubbed so the compile CLI's
    host-availability check passes for every EP — tests that exercise the
    negative path patch the registry singleton locally with a tighter mock.
    """
    mock_registry = MagicMock()
    mock_registry.is_ep_available.return_value = True

    with (
        patch(
            "winml.modelkit.commands.compile.resolve_device",
            side_effect=lambda device, ep=None: (
                "npu" if device == "auto" else device.lower(),
                ["npu", "gpu", "cpu"],
            ),
        ),
        patch(
            "winml.modelkit.commands.compile.resolve_eps",
            side_effect=lambda device: list(_DEVICE_TO_EPS.get(device, [])),
        ),
        patch(
            "winml.modelkit.session.ep_registry.WinMLEPRegistry.get_instance",
            return_value=mock_registry,
        ),
    ):
        yield
