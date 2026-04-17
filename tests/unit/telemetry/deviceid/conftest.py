# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Shared fixtures for deviceid unit tests."""

from __future__ import annotations

import os

import pytest

from winml.modelkit.telemetry.deviceid import _store


@pytest.fixture
def isolated_store(monkeypatch):
    """Redirect _store to a per-pid registry subkey so tests don't touch
    real state. Best-effort cleanup on teardown."""
    subkey = rf"SOFTWARE\Microsoft\DeveloperTools\.modelkit-test-{os.getpid()}"
    monkeypatch.setattr(_store, "_REGISTRY_KEY", subkey)
    yield
    import winreg

    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, subkey)
    except OSError:
        # Best-effort fixture cleanup; a teardown failure must not mask
        # the test result.
        return
