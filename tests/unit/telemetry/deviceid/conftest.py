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
def isolated_store(monkeypatch, tmp_path):
    """Redirect _store to per-test scratch space.

    On Windows: uses a per-pid registry subkey to avoid test pollution.
    On other platforms: redirects HOME to a temp dir.
    """
    if os.name == "nt":
        subkey = rf"SOFTWARE\Microsoft\DeveloperTools\.modelkit-test-{os.getpid()}"
        monkeypatch.setattr(_store, "_REGISTRY_KEY", subkey)
        yield
        import winreg

        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, subkey)
        except OSError:
            # Best-effort fixture cleanup; a teardown failure must not
            # mask the test result.
            return
    else:
        monkeypatch.setenv("HOME", str(tmp_path))
        yield
