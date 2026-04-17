# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from winml.modelkit.telemetry.deviceid import _store


def test_get_telemetry_base_dir_is_callable():
    # Regression guard: helpers must not be @property on the module — a
    # property descriptor is not callable and silently breaks on non-Windows.
    assert callable(_store.get_telemetry_base_dir)
    result = _store.get_telemetry_base_dir()
    assert isinstance(result, Path)


def test_get_telemetry_base_dir_falls_back_when_home_unset(monkeypatch, tmp_path):
    # Simulate a container with HOME unset on Linux/macOS
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("USERPROFILE", raising=False)
    with patch.object(Path, "home", side_effect=RuntimeError("no home")):
        # On Linux/macOS the fallback is /var/tmp; on Windows the registry
        # helpers don't hit this path, so Windows skips.
        if os.name == "nt":
            pytest.skip("Windows uses registry; fallback path not exercised")
        result = _store.get_telemetry_base_dir()
        assert result == Path("/var/tmp") / "DeveloperTools" / ".modelkit"  # noqa: S108


@pytest.fixture
def isolated_store(monkeypatch, tmp_path):
    """Redirect _store to a temp directory so tests don't touch real state.

    On Windows we still use the registry but under a unique per-test subkey
    prefix; on non-Windows we point LOCALAPPDATA/HOME at tmp_path.
    """
    if os.name == "nt":
        # Use a per-test registry subkey to avoid test pollution.
        subkey = rf"SOFTWARE\Microsoft\DeveloperTools\.modelkit-test-{os.getpid()}"
        monkeypatch.setattr(_store, "_REGISTRY_KEY", subkey)
        yield
        import winreg

        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, subkey)
        except OSError:
            pass
    else:
        monkeypatch.setenv("HOME", str(tmp_path))
        yield


def test_write_then_read_returns_value(isolated_store):
    _store.write_key("deviceid", "abc123")
    assert _store.read_key("deviceid") == "abc123"


def test_read_absent_returns_none(isolated_store):
    assert _store.read_key("missing") is None


def test_delete_removes_value(isolated_store):
    _store.write_key("deviceid", "abc123")
    _store.delete_key("deviceid")
    assert _store.read_key("deviceid") is None


def test_delete_absent_is_silent(isolated_store):
    _store.delete_key("never-existed")  # must not raise


def test_overwrite_updates_value(isolated_store):
    _store.write_key("deviceid", "first")
    _store.write_key("deviceid", "second")
    assert _store.read_key("deviceid") == "second"
