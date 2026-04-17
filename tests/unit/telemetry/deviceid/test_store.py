# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path

from winml.modelkit.telemetry.deviceid import _store


def test_get_telemetry_base_dir_is_callable():
    # Regression guard: helpers must not be @property on the module — a
    # property descriptor is not callable and silently breaks on import.
    assert callable(_store.get_telemetry_base_dir)
    result = _store.get_telemetry_base_dir()
    assert isinstance(result, Path)


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


def test_read_registry_ignores_non_string_values(isolated_store):
    """Regression guard: externally planted REG_DWORD or REG_BINARY must not
    be coerced to a surprise string."""
    import winreg

    # Write a REG_DWORD under our per-pid test key.
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, _store._REGISTRY_KEY) as key:
        winreg.SetValueEx(key, "deviceid", 0, winreg.REG_DWORD, 12345)
    # Reading via _store.read_key should treat this as absent (None).
    assert _store.read_key("deviceid") is None
