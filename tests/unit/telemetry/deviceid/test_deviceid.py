# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import re

from winml.modelkit.telemetry.deviceid import IdStatus, _store, get_or_create_device_id


# CS 4.0 ext.device.localId format: <scope>:<canonical-uuid>. We use the 'r'
# (random) scope with a lowercase hyphenated UUID4.
_RANDOM_LOCAL_ID_RE = re.compile(
    r"^r:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def test_fresh_state_generates_new_device_id(isolated_store):
    device_id, status = get_or_create_device_id()
    assert status is IdStatus.NEW
    assert _RANDOM_LOCAL_ID_RE.match(device_id)


def test_subsequent_call_returns_existing(isolated_store):
    first_id, _ = get_or_create_device_id()
    second_id, status = get_or_create_device_id()
    assert status is IdStatus.EXISTING
    assert second_id == first_id


def test_status_is_string_compatible_for_otel_resource(isolated_store):
    # IdStatus subclasses str so callers can stuff it into an OTel Resource
    # or the CS 4.0 ext.device.authId slot without .value conversion.
    _, status = get_or_create_device_id()
    assert isinstance(status, str)
    assert status == "NEW"


def test_storage_write_failure_returns_failed(isolated_store, monkeypatch):
    def boom(name, value):
        raise OSError("disk full")

    monkeypatch.setattr(_store, "write_key", boom)
    device_id, status = get_or_create_device_id()
    assert status is IdStatus.FAILED
    assert device_id == ""


def test_storage_read_failure_falls_through_to_new(isolated_store, monkeypatch):
    def boom(name):
        raise OSError("registry down")

    monkeypatch.setattr(_store, "read_key", boom)
    # Writing still works; we should generate a NEW id rather than FAILED
    device_id, status = get_or_create_device_id()
    assert status is IdStatus.NEW
    assert _RANDOM_LOCAL_ID_RE.match(device_id)


def test_legacy_deviceid_key_is_ignored(isolated_store):
    # Releases <= 0.0.4 wrote a 64-char SHA256 hex digest under the
    # legacy "deviceid" key. The new key name ("device_id") sidesteps it
    # entirely — first read sees nothing, so we generate a fresh value
    # in the CS 4.0 format. The legacy REG_SZ stays as a harmless orphan.
    legacy_hex = "a" * 64
    _store.write_key("deviceid", legacy_hex)
    device_id, status = get_or_create_device_id()
    assert status is IdStatus.NEW
    assert _RANDOM_LOCAL_ID_RE.match(device_id)
    assert device_id != legacy_hex
    # Legacy key untouched.
    assert _store.read_key("deviceid") == legacy_hex
    # Replacement is persisted so the next call returns EXISTING.
    second_id, second_status = get_or_create_device_id()
    assert second_status is IdStatus.EXISTING
    assert second_id == device_id
