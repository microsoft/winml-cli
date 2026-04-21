# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import hashlib
import re

from winml.modelkit.telemetry.deviceid import IdStatus, _store, get_or_create_device_id


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def test_fresh_state_generates_new_device_id(isolated_store):
    device_id, status = get_or_create_device_id()
    assert status is IdStatus.NEW
    # SHA256 hex digest: 64 lowercase hex chars
    assert _HEX64_RE.match(device_id)


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
    assert _HEX64_RE.match(device_id)


def test_hash_stability_across_calls(isolated_store):
    # Same underlying UUID → same hex digest (regression guard against
    # accidental dependence on bytes representation).
    import uuid

    from winml.modelkit.telemetry.deviceid.deviceid import _hash_uuid

    u = uuid.UUID("12345678-1234-5678-1234-567812345678")
    assert _hash_uuid(u) == _hash_uuid(u)
    assert _hash_uuid(u) == hashlib.sha256(str(u).encode("utf-8")).hexdigest()
