# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import base64

import pytest

from winml.modelkit.telemetry.utils import (
    _decode_cache_entry,
    _encode_cache_entry,
    _ExclusiveFileLock,
)


def test_encode_decode_roundtrip():
    original = {"name": "WinMLCLIAction", "data": {"success": True}, "n": 42}
    encoded = _encode_cache_entry(original)
    assert isinstance(encoded, str)  # storable as a single line
    decoded = _decode_cache_entry(encoded)
    assert decoded == original


def test_encode_then_decode_unicode():
    original = {"note": "café - λ"}
    assert _decode_cache_entry(_encode_cache_entry(original)) == original


def test_decode_invalid_input_returns_none():
    # Cache read path must be robust to malformed entries (base64 garbage,
    # truncated lines). Loss of one entry is acceptable; crashing the
    # process is not.
    assert _decode_cache_entry("not-base64!!!") is None
    assert _decode_cache_entry("") is None


def test_decode_json_parse_error_returns_none():
    # Regression guard: the decode path must call json.loads AFTER base64
    # decode, not treat the decoded bytes as the final value.
    bogus = base64.b64encode(b"this is not json").decode("ascii")
    assert _decode_cache_entry(bogus) is None


def test_exclusive_file_lock_acquires_and_releases(tmp_path):
    lock_file = tmp_path / "telemetry.lock"
    with _ExclusiveFileLock(lock_file):
        assert lock_file.exists()
    # After exit the lock is released - re-acquiring proves it.
    with _ExclusiveFileLock(lock_file):
        pass


def test_exclusive_file_lock_closes_handle_on_lock_failure(tmp_path, monkeypatch):
    """If lock acquisition fails, the underlying file handle must still be
    closed (otherwise Windows would refuse to delete the lockfile)."""
    lock_file = tmp_path / "telemetry.lock"

    import msvcrt

    def failing_locking(fd, mode, nbytes):
        raise OSError("simulated lock failure")

    monkeypatch.setattr(msvcrt, "locking", failing_locking)

    with pytest.raises(OSError), _ExclusiveFileLock(lock_file):
        pass  # pragma: no cover - should not be entered

    # File handle not leaked - unlink would fail with PermissionError on leak.
    try:
        lock_file.unlink()
    except PermissionError:  # pragma: no cover - would indicate a leak
        pytest.fail("file handle was leaked on lock failure")
