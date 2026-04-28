# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for the file-backed telemetry envelope cache."""

from pathlib import Path

import pytest

from winml.modelkit.telemetry._cache import (
    _CACHE_FILE_NAME,
    _cache_dir,
    _cache_file,
    _PersistentCache,
)


def test_cache_dir_honors_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELKIT_TELEMETRY_CACHE_DIR", str(tmp_path / "custom"))
    assert _cache_dir() == tmp_path / "custom"
    assert _cache_file() == tmp_path / "custom" / _CACHE_FILE_NAME


def test_cache_dir_default_uses_userprofile(monkeypatch, tmp_path):
    monkeypatch.delenv("MODELKIT_TELEMETRY_CACHE_DIR", raising=False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert _cache_dir() == tmp_path / ".winml" / "telemetry"


def test_cache_dir_returns_none_when_no_user_home(monkeypatch):
    """Regression: with no USERPROFILE / HOMEDRIVE+HOMEPATH and no
    override, _cache_dir must return None rather than silently
    resolving to a CWD-relative path."""
    monkeypatch.delenv("MODELKIT_TELEMETRY_CACHE_DIR", raising=False)
    monkeypatch.delenv("USERPROFILE", raising=False)
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    assert _cache_dir() is None
    from winml.modelkit.telemetry._cache import _cache_file

    assert _cache_file() is None


def test_cache_no_op_when_no_user_home(monkeypatch):
    """A cache built with a None path must silently no-op on every
    operation rather than crash."""
    monkeypatch.delenv("MODELKIT_TELEMETRY_CACHE_DIR", raising=False)
    monkeypatch.delenv("USERPROFILE", raising=False)
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    cache = _PersistentCache()
    assert cache._path is None
    cache.append([{"a": 1}])  # no-op, no raise
    assert cache.drain() == []
    cache.clear()  # no-op, no raise


@pytest.fixture
def cache(tmp_path):
    return _PersistentCache(path=tmp_path / "modelkit.cache")


def test_append_then_drain_roundtrip(cache):
    e1 = {"name": "ModelKitHeartbeat", "iKey": "o:test", "data": {}}
    e2 = {"name": "ModelKitAction", "iKey": "o:test", "data": {"a": 1}}
    cache.append([e1, e2])
    drained = cache.drain()
    assert drained == [e1, e2]
    # File is removed after drain.
    assert not cache._path.exists()


def test_drain_empty_when_file_missing(cache):
    assert cache.drain() == []


def test_append_creates_parent_dir(tmp_path):
    nested = tmp_path / "deep" / "nested" / "modelkit.cache"
    cache = _PersistentCache(path=nested)
    cache.append([{"a": 1}])
    assert nested.exists()
    assert cache.drain() == [{"a": 1}]


def test_drain_skips_malformed_lines(cache):
    """A single corrupt line must not poison the rest of the cache."""
    good = {"name": "ok", "iKey": "o:test"}
    cache.append([good])
    # Corrupt line injected directly.
    with cache._path.open("a", encoding="utf-8") as f:
        f.write("!not-base64!\n")
        f.write("\n")  # empty line
    cache.append([good])
    drained = cache.drain()
    assert drained == [good, good]


def test_clear_deletes_file(cache):
    cache.append([{"a": 1}])
    assert cache._path.exists()
    cache.clear()
    assert not cache._path.exists()


def test_clear_on_missing_file_is_noop(cache):
    assert not cache._path.exists()
    cache.clear()  # must not raise
    assert not cache._path.exists()


def test_append_empty_list_is_noop(cache):
    cache.append([])
    assert not cache._path.exists()


def test_append_swallows_io_errors(monkeypatch, cache):
    """A cache I/O failure must not propagate to the caller."""

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", boom)
    # Must not raise.
    cache.append([{"a": 1}])


def test_drain_swallows_io_errors(monkeypatch, cache):
    cache.append([{"a": 1}])

    def boom(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", boom)
    assert cache.drain() == []


def test_lock_path_is_sibling(cache):
    """The lockfile lives next to the cache file so the lock guards
    exactly the cache, not anything broader."""
    assert cache._lock_path.parent == cache._path.parent
    assert cache._lock_path.name.startswith(cache._path.name)
