# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

r"""File-backed cache for telemetry envelopes that failed to send.

Persists envelopes one-per-line as ``base64(json(envelope))`` so a
single corrupt line doesn't poison the rest. Multi-process safe via
:class:`utils._ExclusiveFileLock`.

Default location: ``%USERPROFILE%\.winml\telemetry\modelkit.json``.
Override with the ``MODELKIT_TELEMETRY_CACHE_DIR`` env var (developer-
facing only; not for end users).

The cache is intentionally append-only on the failure path and
drain-then-resend on the recovery path. There is no in-place mutation,
no rotation, and no size cap — telemetry traffic is sparse, the cache
fits in a few KB even after weeks of intermittent failures.

When telemetry transitions from enabled to disabled (consent change, or
empty iKey on a build that previously had one), :meth:`clear` is called
so a disabled session never resends events the user has since opted out
of.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .utils import (
    _decode_cache_entry,
    _encode_cache_entry,
    _ExclusiveFileLock,
    _resolve_user_home,
)


_CACHE_FILE_NAME = "modelkit.json"


def _cache_dir() -> Path | None:
    r"""Resolve the cache directory.

    Honors the ``MODELKIT_TELEMETRY_CACHE_DIR`` env var; otherwise
    falls back to ``%USERPROFILE%\.winml\telemetry``. Returns ``None``
    when no user home is resolvable — the cache becomes inert in that
    case rather than writing to a CWD-relative path.
    """
    override = os.environ.get("MODELKIT_TELEMETRY_CACHE_DIR")
    if override:
        return Path(override)
    home = _resolve_user_home()
    if home is None:
        return None
    return Path(home) / ".winml" / "telemetry"


def _cache_file() -> Path | None:
    base = _cache_dir()
    if base is None:
        return None
    return base / _CACHE_FILE_NAME


class _PersistentCache:
    """Append-or-drain file-backed cache for failed envelopes.

    All operations are best-effort: any I/O failure is swallowed because
    a cache failure must never affect telemetry emission (which itself
    must never affect the CLI).
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path: Path | None = path if path is not None else _cache_file()
        self._lock_path: Path | None = (
            self._path.with_suffix(self._path.suffix + ".lock") if self._path is not None else None
        )

    def append(self, envelopes: list[dict[str, Any]]) -> None:
        """Append the given envelopes to the cache."""
        if not envelopes or self._path is None or self._lock_path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with (
                _ExclusiveFileLock(self._lock_path),
                self._path.open("a", encoding="utf-8") as f,
            ):
                for env in envelopes:
                    f.write(_encode_cache_entry(env) + "\n")
        except Exception:
            # Cache failure is silent; the calling exporter has already
            # logged its own failure context.
            pass

    def drain(self) -> list[dict[str, Any]]:
        """Read all cached envelopes and delete the cache file.

        Returns an empty list on missing file, no-home, lock timeout, or
        any other I/O error. Malformed lines are skipped silently.
        """
        if self._path is None or self._lock_path is None:
            return []
        try:
            with _ExclusiveFileLock(self._lock_path):
                if not self._path.exists():
                    return []
                lines = self._path.read_text(encoding="utf-8").splitlines()
                self._path.unlink()
        except Exception:
            return []

        envelopes: list[dict[str, Any]] = []
        for line in lines:
            decoded = _decode_cache_entry(line)
            if decoded is not None:
                envelopes.append(decoded)
        return envelopes

    def clear(self) -> None:
        """Best-effort delete the cache file (e.g. on opt-out)."""
        if self._path is None or self._lock_path is None:
            return
        try:
            with _ExclusiveFileLock(self._lock_path):
                if self._path.exists():
                    self._path.unlink()
        except Exception:
            pass
