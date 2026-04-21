# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Utility helpers for the telemetry module.

Exception scrubbing, structured traceback extraction, file locking, and
cache encoding. Private-by-convention (``_``-prefixed) functions are
exported so ``telemetry.py`` can use them; they are not part of the
public package API.
"""

from __future__ import annotations

import base64
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any


_PACKAGE_ROOT = "winml/modelkit"


def _trim_path(path: str) -> str:
    """Rewrite a path to a package-relative form (forward slashes).

    If the path contains the package root, return the slice from that root
    onwards. Otherwise fall back to the basename (stdlib / third-party
    frames). Empty input returns empty.
    """
    if not path:
        return ""
    normalized = path.replace("\\", "/")
    idx = normalized.find(_PACKAGE_ROOT)
    if idx >= 0:
        return normalized[idx:]
    # No package prefix - basename only (e.g., stdlib or third-party frames).
    return normalized.rsplit("/", 1)[-1]


_PII_PATTERNS: list[re.Pattern[str]] = [
    # Email
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    # GUID (8-4-4-4-12 hex)
    re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
        r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    ),
    # IPv4
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # IPv6. Two alternatives:
    #   1) compressed form with "::" (e.g. 2001:db8::1)
    #   2) full uncompressed form (e.g. 2001:db8:85a3:0:0:8a2e:370:7334)
    re.compile(
        r"\b[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{0,4}){1,7}::[0-9a-fA-F:]*\b"
        r"|\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"
        r"|\b[0-9a-fA-F]{1,4}::[0-9a-fA-F:]*\b"
    ),
    # Long opaque token (API keys, JWTs, base64 blobs). Runs last so it
    # doesn't swallow already-scrubbed markers.
    re.compile(r"\b[A-Za-z0-9_\-]{24,}\b"),
]

_SCRUB_PLACEHOLDER = "<scrubbed>"


def _scrub_pii(text: str) -> str:
    """Replace known PII / secret patterns with ``<scrubbed>``.

    Order matters: specific patterns (email, GUID, IP) run before the
    generic long-token pattern so a specific match is not subsumed.
    """
    if not text:
        return text
    for pattern in _PII_PATTERNS:
        text = pattern.sub(_SCRUB_PLACEHOLDER, text)
    return text


_MESSAGE_CAP = 200


def _format_exception_message(message: str | None) -> str:
    """Run the scrubbing pipeline: path trim -> length cap -> PII scrub.

    Path trim is a token-level operation that only rewrites absolute paths
    found in the message. Length cap truncates to ``_MESSAGE_CAP`` chars
    with a trailing ``…`` marker. PII scrub replaces matching patterns
    with ``<scrubbed>``.
    """
    if not message:
        return ""
    # Trim absolute paths token-by-token (keeps surrounding text intact).
    tokens = [_trim_path(tok) if _looks_like_path(tok) else tok for tok in message.split(" ")]
    result = " ".join(tokens)
    # Length cap first (so PII runs on a bounded string; also prevents huge
    # messages from dominating the regex cost).
    if len(result) > _MESSAGE_CAP:
        result = result[: _MESSAGE_CAP - 1] + "…"
    # PII scrub last (so truncation won't reveal partial PII).
    return _scrub_pii(result)


def _looks_like_path(token: str) -> bool:
    # Heuristic: absolute path if it has a drive letter, starts with /,
    # or contains both a separator and the package root fragment.
    if not token:
        return False
    if re.match(r"^[A-Za-z]:[\\/]", token):
        return True
    if token.startswith("/"):
        return True
    return "\\" in token and _PACKAGE_ROOT in token.replace("\\", "/")


def _encode_cache_entry(entry: dict[str, Any]) -> str:
    """Encode a cache entry to a single-line string: ``base64(json(dict))``."""
    raw = json.dumps(entry, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _decode_cache_entry(line: str) -> dict[str, Any] | None:
    """Decode a line produced by :func:`_encode_cache_entry`.

    Returns ``None`` on any failure (malformed base64 or JSON). Callers
    treat ``None`` as "skip this entry" - a single bad line must not
    disturb the rest of the cache.
    """
    if not line:
        return None
    try:
        raw = base64.b64decode(line.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError):
        return None
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    return decoded


class _ExclusiveFileLock:
    """Windows-only multi-process-safe exclusive lock on a lockfile.

    Uses ``msvcrt.locking``. If lock acquisition fails, the underlying
    file handle is closed before the exception propagates (no handle leak).
    """

    def __init__(self, lockfile: Path) -> None:
        self._path = Path(lockfile)
        self._fd: int | None = None

    def __enter__(self) -> _ExclusiveFileLock:
        import msvcrt

        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        import msvcrt

        if self._fd is None:
            return
        try:
            try:
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                # Already unlocked (e.g. process exit); close anyway.
                pass
        finally:
            os.close(self._fd)
            self._fd = None


def _extract_exception_stack(tb: Any) -> list[dict[str, Any]]:
    """Return a list of ``{file, line, function}`` dicts for ``tb``.

    Contains only structural info: no exception message, no source line
    text, no local variable values. File paths are trimmed to package-
    relative form via :func:`_trim_path`.
    """
    if tb is None:
        return []
    frames = traceback.extract_tb(tb)
    return [
        {
            "file": _trim_path(frame.filename or ""),
            "line": frame.lineno or 0,
            "function": frame.name or "",
        }
        for frame in frames
    ]
