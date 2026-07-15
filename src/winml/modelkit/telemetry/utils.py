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
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from types import TracebackType


_PACKAGE_ROOT = "winml/modelkit"


def _resolve_user_home() -> str | None:
    r"""Resolve the user home directory on Windows.

    Prefers ``%USERPROFILE%`` (virtually always set on Windows). Falls
    back to the Windows-native ``%HOMEDRIVE%%HOMEPATH%`` pair for stripped
    service-account / container environments where ``USERPROFILE`` may be
    missing. Returns ``None`` when neither is available — callers should
    treat ``None`` as "no per-user persistence" and skip the operation
    rather than silently resolve to a CWD-relative path.
    """
    profile = os.environ.get("USERPROFILE")
    if profile:
        return profile
    combined = os.environ.get("HOMEDRIVE", "") + os.environ.get("HOMEPATH", "")
    return combined or None


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
    # IPv4. Octets are validated to the 0-255 range so obvious nonsense
    # (e.g. "999.0.0.1") is not scrubbed. 4-part version strings like
    # "1.18.0.1" will still match - we accept this false positive per
    # the spec's "conservative over leaky" philosophy; version numbers
    # in error messages are secondary signal and the user still has
    # exception_type + stack for triage.
    re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
    ),
    # IPv6. Two alternatives:
    #   1) compressed form with "::" (e.g. 2001:db8::1)
    #   2) full uncompressed form (e.g. 2001:db8:85a3:0:0:8a2e:370:7334)
    re.compile(
        r"\b[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{0,4}){1,7}::[0-9a-fA-F:]*\b"
        r"|\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"
        r"|\b[0-9a-fA-F]{1,4}::[0-9a-fA-F:]*\b"
    ),
    # Long opaque token (API keys, JWTs, base64 blobs). Runs last so it
    # doesn't swallow already-scrubbed markers. Requires at least one
    # digit so we don't scrub long Pythonic class names (e.g.
    # "WinMLImageFeatureExtractionEvaluator") that carry diagnostic value
    # in error messages.
    re.compile(r"\b(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{24,}\b"),
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


def _format_exception_message(message: str | None, cap: int = _MESSAGE_CAP) -> str:
    """Run the scrubbing pipeline: path trim -> PII scrub -> length cap.

    Scrub runs *before* the length cap so PII straddling the cap boundary
    is still recognized by the regexes. Capping first would split a token
    or email mid-string and leak the surviving prefix (e.g. ``alice@exa…``
    leaves ``alice`` exposed because the cropped fragment no longer
    matches the email pattern). ``cap`` is parameterized so the root-cause
    message can use a larger limit than the outer message.
    """
    if not message:
        return ""
    # Trim absolute paths token-by-token (keeps surrounding text intact).
    tokens = [_trim_path(tok) if _looks_like_path(tok) else tok for tok in message.split(" ")]
    result = " ".join(tokens)
    # Scrub PII first so the cap can't split a sensitive token.
    result = _scrub_pii(result)
    # Cap last - bounds final size even if scrub expanded the string
    # (each match becomes the 11-char ``<scrubbed>`` placeholder).
    if len(result) > cap:
        result = result[: cap - 1] + "…"
    return result


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


# A clean HuggingFace Hub id: one or two segments of the Hub charset, with
# at most one slash (``bert-base-uncased`` or ``org/name``). Anything with a
# path separator beyond a single ``/``, a drive letter, an ``=`` (eval's
# ``role=path`` form), or any other character falls through to the local
# marker.
_HUB_ID_RE = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)?$")


def _model_ref_local_marker(value: str) -> str:
    """Anonymized marker for a local model reference.

    Emits only the file extension or the literal ``dir`` — never any path
    fragment (no basename, no directory names, no username).
    """
    tail = value.replace("\\", "/").rsplit("/", 1)[-1]
    ext = Path(tail).suffix
    return f"<local:{ext}>" if ext else "<local:dir>"


def _scrub_model_ref(value: str | tuple[str, ...] | None) -> str | None:
    """Classify a ``-m/--model`` reference for telemetry.

    Clean HuggingFace Hub ids — one or two Hub-charset segments with at most
    one slash, not present on disk (``bert-base-uncased``, ``org/name``) —
    pass through verbatim. Everything else — drive-letter paths, absolute
    paths, on-disk paths, ``role=path`` composites, and names with unexpected
    characters — collapses to a ``<local:...>`` marker that carries no path
    content.
    """
    if isinstance(value, tuple):
        value = value[0] if value else None
    if not value:
        return None
    normalized = value.replace("\\", "/")
    if re.match(r"^[A-Za-z]:[\\/]", value) or normalized.startswith("/"):
        return _model_ref_local_marker(value)
    if Path(value).exists():
        return _model_ref_local_marker(value)
    # A single-segment name that carries a file extension (e.g. ``model.onnx``)
    # is treated as a local file reference, not a Hub id — Hub ids don't carry
    # file extensions, and this is far more likely a path the user typed for a
    # file that isn't on *this* disk. Two-segment ``org/name`` is exempt: a dot
    # there is part of the id (e.g. ``org/model.v2``), not a file extension.
    if "/" not in normalized and Path(value).suffix:
        return _model_ref_local_marker(value)
    if _HUB_ID_RE.match(value):
        return value
    return _model_ref_local_marker(value)


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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
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


def _root_cause(exc: BaseException) -> BaseException:
    """Return the innermost cause of an exception chain.

    Follows ``__cause__`` (explicit ``raise ... from e``) in preference to
    ``__context__`` (implicit, set when raising inside an ``except`` block),
    repeatedly, until neither is set. A ``__context__`` explicitly suppressed
    by ``raise ... from None`` (``__suppress_context__``) is honored — the
    walk stops there, matching Python's own traceback printing and respecting
    the developer's intent to hide that inner error. Returns ``exc`` itself
    when there is no chain. Cycle-safe: a chain that loops back on itself
    terminates rather than spinning forever.
    """
    seen: set[int] = {id(exc)}
    current = exc
    while True:
        nxt = current.__cause__
        if nxt is None and not current.__suppress_context__:
            nxt = current.__context__
        if nxt is None or id(nxt) in seen:
            return current
        seen.add(id(nxt))
        current = nxt
