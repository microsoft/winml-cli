# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

r"""Consent decision for ModelKit telemetry.

A first-run interactive prompt collects user consent (default: accept)
and persists it to ``%USERPROFILE%\.winml\config.json``. This module
owns: CI/CD detection, config-file read/write, and the prompt. There
are **no** environment-variable overrides and **no** ``winml telemetry``
subcommands - to change consent after first run, users edit the config
file directly.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Literal

from .utils import _resolve_user_home


def _default_config_path() -> Path | None:
    home = _resolve_user_home()
    if home is None:
        return None
    return Path(home) / ".winml" / "config.json"


# ``None`` means we couldn't resolve a user home at import time — read /
# write paths below treat that as "no persistence" rather than silently
# falling through to CWD. Tests monkeypatch this to a ``tmp_path``.
_CONFIG_PATH: Path | None = _default_config_path()

# Consent notice version + text are a pair: bump the version whenever
# _PROMPT_TEXT's scope materially changes (new data category, widened
# scope). The prompt always describes the full current scope - it is
# NOT a delta vs. prior versions - so whatever vN's text lists is
# exactly what the user consents to when they answer. On a bump,
# stored records with an older version are treated as unrecorded on
# read so the user sees the updated notice and re-consents. Records
# predating the version field are grandfathered as the current version.
_CONSENT_VERSION: int = 1

_PROMPT_TEXT = """\
ModelKit can collect anonymous usage data to help improve the product.

What is collected:
  - Command name, duration, success/failure
  - Target device/EP (when the command specifies them)
  - OS, architecture, ModelKit version
  - Unhandled exception types, code locations, and scrubbed error
    messages (paths trimmed, length capped, PII patterns scrubbed)

What is never collected:
  - File paths, model contents, command arguments, credentials

Enable telemetry? [Y/n]: """

_CI_ENV_VARS = (
    "CI",
    "TF_BUILD",
    "GITHUB_ACTIONS",
    "JENKINS_URL",
    "CODEBUILD_BUILD_ID",
    "BUILDKITE",
    "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",
)

Consent = Literal["enabled", "disabled"]


def _is_ci_environment() -> bool:
    return any(os.environ.get(v) for v in _CI_ENV_VARS)


def _load_config() -> dict:
    """Read the full config.json. Return ``{}`` if missing or unreadable."""
    if _CONFIG_PATH is None:
        return {}
    try:
        raw = _CONFIG_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_stored_consent() -> Consent | None:
    """Return the stored consent value, or ``None`` if not recorded.

    ``None`` is returned for missing / malformed / unknown values, or
    when the stored ``consent_version`` is strictly older than the
    current ``_CONSENT_VERSION`` (triggers a re-prompt after a notice
    update). Records without a ``consent_version`` field are grandfathered
    as the current version so introducing the field doesn't re-prompt
    existing users.
    """
    data = _load_config()
    tele = data.get("telemetry")
    if not isinstance(tele, dict):
        return None
    value = tele.get("consent")
    if value not in ("enabled", "disabled"):
        return None
    stored_version = tele.get("consent_version")
    # `bool` is a subclass of `int` in Python; exclude explicitly so a
    # stray `True` isn't silently interpreted as version 1.
    if (
        isinstance(stored_version, int)
        and not isinstance(stored_version, bool)
        and stored_version < _CONSENT_VERSION
    ):
        return None
    return value  # type: ignore[return-value]


def _write_stored_consent(value: Consent) -> None:
    """Persist consent to config.json. Atomic: temp file + replace.

    Preserves any unrelated top-level keys the user (or future features)
    may have added. No-op if no user home is resolvable.
    """
    if _CONFIG_PATH is None:
        return
    data = _load_config()
    tele = data.get("telemetry") if isinstance(data.get("telemetry"), dict) else {}
    tele["consent"] = value
    tele["consent_version"] = _CONSENT_VERSION
    data["telemetry"] = tele

    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file in the same directory, then atomic replace.
    fd, tmp_name = tempfile.mkstemp(prefix=".config-", suffix=".json.tmp", dir=_CONFIG_PATH.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(_CONFIG_PATH)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            # Best-effort cleanup; temp file may already be gone or on a
            # read-only volume. The real failure is re-raised below.
            pass
        raise


def _prompt_for_consent() -> Consent:
    """Show the first-run prompt and return the user's decision.

    Default on empty / unknown input is ``'enabled'`` (accept-by-default).
    Only an explicit ``n`` / ``no`` declines.
    """
    try:
        sys.stdout.write(_PROMPT_TEXT)
        sys.stdout.flush()
        answer = sys.stdin.readline().strip().lower()
    except (OSError, EOFError):
        # If we can't read the answer, fall back to disabled - silent
        # environments must not default to emission.
        return "disabled"
    if answer in ("n", "no"):
        return "disabled"
    return "enabled"


def resolve_consent() -> Consent:
    r"""Compute the effective consent decision for this invocation.

    Precedence (first match wins):

    1. CI environment -> ``disabled`` (does not touch stored state)
    2. Non-TTY stdin  -> ``disabled`` (does not touch stored state)
    3. Stored decision -> honored
    4. Interactive prompt -> asks user, persists answer

    Accept-by-default applies **only** to step 4. Steps 1 and 2 fail
    closed - silent environments never default to emission even though
    interactive users see an accept-by-default prompt.

    To change a stored decision, users edit ``telemetry.consent`` in
    ``%USERPROFILE%\.winml\config.json`` directly. There are no CLI
    subcommands for this.
    """
    if _is_ci_environment():
        return "disabled"

    if not sys.stdin.isatty():
        return "disabled"

    stored = _read_stored_consent()
    if stored is not None:
        return stored

    answer = _prompt_for_consent()
    try:
        _write_stored_consent(answer)
    except Exception:
        # Never crash the CLI on storage failure - next run will re-prompt.
        pass
    return answer
