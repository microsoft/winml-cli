# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""ModelKit telemetry singleton.

Lazily constructed: on first call, consults :func:`consent.resolve_consent`,
builds a :class:`LoggerProvider` with device ID + OS + app context, and
exposes :meth:`log_heartbeat` / :meth:`log_action` / :meth:`log_error`. If
consent is disabled OR the iKey is empty, ``disabled=True`` and all
emission calls are no-ops.
"""

from __future__ import annotations

import logging
import platform
import sys
import time
import uuid
from typing import Any

from . import consent as consent_mod
from . import constants
from .deviceid import get_or_create_device_id
from .utils import _extract_exception_stack, _format_exception_message


_LOGGER = logging.getLogger(__name__)

_INSTANCE: Telemetry | None = None

_HEARTBEAT_EVENT = "ModelKitHeartbeat"
_ACTION_EVENT = "ModelKitAction"
_ERROR_EVENT = "ModelKitError"

_ALLOWED_KEYS: dict[str, set[str]] = {
    _HEARTBEAT_EVENT: set(),
    _ACTION_EVENT: {
        "invoked_from",
        "action_name",
        "device",
        "ep",
        "duration_ms",
        "success",
    },
    _ERROR_EVENT: {
        "exception_type",
        "exception_message",
        "exception_stack",
    },
}


def _filter_allowlist(event_name: str, attrs: dict[str, Any]) -> dict[str, Any]:
    allowed = _ALLOWED_KEYS[event_name]
    return {k: v for k, v in attrs.items() if k in allowed}


def _clear_cache_quietly() -> None:
    """Best-effort delete the persistent cache.

    Called when telemetry init resolves to disabled (empty iKey, consent
    declined, or init crashed) so a disabled session never resends events
    the user has since opted out of.
    """
    try:
        from ._cache import _PersistentCache

        _PersistentCache().clear()
    except Exception:
        _LOGGER.debug("cache clear failed", exc_info=True)


class Telemetry:
    """Process-wide telemetry singleton. Use :meth:`get_or_init` to access.

    The caller owns the lifecycle: :meth:`shutdown` must be invoked before
    the process exits to flush any events still queued in the underlying
    ``BatchLogRecordProcessor``. Phase 3's ``ActionGroup`` will own this
    in its Click teardown path; until then, direct callers risk losing
    the last batch on process exit.
    """

    def __init__(self) -> None:
        self._logger = None  # set when enabled; None when disabled
        self._provider = None
        self._disabled = True  # set to False only after successful init
        self._init_ts = time.time()
        self._app_instance_id = str(uuid.uuid4())
        # Kept in the event schema for forward-compat: today
        # `consent.resolve_consent()` returns "disabled" for non-TTY, so
        # only "Interactive" reaches the wire. If Phase 3 relaxes that
        # rule (e.g. honors stored consent in non-TTY), no schema change
        # is needed.
        self._invoked_from = "Script" if not sys.stdin.isatty() else "Interactive"
        self._try_init()

    @classmethod
    def get_or_init(cls) -> Telemetry:
        """Return the cached singleton, constructing it on first call."""
        global _INSTANCE
        if _INSTANCE is None:
            _INSTANCE = cls()
        return _INSTANCE

    @property
    def disabled(self) -> bool:
        """Whether telemetry is inactive (consent off, empty iKey, or init failed)."""
        return self._disabled

    def _try_init(self) -> None:
        """Wire up the LoggerProvider, swallowing any setup error.

        Every step that can fail is guarded - a telemetry init failure
        must never propagate to the CLI.
        """
        try:
            if not constants.INSTRUMENTATION_KEY:
                _clear_cache_quietly()
                return
            if consent_mod.resolve_consent() != "enabled":
                _clear_cache_quietly()
                return
            # Import lazily so tests that never reach this branch don't
            # pay the OTel SDK import cost.
            from .library import create_logger_provider

            resource = self._build_resource()
            self._provider = create_logger_provider(
                ikey=constants.INSTRUMENTATION_KEY,
                resource=resource,
            )
            self._logger = self._provider.get_logger("modelkit")
            self._disabled = False
        except Exception:
            _LOGGER.debug("telemetry init failed", exc_info=True)
            self._logger = None
            self._provider = None
            self._disabled = True
            _clear_cache_quietly()

    def _build_resource(self):
        from opentelemetry.sdk.resources import Resource

        device_id, id_status = get_or_create_device_id()
        uname = platform.uname()
        try:
            from winml.modelkit import __version__ as app_version
        except Exception:
            app_version = "0.0.0"
        return Resource.create(
            {
                "device_id": device_id,
                "id_status": id_status,
                "os.name": uname.system,
                "os.version": uname.version,
                "os.release": uname.release,
                "os.arch": uname.machine,
                "app_version": app_version,
                "app_instance_id": self._app_instance_id,
                "initTs": self._init_ts,
            }
        )

    def log_heartbeat(self) -> None:
        """Emit a ``ModelKitHeartbeat`` event (session-started signal)."""
        try:
            if self._logger is None:
                return
            self._emit(_HEARTBEAT_EVENT, {})
        except Exception:
            _LOGGER.debug("log_heartbeat failed", exc_info=True)

    def log_action(
        self,
        action_name: str,
        device: str | None,
        ep: str | None,
        duration_ms: int,
        success: bool,
        **_unused: Any,
    ) -> None:
        """Emit a ``ModelKitAction`` event for a completed CLI command."""
        try:
            if self._logger is None:
                return
            attrs = {
                "invoked_from": self._invoked_from,
                "action_name": action_name,
                "device": device,
                "ep": ep,
                "duration_ms": duration_ms,
                "success": success,
            }
            self._emit(_ACTION_EVENT, attrs)
        except Exception:
            _LOGGER.debug("log_action failed", exc_info=True)

    def log_error(self, exc: BaseException) -> None:
        """Emit a ``ModelKitError`` event for an unhandled exception."""
        try:
            if self._logger is None:
                return
            attrs = {
                "exception_type": type(exc).__name__,
                "exception_message": _format_exception_message(str(exc)),
                "exception_stack": _extract_exception_stack(exc.__traceback__),
            }
            self._emit(_ERROR_EVENT, attrs)
        except Exception:
            _LOGGER.debug("log_error failed", exc_info=True)

    def _emit(self, event_name: str, attrs: dict[str, Any]) -> None:
        from opentelemetry._logs import LogRecord

        filtered = _filter_allowlist(event_name, attrs)
        record = LogRecord(
            timestamp=time.time_ns(),
            body=event_name,
            attributes=filtered,
        )
        self._logger.emit(record)

    def shutdown(self) -> None:
        """Flush the provider and mark the instance disabled. Idempotent."""
        try:
            if self._provider is None:
                return
            self._provider.shutdown()
        except Exception:
            _LOGGER.debug("telemetry shutdown failed", exc_info=True)
        finally:
            self._logger = None
            self._provider = None
            self._disabled = True
