# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""ActionGroup auto-instruments every registered Click subcommand.

On subcommand invocation:

  * Emits ``ModelKitHeartbeat`` once per CLI invocation, just before the
    subcommand body runs.
  * Wraps the subcommand's ``invoke`` to time execution and emit
    ``ModelKitAction`` on completion (success or failure) and
    ``ModelKitError`` on unhandled exception.

No per-command decoration is needed; any command registered on an
``ActionGroup`` is instrumented automatically.

The Click ``--help`` / ``--version`` paths short-circuit inside Click's
own argument parsing before the wrapped invoke is reached, so they are
never instrumented and the Telemetry singleton is not materialized for
them.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import click

from .telemetry import Telemetry


_LOGGER = logging.getLogger(__name__)
_INSTRUMENTED_ATTR = "_modelkit_instrumented"
_HEARTBEAT_FLAG = "_modelkit_heartbeat_sent"


class ActionGroup(click.Group):
    """Click group that auto-instruments every registered command."""

    def resolve_command(self, ctx, args):  # noqa: D102 - Click override
        cmd_name, cmd, remaining = super().resolve_command(ctx, args)
        if cmd is None:
            return cmd_name, cmd, remaining
        return cmd_name, _instrument(cmd), remaining


def _instrument(cmd: click.Command) -> click.Command:
    """Wrap ``cmd.invoke`` with ModelKit telemetry emission.

    Every call emits ``ModelKitHeartbeat`` (once per CLI invocation),
    ``ModelKitAction`` on completion, and ``ModelKitError`` on
    exception. Idempotent via a marker attribute so repeated resolutions
    don't stack wrappers.
    """
    if getattr(cmd, _INSTRUMENTED_ATTR, False):
        return cmd
    original_invoke = cmd.invoke

    def wrapped_invoke(ctx: click.Context) -> Any:
        try:
            telemetry = Telemetry.get_or_init()
        except Exception:
            return original_invoke(ctx)

        if telemetry.disabled:
            return original_invoke(ctx)

        _emit_heartbeat_once(ctx, telemetry)

        start = time.perf_counter()
        success = True
        try:
            return original_invoke(ctx)
        except Exception as exc:
            success = False
            try:
                telemetry.log_error(exc)
            except Exception:
                _LOGGER.debug("error emit failed", exc_info=True)
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            try:
                telemetry.log_action(
                    action_name=cmd.name or "",
                    device=_param(ctx, "device"),
                    ep=_param(ctx, "ep"),
                    duration_ms=duration_ms,
                    success=success,
                )
            except Exception:
                _LOGGER.debug("action emit failed", exc_info=True)

    cmd.invoke = wrapped_invoke
    setattr(cmd, _INSTRUMENTED_ATTR, True)
    return cmd


def _emit_heartbeat_once(ctx: click.Context, telemetry: Telemetry) -> None:
    """Emit ``ModelKitHeartbeat`` once per CLI invocation.

    Subsequent calls (e.g. nested ``ActionGroup`` chain) are no-ops. The
    flag lives on the root Click context so it is naturally scoped to
    one CLI process.
    """
    root = ctx.find_root()
    if getattr(root, _HEARTBEAT_FLAG, False):
        return
    try:
        telemetry.log_heartbeat()
    except Exception:
        _LOGGER.debug("heartbeat emit failed", exc_info=True)
    setattr(root, _HEARTBEAT_FLAG, True)


def _param(ctx: click.Context, name: str) -> Any:
    if ctx.params is None:
        return None
    return ctx.params.get(name)
