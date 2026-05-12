# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""ActionGroup auto-instruments every registered Click subcommand.

On subcommand invocation:

  * Emits ``WinMLCLIHeartbeat`` once per CLI invocation, just before the
    subcommand body runs.
  * Wraps the subcommand's ``invoke`` to time execution and emit
    ``WinMLCLIAction`` on completion (success or failure) and
    ``WinMLCLIError`` on unhandled exception.

No per-command decoration is needed; any command registered on an
``ActionGroup`` is instrumented automatically.

The Click ``--help`` / ``--version`` paths short-circuit inside Click's
own argument parsing before the wrapped invoke is reached, so they are
never instrumented and the Telemetry singleton is not materialized for
them.
"""

from __future__ import annotations

import time
from typing import Any

import click

from .telemetry import Telemetry


_INSTRUMENTED_ATTR = "_winmlcli_instrumented"
_HEARTBEAT_FLAG = "_winmlcli_heartbeat_sent"


class ActionGroup(click.Group):
    """Click group that auto-instruments every registered command."""

    def resolve_command(self, ctx, args):
        """Wrap the resolved subcommand with telemetry instrumentation."""
        cmd_name, cmd, remaining = super().resolve_command(ctx, args)
        if cmd is None:
            return cmd_name, cmd, remaining
        return cmd_name, _instrument(cmd), remaining


def _instrument(cmd: click.Command) -> click.Command:
    """Wrap ``cmd.invoke`` with WinML CLI telemetry emission.

    Every call emits ``WinMLCLIHeartbeat`` (once per CLI invocation),
    ``WinMLCLIAction`` on completion, and ``WinMLCLIError`` on
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
        except SystemExit as exc:
            # SystemExit inherits from BaseException, not Exception, so
            # `sys.exit(1)` would otherwise slip past the Exception
            # handler and be reported as a success in the finally block.
            # Treat any non-zero code as failure but don't log a Python
            # exception — the command exited intentionally.
            if exc.code not in (None, 0):
                success = False
            raise
        except click.exceptions.Exit as exc:
            # ``ctx.exit(N)`` raises this; it inherits from RuntimeError
            # so the Exception handler below would otherwise mis-fire
            # ``log_error`` and emit a ``WinMLCLIError`` event for what
            # is really a clean intentional exit. Mirror SystemExit:
            # success reflects the exit code, no log_error.
            if exc.exit_code != 0:
                success = False
            raise
        except Exception as exc:
            success = False
            telemetry.log_error(exc)
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            telemetry.log_action(
                action_name=cmd.name or "",
                device=_param(ctx, "device"),
                ep=_param(ctx, "ep"),
                duration_ms=duration_ms,
                success=success,
            )

    cmd.invoke = wrapped_invoke
    setattr(cmd, _INSTRUMENTED_ATTR, True)
    return cmd


def _emit_heartbeat_once(ctx: click.Context, telemetry: Telemetry) -> None:
    """Emit ``WinMLCLIHeartbeat`` once per CLI invocation.

    Subsequent calls (e.g. nested ``ActionGroup`` chain) are no-ops. The
    flag lives on the root Click context so it is naturally scoped to
    one CLI process.
    """
    root = ctx.find_root()
    if getattr(root, _HEARTBEAT_FLAG, False):
        return
    telemetry.log_heartbeat()
    setattr(root, _HEARTBEAT_FLAG, True)


def _param(ctx: click.Context, name: str) -> Any:
    return ctx.params.get(name)
