# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Logging utilities for WinML CLI.

Verbosity Convention (adopted from pip, ansible, pytest):
=========================================================

    Flag        Level       Value   Use case
    ----        -----       -----   --------
    -q          ERROR       40      Errors only (quiet / scripting)
    (default)   WARNING     30      Warnings + errors (production default)
    -v          INFO        20      Operational progress messages
    -vv         DEBUG       10      Developer-level tracing
    --debug     DEBUG       10      Alias for -vv (backward compat)

    Formula: level = WARNING - (verbosity * 10)  ->  30, 20, 10
    Quiet:   level = ERROR (40)

All log output goes to stderr so stdout stays clean for structured data
(JSON, compact output, piped commands). Format:

    [%(asctime)s %(levelname)-7s %(name)s] %(message)s

Sample line: ``[14:32:11 INFO    winml.modelkit.export] Loaded config.json``
"""

import logging
import sys


_HANDLER_MARKER = "_winml_cli_handler"
_LOG_FORMAT = "[%(asctime)s %(levelname)-7s %(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(
    verbosity: int = 0,
    quiet: bool = False,
    *,
    # Backward-compat: accept old bool signature
    verbose: bool = False,
) -> None:
    """Configure root logger based on verbosity level.

    Idempotent: subcommands re-call this after merging top-level + subcommand
    ``-v``/``-q``. The first call installs the WinML stderr handler; later
    calls only adjust the level. Existing non-WinML handlers (notably pytest's
    ``caplog`` propagate-handler) are preserved.

    Args:
        verbosity: Number of ``-v`` flags (0=WARNING, 1=INFO, 2+=DEBUG).
        quiet: If True, override to ERROR level regardless of verbosity.
        verbose: **Deprecated bool compat** — treated as verbosity=1 when
                 True and verbosity is 0. Existing callers that pass
                 ``verbose=True`` keep working without changes.
    """
    # Backward compat: bool verbose → int, also handles count passthrough
    if verbose and verbosity == 0:
        verbosity = int(verbose)

    # Clamp between DEBUG (10) and WARNING (30); quiet overrides to ERROR
    log_level = logging.ERROR if quiet else max(logging.DEBUG, logging.WARNING - verbosity * 10)

    root = logging.getLogger()
    # Drop any prior WinML handler and install a fresh one bound to the
    # *current* ``sys.stderr``. Click's ``CliRunner.invoke()`` swaps the
    # process stderr for each test, so a cached handler from an earlier
    # invocation would write to a stream the test no longer captures.
    # We leave non-WinML handlers (notably pytest's caplog handler) alone.
    for h in list(root.handlers):
        if getattr(h, _HANDLER_MARKER, False):
            root.removeHandler(h)
    own_handler = logging.StreamHandler(sys.stderr)
    own_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    setattr(own_handler, _HANDLER_MARKER, True)
    own_handler.setLevel(log_level)
    root.addHandler(own_handler)
    root.setLevel(log_level)


def flush_ort_startup_logs() -> None:
    """No-op kept for backward compatibility.

    ORT startup stderr is now discarded to devnull (not captured), so there
    is nothing to replay.
    """
