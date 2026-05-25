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
(JSON, compact output, piped commands).
"""

import logging
import sys


def configure_logging(
    verbosity: int = 0,
    quiet: bool = False,
    *,
    # Backward-compat: accept old bool signature
    verbose: bool = False,
) -> None:
    """Configure root logger based on verbosity level.

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

    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


def flush_ort_startup_logs() -> None:
    """Replay ORT stderr messages captured during onnxruntime import.

    onnxruntime is imported lazily (when a command module is first loaded),
    which happens after :func:`configure_logging` has already run.  This
    function must therefore be called from the command-dispatch path — after
    the command module import but before the command handler runs — so that
    the messages are replayed against the already-configured logger.

    Messages are emitted at DEBUG level (visible with ``-vv`` / ``--debug``).
    The buffer is drained on first call; subsequent calls are no-ops.
    """
    try:
        from winml.modelkit.utils.native_stderr import replay_ort_startup_logs
    except ImportError:
        return

    replay_ort_startup_logs()
