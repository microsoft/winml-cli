# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared test helpers for WinML CLI invocation.

Provides ``run_inspect``, a thin wrapper around ``CliRunner.invoke`` used
by both ``tests/cli/test_inspect_cli.py`` and ``tests/e2e/test_inspect_e2e.py``
so that the invocation envelope (``obj={}``, ``mix_stderr`` defaults, etc.)
lives in a single place.
"""

from __future__ import annotations

from click.testing import CliRunner, Result

from winml.modelkit.commands.inspect import inspect


def run_inspect(*args: str) -> Result:
    """Invoke the ``inspect`` Click command with *args and return the Result."""
    return CliRunner().invoke(inspect, list(args), obj={})
