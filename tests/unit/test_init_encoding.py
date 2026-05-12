# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Reproduces the cp1252 emoji UnicodeEncodeError that occurs on Windows
consoles initialized with PYTHONIOENCODING=cp1252 (the common default for
non-utf-8 Windows shells).

Importing winml.modelkit must reconfigure sys.stdout/sys.stderr to utf-8
so downstream emoji / Unicode output (rich console, log messages, etc.)
never raises UnicodeEncodeError at runtime.
"""

from __future__ import annotations

import os
import subprocess
import sys


EMOJI_PROGRAM = "import winml.modelkit; print('\\U0001f680')"


def test_winml_modelkit_import_makes_emoji_print_safe_under_cp1252() -> None:
    """A subprocess started with PYTHONIOENCODING=cp1252 must be able to
    print an emoji after ``import winml.modelkit``.

    Without the fix in winml.modelkit.__init__, the child process raises:
        UnicodeEncodeError: 'charmap' codec can't encode character '\\U0001f680'
    """
    saved = os.environ.get("PYTHONIOENCODING")
    os.environ["PYTHONIOENCODING"] = "cp1252"
    try:
        result = subprocess.run(  # noqa: S603 -- trusted args (sys.executable + constant)
            [sys.executable, "-c", EMOJI_PROGRAM],
            capture_output=True,
            check=False,
        )
    finally:
        if saved is None:
            os.environ.pop("PYTHONIOENCODING", None)
        else:
            os.environ["PYTHONIOENCODING"] = saved

    stderr = result.stderr.decode("utf-8", errors="replace")
    assert result.returncode == 0, (
        f"emoji print under cp1252 failed (rc={result.returncode}):\n{stderr}"
    )
    assert "UnicodeEncodeError" not in stderr, stderr
