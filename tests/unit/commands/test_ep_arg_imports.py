# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Import-boundary tests for the ``--ep <name>[@<source>]`` parser."""

from __future__ import annotations

import subprocess
import sys


def test_ep_arg_import_does_not_import_onnxruntime() -> None:
    """The parser is used while Click loads commands, before warning filters run."""
    code = "import sys; import winml.modelkit.commands._ep_arg; print('onnxruntime' in sys.modules)"
    result = subprocess.run(  # noqa: S603 - fixed interpreter and inline test code
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_perf_command_import_does_not_import_onnxruntime() -> None:
    """Click imports command modules before entering the command body."""
    code = "import sys; import winml.modelkit.commands.perf; print('onnxruntime' in sys.modules)"
    result = subprocess.run(  # noqa: S603 - fixed interpreter and inline test code
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
