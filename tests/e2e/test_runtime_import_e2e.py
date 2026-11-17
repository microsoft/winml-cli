# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Smoke-test installed runtime wheels without downloading models or running inference."""

import subprocess
import sys

import pytest


pytestmark = [pytest.mark.e2e]

_IMPORT_PROBE = """
import importlib
from importlib.metadata import version

for distribution, module_name in (
    ("windowsml", "windowsml"),
    ("onnxruntime-windowsml", "onnxruntime"),
):
    print(f"Importing {module_name}...", flush=True)
    module = importlib.import_module(module_name)
    print(f"{distribution}=={version(distribution)}", flush=True)
    print(f"Module path: {module.__file__}", flush=True)
"""


def test_runtime_imports(capsys: pytest.CaptureFixture[str]) -> None:
    result = subprocess.run(  # noqa: S603 -- current interpreter and a constant import probe
        [sys.executable, "-I", "-X", "utf8", "-c", _IMPORT_PROBE],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    with capsys.disabled():
        print(result.stdout, end="")
    assert result.returncode == 0, (
        f"Runtime imports failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
