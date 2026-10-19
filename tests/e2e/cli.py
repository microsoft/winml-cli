# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Load the standalone E2E CLI helper independently of the pytest entry point."""

import importlib.util
from pathlib import Path


# `pytest` with importlib mode does not add the repository root to sys.path.
# Unlike `python -m pytest`, it cannot import our unpackaged scripts directory
# (and a third-party package named scripts may shadow it). Load the exact file.
_path = Path(__file__).resolve().parents[2] / "scripts" / "e2e_eval" / "winml_cli.py"
_spec = importlib.util.spec_from_file_location("_winml_e2e_cli", _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load E2E CLI helper from {_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

cli_command = _module.cli_command
openvino_gpu_allocators = _module.openvino_gpu_allocators
