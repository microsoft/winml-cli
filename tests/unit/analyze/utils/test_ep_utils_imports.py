# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Regression pin for P0-B: analyze/utils/ep_utils.py had a dangling
`from ...sysinfo.device import get_ep_device_map` inside
`get_devices_with_rule_data` after `sysinfo.device` was deleted.

Bare module import passes (the import is lazy). We exercise the actual
call path to catch the ImportError.
"""
from __future__ import annotations

import importlib


def test_module_imports_cleanly() -> None:
    """Importing analyze.utils.ep_utils must not raise (regression pin for P0-B)."""
    importlib.import_module("winml.modelkit.analyze.utils.ep_utils")


def test_get_devices_with_rule_data_does_not_importerror(tmp_path) -> None:
    """Calling get_devices_with_rule_data must not raise ImportError.

    Before the fix, the fallback branch imported `get_ep_device_map`
    from a deleted `sysinfo.device` module. Point rule search at an
    empty dir to guarantee the fallback path is taken.
    """
    from winml.modelkit.analyze.utils import ep_utils

    with importlib.import_module("pytest").MonkeyPatch.context() as mp:
        mp.setattr(
            "winml.modelkit.analyze.utils.rule_loader.get_runtime_rules_search_dirs",
            lambda: [tmp_path],
        )
        # Any known EP forces the fallback branch — QNN + NPU is a known catalog row.
        result = ep_utils.get_devices_with_rule_data("QNNExecutionProvider")
        assert isinstance(result, list)
        assert "NPU" in result, f"expected NPU in QNN device list, got {result!r}"
