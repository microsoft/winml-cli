# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compatibility tests for inspect config loading."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from transformers import PretrainedConfig

from winml.modelkit.inspect import InspectError, inspect_model


def test_inspect_rejects_generic_fallback_with_task_override() -> None:
    config = PretrainedConfig()
    config._winml_generic_fallback = True

    with (
        patch("winml.modelkit.inspect.load_hf_config", return_value=config),
        patch(
            "winml.modelkit.inspect.resolve_loader",
            side_effect=AssertionError("generic config was not rejected"),
        ),
        pytest.raises(InspectError, match="overrides are not enough"),
    ):
        inspect_model("local-model", task_override="fill-mask")
