# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression test for ModelValidatorManager device fallback.

Guards against a regression of the double-assignment bug where
``self.device = device or "NPU"`` was immediately clobbered by
``self.device = device`` on the next line.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from winml.modelkit.analyze.core.model_validators import ModelValidatorManager


def test_none_device_falls_back_to_npu() -> None:
    """Passing device=None must resolve to the documented "NPU" fallback."""
    fake_proto = MagicMock()
    fake_model = MagicMock()
    fake_model.get_model.return_value = fake_proto

    manager = ModelValidatorManager(
        model=fake_model,
        enabled_validators=[],  # skip validator instantiation for isolation
        device=None,
    )
    assert manager.device == "NPU"
