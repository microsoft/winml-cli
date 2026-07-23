# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Regression pin for P0-C: composite dispatch in WinMLAutoModel.from_onnx
referenced `device`/`ep` after the pre-bench refactor collapsed them into
`ep_device`. Passing a dict `onnx_path` hit lines 141-157 and blew up
with NameError.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.models.auto import WinMLAutoModel


_FIXTURE = "tests/_fixtures/identity.onnx"


def test_composite_dispatch_does_not_nameerror() -> None:
    """Composite dict input must not raise NameError at dispatch.

    We stub the inner composite build so the test doesn't need a real
    ONNX runtime — the goal is to confirm the dispatch line evaluates
    without hitting `device` / `ep` name-lookup failures.
    """
    dispatched: dict[str, object] = {}

    def _fake_composite_from_onnx(*args, **kwargs):
        dispatched["args"] = args
        dispatched["kwargs"] = kwargs
        return MagicMock()

    fake_ep_device = MagicMock(name="WinMLEPDevice")

    with patch(
        "winml.modelkit.models.winml.composite_model.WinMLCompositeModel.from_onnx",
        side_effect=_fake_composite_from_onnx,
    ):
        try:
            WinMLAutoModel.from_onnx(
                {"encoder": _FIXTURE, "decoder": _FIXTURE},
                ep_device=fake_ep_device,
            )
        except NameError as exc:
            pytest.fail(f"NameError at composite dispatch: {exc}")

    # Confirm the dispatch actually ran (not that we short-circuited elsewhere).
    assert "args" in dispatched, "composite dispatch did not fire"
