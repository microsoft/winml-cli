# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests for WinMLCompositeModel.from_pretrained (ep_device signature).

Guards:
  * Sub-model construction dispatches to WinMLAutoModel.from_pretrained without
    a TypeError from missing ``ep_device`` (B3).
  * The caller-supplied ``device`` value is forwarded into ``__init__`` rather
    than being silently defaulted to ``"cpu"`` (B6).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.models.winml.composite_model import (
    COMPOSITE_MODEL_REGISTRY,
    WinMLCompositeModel,
)


class _StubComposite(WinMLCompositeModel):
    """Concrete subclass with a single sub-component for isolation."""

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {"encoder": "feature-extraction"}


@pytest.fixture
def stub_registry_hit():
    """Register _StubComposite under a synthetic (model_type, task) pair."""
    key = ("_test_model_type", "_test_task")
    COMPOSITE_MODEL_REGISTRY[key] = _StubComposite
    try:
        yield key
    finally:
        COMPOSITE_MODEL_REGISTRY.pop(key, None)


def _fake_ep_device() -> object:
    device = SimpleNamespace(device_type="NPU", ep_name="QNNExecutionProvider")
    return SimpleNamespace(device=device)


def test_from_pretrained_does_not_raise_typeerror() -> None:
    """Sub-model dispatch must call WinMLAutoModel.from_pretrained without TypeError.

    Mirrors the production caller (:meth:`WinMLAutoModel.from_pretrained` at
    ``models/auto.py:329-341``) which passes ``device=`` and no ``ep_device``.
    The composite must derive/forward ``ep_device`` to the sub-model call.
    """
    hf_cfg = SimpleNamespace(model_type="_test_model_type")
    fake_ep_device = _fake_ep_device()

    with (
        patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=hf_cfg,
        ),
        patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_pretrained",
        ) as mock_from_pretrained,
        patch(
            "winml.modelkit.session.resolve_device",
            return_value=SimpleNamespace(ep="cpu", device="cpu", source=None),
        ),
        patch("winml.modelkit.session.WinMLEPRegistry.instance") as mock_instance,
    ):
        mock_from_pretrained.return_value = MagicMock()
        mock_instance.return_value.auto_device.return_value = fake_ep_device
        # Direct-subclass path (skips the registry lookup at line 156).
        # Production callers pass device= only — the composite must resolve
        # ep_device internally and pass it to the sub-model dispatch.
        _StubComposite.from_pretrained(
            "hf/id",
            task="_test_task",
            device="cpu",
        )

    # WinMLAutoModel.from_pretrained must have been reached for the sub-model
    # without raising TypeError on ep_device.
    assert mock_from_pretrained.called
    call = mock_from_pretrained.call_args
    assert fake_ep_device in call.args or call.kwargs.get("ep_device") is fake_ep_device, (
        f"Expected ep_device in call, got args={call.args!r} kwargs={list(call.kwargs)!r}"
    )


def test_device_is_forwarded_from_from_pretrained() -> None:
    """The device kwarg supplied to from_pretrained must reach __init__ (not default to cpu)."""
    hf_cfg = SimpleNamespace(model_type="_test_model_type")
    fake_ep_device = _fake_ep_device()

    with (
        patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=hf_cfg,
        ),
        patch(
            "winml.modelkit.models.auto.WinMLAutoModel.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        result = _StubComposite.from_pretrained(
            "hf/id",
            task="_test_task",
            device="npu",
            ep_device=fake_ep_device,
        )

    # Caller intent must survive — not clobbered by the "cpu" default.
    assert result._device == "npu"
