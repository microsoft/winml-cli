# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests for _InferenceEngine loader paths (ep_device signature).

Guards against a regression where the three private loader paths in
``InferenceEngine`` still pass ``device=``/``ep=`` kwargs to the
``WinMLAutoModel`` and ``WinMLPreTrainedModel`` APIs — which now require
``ep_device: WinMLEPDevice`` instead.

Each loader must:
  1. Resolve a ``WinMLEPDevice`` via ``resolve_device`` + ``auto_device``.
  2. Reach the downstream callable with ``ep_device=`` (no ``TypeError``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from winml.modelkit.inference import InferenceEngine


if TYPE_CHECKING:
    from pathlib import Path


def _fake_ep_device() -> object:
    """A minimal stand-in for WinMLEPDevice; loader paths only read no attrs."""
    device = SimpleNamespace(device_type="CPU", ep_name="CPUExecutionProvider")
    return SimpleNamespace(device=device)


def test_load_from_onnx_uses_ep_device_kwarg(tmp_path: Path) -> None:
    """_load_from_onnx must call WinMLAutoModel.from_onnx with ep_device=, not device=/ep=."""
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"")

    engine = InferenceEngine()
    fake_ep_device = _fake_ep_device()

    with (
        patch(
            "winml.modelkit.session.resolve_device",
            return_value=SimpleNamespace(ep="cpu", device="cpu", source=None),
        ),
        patch("winml.modelkit.session.WinMLEPRegistry.instance") as mock_instance,
        patch("winml.modelkit.models.auto.WinMLAutoModel.from_onnx") as mock_from_onnx,
    ):
        mock_instance.return_value.auto_device.return_value = fake_ep_device
        mock_from_onnx.return_value = MagicMock()
        # Must reach the mock without raising TypeError.
        engine._load_from_onnx(onnx, task="text-classification", device="cpu", ep=None)

    assert mock_from_onnx.called
    call = mock_from_onnx.call_args
    assert "ep_device" in call.kwargs
    assert "device" not in call.kwargs
    assert "ep" not in call.kwargs


def test_load_from_hf_uses_ep_device_kwarg() -> None:
    """_load_from_hf must call WinMLAutoModel.from_pretrained with ep_device= (positional or kw)."""
    engine = InferenceEngine()
    fake_ep_device = _fake_ep_device()

    with (
        patch(
            "winml.modelkit.session.resolve_device",
            return_value=SimpleNamespace(ep="cpu", device="cpu", source=None),
        ),
        patch("winml.modelkit.session.WinMLEPRegistry.instance") as mock_instance,
        patch("winml.modelkit.models.auto.WinMLAutoModel.from_pretrained") as mock_from_pretrained,
    ):
        mock_instance.return_value.auto_device.return_value = fake_ep_device
        mock_from_pretrained.return_value = MagicMock(task=None, config=None)
        engine._load_from_hf("hf/id", task="text-classification", device="cpu", ep=None)

    assert mock_from_pretrained.called
    call = mock_from_pretrained.call_args
    # ep_device may be positional or keyword — as long as one of them
    # holds the object we constructed above.
    assert fake_ep_device in call.args or call.kwargs.get("ep_device") is fake_ep_device
    assert "device" not in call.kwargs
    assert "ep" not in call.kwargs


def test_load_from_build_dir_constructs_with_ep_device(tmp_path: Path) -> None:
    """_load_from_build_dir must construct WinMLPreTrainedModel with ep_device=, not device=."""
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"")
    manifest = tmp_path / "build_manifest.json"
    manifest.write_text(
        '{"model_id": "hf/id", "task": "text-classification"}',
        encoding="utf-8",
    )

    engine = InferenceEngine()
    fake_ep_device = _fake_ep_device()

    winml_class = MagicMock()
    winml_class.return_value = MagicMock(task="text-classification")

    with (
        patch(
            "winml.modelkit.inference.engine._find_build_artifacts",
            return_value=(onnx, {"model_id": "hf/id", "task": "text-classification"}),
        ),
        patch(
            "winml.modelkit.session.resolve_device",
            return_value=SimpleNamespace(ep="cpu", device="cpu", source=None),
        ),
        patch("winml.modelkit.session.WinMLEPRegistry.instance") as mock_instance,
        patch("winml.modelkit.models.winml.get_winml_class", return_value=winml_class),
        patch.object(engine, "_attach_hf_config"),
    ):
        mock_instance.return_value.auto_device.return_value = fake_ep_device
        engine._load_from_build_dir(tmp_path, task="text-classification", device="cpu", ep=None)

    assert winml_class.called
    call = winml_class.call_args
    assert call.kwargs.get("ep_device") is fake_ep_device
    assert "device" not in call.kwargs
