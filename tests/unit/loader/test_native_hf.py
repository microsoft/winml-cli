# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for shared native Hugging Face loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from winml.modelkit.loader import load_native_hf_model, resolve_native_device


class TestResolveNativeDevice:
    def test_auto_uses_cpu_without_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        resolved = resolve_native_device("auto")

        assert resolved.name == "cpu"
        assert resolved.torch_device == torch.device("cpu")

    def test_auto_uses_cuda_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        resolved = resolve_native_device("auto")

        assert resolved.name == "gpu"
        assert resolved.torch_device == torch.device("cuda")

    def test_gpu_requires_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(ValueError, match="requires a CUDA-enabled PyTorch"):
            resolve_native_device("gpu")


def test_load_native_hf_model_preserves_dtype_and_task_selection() -> None:
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    with patch(
        "winml.modelkit.loader.hf.load_hf_model",
        return_value=(model, MagicMock(), "image-classification"),
    ) as load:
        result = load_native_hf_model(
            "fake/model",
            task="image-classification",
            device="cpu",
            trust_remote_code=True,
        )

    load.assert_called_once_with(
        "fake/model",
        task="image-classification",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.to.assert_called_once_with(torch.device("cpu"))
    model.eval.assert_called_once_with()
    assert result.model is model
    assert result.device.name == "cpu"
