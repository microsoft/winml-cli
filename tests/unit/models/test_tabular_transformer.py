# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import inspect
from typing import ClassVar

import torch

from winml.modelkit.loader.task import TASK_SYNONYM_EXTENSIONS, to_optimum_task
from winml.modelkit.models.hf import MODEL_CLASS_MAPPING
from winml.modelkit.models.hf.transformer import (
    TabularTransformerWrapper,
    _ensure_legacy_remote_model_compat,
)
from winml.modelkit.models.winml.tabular_classification import (
    WinMLModelForTabularClassification,
)


class _Config:
    mean: ClassVar[list[float]] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    std: ClassVar[list[float]] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


class _RemoteTabularModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.input_proj = torch.nn.Linear(7, 3, bias=False)
        self.transformer = torch.nn.Identity()
        self.head = torch.nn.Linear(3, 1, bias=False)


def test_tabular_task_maps_to_optimum_text_classification() -> None:
    assert TASK_SYNONYM_EXTENSIONS["tabular-classification"] == "text-classification"
    assert to_optimum_task("tabular-classification") == "text-classification"


def test_transformer_model_class_mapping_registers_tabular_wrapper() -> None:
    assert (
        MODEL_CLASS_MAPPING[("transformer", "tabular-classification")]
        is TabularTransformerWrapper
    )
    assert MODEL_CLASS_MAPPING[("transformer", "text-classification")] is TabularTransformerWrapper
    assert MODEL_CLASS_MAPPING[("transformer", None)] is TabularTransformerWrapper


def test_runtime_wrapper_forward_matches_generic_base_signature() -> None:
    parameters = list(
        inspect.signature(WinMLModelForTabularClassification.forward).parameters.values()
    )

    assert [parameter.kind for parameter in parameters] == [
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_KEYWORD,
    ]


def test_tabular_wrapper_exposes_tensor_logits() -> None:
    remote = _RemoteTabularModel()
    wrapper = TabularTransformerWrapper(remote)

    features = torch.tensor([[1.0, 4.0, 7.0, 12.0, 21.0, 38.0, 71.0]])
    logits = wrapper(features)

    normalized = (features - torch.tensor(_Config.mean)) / (torch.tensor(_Config.std) + 1e-8)
    hidden = remote.input_proj(normalized).unsqueeze(1)
    expected = remote.head(remote.transformer(hidden).squeeze(1))
    torch.testing.assert_close(logits, expected)
    assert logits.shape == (1, 1)


def test_tabular_wrapper_restores_mha_fastpath_after_export(monkeypatch) -> None:
    wrapper = TabularTransformerWrapper(_RemoteTabularModel())
    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
    torch.backends.mha.set_fastpath_enabled(True)

    wrapper(torch.zeros((1, 7)))

    assert torch.backends.mha.get_fastpath_enabled() is True


def test_loader_shims_legacy_remote_model_class() -> None:
    class LegacyRemoteModel(_RemoteTabularModel):
        pass

    probe = LegacyRemoteModel()

    class FakeAutoModel:
        @staticmethod
        def from_config(*_args, **_kwargs):
            return probe

    _ensure_legacy_remote_model_compat(
        FakeAutoModel,
        _Config(),
        trust_remote_code=True,
    )

    assert LegacyRemoteModel.all_tied_weights_keys == {}
