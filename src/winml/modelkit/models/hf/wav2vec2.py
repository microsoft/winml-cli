# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Audeering wav2vec2 emotion-regression export variant."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyAudioInputGenerator
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from ...export import register_onnx_overwrite
from ..winml import register_specialization


EMOTION_REGRESSION_MODEL_TYPE = "wav2vec2_emotion_regression"


class RegressionHead(nn.Module):
    """Audeering dimensional-emotion regression head."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return cast("torch.Tensor", self.out_proj(x))


class EmotionModel(Wav2Vec2PreTrainedModel):
    """Audeering wav2vec2 mean-pooling regression model."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.config.model_type = EMOTION_REGRESSION_MODEL_TYPE
        self.init_weights()

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D102
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


_EMOTION_NORMALIZED_CONFIG = NormalizedConfig.with_args(
    hidden_size="hidden_size",
    num_labels="num_labels",
    allow_new=True,
)


@register_onnx_overwrite(
    EMOTION_REGRESSION_MODEL_TYPE,
    "audio-classification",
    library_name="transformers",
)
class Wav2Vec2EmotionRegressionIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for raw waveform -> hidden_states, logits."""

    NORMALIZED_CONFIG_CLASS = _EMOTION_NORMALIZED_CONFIG
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyAudioInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return {"input_values": {0: "batch_size", 1: "audio_sequence_length"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return {
            "hidden_states": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }


MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("wav2vec2-emotion-regression", "audio-classification"): EmotionModel,
}

register_specialization(
    EMOTION_REGRESSION_MODEL_TYPE,
    "audio-classification",
    "WinMLModelForGenericTask",
)


__all__ = [
    "EMOTION_REGRESSION_MODEL_TYPE",
    "MODEL_CLASS_MAPPING",
    "EmotionModel",
    "RegressionHead",
    "Wav2Vec2EmotionRegressionIOConfig",
]
