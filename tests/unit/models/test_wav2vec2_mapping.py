# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for Wav2Vec2 task-specific model-class routing."""

from __future__ import annotations

from typing import ClassVar

from winml.modelkit.loader import resolve_task
from winml.modelkit.loader.resolution import _get_custom_model_class
from winml.modelkit.models.hf import MODEL_CLASS_MAPPING
from winml.modelkit.models.hf.wav2vec2 import MODEL_CLASS_MAPPING as WAV2VEC2_MAPPING


class _Config:
    model_type = "wav2vec2"
    architectures: ClassVar[list[str]] = ["Wav2Vec2ForCTC"]
    _name_or_path = "facebook/mms-1b-all"
    is_encoder_decoder = False


class TestWav2Vec2Mapping:
    """Wav2Vec2 ASR and audio-classification loaders use architecture-appropriate classes."""

    def test_asr_mapping_returns_ctc_model(self):
        """Wav2Vec2 ASR checkpoints should resolve to the CTC loader."""
        result = _get_custom_model_class("wav2vec2", "automatic-speech-recognition")

        assert result is not None
        assert result.__name__ == "AutoModelForCTC"

    def test_audio_classification_mapping_returns_audio_model(self):
        """Wav2Vec2 sequence-classification checkpoints keep the audio loader."""
        result = _get_custom_model_class("wav2vec2", "audio-classification")

        assert result is not None
        assert result.__name__ == "AutoModelForAudioClassification"

    def test_mapping_merged_into_aggregate(self):
        """The module-level mapping is included in the aggregated mapping."""
        assert WAV2VEC2_MAPPING.items() <= MODEL_CLASS_MAPPING.items()

    def test_detected_ctc_asr_uses_ctc_loader(self):
        """Detected Wav2Vec2ForCTC ASR resolves to AutoModelForCTC."""
        resolution = resolve_task(_Config())

        assert resolution.task == "automatic-speech-recognition"
        assert resolution.model_class.__name__ == "AutoModelForCTC"
