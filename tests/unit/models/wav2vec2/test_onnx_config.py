# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the audeering wav2vec2 dimensional-emotion export variant.

The audeering ``wav2vec2-large-robust-12-ft-emotion-msp-dim`` checkpoint ships a
custom mean-pooling ``RegressionHead`` that transformers/Optimum cannot route on
their own. Support hinges on three contracts these tests lock in:

- the ``("wav2vec2-emotion-regression", "audio-classification")`` entry in
  ``MODEL_CLASS_MAPPING`` resolving to ``EmotionModel``;
- ``resolve_task`` honouring the build variant's underscore ``model_type``
  (``wav2vec2_emotion_regression``) via the ``_`` -> ``-`` normalization the
  mapping key depends on;
- ``Wav2Vec2EmotionRegressionIOConfig`` registering for the
  ``audio-classification`` ONNX export with the expected input/output axes.

A rename of ``EMOTION_REGRESSION_MODEL_TYPE`` or a resolver normalization change
would otherwise break routing with nothing failing in CI.
"""

from __future__ import annotations

import pytest
from optimum.exporters.tasks import TasksManager
from optimum.utils.input_generators import DummyAudioInputGenerator
from transformers import Wav2Vec2Config

from winml.modelkit.loader import resolve_task
from winml.modelkit.models.hf import MODEL_CLASS_MAPPING
from winml.modelkit.models.hf.wav2vec2 import (
    EMOTION_REGRESSION_MODEL_TYPE,
    EmotionModel,
    Wav2Vec2EmotionRegressionIOConfig,
)
from winml.modelkit.models.hf.wav2vec2 import MODEL_CLASS_MAPPING as WAV2VEC2_MAPPING


# =============================================================================
# Test Constants
# =============================================================================

TASK = "audio-classification"
MAPPING_KEY = ("wav2vec2-emotion-regression", TASK)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def emotion_config():
    """Minimal Wav2Vec2Config exercising the emotion-regression head dimensions."""
    return Wav2Vec2Config(hidden_size=16, num_labels=3)


# =============================================================================
# MODEL_CLASS_MAPPING — routing to the custom regression wrapper
# =============================================================================


class TestWav2Vec2EmotionModelClassMapping:
    """The emotion-regression variant routes to EmotionModel."""

    def test_mapping_entry_registered(self):
        """The aggregated mapping exposes the emotion-regression entry."""
        assert MAPPING_KEY in MODEL_CLASS_MAPPING
        assert MODEL_CLASS_MAPPING[MAPPING_KEY] is EmotionModel

    def test_module_mapping_merged_into_aggregate(self):
        """The module-level mapping is included in the aggregated mapping."""
        assert WAV2VEC2_MAPPING.items() <= MODEL_CLASS_MAPPING.items()


# =============================================================================
# resolve_task — underscore model_type normalization contract
# =============================================================================


class TestWav2Vec2EmotionTaskResolution:
    """resolve_task honours the underscore build-variant model_type."""

    def test_model_type_override_resolves_emotion_model(self, emotion_config):
        """An underscore model_type override normalizes and resolves EmotionModel."""
        resolution = resolve_task(
            emotion_config,
            task=TASK,
            model_class="EmotionModel",
            model_type_override=EMOTION_REGRESSION_MODEL_TYPE,
        )

        assert resolution.task == TASK
        assert resolution.model_class is EmotionModel

    def test_registered_model_type_uses_underscores(self):
        """The registered model_type uses underscores; the mapping key uses dashes."""
        assert EMOTION_REGRESSION_MODEL_TYPE == "wav2vec2_emotion_regression"
        assert EMOTION_REGRESSION_MODEL_TYPE.replace("_", "-") == MAPPING_KEY[0]


# =============================================================================
# Wav2Vec2EmotionRegressionIOConfig — registration and I/O axes
# =============================================================================


class TestWav2Vec2EmotionRegressionIOConfig:
    """The ONNX export config is registered with the expected axes."""

    def test_onnx_config_registered(self):
        """Config is registered with TasksManager for audio-classification."""
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type=EMOTION_REGRESSION_MODEL_TYPE,
            exporter="onnx",
            task=TASK,
            library_name="transformers",
        )
        assert config_cls.func.__name__ == Wav2Vec2EmotionRegressionIOConfig.__name__

    def test_inputs_axes(self, emotion_config):
        """Inputs expose input_values with batch and audio-length dynamic axes."""
        io_config = Wav2Vec2EmotionRegressionIOConfig(emotion_config, task=TASK)

        assert io_config.inputs == {
            "input_values": {0: "batch_size", 1: "audio_sequence_length"},
        }

    def test_outputs_axes(self, emotion_config):
        """Outputs expose hidden_states and logits with a batch dynamic axis."""
        io_config = Wav2Vec2EmotionRegressionIOConfig(emotion_config, task=TASK)

        assert io_config.outputs == {
            "hidden_states": {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    def test_dummy_input_generator_class(self):
        """Uses DummyAudioInputGenerator for raw-waveform dummy inputs."""
        assert (
            DummyAudioInputGenerator,
        ) == Wav2Vec2EmotionRegressionIOConfig.DUMMY_INPUT_GENERATOR_CLASSES
