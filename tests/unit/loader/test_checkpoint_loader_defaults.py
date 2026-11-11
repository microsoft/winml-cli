# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Checkpoint loader identity must not depend on performance recipes."""

from unittest.mock import patch

import pytest
from transformers import Wav2Vec2Config

from winml.modelkit.loader import resolve_loader_config, resolve_task


MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


@pytest.mark.parametrize("task", [None, "audio-classification"])
def test_checkpoint_loader_defaults_resolve_custom_head(task):
    config = Wav2Vec2Config(
        architectures=["Wav2Vec2ForSpeechClassification"], problem_type="regression"
    )
    config._name_or_path = MODEL_ID
    with patch("winml.modelkit.loader._autoconfig.load_hf_config", return_value=config):
        loader, resolved_config, cls, resolution = resolve_loader_config(MODEL_ID, task=task)
    assert loader.task == "audio-classification"
    assert loader.model_type == "wav2vec2_emotion_regression"
    assert cls.__name__ == "EmotionModel"
    assert resolution.model_class is cls
    assert resolved_config.model_type == "wav2vec2"
    direct = resolve_task(config, task=task)
    assert direct.task == loader.task
    assert direct.model_class is cls


def test_explicit_other_task_does_not_select_checkpoint_head():
    config = Wav2Vec2Config(architectures=["Wav2Vec2ForCTC"])
    config._name_or_path = MODEL_ID
    result = resolve_task(config, task="automatic-speech-recognition")
    assert result.model_class.__name__ != "EmotionModel"


def test_unregistered_checkpoint_does_not_select_custom_head():
    config = Wav2Vec2Config(architectures=["Wav2Vec2ForSequenceClassification"])
    config._name_or_path = "another/checkpoint"
    result = resolve_task(config, task="audio-classification")
    assert result.model_class.__name__ != "EmotionModel"


def test_explicit_model_type_keeps_native_loader():
    config = Wav2Vec2Config(architectures=["Wav2Vec2ForSequenceClassification"])
    config._name_or_path = MODEL_ID
    loader, _, cls, _ = resolve_loader_config(
        MODEL_ID, hf_config=config, model_type="wav2vec2", task="audio-classification"
    )
    assert loader.model_type == "wav2vec2"
    assert cls.__name__ != "EmotionModel"


def test_generated_build_config_keeps_default_optimizations():
    from winml.modelkit.config import generate_hf_build_config

    config = Wav2Vec2Config(
        architectures=["Wav2Vec2ForSpeechClassification"], problem_type="regression"
    )
    config._name_or_path = MODEL_ID
    with patch("winml.modelkit.loader._autoconfig.load_hf_config", return_value=config):
        build = generate_hf_build_config(MODEL_ID, device="cpu", ep="cpu")
    assert build.loader.model_class == "EmotionModel"
    assert build.loader.task == "audio-classification"
    assert build.loader.model_type == "wav2vec2_emotion_regression"
    assert build.optim.to_dict() == {}
    assert [tensor.name for tensor in build.export.input_tensors] == ["input_values"]
    assert [tensor.name for tensor in build.export.output_tensors] == ["hidden_states", "logits"]


def test_preloaded_config_preserves_explicit_native_type():
    config = Wav2Vec2Config(architectures=["Wav2Vec2ForSequenceClassification"])
    config._name_or_path = MODEL_ID
    loader, _, cls, _ = resolve_loader_config(
        hf_config=config, model_type="wav2vec2", task="audio-classification"
    )
    assert loader.model_type == "wav2vec2"
    assert cls.__name__ != "EmotionModel"


def test_automatic_loader_provenance_is_not_user_task():
    config = Wav2Vec2Config(
        architectures=["Wav2Vec2ForSpeechClassification"], problem_type="regression"
    )
    config._name_or_path = MODEL_ID
    _, _, _, resolution = resolve_loader_config(MODEL_ID, hf_config=config)
    assert resolution.source.value == "architecture-default"


def test_direct_load_preserves_explicit_model_type():
    from winml.modelkit.loader import load_hf_model

    config = Wav2Vec2Config(architectures=["Wav2Vec2ForSequenceClassification"])
    config._name_or_path = MODEL_ID
    with (
        patch(
            "winml.modelkit.loader.resolution.resolve_task", side_effect=RuntimeError("stop")
        ) as resolve,
        pytest.raises(RuntimeError, match="stop"),
    ):
        load_hf_model(
            MODEL_ID, hf_config=config, model_type="wav2vec2", task="audio-classification"
        )
    assert resolve.call_args.kwargs["model_type_override"] == "wav2vec2"


@pytest.mark.parametrize("identity", ["other/renamed", "C:/models/local-copy"])
def test_architecture_resolution_is_independent_of_model_id(identity):
    config = Wav2Vec2Config(
        architectures=["Wav2Vec2ForSpeechClassification"], problem_type="regression"
    )
    config._name_or_path = identity
    loader, _, cls, _ = resolve_loader_config(identity, hf_config=config)
    assert cls.__name__ == "EmotionModel"
    assert loader.model_type == "wav2vec2_emotion_regression"


def test_registered_architecture_rejects_incompatible_config():
    config = Wav2Vec2Config(
        architectures=["Wav2Vec2ForSpeechClassification"],
        problem_type="single_label_classification",
    )
    with pytest.raises(ValueError, match="incompatible"):
        resolve_loader_config("other/checkpoint", hf_config=config)


@pytest.mark.parametrize(
    "field", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
)
def test_custom_architecture_rejects_incompatible_weights(field):
    from unittest.mock import MagicMock

    from winml.modelkit.loader import load_hf_model
    from winml.modelkit.models.hf.wav2vec2 import EmotionModel

    config = Wav2Vec2Config(
        architectures=["Wav2Vec2ForSpeechClassification"], problem_type="regression"
    )
    info = {
        name: [] for name in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
    }
    info[field] = ["incompatible.weight"]
    with (
        patch.object(EmotionModel, "from_pretrained", return_value=(MagicMock(), info)) as load,
        pytest.raises(ValueError, match="incompatible checkpoint"),
    ):
        load_hf_model("renamed/model", hf_config=config)
    assert load.call_args.kwargs["output_loading_info"] is True


def test_custom_architecture_accepts_complete_weight_load():
    from unittest.mock import MagicMock

    from winml.modelkit.loader import load_hf_model
    from winml.modelkit.models.hf.wav2vec2 import EmotionModel

    config = Wav2Vec2Config(
        architectures=["Wav2Vec2ForSpeechClassification"], problem_type="regression"
    )
    model = MagicMock()
    info = {
        name: [] for name in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
    }
    with patch.object(EmotionModel, "from_pretrained", return_value=(model, info)):
        actual, _, task = load_hf_model("local-copy", hf_config=config)
    assert actual is model
    assert task == "audio-classification"


def test_explicit_task_bypasses_ambiguous_architecture_default():
    config = Wav2Vec2Config(
        architectures=["Wav2Vec2ForSpeechClassification", "Wav2Vec2ForCTC"],
        problem_type="regression",
    )
    result = resolve_task(config, task="automatic-speech-recognition")
    assert result.model_class.__name__ != "EmotionModel"


def test_ambiguous_architecture_requires_explicit_choice():
    config = Wav2Vec2Config(
        architectures=["Wav2Vec2ForSpeechClassification", "Wav2Vec2ForCTC"],
        problem_type="regression",
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_task(config)


def test_explicit_native_type_bypasses_registered_architecture():
    config = Wav2Vec2Config(
        architectures=["Wav2Vec2ForSpeechClassification"], problem_type="regression"
    )
    result = resolve_task(config, task="audio-classification", model_type_override="wav2vec2")
    assert result.model_class.__name__ != "EmotionModel"
