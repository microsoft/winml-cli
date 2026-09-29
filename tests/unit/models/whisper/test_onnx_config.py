# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for split Whisper encoder/decoder support."""

from __future__ import annotations

from types import SimpleNamespace

from optimum.exporters.tasks import TasksManager
from transformers import GenerationConfig, WhisperConfig
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin

from winml.modelkit.loader.resolution import (
    _composite_components_for_task,
    resolve_composite,
)
from winml.modelkit.models.hf.whisper import (
    WhisperDecoderIOConfig,
    WhisperEncoderInputGenerator,
    WhisperEncoderIOConfig,
    WinMLWhisperModel,
    _WhisperDecoderNormalizedConfig,
)
from winml.modelkit.models.winml.composite_model import COMPOSITE_MODEL_REGISTRY
from winml.modelkit.models.winml.encoder_decoder import EncoderDecoderInputGenerator
from winml.modelkit.models.winml.kv_cache import PastKeyValueInputGenerator, WinMLStaticCache


def _config() -> WhisperConfig:
    return WhisperConfig(
        vocab_size=128,
        num_mel_bins=80,
        d_model=32,
        encoder_layers=2,
        decoder_layers=3,
        encoder_attention_heads=2,
        decoder_attention_heads=4,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_source_positions=20,
        max_target_positions=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )


def test_encoder_registration_and_contract() -> None:
    constructor = TasksManager.get_exporter_config_constructor(
        model_type="whisper",
        exporter="onnx",
        task="feature-extraction",
        library_name="transformers",
    )
    assert constructor.func is WhisperEncoderIOConfig
    onnx_config = WhisperEncoderIOConfig(_config(), task="feature-extraction")
    assert onnx_config.inputs == {"input_features": {0: "batch_size"}}
    assert onnx_config.outputs == {"encoder_hidden_states": {0: "batch_size"}}


def test_encoder_dummy_input_uses_fixed_audio_geometry() -> None:
    onnx_config = WhisperEncoderIOConfig(_config(), task="feature-extraction")
    inputs = onnx_config.generate_dummy_inputs(framework="pt")
    assert inputs["input_features"].shape == (1, 80, 40)
    assert (WhisperEncoderInputGenerator,) == WhisperEncoderIOConfig.DUMMY_INPUT_GENERATOR_CLASSES


def test_decoder_contract_uses_decoder_dimensions() -> None:
    config = _config()
    normalized = _WhisperDecoderNormalizedConfig(config)
    assert normalized.num_layers == 3
    assert normalized.num_attention_heads == 4
    assert normalized.head_dim == 8
    assert normalized.max_cache_len == 16
    assert normalized.sequence_length == 20

    onnx_config = WhisperDecoderIOConfig(config, task="text2text-generation")
    assert (
        EncoderDecoderInputGenerator,
        PastKeyValueInputGenerator,
    ) == WhisperDecoderIOConfig.DUMMY_INPUT_GENERATOR_CLASSES
    assert set(onnx_config.inputs).issuperset(
        {
            "decoder_input_ids",
            "encoder_hidden_states",
            "decoder_attention_mask",
            "cache_position",
            "past_0_key",
            "past_2_value",
        }
    )
    assert "past_3_key" not in onnx_config.inputs
    assert set(onnx_config.outputs).issuperset({"logits", "present_0_key", "present_2_value"})


def test_asr_composite_registration_and_detection_bridge() -> None:
    expected = {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }
    assert (
        COMPOSITE_MODEL_REGISTRY[("whisper", "automatic-speech-recognition")] is WinMLWhisperModel
    )
    assert resolve_composite("whisper", "automatic-speech-recognition") == expected
    assert _composite_components_for_task("whisper", "automatic-speech-recognition") == expected
    assert WinMLWhisperModel.main_input_name == "input_features"
    assert WinMLWhisperModel.get_cache_class() is WinMLStaticCache
    assert issubclass(WinMLWhisperModel, WhisperGenerationMixin)
    assert WinMLWhisperModel.generate is WhisperGenerationMixin.generate


def test_generation_config_preserves_whisper_tokens() -> None:
    model = object.__new__(WinMLWhisperModel)
    model.config = _config()
    model._max_dec = 16
    generation_config = model.generation_config
    assert generation_config.decoder_start_token_id == 1
    assert generation_config.eos_token_id == 2
    assert generation_config.max_new_tokens == 15


def test_runtime_loads_generation_metadata_and_derives_input_stride(monkeypatch) -> None:
    config = _config()
    config._name_or_path = "test/whisper"
    loaded_generation_config = GenerationConfig(decoder_start_token_id=7)
    monkeypatch.setattr(
        GenerationConfig,
        "from_pretrained",
        lambda model_name_or_path: loaded_generation_config,
    )
    encoder = SimpleNamespace(
        io_config={
            "input_names": ["input_features"],
            "input_shapes": [[1, 80, 40]],
        }
    )
    decoder = SimpleNamespace(
        io_config={
            "input_names": ["decoder_input_ids", "past_0_key"],
            "input_shapes": [[1, 1], [1, 4, 16, 8]],
            "input_types": ["int64", "float32"],
        }
    )

    model = WinMLWhisperModel({"encoder": encoder, "decoder": decoder}, config)

    assert model.generation_config is loaded_generation_config
    assert model.model.encoder.conv1.stride == (1,)
    assert model.model.encoder.conv2.stride == (2,)
