# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for native Florence-2 split image-to-text export."""

from unittest.mock import patch

import torch
from optimum.exporters.tasks import TasksManager
from transformers import Florence2Config

from winml.modelkit.export import generate_dummy_inputs, resolve_io_specs
from winml.modelkit.export.config import _resolve_export_config_from_specs
from winml.modelkit.loader.resolution import (
    resolve_composite,
    resolve_composite_components,
    resolve_composite_precision_overrides,
    resolve_task,
)
from winml.modelkit.models import HF_MODEL_CLASS_MAPPING
from winml.modelkit.models.hf.florence2 import (
    Florence2DecoderWrapper,
    WinMLFlorence2ImageToText,
    _legacy_florence2_weight_conversions,
    _load_florence2_model,
    _WinMLFlorence2ForConditionalGeneration,
)


def _convert_legacy_key(key: str) -> str:
    for conversion in _legacy_florence2_weight_conversions():
        key, _ = conversion.rename_source_key(key)
    return key


def test_florence2_legacy_checkpoint_key_conversion() -> None:
    assert _convert_legacy_key("vision_tower.convs.0.proj.weight") == (
        "model.vision_tower.convs.0.conv.weight"
    )
    assert _convert_legacy_key(
        "vision_tower.blocks.0.0.spatial_block.window_attn.fn.qkv.weight"
    ) == "model.vision_tower.blocks.0.0.spatial_block.window_attn.qkv.weight"
    assert _convert_legacy_key(
        "vision_tower.blocks.0.0.channel_block.ffn.norm.weight"
    ) == "model.vision_tower.blocks.0.0.channel_block.norm2.weight"
    assert _convert_legacy_key("language_model.model.encoder.layers.0.fc1.weight") == (
        "model.language_model.encoder.layers.0.fc1.weight"
    )
    assert _convert_legacy_key("image_pos_embed.row_embeddings.weight") == (
        "model.multi_modal_projector.image_position_embed.row_embeddings.weight"
    )
    assert _convert_legacy_key("image_projection") == (
        "model.multi_modal_projector.image_projection.weight"
    )


def test_florence2_legacy_checkpoint_loads_float32_source_model() -> None:
    config = Florence2Config()
    config.image_token_id = 50265
    config.text_config.vocab_size = 51289
    expected_model = object()
    loading_info = {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }

    with patch.object(
        _WinMLFlorence2ForConditionalGeneration,
        "from_pretrained",
        return_value=(expected_model, loading_info),
    ) as from_pretrained:
        model = _load_florence2_model("florence-checkpoint", config=config)

    assert model is expected_model
    assert from_pretrained.call_args.kwargs["dtype"] is torch.float32


def test_build_config_roundtrips_explicit_precision() -> None:
    from winml.modelkit.config import WinMLBuildConfig

    config = WinMLBuildConfig(precision="fp32")

    assert config.to_dict()["precision"] == "fp32"
    assert WinMLBuildConfig.from_dict(config.to_dict()).precision == "fp32"


def test_florence2_decoder_wrapper_initializes_shared_export_state() -> None:
    config = Florence2Config()
    loaded_model = torch.nn.Module()
    loaded_model.config = config

    with patch(
        "winml.modelkit.models.hf.florence2._load_florence2_model",
        return_value=loaded_model,
    ):
        wrapper = Florence2DecoderWrapper.from_pretrained("florence-checkpoint")

    assert wrapper.model is loaded_model
    assert wrapper.config is config
    assert wrapper.num_layers == config.text_config.decoder_layers
    assert wrapper.training is False


def test_florence2_family_registrations() -> None:
    config = Florence2Config()
    config._name_or_path = "microsoft/Florence-2-base"

    assert ("florence2", None) in HF_MODEL_CLASS_MAPPING
    assert ("florence2", "image-text-to-text") in HF_MODEL_CLASS_MAPPING
    assert ("florence2", "image-to-text") in HF_MODEL_CLASS_MAPPING
    assert ("florence2", "feature-extraction") in HF_MODEL_CLASS_MAPPING
    assert ("florence2", "text2text-generation") in HF_MODEL_CLASS_MAPPING
    assert resolve_composite("florence2", "image-text-to-text") == {
        "encoder": "image-feature-extraction",
        "decoder": "text2text-generation",
    }
    assert resolve_composite_components(None, model_type="florence2") == {
        "encoder": "image-feature-extraction",
        "decoder": "text2text-generation",
    }
    assert resolve_composite_precision_overrides(
        "florence2",
        {
            "encoder": "image-feature-extraction",
            "decoder": "text2text-generation",
        },
    ) == {"encoder": "fp32"}
    assert resolve_task(config).task == "image-to-text"

    encoder = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model_type="florence2",
        task="feature-extraction",
        library_name="transformers",
    )
    decoder = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model_type="florence2",
        task="text2text-generation",
        library_name="transformers",
    )

    assert encoder.func.__name__ == "Florence2EncoderIOConfig"
    assert decoder.func.__name__ == "Florence2DecoderIOConfig"


def test_florence2_encoder_preserves_image_placeholder_tokens() -> None:
    config = Florence2Config()
    generated = generate_dummy_inputs("florence2", "feature-extraction", config)
    specs = resolve_io_specs("florence2", "feature-extraction", config)
    export_config = _resolve_export_config_from_specs("florence2", "feature-extraction", config)

    assert generated["input_ids"].shape == (1, 585)
    assert torch.count_nonzero(generated["input_ids"] == 50265) == 577
    assert generated["input_ids"][0, 577:].tolist() == [0, 2264, 473, 5, 2274, 6190, 116, 2]
    assert specs["dummy_value_runs"]["input_ids"][0] == (577, 50265)
    assert torch.equal(export_config.generate_dummy_inputs()["input_ids"], generated["input_ids"])


def test_florence2_decoder_declares_six_cache_layers() -> None:
    config = Florence2Config(
        text_config={
            "d_model": 768,
            "decoder_attention_heads": 12,
            "decoder_layers": 6,
            "vocab_size": 51289,
        }
    )
    specs = resolve_io_specs("florence2", "text2text-generation", config)
    generated = generate_dummy_inputs("florence2", "text2text-generation", config)

    assert specs["input_names"][:4] == [
        "decoder_input_ids",
        "encoder_hidden_states",
        "decoder_attention_mask",
        "cache_position",
    ]
    assert [name for name in specs["input_names"] if name.endswith("_key")] == [
        f"past_{index}_key" for index in range(6)
    ]
    assert generated["encoder_hidden_states"].shape == (1, 585, 768)
    assert specs["output_names"][0] == "logits"


def test_florence2_generation_keeps_encoder_prompt_out_of_decoder_history() -> None:
    model = object.__new__(WinMLFlorence2ImageToText)
    model.config = Florence2Config()
    model.config.is_encoder_decoder = True
    encoder_input_ids = torch.full((1, 585), 50265, dtype=torch.long)
    pixel_values = torch.zeros(1, 3, 768, 768)

    inputs, input_name, model_kwargs = model._prepare_model_inputs(
        None,
        torch.tensor(model.config.text_config.bos_token_id),
        {
            "input_ids": encoder_input_ids,
            "pixel_values": pixel_values,
        },
    )
    decoder_input_ids, model_kwargs = model._prepare_decoder_input_ids_for_generation(
        batch_size=1,
        model_input_name=input_name,
        model_kwargs=model_kwargs,
        decoder_start_token_id=torch.tensor(
            model.config.text_config.decoder_start_token_id
        ),
        device=encoder_input_ids.device,
    )

    assert input_name == "input_ids"
    assert inputs is encoder_input_ids
    assert model_kwargs["pixel_values"] is pixel_values
    assert decoder_input_ids.tolist() == [[model.config.text_config.decoder_start_token_id]]


def test_full_static_cache_output_advances_by_logical_query_length() -> None:
    from transformers import PretrainedConfig

    from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

    config = PretrainedConfig()
    config.num_hidden_layers = 1
    cache = WinMLStaticCache.create(config, [1, 12, 1024, 64], torch.float32)
    present_key = torch.ones(1, 12, 1024, 64)
    present_value = torch.full_like(present_key, 2)

    cache.update_all_layers(
        {
            "logits": torch.zeros(1, 1, 10),
            "present_0_key": present_key,
            "present_0_value": present_value,
        }
    )

    assert cache.step == 1
    assert cache.get_seq_length() == 1
    assert torch.equal(cache._layer(0).keys, present_key)
    assert torch.equal(cache._layer(0).values, present_value)


def test_static_cache_restores_exported_logical_position() -> None:
    from transformers import PretrainedConfig

    from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

    config = PretrainedConfig()
    config.num_hidden_layers = 1
    cache = WinMLStaticCache.create(config, [1, 12, 1024, 64], torch.float32)
    cache.set_trace_position(torch.tensor([7], dtype=torch.int64))

    assert cache.get_seq_length() == 7
