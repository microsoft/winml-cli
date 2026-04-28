# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for vision-encoder-decoder split image-to-text export.

Covers the export surface shared by TrOCR / Donut / Nougat / ViT-GPT2.

See also: modelkit/models/hf/vision_encoder_decoder.py
"""

from __future__ import annotations

import pytest
from optimum.exporters.tasks import TasksManager

import winml.modelkit.models  # noqa: F401 — triggers OnnxConfig registration
from winml.modelkit.export import generate_dummy_inputs, resolve_io_specs


@pytest.fixture(scope="module")
def ved_config():
    """Minimal ``VisionEncoderDecoderConfig`` (ViT encoder + TrOCR decoder)."""
    from transformers import TrOCRConfig, VisionEncoderDecoderConfig, ViTConfig

    encoder = ViTConfig(
        image_size=32,
        patch_size=8,
        num_channels=3,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
    )
    decoder = TrOCRConfig(
        vocab_size=100,
        d_model=64,
        decoder_layers=2,
        decoder_attention_heads=2,
        cross_attention_hidden_size=64,
        max_position_embeddings=32,
    )
    return VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)


class TestVEDRegistration:
    def test_encoder_config_registered(self) -> None:
        c = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model_type="vision-encoder-decoder",
            task="feature-extraction",
            library_name="transformers",
        )
        assert c.func.__name__ == "VisionEncoderIOConfig"

    def test_decoder_config_registered(self) -> None:
        c = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model_type="vision-encoder-decoder",
            task="text2text-generation",
            library_name="transformers",
        )
        assert c.func.__name__ == "VisionDecoderIOConfig"

    def test_composite_registered(self) -> None:
        from winml.modelkit.models.winml.composite_model import COMPOSITE_MODEL_REGISTRY

        assert ("vision-encoder-decoder", "image-to-text") in COMPOSITE_MODEL_REGISTRY
        cls = COMPOSITE_MODEL_REGISTRY[("vision-encoder-decoder", "image-to-text")]
        assert cls.__name__ == "WinMLVEDImageToText"


class TestVEDEncoderIO:
    def test_input_is_pixel_values_only(self, ved_config) -> None:
        specs = resolve_io_specs("vision-encoder-decoder", "feature-extraction", ved_config)
        assert specs["input_names"] == ["pixel_values"]

    def test_output_is_encoder_hidden_states(self, ved_config) -> None:
        specs = resolve_io_specs("vision-encoder-decoder", "feature-extraction", ved_config)
        assert specs["output_names"] == ["encoder_hidden_states"]

    def test_pixel_values_shape(self, ved_config) -> None:
        inputs = generate_dummy_inputs("vision-encoder-decoder", "feature-extraction", ved_config)
        shape = inputs["pixel_values"].shape
        assert shape[1] == ved_config.encoder.num_channels
        assert shape[2] == ved_config.encoder.image_size
        assert shape[3] == ved_config.encoder.image_size


class TestVEDDecoderIO:
    def test_inputs_include_kv_per_layer(self, ved_config) -> None:
        specs = resolve_io_specs("vision-encoder-decoder", "text2text-generation", ved_config)
        names = specs["input_names"]
        for required in ("decoder_input_ids", "encoder_hidden_states",
                         "decoder_attention_mask", "cache_position"):
            assert required in names
        n_layers = ved_config.decoder.num_hidden_layers
        for i in range(n_layers):
            assert f"past_{i}_key" in names
            assert f"past_{i}_value" in names

    def test_outputs_include_present_kv_per_layer(self, ved_config) -> None:
        specs = resolve_io_specs("vision-encoder-decoder", "text2text-generation", ved_config)
        names = specs["output_names"]
        assert names[0] == "logits"
        n_layers = ved_config.decoder.num_hidden_layers
        for i in range(n_layers):
            assert f"present_{i}_key" in names
            assert f"present_{i}_value" in names

    def test_encoder_hidden_states_uses_vision_sequence(self, ved_config) -> None:
        inputs = generate_dummy_inputs("vision-encoder-decoder", "text2text-generation", ved_config)
        enc_h = inputs["encoder_hidden_states"]
        ec = ved_config.encoder
        expected_seq = (ec.image_size // ec.patch_size) ** 2 + 1
        assert enc_h.shape[1] == expected_seq
        assert enc_h.shape[2] == ec.hidden_size

    def test_past_kv_is_full_buffer(self, ved_config) -> None:
        inputs = generate_dummy_inputs("vision-encoder-decoder", "text2text-generation", ved_config)
        dc = ved_config.decoder
        expected_head_dim = dc.hidden_size // dc.num_attention_heads
        expected = (1, dc.num_attention_heads, dc.max_position_embeddings, expected_head_dim)
        assert tuple(inputs["past_0_key"].shape) == expected
        assert tuple(inputs["past_0_value"].shape) == expected
