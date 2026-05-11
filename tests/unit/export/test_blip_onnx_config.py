# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for BLIP split image-to-text export.

Verifies the encoder (``feature-extraction``) and decoder
(``text2text-generation``) IOConfigs are registered, produce the expected
I/O names, and generate correctly-shaped dummy inputs.

See also: modelkit/models/hf/blip.py
"""

from __future__ import annotations

import pytest
from optimum.exporters.tasks import TasksManager

from winml.modelkit.export import generate_dummy_inputs, resolve_io_specs
from winml.modelkit.models import HF_MODEL_CLASS_MAPPING  # registers OnnxConfigs (side-effect)


@pytest.fixture(scope="module")
def blip_config():
    """Minimal ``BlipConfig`` for testing (nested vision_config + text_config)."""
    from transformers import BlipConfig

    return BlipConfig(
        vision_config={
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 128,
        },
        text_config={
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 128,
            "max_position_embeddings": 32,
        },
    )


class TestBlipRegistration:
    def test_class_mapping(self) -> None:
        """BLIP wrapper classes appear in the aggregated HF MODEL_CLASS_MAPPING."""
        assert ("blip", "feature-extraction") in HF_MODEL_CLASS_MAPPING
        assert ("blip", "text2text-generation") in HF_MODEL_CLASS_MAPPING

    def test_encoder_config_registered(self) -> None:
        c = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model_type="blip",
            task="feature-extraction",
            library_name="transformers",
        )
        assert c.func.__name__ == "BlipVisionEncoderIOConfig"

    def test_decoder_config_registered(self) -> None:
        c = TasksManager.get_exporter_config_constructor(
            exporter="onnx",
            model_type="blip",
            task="text2text-generation",
            library_name="transformers",
        )
        assert c.func.__name__ == "BlipDecoderIOConfig"

    def test_composite_registered(self) -> None:
        from winml.modelkit.models.winml.composite_model import COMPOSITE_MODEL_REGISTRY

        assert ("blip", "image-to-text") in COMPOSITE_MODEL_REGISTRY
        cls = COMPOSITE_MODEL_REGISTRY[("blip", "image-to-text")]
        assert cls.__name__ == "WinMLBlipImageToText"


class TestBlipEncoderIO:
    def test_input_is_pixel_values_only(self, blip_config) -> None:
        specs = resolve_io_specs("blip", "feature-extraction", blip_config)
        assert specs["input_names"] == ["pixel_values"]

    def test_output_is_encoder_hidden_states(self, blip_config) -> None:
        specs = resolve_io_specs("blip", "feature-extraction", blip_config)
        assert specs["output_names"] == ["encoder_hidden_states"]

    def test_pixel_values_shape(self, blip_config) -> None:
        inputs = generate_dummy_inputs("blip", "feature-extraction", blip_config)
        shape = inputs["pixel_values"].shape
        assert shape[1] == blip_config.vision_config.num_channels
        assert shape[2] == blip_config.vision_config.image_size
        assert shape[3] == blip_config.vision_config.image_size


class TestBlipDecoderIO:
    def test_inputs_include_kv_per_layer(self, blip_config) -> None:
        specs = resolve_io_specs("blip", "text2text-generation", blip_config)
        names = specs["input_names"]
        for required in ("decoder_input_ids", "encoder_hidden_states",
                         "decoder_attention_mask", "cache_position"):
            assert required in names
        n_layers = blip_config.text_config.num_hidden_layers
        for i in range(n_layers):
            assert f"past_{i}_key" in names
            assert f"past_{i}_value" in names

    def test_outputs_include_present_kv_per_layer(self, blip_config) -> None:
        specs = resolve_io_specs("blip", "text2text-generation", blip_config)
        names = specs["output_names"]
        assert names[0] == "logits"
        n_layers = blip_config.text_config.num_hidden_layers
        for i in range(n_layers):
            assert f"present_{i}_key" in names
            assert f"present_{i}_value" in names

    def test_encoder_hidden_states_uses_vision_sequence(self, blip_config) -> None:
        inputs = generate_dummy_inputs("blip", "text2text-generation", blip_config)
        enc_h = inputs["encoder_hidden_states"]
        vc = blip_config.vision_config
        expected_seq = (vc.image_size // vc.patch_size) ** 2 + 1
        assert enc_h.shape[1] == expected_seq
        assert enc_h.shape[2] == vc.hidden_size

    def test_past_kv_is_full_buffer(self, blip_config) -> None:
        inputs = generate_dummy_inputs("blip", "text2text-generation", blip_config)
        tc = blip_config.text_config
        expected_head_dim = tc.hidden_size // tc.num_attention_heads
        expected = (1, tc.num_attention_heads, tc.max_position_embeddings, expected_head_dim)
        assert tuple(inputs["past_0_key"].shape) == expected
        assert tuple(inputs["past_0_value"].shape) == expected
