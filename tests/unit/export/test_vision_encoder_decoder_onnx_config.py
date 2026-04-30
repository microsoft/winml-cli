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

from winml.modelkit.export import generate_dummy_inputs, resolve_io_specs
from winml.modelkit.models import HF_MODEL_CLASS_MAPPING  # registers OnnxConfigs (side-effect)


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
    def test_class_mapping(self) -> None:
        """VED wrapper classes appear in the aggregated HF MODEL_CLASS_MAPPING."""
        assert ("vision-encoder-decoder", "feature-extraction") in HF_MODEL_CLASS_MAPPING
        assert ("vision-encoder-decoder", "text2text-generation") in HF_MODEL_CLASS_MAPPING

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


@pytest.fixture(scope="module")
def swin_mbart_ved_config():
    """Donut/Nougat-style config: Swin (HxW image_size) encoder + MBart decoder.

    MBart's ``num_hidden_layers`` aliases the encoder side (12) while
    ``decoder_layers`` is the authoritative decoder count (4) — this
    asymmetry is what the ``num_layers`` fallback logic addresses.
    """
    from transformers import MBartConfig, VisionEncoderDecoderConfig
    from transformers.models.donut.configuration_donut_swin import DonutSwinConfig

    encoder = DonutSwinConfig(
        image_size=[64, 32],          # rectangular HxW
        patch_size=4,
        num_channels=3,
        embed_dim=16,
        depths=[2, 2, 2, 2],          # 4 stages → shrink = 2**3 = 8
        num_heads=[1, 2, 4, 8],
        window_size=4,
    )
    decoder = MBartConfig(
        vocab_size=100,
        d_model=64,
        encoder_layers=12,            # mbart's encoder side (irrelevant for VED)
        decoder_layers=4,             # the authoritative decoder count
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        max_position_embeddings=32,
    )
    return VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)


class TestVedDecoderNormalizedConfig:
    """Targeted tests for ``_VedDecoderNormalizedConfig`` dispatch logic.

    Covers two pieces of dispatch:
    - ``num_layers``: prefer ``decoder.decoder_layers`` when present,
      else fall back to Optimum's per-family ``num_layers``.
    - ``encoder_seq_length``: scalar (square ViT) vs ``[H, W]``
      (hierarchical Swin) image_size.
    """

    def test_num_layers_prefers_decoder_layers_for_bart_family(self, swin_mbart_ved_config) -> None:
        """MBart exposes both ``decoder_layers`` (4) and ``num_hidden_layers``
        (=encoder_layers, 12).  We must pick the former."""
        from winml.modelkit.models.hf.vision_encoder_decoder import _VedDecoderNormalizedConfig

        nc = _VedDecoderNormalizedConfig(swin_mbart_ved_config)
        # Sanity-check the asymmetry our override is meant to handle.
        assert swin_mbart_ved_config.decoder.decoder_layers == 4
        assert swin_mbart_ved_config.decoder.num_hidden_layers == 12
        # The override picks decoder_layers, not num_hidden_layers.
        assert nc.num_layers == 4

    def test_num_layers_falls_back_to_optimum_when_no_decoder_layers(self) -> None:
        """Non-BART decoder (BERT) has no ``decoder_layers`` field at all;
        the override must fall back to Optimum's NormalizedConfig.num_layers,
        which for BERT reads ``num_hidden_layers``.
        """
        from transformers import BertConfig, VisionEncoderDecoderConfig, ViTConfig

        from winml.modelkit.models.hf.vision_encoder_decoder import _VedDecoderNormalizedConfig

        encoder = ViTConfig(
            image_size=32, patch_size=8, num_channels=3,
            hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        )
        decoder = BertConfig(
            vocab_size=100, hidden_size=64, num_hidden_layers=3,
            num_attention_heads=2, intermediate_size=128,
            max_position_embeddings=32, is_decoder=True,
        )
        cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)
        # Sanity: BertConfig has no ``decoder_layers`` attribute.
        assert not hasattr(cfg.decoder, "decoder_layers")
        nc = _VedDecoderNormalizedConfig(cfg)
        # Falls back to Optimum's BertNormalizedConfig.num_layers = num_hidden_layers.
        assert nc.num_layers == 3

    def test_encoder_seq_length_scalar_image_size(self, ved_config) -> None:
        """Scalar image_size: ``(image_size / patch_size)**2 + 1`` (CLS token)."""
        from winml.modelkit.models.hf.vision_encoder_decoder import _VedDecoderNormalizedConfig

        nc = _VedDecoderNormalizedConfig(ved_config)
        ec = ved_config.encoder
        assert nc.encoder_seq_length == (ec.image_size // ec.patch_size) ** 2 + 1

    def test_encoder_seq_length_hxw_image_size_with_depths(self, swin_mbart_ved_config) -> None:
        """``[H, W]`` image_size + ``depths``: hierarchical Swin formula
        without CLS token.  shrink = 2**(N-1) per spatial dim.
        """
        from winml.modelkit.models.hf.vision_encoder_decoder import _VedDecoderNormalizedConfig

        nc = _VedDecoderNormalizedConfig(swin_mbart_ved_config)
        ec = swin_mbart_ved_config.encoder
        h, w = ec.image_size  # [64, 32]
        shrink = 2 ** (len(ec.depths) - 1)  # 8
        expected = (h // ec.patch_size // shrink) * (w // ec.patch_size // shrink)
        assert nc.encoder_seq_length == expected
        # Sanity: must NOT add the +1 CLS token for hierarchical encoders.
        assert nc.encoder_seq_length != expected + 1
