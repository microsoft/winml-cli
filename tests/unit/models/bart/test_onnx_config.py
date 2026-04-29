# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for BART ONNX export configs.

Tests for BART IOConfig classes and the WinML composite model:
- BartEncoderIOConfig: encoder-only (feature-extraction task)
- BartDecoderIOConfig: decoder with KV cache (text2text-generation task)
- _BartDecoderNormalizedConfig: derives ``head_dim`` and maps decoder-side attrs
- WinMLBartModel: composite (encoder + decoder) for the ``summarization`` task

Includes regression coverage for asymmetric encoder/decoder layer counts
(e.g., distilbart-cnn-12-6: encoder_layers=12, decoder_layers=6).
"""

from __future__ import annotations

import pytest
from optimum.exporters.tasks import TasksManager
from optimum.utils.input_generators import DummyTextInputGenerator
from transformers import BartConfig

# Import triggers registration
from winml.modelkit.models.hf.bart import (
    BartDecoderIOConfig,
    BartEncoderIOConfig,
    WinMLBartModel,
    _BartDecoderNormalizedConfig,
)
from winml.modelkit.models.winml.composite_model import COMPOSITE_MODEL_REGISTRY
from winml.modelkit.models.winml.encoder_decoder import EncoderDecoderInputGenerator
from winml.modelkit.models.winml.kv_cache import (
    PastKeyValueInputGenerator,
    WinMLStaticCache,
)


# =============================================================================
# Test Constants
# =============================================================================

D_MODEL = 32
DECODER_LAYERS = 4
DECODER_ATTENTION_HEADS = 2
ENCODER_LAYERS = 4
ENCODER_ATTENTION_HEADS = 2
VOCAB_SIZE = 100
MAX_POSITION_EMBEDDINGS = 16
HEAD_DIM = D_MODEL // DECODER_ATTENTION_HEADS  # 16

# Asymmetric (distilbart-cnn-12-6 shape) — encoder layers > decoder layers
ASYM_DECODER_LAYERS = 6
ASYM_ENCODER_LAYERS = 12


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def bart_config() -> BartConfig:
    """Synthetic symmetric BartConfig — small dims, no network."""
    return BartConfig(
        d_model=D_MODEL,
        decoder_layers=DECODER_LAYERS,
        decoder_attention_heads=DECODER_ATTENTION_HEADS,
        encoder_layers=ENCODER_LAYERS,
        encoder_attention_heads=ENCODER_ATTENTION_HEADS,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    )


@pytest.fixture(scope="module")
def bart_config_asymmetric() -> BartConfig:
    """Synthetic asymmetric BartConfig — distilbart-cnn-12-6 shape."""
    return BartConfig(
        d_model=D_MODEL,
        decoder_layers=ASYM_DECODER_LAYERS,
        decoder_attention_heads=DECODER_ATTENTION_HEADS,
        encoder_layers=ASYM_ENCODER_LAYERS,
        encoder_attention_heads=ENCODER_ATTENTION_HEADS,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    )


# =============================================================================
# _BartDecoderNormalizedConfig Tests
# =============================================================================


class TestBartDecoderNormalizedConfig:
    """Tests for _BartDecoderNormalizedConfig.

    The PR replaced ``NormalizedConfig.with_args(...)`` (a functools.partial)
    with a real subclass so ``head_dim`` could be exposed as a property.
    These tests pin the contract: ``num_layers`` follows ``decoder_layers``
    (not the outer ``num_hidden_layers``, which on BART is the encoder
    count) and ``head_dim`` is derived from ``d_model // decoder_attention_heads``.
    """

    def test_num_layers_uses_decoder_layers(self, bart_config) -> None:
        nc = _BartDecoderNormalizedConfig(bart_config)
        assert nc.num_layers == DECODER_LAYERS

    def test_num_attention_heads_uses_decoder_attention_heads(self, bart_config) -> None:
        nc = _BartDecoderNormalizedConfig(bart_config)
        assert nc.num_attention_heads == DECODER_ATTENTION_HEADS

    def test_hidden_size_uses_d_model(self, bart_config) -> None:
        nc = _BartDecoderNormalizedConfig(bart_config)
        assert nc.hidden_size == D_MODEL

    def test_max_cache_len_uses_max_position_embeddings(self, bart_config) -> None:
        nc = _BartDecoderNormalizedConfig(bart_config)
        assert nc.max_cache_len == MAX_POSITION_EMBEDDINGS

    def test_head_dim_derived(self, bart_config) -> None:
        """``head_dim`` is derived (no native ``head_dim`` attr on BartConfig)."""
        nc = _BartDecoderNormalizedConfig(bart_config)
        assert nc.head_dim == HEAD_DIM
        assert nc.head_dim == nc.hidden_size // nc.num_attention_heads

    def test_asymmetric_num_layers_uses_decoder_count(self, bart_config_asymmetric) -> None:
        """Distilbart-style: outer ``num_hidden_layers`` is 12 but decoder has 6 layers.

        Pre-fix, the cache walked past ``self.layers`` and crashed.  At the
        NormalizedConfig level the contract is: ``num_layers`` must read
        ``decoder_layers``, not the outer ``num_hidden_layers``.
        """
        # Sanity check: the outer config still reports the encoder count.
        assert bart_config_asymmetric.num_hidden_layers == ASYM_ENCODER_LAYERS

        nc = _BartDecoderNormalizedConfig(bart_config_asymmetric)
        assert nc.num_layers == ASYM_DECODER_LAYERS


# =============================================================================
# BartEncoderIOConfig Tests
# =============================================================================


class TestBartEncoderIOConfig:
    """Tests for BartEncoderIOConfig (encoder-only, feature-extraction)."""

    def test_registration(self) -> None:
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type="bart",
            exporter="onnx",
            task="feature-extraction",
            library_name="transformers",
        )
        assert config_cls.func is BartEncoderIOConfig

    def test_inputs(self, bart_config) -> None:
        onnx_config = BartEncoderIOConfig(bart_config, task="feature-extraction")

        inputs = onnx_config.inputs
        assert set(inputs.keys()) == {"input_ids", "attention_mask"}
        assert inputs["input_ids"] == {0: "batch_size", 1: "sequence_length"}
        assert inputs["attention_mask"] == {0: "batch_size", 1: "sequence_length"}

    def test_outputs(self, bart_config) -> None:
        onnx_config = BartEncoderIOConfig(bart_config, task="feature-extraction")

        outputs = onnx_config.outputs
        assert set(outputs.keys()) == {"encoder_hidden_states"}
        assert outputs["encoder_hidden_states"] == {0: "batch_size", 1: "sequence_length"}

    def test_dummy_input_generator_classes(self) -> None:
        assert (DummyTextInputGenerator,) == BartEncoderIOConfig.DUMMY_INPUT_GENERATOR_CLASSES


# =============================================================================
# BartDecoderIOConfig Tests
# =============================================================================


class TestBartDecoderIOConfig:
    """Tests for BartDecoderIOConfig (decoder with KV cache, text2text-generation)."""

    def test_registration(self) -> None:
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type="bart",
            exporter="onnx",
            task="text2text-generation",
            library_name="transformers",
        )
        assert config_cls.func is BartDecoderIOConfig

    def test_normalized_config_class(self) -> None:
        assert BartDecoderIOConfig.NORMALIZED_CONFIG_CLASS is _BartDecoderNormalizedConfig

    def test_dummy_input_generator_classes(self) -> None:
        assert (
            EncoderDecoderInputGenerator,
            PastKeyValueInputGenerator,
        ) == BartDecoderIOConfig.DUMMY_INPUT_GENERATOR_CLASSES

    def test_non_kv_inputs(self, bart_config) -> None:
        onnx_config = BartDecoderIOConfig(bart_config, task="text2text-generation")

        inputs = onnx_config.inputs
        for name in (
            "decoder_input_ids",
            "encoder_hidden_states",
            "attention_mask",
            "decoder_attention_mask",
            "cache_position",
        ):
            assert name in inputs

    def test_kv_inputs_match_decoder_layers(self, bart_config) -> None:
        onnx_config = BartDecoderIOConfig(bart_config, task="text2text-generation")

        inputs = onnx_config.inputs
        for i in range(DECODER_LAYERS):
            assert f"past_{i}_key" in inputs
            assert f"past_{i}_value" in inputs
        assert f"past_{DECODER_LAYERS}_key" not in inputs

    def test_outputs_include_logits_and_present_kv(self, bart_config) -> None:
        onnx_config = BartDecoderIOConfig(bart_config, task="text2text-generation")

        outputs = onnx_config.outputs
        assert "logits" in outputs
        for i in range(DECODER_LAYERS):
            assert f"present_{i}_key" in outputs
            assert f"present_{i}_value" in outputs

    def test_asymmetric_kv_inputs_use_decoder_layers(self, bart_config_asymmetric) -> None:
        """KV input count tracks ``decoder_layers``, not the outer ``num_hidden_layers``."""
        onnx_config = BartDecoderIOConfig(bart_config_asymmetric, task="text2text-generation")

        kv_inputs = [name for name in onnx_config.inputs if name.startswith("past_")]
        # decoder_layers=6 → 12 KV tensors; encoder_layers=12 would give 24 (wrong).
        assert len(kv_inputs) == ASYM_DECODER_LAYERS * 2
        assert "past_5_key" in onnx_config.inputs
        assert "past_6_key" not in onnx_config.inputs


# =============================================================================
# WinMLBartModel Tests
# =============================================================================


class TestWinMLBartModel:
    """Tests for WinMLBartModel (composite model registered for summarization)."""

    def test_composite_registration(self) -> None:
        assert COMPOSITE_MODEL_REGISTRY[("bart", "summarization")] is WinMLBartModel

    def test_sub_model_config(self) -> None:
        assert WinMLBartModel._SUB_MODEL_CONFIG == {
            "encoder": "feature-extraction",
            "decoder": "text2text-generation",
        }

    def test_get_cache_class_returns_static_cache(self) -> None:
        """BART currently uses ``WinMLStaticCache`` (index_put_ → ScatterND)."""
        assert WinMLBartModel.get_cache_class() is WinMLStaticCache
