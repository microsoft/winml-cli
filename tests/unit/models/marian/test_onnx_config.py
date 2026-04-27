# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for Marian ONNX export configs.

Tests for Marian IOConfig classes and the WinML composite model:
- MarianEncoderIOConfig: encoder-only (feature-extraction task)
- MarianDecoderIOConfig: decoder with KV cache (text2text-generation task)
- _MarianDecoderNormalizedConfig: derives ``head_dim`` and maps decoder-side attrs
- WinMLMarianModel: composite (encoder + decoder) for the ``translation`` task
"""

from __future__ import annotations

import pytest
from optimum.exporters.tasks import TasksManager
from optimum.utils.input_generators import DummyTextInputGenerator
from transformers import MarianConfig

# Import triggers registration
from winml.modelkit.models.hf.marian import (
    MarianDecoderIOConfig,
    MarianEncoderIOConfig,
    WinMLMarianModel,
    _MarianDecoderNormalizedConfig,
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
DECODER_LAYERS = 2
DECODER_ATTENTION_HEADS = 2
ENCODER_LAYERS = 2
ENCODER_ATTENTION_HEADS = 2
VOCAB_SIZE = 100
MAX_POSITION_EMBEDDINGS = 16
HEAD_DIM = D_MODEL // DECODER_ATTENTION_HEADS  # 16


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def marian_config() -> MarianConfig:
    """Synthetic MarianConfig — small dims, no network."""
    return MarianConfig(
        d_model=D_MODEL,
        decoder_layers=DECODER_LAYERS,
        decoder_attention_heads=DECODER_ATTENTION_HEADS,
        encoder_layers=ENCODER_LAYERS,
        encoder_attention_heads=ENCODER_ATTENTION_HEADS,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    )


# =============================================================================
# _MarianDecoderNormalizedConfig Tests
# =============================================================================


class TestMarianDecoderNormalizedConfig:
    """Tests for _MarianDecoderNormalizedConfig.

    The PR replaced ``NormalizedConfig.with_args(...)`` (a functools.partial)
    with a real subclass so ``head_dim`` could be exposed as a property.
    These tests pin the contract: ``num_layers`` follows ``decoder_layers``
    (not the outer ``num_hidden_layers``, which on Marian is the encoder
    count) and ``head_dim`` is derived from ``d_model // decoder_attention_heads``.
    """

    def test_num_layers_uses_decoder_layers(self, marian_config) -> None:
        nc = _MarianDecoderNormalizedConfig(marian_config)
        assert nc.num_layers == DECODER_LAYERS

    def test_num_attention_heads_uses_decoder_attention_heads(self, marian_config) -> None:
        nc = _MarianDecoderNormalizedConfig(marian_config)
        assert nc.num_attention_heads == DECODER_ATTENTION_HEADS

    def test_hidden_size_uses_d_model(self, marian_config) -> None:
        nc = _MarianDecoderNormalizedConfig(marian_config)
        assert nc.hidden_size == D_MODEL

    def test_max_cache_len_uses_max_position_embeddings(self, marian_config) -> None:
        nc = _MarianDecoderNormalizedConfig(marian_config)
        assert nc.max_cache_len == MAX_POSITION_EMBEDDINGS

    def test_head_dim_derived(self, marian_config) -> None:
        """``head_dim`` is derived (no native ``head_dim`` attr on MarianConfig)."""
        nc = _MarianDecoderNormalizedConfig(marian_config)
        assert nc.head_dim == HEAD_DIM
        assert nc.head_dim == nc.hidden_size // nc.num_attention_heads


# =============================================================================
# MarianEncoderIOConfig Tests
# =============================================================================


class TestMarianEncoderIOConfig:
    """Tests for MarianEncoderIOConfig (encoder-only, feature-extraction)."""

    def test_registration(self):
        """Config is registered with TasksManager for marian feature-extraction."""
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type="marian",
            exporter="onnx",
            task="feature-extraction",
            library_name="transformers",
        )
        assert config_cls.func is MarianEncoderIOConfig

    def test_inputs(self, marian_config) -> None:
        onnx_config = MarianEncoderIOConfig(marian_config, task="feature-extraction")

        inputs = onnx_config.inputs
        assert set(inputs.keys()) == {"input_ids", "attention_mask"}
        assert inputs["input_ids"] == {0: "batch_size", 1: "sequence_length"}
        assert inputs["attention_mask"] == {0: "batch_size", 1: "sequence_length"}

    def test_outputs(self, marian_config) -> None:
        onnx_config = MarianEncoderIOConfig(marian_config, task="feature-extraction")

        outputs = onnx_config.outputs
        assert set(outputs.keys()) == {"encoder_hidden_states"}
        assert outputs["encoder_hidden_states"] == {0: "batch_size", 1: "sequence_length"}

    def test_dummy_input_generator_classes(self) -> None:
        assert (DummyTextInputGenerator,) == MarianEncoderIOConfig.DUMMY_INPUT_GENERATOR_CLASSES


# =============================================================================
# MarianDecoderIOConfig Tests
# =============================================================================


class TestMarianDecoderIOConfig:
    """Tests for MarianDecoderIOConfig (decoder with KV cache, text2text-generation)."""

    def test_registration(self):
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type="marian",
            exporter="onnx",
            task="text2text-generation",
            library_name="transformers",
        )
        assert config_cls.func is MarianDecoderIOConfig

    def test_normalized_config_class(self) -> None:
        assert MarianDecoderIOConfig.NORMALIZED_CONFIG_CLASS is _MarianDecoderNormalizedConfig

    def test_dummy_input_generator_classes(self) -> None:
        assert (
            EncoderDecoderInputGenerator,
            PastKeyValueInputGenerator,
        ) == MarianDecoderIOConfig.DUMMY_INPUT_GENERATOR_CLASSES

    def test_non_kv_inputs(self, marian_config) -> None:
        onnx_config = MarianDecoderIOConfig(marian_config, task="text2text-generation")

        inputs = onnx_config.inputs
        for name in (
            "decoder_input_ids",
            "encoder_hidden_states",
            "attention_mask",
            "decoder_attention_mask",
            "cache_position",
        ):
            assert name in inputs

    def test_kv_inputs_match_decoder_layers(self, marian_config) -> None:
        onnx_config = MarianDecoderIOConfig(marian_config, task="text2text-generation")

        inputs = onnx_config.inputs
        for i in range(DECODER_LAYERS):
            assert f"past_{i}_key" in inputs
            assert f"past_{i}_value" in inputs
        assert f"past_{DECODER_LAYERS}_key" not in inputs

    def test_outputs_include_logits_and_present_kv(self, marian_config) -> None:
        onnx_config = MarianDecoderIOConfig(marian_config, task="text2text-generation")

        outputs = onnx_config.outputs
        assert "logits" in outputs
        for i in range(DECODER_LAYERS):
            assert f"present_{i}_key" in outputs
            assert f"present_{i}_value" in outputs


# =============================================================================
# WinMLMarianModel Tests
# =============================================================================


class TestWinMLMarianModel:
    """Tests for WinMLMarianModel (composite model registered for translation)."""

    def test_composite_registration(self) -> None:
        assert COMPOSITE_MODEL_REGISTRY[("marian", "translation")] is WinMLMarianModel

    def test_sub_model_config(self) -> None:
        assert WinMLMarianModel._SUB_MODEL_CONFIG == {
            "encoder": "feature-extraction",
            "decoder": "text2text-generation",
        }

    def test_get_cache_class_returns_static_cache(self) -> None:
        """Marian currently uses ``WinMLStaticCache`` (index_put_ → ScatterND)."""
        assert WinMLMarianModel.get_cache_class() is WinMLStaticCache
