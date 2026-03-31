# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Export test fixtures - minimal HF configs for I/O spec testing.

Provides module-scoped HF config fixtures for fast parametrized testing
without instantiating model weights.
"""

from __future__ import annotations

import pytest

# CRITICAL: Trigger OnnxConfig registration with TasksManager.
# Without this import, custom configs (BertIOConfig, CLIPTextModelIOConfig, etc.)
# are NOT registered, and Optimum's defaults are used instead.
import winml.modelkit.models  # noqa: F401


# --- Text Encoder Configs ---


@pytest.fixture(scope="module")
def bert_config():
    """Minimal BertConfig for testing (max_position_embeddings=32)."""
    from transformers import BertConfig

    return BertConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=32,
    )


@pytest.fixture(scope="module")
def albert_config():
    """Minimal AlbertConfig for testing."""
    from transformers import AlbertConfig

    return AlbertConfig(
        vocab_size=100,
        embedding_size=32,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
    )


@pytest.fixture(scope="module")
def distilbert_config():
    """Minimal DistilBertConfig for testing."""
    from transformers import DistilBertConfig

    return DistilBertConfig(
        vocab_size=100,
        dim=64,
        n_layers=2,
        n_heads=2,
        hidden_dim=128,
    )


# --- Roberta-family Configs (position offset: max_pos = usable + pad_token_id + 1) ---
# NOTE: Function-scoped (not module) because _adjust_position_embeddings mutates
# config.max_position_embeddings in-place. Each test needs a fresh config.


@pytest.fixture
def roberta_config():
    """Minimal RobertaConfig for testing (max_position_embeddings=34, usable=32).

    Roberta: max_position_embeddings = usable_length + pad_token_id + 1 = 32 + 1 + 1 = 34.
    """
    from transformers import RobertaConfig

    return RobertaConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=34,
        pad_token_id=1,
        type_vocab_size=1,
    )


@pytest.fixture
def xlm_roberta_config():
    """Minimal XLMRobertaConfig for testing (max_position_embeddings=34, usable=32)."""
    from transformers import XLMRobertaConfig

    return XLMRobertaConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=34,
        pad_token_id=1,
        type_vocab_size=1,
    )


@pytest.fixture
def camembert_config():
    """Minimal CamembertConfig for testing (max_position_embeddings=34, usable=32)."""
    from transformers import CamembertConfig

    return CamembertConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=34,
        pad_token_id=1,
        type_vocab_size=1,
    )


@pytest.fixture
def mpnet_config():
    """Minimal MPNetConfig for testing (max_position_embeddings=34, usable=32).

    MPNet: max_position_embeddings = usable_length + pad_token_id + 1 = 32 + 1 + 1 = 34.
    """
    from transformers import MPNetConfig

    return MPNetConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=34,
        pad_token_id=1,
    )


# --- Text Decoder Configs ---


@pytest.fixture(scope="module")
def gpt2_config():
    """Minimal GPT2Config for testing (n_positions=32)."""
    from transformers import GPT2Config

    return GPT2Config(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=32,
    )


# --- Vision Configs ---


@pytest.fixture(scope="module")
def resnet_config():
    """Minimal ResNetConfig for testing."""
    from transformers import ResNetConfig

    return ResNetConfig(
        num_channels=3,
        hidden_sizes=[64, 128],
        depths=[1, 1],
        layer_type="basic",
    )


@pytest.fixture(scope="module")
def vit_config():
    """Minimal ViTConfig for testing (image_size=32, patch_size=8)."""
    from transformers import ViTConfig

    return ViTConfig(
        image_size=32,
        patch_size=8,
        num_channels=3,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
    )


@pytest.fixture(scope="module")
def convnext_config():
    """Minimal ConvNextConfig for testing."""
    from transformers import ConvNextConfig

    return ConvNextConfig(
        num_channels=3,
        hidden_sizes=[64, 128],
        depths=[1, 1],
    )


# --- Multimodal Configs ---


@pytest.fixture(scope="module")
def clip_vision_config():
    """Minimal CLIPVisionConfig for testing (image_size=32, patch_size=8)."""
    from transformers import CLIPVisionConfig

    return CLIPVisionConfig(
        image_size=32,
        patch_size=8,
        num_channels=3,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
    )


@pytest.fixture(scope="module")
def clip_text_config():
    """Minimal CLIPTextConfig for testing (max_position_embeddings=32)."""
    from transformers import CLIPTextConfig

    return CLIPTextConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=32,
    )


# --- Detection Configs ---


@pytest.fixture(scope="module")
def detr_config():
    """Minimal DetrConfig for testing."""
    from transformers import DetrConfig

    return DetrConfig(
        num_channels=3,
        d_model=64,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
    )
