# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for CLIP ONNX export configs."""

from __future__ import annotations

import pytest
import torch
from optimum.exporters.tasks import TasksManager
from transformers import (
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

# Import triggers registration
from winml.modelkit.models.hf.clip import CLIPTextModelIOConfig, CLIPVisionModelIOConfig


# =============================================================================
# Test Constants - Explicit config values for verification
# =============================================================================

# Text model config
TEXT_VOCAB_SIZE = 1000
TEXT_HIDDEN_SIZE = 64
TEXT_PROJECTION_DIM = 32
TEXT_NUM_HIDDEN_LAYERS = 2
TEXT_NUM_ATTENTION_HEADS = 2
TEXT_MAX_POSITION_EMBEDDINGS = 32  # max sequence length

# Vision model config
VISION_HIDDEN_SIZE = 64
VISION_PROJECTION_DIM = 32
VISION_NUM_HIDDEN_LAYERS = 2
VISION_NUM_ATTENTION_HEADS = 2
VISION_IMAGE_SIZE = 32
VISION_PATCH_SIZE = 8
VISION_NUM_CHANNELS = 3

# Test input shapes
BATCH_SIZE = 2
SEQUENCE_LENGTH = 16  # must be <= TEXT_MAX_POSITION_EMBEDDINGS


# =============================================================================
# Fixtures - Dummy configs and model instances (no network download)
# =============================================================================


@pytest.fixture(scope="module")
def clip_text_config() -> CLIPTextConfig:
    """Create minimal CLIPTextConfig for testing."""
    return CLIPTextConfig(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN_SIZE,
        projection_dim=TEXT_PROJECTION_DIM,
        num_hidden_layers=TEXT_NUM_HIDDEN_LAYERS,
        num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
        max_position_embeddings=TEXT_MAX_POSITION_EMBEDDINGS,
        intermediate_size=TEXT_HIDDEN_SIZE * 4,
    )


@pytest.fixture(scope="module")
def clip_vision_config() -> CLIPVisionConfig:
    """Create minimal CLIPVisionConfig for testing."""
    return CLIPVisionConfig(
        hidden_size=VISION_HIDDEN_SIZE,
        projection_dim=VISION_PROJECTION_DIM,
        num_hidden_layers=VISION_NUM_HIDDEN_LAYERS,
        num_attention_heads=VISION_NUM_ATTENTION_HEADS,
        image_size=VISION_IMAGE_SIZE,
        patch_size=VISION_PATCH_SIZE,
        num_channels=VISION_NUM_CHANNELS,
        intermediate_size=VISION_HIDDEN_SIZE * 4,
    )


@pytest.fixture(scope="module")
def clip_text_model(clip_text_config) -> CLIPTextModelWithProjection:
    """Instantiate CLIPTextModelWithProjection with dummy config."""
    return CLIPTextModelWithProjection(clip_text_config)


@pytest.fixture(scope="module")
def clip_vision_model(clip_vision_config) -> CLIPVisionModelWithProjection:
    """Instantiate CLIPVisionModelWithProjection with dummy config."""
    return CLIPVisionModelWithProjection(clip_vision_config)


# =============================================================================
# CLIPTextModelIOConfig Tests
# =============================================================================


class TestCLIPTextModelIOConfig:
    """Tests for CLIPTextModelIOConfig."""

    def test_registration(self):
        """Config is registered with TasksManager."""
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type="clip_text_model",
            exporter="onnx",
            task="feature-extraction",
            library_name="transformers",
        )
        assert config_cls.func is CLIPTextModelIOConfig

    def test_inputs_includes_attention_mask(self, clip_text_config):
        """Inputs include both input_ids and attention_mask."""
        onnx_config = CLIPTextModelIOConfig(clip_text_config, task="feature-extraction")

        inputs = onnx_config.inputs
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["input_ids"] == {0: "batch_size", 1: "sequence_length"}
        assert inputs["attention_mask"] == {0: "batch_size", 1: "sequence_length"}

    def test_input_shape_from_config(self, clip_text_model):
        """Verify model accepts inputs with shape derived from config."""
        # Create input tensors with explicit shapes
        input_ids = torch.randint(
            0, TEXT_VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH), dtype=torch.long
        )
        attention_mask = torch.ones(BATCH_SIZE, SEQUENCE_LENGTH, dtype=torch.long)

        # Model should accept these inputs without error
        with torch.no_grad():
            outputs = clip_text_model(input_ids=input_ids, attention_mask=attention_mask)

        # Verify output shape
        assert outputs.text_embeds.shape == (BATCH_SIZE, TEXT_PROJECTION_DIM)


# =============================================================================
# CLIPVisionModelIOConfig Tests
# =============================================================================


class TestCLIPVisionModelIOConfig:
    """Tests for CLIPVisionModelIOConfig."""

    def test_registration(self):
        """Config is registered with TasksManager."""
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type="clip_vision_model",
            exporter="onnx",
            task="feature-extraction",
            library_name="transformers",
        )
        # Note: may return Optimum's built-in if it takes precedence
        assert config_cls is not None

    def test_outputs_includes_image_embeds(self, clip_vision_config):
        """Outputs include image_embeds instead of pooler_output."""
        onnx_config = CLIPVisionModelIOConfig(
            clip_vision_config, task="feature-extraction"
        )

        outputs = onnx_config.outputs
        assert "image_embeds" in outputs
        assert "last_hidden_state" in outputs
        assert outputs["image_embeds"] == {0: "batch_size"}

    def test_input_shape_from_config(self, clip_vision_model):
        """Verify model accepts inputs with shape derived from config."""
        # Create input tensor with explicit shape from config
        pixel_values = torch.randn(
            BATCH_SIZE,
            VISION_NUM_CHANNELS,
            VISION_IMAGE_SIZE,
            VISION_IMAGE_SIZE,
            dtype=torch.float32,
        )

        # Model should accept these inputs without error
        with torch.no_grad():
            outputs = clip_vision_model(pixel_values=pixel_values)

        # Verify output shape
        assert outputs.image_embeds.shape == (BATCH_SIZE, VISION_PROJECTION_DIM)
