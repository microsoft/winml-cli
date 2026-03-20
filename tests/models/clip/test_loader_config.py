# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for CLIP loader configuration and task resolution."""

from __future__ import annotations

import pytest
from transformers import (
    CLIPConfig,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from winml.modelkit.loader.task import _get_custom_model_class, resolve_task_and_model_class
from winml.modelkit.models.hf.clip import MODEL_CLASS_MAPPING


# =============================================================================
# Test Constants - Match real CLIP config structure
# =============================================================================

# Shared projection dim (must match between text and vision)
PROJECTION_DIM = 32

# Text config (minimal but valid)
TEXT_VOCAB_SIZE = 1000
TEXT_HIDDEN_SIZE = 64
TEXT_NUM_HIDDEN_LAYERS = 2
TEXT_NUM_ATTENTION_HEADS = 2
TEXT_MAX_POSITION_EMBEDDINGS = 32
TEXT_INTERMEDIATE_SIZE = TEXT_HIDDEN_SIZE * 4

# Vision config (minimal but valid)
VISION_HIDDEN_SIZE = 64
VISION_NUM_HIDDEN_LAYERS = 2
VISION_NUM_ATTENTION_HEADS = 2
VISION_IMAGE_SIZE = 32
VISION_PATCH_SIZE = 8
VISION_NUM_CHANNELS = 3
VISION_INTERMEDIATE_SIZE = VISION_HIDDEN_SIZE * 4

# Test input shapes
BATCH_SIZE = 2
SEQUENCE_LENGTH = 16


# =============================================================================
# Fixtures - Dummy configs matching real CLIP structure
# =============================================================================


@pytest.fixture(scope="module")
def clip_text_config() -> CLIPTextConfig:
    """Create minimal CLIPTextConfig for testing.

    model_type = "clip_text_model"
    """
    return CLIPTextConfig(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN_SIZE,
        projection_dim=PROJECTION_DIM,
        num_hidden_layers=TEXT_NUM_HIDDEN_LAYERS,
        num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
        max_position_embeddings=TEXT_MAX_POSITION_EMBEDDINGS,
        intermediate_size=TEXT_INTERMEDIATE_SIZE,
    )


@pytest.fixture(scope="module")
def clip_vision_config() -> CLIPVisionConfig:
    """Create minimal CLIPVisionConfig for testing.

    model_type = "clip_vision_model"
    """
    return CLIPVisionConfig(
        hidden_size=VISION_HIDDEN_SIZE,
        projection_dim=PROJECTION_DIM,
        num_hidden_layers=VISION_NUM_HIDDEN_LAYERS,
        num_attention_heads=VISION_NUM_ATTENTION_HEADS,
        image_size=VISION_IMAGE_SIZE,
        patch_size=VISION_PATCH_SIZE,
        num_channels=VISION_NUM_CHANNELS,
        intermediate_size=VISION_INTERMEDIATE_SIZE,
    )


@pytest.fixture(scope="module")
def clip_config(clip_text_config, clip_vision_config) -> CLIPConfig:
    """Create CLIPConfig (combined) for loader resolution tests.

    model_type = "clip"
    This is what you get when loading "openai/clip-vit-base-patch32".
    """
    return CLIPConfig(
        text_config=clip_text_config.to_dict(),
        vision_config=clip_vision_config.to_dict(),
        projection_dim=PROJECTION_DIM,
    )


# =============================================================================
# Tests: MODEL_CLASS_MAPPING
# =============================================================================


class TestModelClassMapping:
    """Tests for MODEL_CLASS_MAPPING lookup table."""

    def test_mapping_has_feature_extraction(self):
        """Mapping contains (clip, feature-extraction) entry."""
        assert ("clip", "feature-extraction") in MODEL_CLASS_MAPPING

    def test_mapping_has_image_feature_extraction(self):
        """Mapping contains (clip, image-feature-extraction) entry."""
        assert ("clip", "image-feature-extraction") in MODEL_CLASS_MAPPING

    def test_feature_extraction_maps_to_text_model(self):
        """feature-extraction maps to CLIPTextModelWithProjection."""
        assert (
            MODEL_CLASS_MAPPING[("clip", "feature-extraction")]
            is CLIPTextModelWithProjection
        )

    def test_image_feature_extraction_maps_to_vision_model(self):
        """image-feature-extraction maps to CLIPVisionModelWithProjection."""
        assert (
            MODEL_CLASS_MAPPING[("clip", "image-feature-extraction")]
            is CLIPVisionModelWithProjection
        )


# =============================================================================
# Tests: _get_custom_model_class()
# =============================================================================


class TestGetCustomModelClass:
    """Tests for _get_custom_model_class() lookup function."""

    def test_clip_feature_extraction(self):
        """Returns CLIPTextModelWithProjection for clip + feature-extraction."""
        from transformers import CLIPTextModelWithProjection

        result = _get_custom_model_class("clip", "feature-extraction")
        assert result is CLIPTextModelWithProjection

    def test_clip_image_feature_extraction(self):
        """Returns CLIPVisionModelWithProjection for clip + image-feature-extraction."""
        from transformers import CLIPVisionModelWithProjection

        result = _get_custom_model_class("clip", "image-feature-extraction")
        assert result is CLIPVisionModelWithProjection

    def test_clip_unknown_task_returns_none(self):
        """Returns None for unknown task (fallback to TasksManager)."""
        result = _get_custom_model_class("clip", "unknown-task")
        assert result is None

    def test_unknown_model_type_returns_none(self):
        """Returns None for unknown model type."""
        result = _get_custom_model_class("unknown-model", "feature-extraction")
        assert result is None

    def test_normalizes_model_type_underscore(self):
        """Handles model_type with underscores (clip_model → clip-model)."""
        from transformers import CLIPTextModelWithProjection

        result = _get_custom_model_class("CLIP", "feature-extraction")
        assert result is CLIPTextModelWithProjection


# =============================================================================
# Tests: resolve_task_and_model_class()
# =============================================================================


class TestResolveTaskAndModelClass:
    """Tests for resolve_task_and_model_class() resolution function."""

    def test_config_has_correct_model_type(self, clip_config):
        """Verify fixture has model_type='clip'."""
        assert clip_config.model_type == "clip"

    def test_feature_extraction_resolves_to_text_model(self, clip_config):
        """task='feature-extraction' resolves to CLIPTextModelWithProjection via specialization."""
        task, resolved_class = resolve_task_and_model_class(
            clip_config, task="feature-extraction"
        )

        assert task == "feature-extraction"
        assert resolved_class is CLIPTextModelWithProjection

    def test_image_feature_extraction_resolves_to_vision_model(self, clip_config):
        """image-feature-extraction resolves to CLIPVisionModelWithProjection."""
        task, resolved_class = resolve_task_and_model_class(
            clip_config, task="image-feature-extraction"
        )

        assert task == "image-feature-extraction"
        assert resolved_class is CLIPVisionModelWithProjection

    def test_preserves_original_task_name(self, clip_config):
        """Returns original task name, not normalized version."""
        task, _ = resolve_task_and_model_class(
            clip_config, task="image-feature-extraction"
        )

        # Should preserve "image-feature-extraction", not normalize to "feature-extraction"
        assert task == "image-feature-extraction"


# =============================================================================
# Tests: Model Instantiation with Resolved Class
# =============================================================================


class TestModelInstantiation:
    """Tests for instantiating models with resolved classes."""

    def test_text_model_instantiation(self, clip_text_config):
        """CLIPTextModelWithProjection instantiates with dummy config."""
        model = CLIPTextModelWithProjection(clip_text_config)

        assert model.config.model_type == "clip_text_model"
        assert model.config.max_position_embeddings == TEXT_MAX_POSITION_EMBEDDINGS
        assert model.config.vocab_size == TEXT_VOCAB_SIZE
        assert model.config.projection_dim == PROJECTION_DIM

    def test_vision_model_instantiation(self, clip_vision_config):
        """CLIPVisionModelWithProjection instantiates with dummy config."""
        model = CLIPVisionModelWithProjection(clip_vision_config)

        assert model.config.model_type == "clip_vision_model"
        assert model.config.image_size == VISION_IMAGE_SIZE
        assert model.config.patch_size == VISION_PATCH_SIZE
        assert model.config.num_channels == VISION_NUM_CHANNELS
        assert model.config.projection_dim == PROJECTION_DIM

    def test_combined_config_contains_both(self, clip_config):
        """CLIPConfig contains both text_config and vision_config."""
        assert hasattr(clip_config, "text_config")
        assert hasattr(clip_config, "vision_config")
        assert clip_config.text_config.model_type == "clip_text_model"
        assert clip_config.vision_config.model_type == "clip_vision_model"
