# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for Segformer ONNX export config and input generator.

Segformer has a unique conflict: config.image_size=224 (backbone pretraining
resolution) differs from preprocessor_config.size=512x512 (finetuned inference
resolution). The _SegformerVisionInputGenerator resolves this by prioritizing
preprocessor-derived height/width over config.image_size.
"""

from __future__ import annotations

import pytest
import torch
from optimum.utils import NormalizedConfig
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# Import triggers ONNX config registration
import winml.modelkit.models  # noqa: F401
from winml.modelkit.export.io import _get_onnx_config, generate_dummy_inputs
from winml.modelkit.models.hf.segformer import (
    MODEL_CLASS_MAPPING,
    SegformerIOConfig,
    _SegformerVisionInputGenerator,
)


# =============================================================================
# Test Constants
# =============================================================================

VISION_IMAGE_SIZE = 32
VISION_NUM_CHANNELS = 3
BATCH_SIZE = 1


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def segformer_config():
    """Minimal SegformerConfig for testing."""
    return SegformerConfig(
        image_size=VISION_IMAGE_SIZE,
        num_channels=VISION_NUM_CHANNELS,
        num_labels=10,
        hidden_sizes=[32, 64],
        num_attention_heads=[1, 2],
        num_encoder_blocks=2,
        depths=[1, 1],
        sr_ratios=[8, 4],
        decoder_hidden_size=32,
    )


@pytest.fixture(scope="module")
def segformer_model(segformer_config):
    """Minimal Segformer model for testing."""
    return SegformerForSemanticSegmentation(segformer_config)


@pytest.fixture()
def normalized_config(segformer_config):
    """NormalizedConfig wrapping SegformerConfig (exposes image_size)."""
    nc = NormalizedConfig.with_args(
        image_size="image_size",
        num_channels="num_channels",
        allow_new=True,
    )
    return nc(segformer_config)


# =============================================================================
# _SegformerVisionInputGenerator — Preprocessor priority tests
# =============================================================================


class TestSegformerVisionInputGenerator:
    """Tests for _SegformerVisionInputGenerator preprocessor priority.

    DummyVisionInputGenerator prioritizes normalized_config.image_size over
    explicit height/width kwargs. For Segformer, the preprocessor resolution
    (e.g. 512x512) must take precedence over config.image_size (224) because
    config.image_size is the backbone pretraining resolution, not the finetuned
    inference resolution.
    """

    def test_preprocessor_size_overrides_config(self, normalized_config):
        """Explicit height/width (from preprocessor) override config.image_size."""
        gen = _SegformerVisionInputGenerator(
            task="image-segmentation",
            normalized_config=normalized_config,
            height=512,
            width=512,
        )

        assert gen.height == 512
        assert gen.width == 512
        assert gen.image_size == (512, 512)

    def test_default_kwargs_keep_config_image_size(self, normalized_config):
        """Without explicit height/width, config.image_size is used."""
        gen = _SegformerVisionInputGenerator(
            task="image-segmentation",
            normalized_config=normalized_config,
        )

        assert gen.height == VISION_IMAGE_SIZE
        assert gen.width == VISION_IMAGE_SIZE

    def test_generated_tensor_uses_preprocessor_size(self, normalized_config):
        """generate() produces tensor with preprocessor dimensions, not config."""
        gen = _SegformerVisionInputGenerator(
            task="image-segmentation",
            normalized_config=normalized_config,
            batch_size=BATCH_SIZE,
            height=512,
            width=512,
        )

        tensor = gen.generate("pixel_values", framework="pt")
        assert tensor.shape == (BATCH_SIZE, VISION_NUM_CHANNELS, 512, 512)

    def test_non_square_preprocessor_size(self, normalized_config):
        """Non-square preprocessor sizes are preserved."""
        gen = _SegformerVisionInputGenerator(
            task="image-segmentation",
            normalized_config=normalized_config,
            height=384,
            width=512,
        )

        assert gen.height == 384
        assert gen.width == 512
        assert gen.image_size == (384, 512)


# =============================================================================
# SegformerIOConfig — Registration and I/O spec tests
# =============================================================================


class TestSegformerIOConfig:
    """Tests for SegformerIOConfig ONNX export registration."""

    def test_onnx_config_registered(self, segformer_model):
        """SegformerIOConfig is registered with TasksManager for image-segmentation."""
        config = _get_onnx_config(
            segformer_model.config.model_type,
            "image-segmentation",
            segformer_model.config,
        )
        assert isinstance(config, SegformerIOConfig)

    def test_inputs_contain_pixel_values(self, segformer_model):
        """Inputs spec includes pixel_values with correct dynamic axes."""
        config = _get_onnx_config(
            segformer_model.config.model_type,
            "image-segmentation",
            segformer_model.config,
        )
        assert "pixel_values" in config.inputs
        assert config.inputs["pixel_values"][0] == "batch_size"

    def test_outputs_contain_logits(self, segformer_model):
        """Outputs spec includes logits."""
        config = _get_onnx_config(
            segformer_model.config.model_type,
            "image-segmentation",
            segformer_model.config,
        )
        assert "logits" in config.outputs

    def test_dummy_inputs_shape(self, segformer_model):
        """Dummy inputs have correct shape from config."""
        inputs = generate_dummy_inputs(
            segformer_model.config.model_type,
            "image-segmentation",
            segformer_model.config,
            batch_size=BATCH_SIZE,
        )
        assert "pixel_values" in inputs
        pv = inputs["pixel_values"]
        assert pv.shape[0] == BATCH_SIZE
        assert pv.shape[1] == VISION_NUM_CHANNELS

    def test_dummy_inputs_can_forward(self, segformer_model):
        """Generated dummy inputs pass through model.forward() without error."""
        segformer_model.eval()
        inputs = generate_dummy_inputs(
            segformer_model.config.model_type,
            "image-segmentation",
            segformer_model.config,
            batch_size=BATCH_SIZE,
        )
        with torch.no_grad():
            outputs = segformer_model(**inputs)
        assert outputs is not None


# =============================================================================
# MODEL_CLASS_MAPPING — Correct AutoModel routing
# =============================================================================


class TestSegformerModelClassMapping:
    """Tests for Segformer model class routing."""

    def test_maps_to_semantic_segmentation(self):
        """image-segmentation task routes to AutoModelForSemanticSegmentation."""
        from transformers import AutoModelForSemanticSegmentation

        assert ("segformer", "image-segmentation") in MODEL_CLASS_MAPPING
        assert (
            MODEL_CLASS_MAPPING[("segformer", "image-segmentation")]
            is AutoModelForSemanticSegmentation
        )
