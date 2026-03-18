"""Tests for SAM2 ONNX export configs.

Tests for SAM2 IOConfig classes:
- Sam2ImageEncoderIOConfig: encoder-only (feature-extraction task)
- Sam2IOConfig: full model encoder+decoder (image-segmentation task)
- Sam2MaskGenerationIOConfig: mask generation decoder (mask-generation task)
"""

from __future__ import annotations

import pytest
from optimum.exporters.tasks import TasksManager

# Import triggers registration
from winml.modelkit.models.hf.sam import (
    Sam2EmbeddingsInputGenerator,
    Sam2ImageEncoderIOConfig,
    Sam2IOConfig,
    Sam2MaskGenerationIOConfig,
    Sam2MaskInputGenerator,
    Sam2NormalizedVisionConfig,
    Sam2PointsInputGenerator,
)


# =============================================================================
# Test Constants
# =============================================================================

BATCH_SIZE = 1


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def sam2_config():
    """Create minimal Sam2Config for testing."""
    from transformers import Sam2Config, Sam2VisionConfig

    # Minimal vision config for faster tests
    vision_config = Sam2VisionConfig(
        hidden_size=32,
        embed_dim_per_stage=[32, 64, 128, 256],
        blocks_per_stage=[1, 1, 1, 1],
        num_attention_heads=2,
        global_attention_blocks=[0],
        window_size=4,
        backbone_channel_list=[256, 128, 64, 32],
    )

    return Sam2Config(vision_config=vision_config)


# =============================================================================
# Sam2NormalizedVisionConfig Tests
# =============================================================================


class TestSam2NormalizedVisionConfig:
    """Tests for Sam2NormalizedVisionConfig."""

    def test_default_image_size_constant(self):
        """Sam2NormalizedVisionConfig.DEFAULT_IMAGE_SIZE is 1024."""
        assert Sam2NormalizedVisionConfig.DEFAULT_IMAGE_SIZE == 1024

    def test_provides_default_image_size(self, sam2_config):
        """NormalizedConfig provides default image_size."""
        norm_config = Sam2NormalizedVisionConfig(sam2_config)

        # Should return Sam2NormalizedVisionConfig.DEFAULT_IMAGE_SIZE
        assert norm_config.image_size == Sam2NormalizedVisionConfig.DEFAULT_IMAGE_SIZE

    def test_has_attribute_image_size(self, sam2_config):
        """has_attribute('image_size') works."""
        norm_config = Sam2NormalizedVisionConfig(sam2_config)

        # This is used by dummy input generators
        assert norm_config.has_attribute("image_size") is True

    def test_other_attributes_still_raise(self, sam2_config):
        """Non-existent attributes still raise AttributeError."""
        norm_config = Sam2NormalizedVisionConfig(sam2_config)

        with pytest.raises(AttributeError):
            _ = norm_config.nonexistent_attribute


# =============================================================================
# Sam2ImageEncoderIOConfig Tests
# =============================================================================


class TestSam2ImageEncoderIOConfig:
    """Tests for Sam2ImageEncoderIOConfig (encoder-only, feature-extraction)."""

    def test_registration_sam2_feature_extraction(self):
        """Config is registered with TasksManager for sam2 feature-extraction."""
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type="sam2",
            exporter="onnx",
            task="feature-extraction",
            library_name="transformers",
        )
        assert config_cls.func.__name__ == Sam2ImageEncoderIOConfig.__name__

    def test_registration_sam2_video_feature_extraction(self):
        """Config is registered with TasksManager for sam2_video feature-extraction.

        All HuggingFace SAM2 models (facebook/sam2-*, facebook/sam2.1-*)
        use model_type='sam2_video', not 'sam2'.
        """
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type="sam2_video",
            exporter="onnx",
            task="feature-extraction",
            library_name="transformers",
        )
        assert config_cls.func.__name__ == Sam2ImageEncoderIOConfig.__name__

    def test_registration_sam2_vision_model_feature_extraction(self):
        """Config is registered for sam2_vision_model (Sam2VisionModel's model_type)."""
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type="sam2_vision_model",
            exporter="onnx",
            task="feature-extraction",
            library_name="transformers",
        )
        assert config_cls.func.__name__ == Sam2ImageEncoderIOConfig.__name__

    def test_inputs_has_pixel_values(self, sam2_config):
        """Inputs include pixel_values with correct axes."""
        onnx_config = Sam2ImageEncoderIOConfig(sam2_config, task="feature-extraction")

        inputs = onnx_config.inputs
        assert "pixel_values" in inputs
        assert inputs["pixel_values"] == {0: "batch_size", 2: "height", 3: "width"}

    def test_outputs_has_embeddings_and_features(self, sam2_config):
        """Outputs include image_embeddings and high_res_features."""
        onnx_config = Sam2ImageEncoderIOConfig(sam2_config, task="feature-extraction")

        outputs = onnx_config.outputs
        assert "image_embeddings" in outputs
        assert "high_res_features1" in outputs
        assert "high_res_features2" in outputs

    def test_uses_custom_normalized_config(self, sam2_config):
        """Uses Sam2NormalizedVisionConfig."""
        onnx_config = Sam2ImageEncoderIOConfig(sam2_config, task="feature-extraction")

        assert isinstance(onnx_config._normalized_config, Sam2NormalizedVisionConfig)

    def test_dummy_input_generator_class(self, sam2_config):
        """Uses DummyVisionInputGenerator."""
        from optimum.utils.input_generators import DummyVisionInputGenerator

        assert (
            DummyVisionInputGenerator,
        ) == Sam2ImageEncoderIOConfig.DUMMY_INPUT_GENERATOR_CLASSES


# =============================================================================
# Sam2IOConfig Tests (Full Model)
# =============================================================================


class TestSam2IOConfig:
    """Tests for Sam2IOConfig (full model encoder+decoder for image-segmentation)."""

    def test_registration_sam2_image_segmentation(self):
        """Config is registered with TasksManager for sam2 image-segmentation."""
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type="sam2",
            exporter="onnx",
            task="image-segmentation",
            library_name="transformers",
        )
        assert config_cls.func.__name__ == Sam2IOConfig.__name__

    def test_registration_sam2_video_image_segmentation(self):
        """Config is registered with TasksManager for sam2_video image-segmentation."""
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type="sam2_video",
            exporter="onnx",
            task="image-segmentation",
            library_name="transformers",
        )
        assert config_cls.func.__name__ == Sam2IOConfig.__name__

    def test_inputs_has_all_required(self, sam2_config):
        """Inputs include pixel_values and point prompts."""
        onnx_config = Sam2IOConfig(sam2_config, task="image-segmentation")

        inputs = onnx_config.inputs
        assert "pixel_values" in inputs
        assert "input_points" in inputs
        assert "input_labels" in inputs

    def test_outputs_has_masks_and_scores(self, sam2_config):
        """Outputs include masks and iou_scores."""
        onnx_config = Sam2IOConfig(sam2_config, task="image-segmentation")

        outputs = onnx_config.outputs
        assert "masks" in outputs
        assert "iou_scores" in outputs

    def test_uses_custom_normalized_config(self, sam2_config):
        """Uses Sam2NormalizedVisionConfig."""
        onnx_config = Sam2IOConfig(sam2_config, task="image-segmentation")

        assert isinstance(onnx_config._normalized_config, Sam2NormalizedVisionConfig)

    def test_dummy_input_generator_classes(self, sam2_config):
        """Uses DummyVisionInputGenerator and Sam2PointsInputGenerator."""
        from optimum.utils.input_generators import DummyVisionInputGenerator

        assert (
            DummyVisionInputGenerator,
            Sam2PointsInputGenerator,
        ) == Sam2IOConfig.DUMMY_INPUT_GENERATOR_CLASSES


# =============================================================================
# Sam2MaskDecoderIOConfig Tests
# =============================================================================


class TestSam2MaskGenerationIOConfig:
    """Tests for Sam2MaskGenerationIOConfig (mask generation decoder)."""

    def test_registration_sam2(self):
        """Config is registered with TasksManager for sam2 mask-generation."""
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type="sam2",
            exporter="onnx",
            task="mask-generation",
            library_name="transformers",
        )
        assert config_cls.func.__name__ == Sam2MaskGenerationIOConfig.__name__

    def test_registration_sam2_video(self):
        """Config is registered with TasksManager for sam2_video mask-generation."""
        config_cls = TasksManager.get_exporter_config_constructor(
            model_type="sam2_video",
            exporter="onnx",
            task="mask-generation",
            library_name="transformers",
        )
        assert config_cls.func.__name__ == Sam2MaskGenerationIOConfig.__name__

    def test_inputs_has_all_required(self, sam2_config):
        """Inputs include points, labels, raw embeddings, and mask inputs."""
        onnx_config = Sam2MaskGenerationIOConfig(sam2_config, task="mask-generation")

        inputs = onnx_config.inputs
        # Point inputs
        assert "input_points" in inputs
        assert "input_labels" in inputs
        # Raw encoder outputs (256-channel, before conv_s0/conv_s1)
        assert "image_embeddings" in inputs
        assert "high_res_features0" in inputs
        assert "high_res_features1" in inputs
        # Mask refinement inputs
        assert "mask_input" in inputs
        assert "use_mask_input" in inputs

    def test_outputs_has_masks_and_scores(self, sam2_config):
        """Outputs include masks, iou_scores, and low_res_masks."""
        onnx_config = Sam2MaskGenerationIOConfig(sam2_config, task="mask-generation")

        outputs = onnx_config.outputs
        assert "masks" in outputs
        assert "iou_scores" in outputs
        assert "low_res_masks" in outputs

    def test_dummy_input_generator_classes(self, sam2_config):
        """Uses correct dummy input generators."""
        assert (
            Sam2PointsInputGenerator,
            Sam2EmbeddingsInputGenerator,
            Sam2MaskInputGenerator,
        ) == Sam2MaskGenerationIOConfig.DUMMY_INPUT_GENERATOR_CLASSES


# =============================================================================
# Dummy Input Generator Tests
# =============================================================================


class TestSam2PointsInputGenerator:
    """Tests for Sam2PointsInputGenerator."""

    def test_generates_input_points(self, sam2_config):
        """Generates input_points with correct shape."""
        norm_config = Sam2NormalizedVisionConfig(sam2_config)
        generator = Sam2PointsInputGenerator(
            task="image-segmentation",
            normalized_config=norm_config,
            batch_size=BATCH_SIZE,
            point_batch_size=1,
            nb_points_per_image=5,
        )

        points = generator.generate("input_points", framework="pt")

        assert points.shape == (BATCH_SIZE, 1, 5, 2)
        # Points should be in 0-1024 range
        assert points.min() >= 0
        assert points.max() <= 1024

    def test_generates_input_labels(self, sam2_config):
        """Generates input_labels with correct shape."""
        norm_config = Sam2NormalizedVisionConfig(sam2_config)
        generator = Sam2PointsInputGenerator(
            task="image-segmentation",
            normalized_config=norm_config,
            batch_size=BATCH_SIZE,
            point_batch_size=1,
            nb_points_per_image=5,
        )

        labels = generator.generate("input_labels", framework="pt")

        assert labels.shape == (BATCH_SIZE, 1, 5)
        assert labels.dtype.is_signed  # int64


class TestSam2EmbeddingsInputGenerator:
    """Tests for Sam2EmbeddingsInputGenerator."""

    def test_generates_image_embeddings(self, sam2_config):
        """Generates image_embeddings with correct shape."""
        norm_config = Sam2NormalizedVisionConfig(sam2_config)
        generator = Sam2EmbeddingsInputGenerator(
            task="image-segmentation",
            normalized_config=norm_config,
            batch_size=BATCH_SIZE,
        )

        embeddings = generator.generate("image_embeddings", framework="pt")

        assert embeddings.shape == (BATCH_SIZE, 256, 64, 64)

    def test_generates_high_res_features(self, sam2_config):
        """Generates raw high_res_features with 256-channel shapes."""
        norm_config = Sam2NormalizedVisionConfig(sam2_config)
        generator = Sam2EmbeddingsInputGenerator(
            task="mask-generation",
            normalized_config=norm_config,
            batch_size=BATCH_SIZE,
        )

        feat0 = generator.generate("high_res_features0", framework="pt")
        feat1 = generator.generate("high_res_features1", framework="pt")

        assert feat0.shape == (BATCH_SIZE, 256, 256, 256)
        assert feat1.shape == (BATCH_SIZE, 256, 128, 128)


class TestSam2MaskInputGenerator:
    """Tests for Sam2MaskInputGenerator."""

    def test_generates_mask_input(self, sam2_config):
        """Generates mask_input with correct shape."""
        norm_config = Sam2NormalizedVisionConfig(sam2_config)
        generator = Sam2MaskInputGenerator(
            task="image-segmentation",
            normalized_config=norm_config,
            batch_size=BATCH_SIZE,
        )

        mask = generator.generate("mask_input", framework="pt")

        assert mask.shape == (BATCH_SIZE, 1, 256, 256)

    def test_generates_use_mask_input(self, sam2_config):
        """Generates use_mask_input flag."""
        norm_config = Sam2NormalizedVisionConfig(sam2_config)
        generator = Sam2MaskInputGenerator(
            task="image-segmentation",
            normalized_config=norm_config,
            batch_size=BATCH_SIZE,
        )

        flag = generator.generate("use_mask_input", framework="pt")

        assert flag.shape == (BATCH_SIZE,)
        # Default is 0.0 (first iteration, don't use mask)
        assert flag[0] == 0.0
