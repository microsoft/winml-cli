# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for HuggingFace Model Class Two-Level Mapping.

This module tests the mapping system that resolves the correct HuggingFace
model class based on (model_type, task) combinations.

Design:
    Level 1 (Specializations): (model_type, task) → specific HF class name
        - ("clip", "feature-extraction") → "CLIPTextModelWithProjection"
        - ("clip", "image-feature-extraction") → "CLIPVisionModelWithProjection"

    Level 2 (Task Defaults): task → default HF class name
        - "next-sentence-prediction" → "AutoModelForNextSentencePrediction"
        (for tasks not supported by TasksManager)

    Level 3 (TasksManager Fallback): Delegate to TasksManager
        - Uses TasksManager.get_model_class_for_task() for standard tasks

Usage:
    model_class_name = _get_custom_model_class(model_type="clip", task="image-feature-extraction")
    # Returns "CLIPVisionModelWithProjection"

    model_class_name = _get_custom_model_class(model_type="bert", task="next-sentence-prediction")
    # Returns "AutoModelForNextSentencePrediction"
"""


class TestHFModelClassMappingDesign:
    """Design tests for HF model class mapping system.

    These tests define the expected behavior of the mapping system.
    Implementation should make these tests pass.
    """

    # =========================================================================
    # Level 1: Specializations (model_type, task) → specific class
    # =========================================================================

    def test_clip_feature_extraction_returns_text_model(self):
        """CLIP with feature-extraction should return CLIPTextModelWithProjection."""
        from winml.modelkit.loader.resolution import _get_custom_model_class

        result = _get_custom_model_class(
            model_type="clip",
            task="feature-extraction",
        )
        assert result.__name__ == "CLIPTextModelWithProjection"

    def test_clip_image_feature_extraction_returns_vision_model(self):
        """CLIP with image-feature-extraction should return CLIPVisionModelWithProjection."""
        from winml.modelkit.loader.resolution import _get_custom_model_class

        result = _get_custom_model_class(
            model_type="clip",
            task="image-feature-extraction",
        )
        assert result.__name__ == "CLIPVisionModelWithProjection"

    def test_segformer_image_segmentation_returns_semantic_segmentation(self):
        """Segformer with image-segmentation should return AutoModelForSemanticSegmentation."""
        from winml.modelkit.loader.resolution import _get_custom_model_class

        result = _get_custom_model_class(
            model_type="segformer",
            task="image-segmentation",
        )
        assert result.__name__ == "AutoModelForSemanticSegmentation"

    # =========================================================================
    # Level 2: Task Defaults (tasks not in TasksManager)
    # =========================================================================

    def test_next_sentence_prediction_returns_auto_model(self):
        """next-sentence-prediction should return AutoModelForNextSentencePrediction.

        NSP is not supported by Optimum's TasksManager, so we provide a default.
        """
        from winml.modelkit.loader.resolution import _get_custom_model_class

        # Should work for any model type (bert, etc.)
        result = _get_custom_model_class(
            model_type="bert",
            task="next-sentence-prediction",
        )
        assert result.__name__ == "AutoModelForNextSentencePrediction"

    def test_next_sentence_prediction_any_model_type(self):
        """NSP mapping should work for any BERT-family model type."""
        from winml.modelkit.loader.resolution import _get_custom_model_class

        for model_type in ["bert", "roberta", "albert", "distilbert"]:
            result = _get_custom_model_class(
                model_type=model_type,
                task="next-sentence-prediction",
            )
            assert result.__name__ == "AutoModelForNextSentencePrediction"

    # =========================================================================
    # Level 3: TasksManager Fallback (standard tasks)
    # =========================================================================

    def test_standard_task_returns_none_for_tasks_manager(self):
        """Standard tasks should return None to use TasksManager default.

        When there's no specialization and no task default, the function
        returns None, signaling to use TasksManager's default behavior.
        """
        from winml.modelkit.loader.resolution import _get_custom_model_class

        # Standard image classification - no special mapping needed
        class_name = _get_custom_model_class(
            model_type="resnet",
            task="image-classification",
        )
        assert class_name is None  # Use TasksManager default

    def test_standard_text_classification(self):
        """Standard text-classification should return None for TasksManager."""
        from winml.modelkit.loader.resolution import _get_custom_model_class

        class_name = _get_custom_model_class(
            model_type="bert",
            task="text-classification",
        )
        assert class_name is None  # Use TasksManager default

    # =========================================================================
    # Edge Cases and Normalization
    # =========================================================================

    def test_model_type_case_insensitive(self):
        """Model type lookup should be case-insensitive."""
        # All these should return the same class
        from transformers import CLIPTextModelWithProjection

        from winml.modelkit.loader.resolution import _get_custom_model_class

        assert _get_custom_model_class("CLIP", "feature-extraction") is CLIPTextModelWithProjection
        assert _get_custom_model_class("Clip", "feature-extraction") is CLIPTextModelWithProjection
        assert _get_custom_model_class("clip", "feature-extraction") is CLIPTextModelWithProjection

    def test_model_type_underscore_normalization(self):
        """Model type with underscores should be normalized to hyphens."""
        from winml.modelkit.loader.resolution import _get_custom_model_class

        # clip_vision_model should match clip-vision-model
        class_name = _get_custom_model_class(
            model_type="clip_vision_model",
            task="feature-extraction",
        )
        # This might return None or a specific class depending on design
        # For now, test that it doesn't crash
        assert class_name is None or isinstance(class_name, type)


class TestHFModelClassMappingRegistry:
    """Tests for the mapping registry structure."""

    def test_class_mapping_registry_exists(self):
        """HF_MODEL_CLASS_MAPPING should be accessible from models package."""
        from winml.modelkit.models import HF_MODEL_CLASS_MAPPING

        assert isinstance(HF_MODEL_CLASS_MAPPING, dict)

    def test_task_defaults_registry_exists(self):
        """HF_TASK_DEFAULTS registry should exist."""
        from winml.modelkit.loader import HF_TASK_DEFAULTS

        assert isinstance(HF_TASK_DEFAULTS, dict)

    def test_clip_class_mapping_registered(self):
        """CLIP class mappings should be in the registry.

        Tests that models/hf/clip/MODEL_CLASS_MAPPING is properly aggregated.
        """
        from winml.modelkit.models import HF_MODEL_CLASS_MAPPING

        assert ("clip", "feature-extraction") in HF_MODEL_CLASS_MAPPING
        assert ("clip", "image-feature-extraction") in HF_MODEL_CLASS_MAPPING

    def test_segformer_class_mapping_registered(self):
        """Segformer class mapping should be in the registry."""
        from winml.modelkit.models import HF_MODEL_CLASS_MAPPING

        assert ("segformer", "image-segmentation") in HF_MODEL_CLASS_MAPPING

    def test_segformer_module_class_mapping_structure(self):
        """Segformer module should export MODEL_CLASS_MAPPING dict."""
        from winml.modelkit.models.hf.segformer import MODEL_CLASS_MAPPING

        assert isinstance(MODEL_CLASS_MAPPING, dict)
        assert ("segformer", "image-segmentation") in MODEL_CLASS_MAPPING

    def test_clip_module_class_mapping_structure(self):
        """CLIP module should export MODEL_CLASS_MAPPING dict.

        Tests the modular design: each model folder exports its class mappings.
        """
        from winml.modelkit.models.hf.clip import MODEL_CLASS_MAPPING

        assert isinstance(MODEL_CLASS_MAPPING, dict)
        assert ("clip", "feature-extraction") in MODEL_CLASS_MAPPING
        assert ("clip", "image-feature-extraction") in MODEL_CLASS_MAPPING

    def test_nsp_task_default_registered(self):
        """NSP task default should be in the registry."""
        from winml.modelkit.loader import HF_TASK_DEFAULTS

        assert "next-sentence-prediction" in HF_TASK_DEFAULTS


class TestHFModelClassMappingIntegration:
    """Integration tests: specializations bypass TasksManager."""

    def test_clip_specialization_returns_class_directly(self):
        """CLIP specialization should return class directly, not go through TasksManager."""
        from winml.modelkit.loader.resolution import _get_custom_model_class

        result = _get_custom_model_class("clip", "image-feature-extraction")
        assert result is not None
        assert result.__name__ == "CLIPVisionModelWithProjection"

    def test_segformer_specialization_returns_class_directly(self):
        """Segformer specialization should return class directly, not go through TasksManager."""
        from winml.modelkit.loader.resolution import _get_custom_model_class

        result = _get_custom_model_class("segformer", "image-segmentation")
        assert result is not None
        assert result.__name__ == "AutoModelForSemanticSegmentation"


class TestCLIIntegration:
    """Tests for CLI integration with task/model_class resolution.

    These tests verify the loader config properly handles:
    - task field
    - model_class field
    - Serialization/deserialization
    """

    def test_cli_export_command_exists(self):
        """CLI export command should be available via click."""
        from click.testing import CliRunner

        from winml.modelkit.commands.export import export

        runner = CliRunner()
        result = runner.invoke(export, ["--help"])

        # Export command should exist and show help
        assert result.exit_code == 0
        assert "--model" in result.output or "-m" in result.output

    def test_loader_config_supports_task_and_model_class(self):
        """WinMLLoaderConfig should support task and model_class fields."""
        from winml.modelkit.loader import WinMLLoaderConfig

        config = WinMLLoaderConfig(
            task="image-classification",
            model_class="AutoModelForImageClassification",
        )

        assert config.task == "image-classification"
        assert config.model_class == "AutoModelForImageClassification"

    def test_loader_config_serializes_task_and_model_class(self):
        """WinMLLoaderConfig should serialize task and model_class."""
        from winml.modelkit.loader import WinMLLoaderConfig

        config = WinMLLoaderConfig(
            task="feature-extraction",
            model_class="CLIPTextModelWithProjection",
        )

        config_dict = config.to_dict()

        assert config_dict.get("task") == "feature-extraction"
        assert config_dict.get("model_class") == "CLIPTextModelWithProjection"

    def test_loader_config_deserializes_task_and_model_class(self):
        """WinMLLoaderConfig should deserialize task and model_class."""
        from winml.modelkit.loader import WinMLLoaderConfig

        config = WinMLLoaderConfig.from_dict(
            {
                "task": "image-feature-extraction",
                "model_class": "CLIPVisionModelWithProjection",
            }
        )

        assert config.task == "image-feature-extraction"
        assert config.model_class == "CLIPVisionModelWithProjection"
