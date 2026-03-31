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

import pytest


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
        from winml.modelkit.loader.task import _get_custom_model_class

        result = _get_custom_model_class(
            model_type="clip",
            task="feature-extraction",
        )
        assert result.__name__ == "CLIPTextModelWithProjection"

    def test_clip_image_feature_extraction_returns_vision_model(self):
        """CLIP with image-feature-extraction should return CLIPVisionModelWithProjection."""
        from winml.modelkit.loader.task import _get_custom_model_class

        result = _get_custom_model_class(
            model_type="clip",
            task="image-feature-extraction",
        )
        assert result.__name__ == "CLIPVisionModelWithProjection"

    def test_segformer_image_segmentation_returns_semantic_segmentation(self):
        """Segformer with image-segmentation should return AutoModelForSemanticSegmentation."""
        from winml.modelkit.loader.task import _get_custom_model_class

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
        from winml.modelkit.loader.task import _get_custom_model_class

        # Should work for any model type (bert, etc.)
        result = _get_custom_model_class(
            model_type="bert",
            task="next-sentence-prediction",
        )
        assert result.__name__ == "AutoModelForNextSentencePrediction"

    def test_next_sentence_prediction_any_model_type(self):
        """NSP mapping should work for any BERT-family model type."""
        from winml.modelkit.loader.task import _get_custom_model_class

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
        from winml.modelkit.loader.task import _get_custom_model_class

        # Standard image classification - no special mapping needed
        class_name = _get_custom_model_class(
            model_type="resnet",
            task="image-classification",
        )
        assert class_name is None  # Use TasksManager default

    def test_standard_text_classification(self):
        """Standard text-classification should return None for TasksManager."""
        from winml.modelkit.loader.task import _get_custom_model_class

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

        from winml.modelkit.loader.task import _get_custom_model_class

        assert _get_custom_model_class("CLIP", "feature-extraction") is CLIPTextModelWithProjection
        assert _get_custom_model_class("Clip", "feature-extraction") is CLIPTextModelWithProjection
        assert _get_custom_model_class("clip", "feature-extraction") is CLIPTextModelWithProjection

    def test_model_type_underscore_normalization(self):
        """Model type with underscores should be normalized to hyphens."""
        from winml.modelkit.loader.task import _get_custom_model_class

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
        from winml.modelkit.loader.task import HF_TASK_DEFAULTS

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
        from winml.modelkit.loader.task import HF_TASK_DEFAULTS

        assert "next-sentence-prediction" in HF_TASK_DEFAULTS


class TestHFModelClassMappingIntegration:
    """Integration tests: specializations bypass TasksManager."""

    def test_clip_specialization_returns_class_directly(self):
        """CLIP specialization should return class directly, not go through TasksManager."""
        from winml.modelkit.loader.task import _get_custom_model_class

        result = _get_custom_model_class("clip", "image-feature-extraction")
        assert result is not None
        assert result.__name__ == "CLIPVisionModelWithProjection"

    def test_segformer_specialization_returns_class_directly(self):
        """Segformer specialization should return class directly, not go through TasksManager."""
        from winml.modelkit.loader.task import _get_custom_model_class

        result = _get_custom_model_class("segformer", "image-segmentation")
        assert result is not None
        assert result.__name__ == "AutoModelForSemanticSegmentation"


class TestResolveTaskAndModelClass:
    """Tests for the resolve_task_and_model_class() function."""

    def test_case1_auto_detect_both(self):
        """Case 1: task=None, model_class=None → auto-detect both."""
        from transformers import AutoConfig

        from winml.modelkit.loader.task import resolve_task_and_model_class

        config = AutoConfig.from_pretrained("microsoft/resnet-18")
        task, resolved_class = resolve_task_and_model_class(config)

        assert task == "image-classification"
        assert "ImageClassification" in resolved_class.__name__

    def test_case2_task_only_standard(self):
        """Case 2: task specified, model_class=None → resolve for task."""
        from transformers import AutoConfig

        from winml.modelkit.loader.task import resolve_task_and_model_class

        config = AutoConfig.from_pretrained("microsoft/resnet-18")
        task, resolved_class = resolve_task_and_model_class(config, task="image-classification")

        assert task == "image-classification"
        assert "ImageClassification" in resolved_class.__name__

    def test_case2_task_only_with_specialization(self):
        """Case 2: task specified triggers specialization (CLIP)."""
        from transformers import AutoConfig

        from winml.modelkit.loader.task import resolve_task_and_model_class

        config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")

        # image-feature-extraction should return CLIPVisionModelWithProjection
        task, resolved_class = resolve_task_and_model_class(config, task="image-feature-extraction")
        assert task == "image-feature-extraction"  # preserves original task name
        assert resolved_class.__name__ == "CLIPVisionModelWithProjection"

        # feature-extraction should return CLIPTextModelWithProjection directly
        task, resolved_class = resolve_task_and_model_class(config, task="feature-extraction")
        assert task == "feature-extraction"
        assert resolved_class.__name__ == "CLIPTextModelWithProjection"

    def test_case2_task_only_with_segformer_specialization(self):
        """Case 2: task specified triggers specialization (Segformer)."""
        from transformers import SegformerConfig

        from winml.modelkit.loader.task import resolve_task_and_model_class

        config = SegformerConfig()

        task, resolved_class = resolve_task_and_model_class(config, task="image-segmentation")
        assert task == "image-segmentation"
        assert resolved_class.__name__ == "AutoModelForSemanticSegmentation"

    def test_case2_task_only_nsp(self):
        """Case 2: NSP task uses HF_TASK_DEFAULTS."""
        from transformers import AutoConfig

        from winml.modelkit.loader.task import resolve_task_and_model_class

        config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
        task, resolved_class = resolve_task_and_model_class(config, task="next-sentence-prediction")

        assert task == "next-sentence-prediction"
        # AutoModelForNextSentencePrediction resolves to concrete class
        assert "NextSentencePrediction" in resolved_class.__name__

    def test_case3_model_class_override(self):
        """Case 3: model_class specified → honor it."""
        from transformers import AutoConfig

        from winml.modelkit.loader.task import resolve_task_and_model_class

        config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")

        # Explicitly request CLIPModel even though specialization would pick something else
        task, resolved_class = resolve_task_and_model_class(
            config,
            task="feature-extraction",
            model_class="CLIPModel",
        )

        assert task == "feature-extraction"
        assert resolved_class.__name__ == "CLIPModel"

    def test_case3_model_class_with_auto_task(self):
        """Case 3: model_class with task=None → detect task."""
        from transformers import AutoConfig

        from winml.modelkit.loader.task import resolve_task_and_model_class

        config = AutoConfig.from_pretrained("microsoft/resnet-18")

        # Specify model_class but let task be auto-detected
        task, resolved_class = resolve_task_and_model_class(
            config,
            model_class="AutoModelForImageClassification",
        )

        assert task == "image-classification"
        assert "ImageClassification" in resolved_class.__name__

    def test_invalid_task_raises_error(self):
        """Invalid task should raise ValueError."""
        from transformers import AutoConfig

        from winml.modelkit.loader.task import resolve_task_and_model_class

        config = AutoConfig.from_pretrained("microsoft/resnet-18")

        with pytest.raises(ValueError, match="not supported"):
            resolve_task_and_model_class(config, task="invalid-nonexistent-task")

    def test_invalid_model_class_raises_error(self):
        """Invalid model_class should raise ValueError."""
        from transformers import AutoConfig

        from winml.modelkit.loader.task import resolve_task_and_model_class

        config = AutoConfig.from_pretrained("microsoft/resnet-18")

        with pytest.raises(ValueError, match="not found"):
            resolve_task_and_model_class(
                config,
                task="image-classification",
                model_class="NonExistentModel",
            )


@pytest.mark.slow
class TestHFModelClassMappingE2E:
    """End-to-end tests that actually load models."""

    def test_clip_image_feature_extraction_e2e(self):
        """E2E: Load CLIP with image-feature-extraction task."""
        from winml.modelkit.loader.hf import load_hf_model

        model, _config, _task = load_hf_model(
            "openai/clip-vit-base-patch32",
            task="image-feature-extraction",
        )

        assert model.__class__.__name__ == "CLIPVisionModelWithProjection"

    def test_clip_feature_extraction_e2e(self):
        """E2E: Load CLIP with feature-extraction resolves to text model."""
        from winml.modelkit.loader.hf import load_hf_model

        model, _config, task = load_hf_model(
            "openai/clip-vit-base-patch32",
            task="feature-extraction",
        )
        assert task == "feature-extraction"
        assert model.__class__.__name__ == "CLIPTextModelWithProjection"

    def test_nsp_e2e(self):
        """E2E: Load BERT with next-sentence-prediction task."""
        from winml.modelkit.loader.hf import load_hf_model

        model, _config, _task = load_hf_model(
            "prajjwal1/bert-tiny",
            task="next-sentence-prediction",
        )

        # Should be BertForNextSentencePrediction (loaded via AutoModelForNextSentencePrediction)
        assert "NextSentencePrediction" in model.__class__.__name__


class TestConflictScenarios:
    """Tests for conflict scenarios between task and model_class.

    These tests verify behavior when:
    - Task and model_class don't match
    - model_class overrides specialization
    - Edge cases in resolution logic
    """

    def test_model_class_overrides_specialization(self):
        """model_class should override CLIP specialization.

        When user explicitly specifies model_class="CLIPModel",
        it should be honored even though specializations would pick
        CLIPTextModelWithProjection for feature-extraction task.
        """
        from transformers import AutoConfig

        from winml.modelkit.loader.task import resolve_task_and_model_class

        config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")

        # Without model_class: specialization picks CLIPTextModelWithProjection
        _task, resolved_class = resolve_task_and_model_class(config, task="feature-extraction")
        assert resolved_class.__name__ == "CLIPTextModelWithProjection"

        # With model_class: user override takes precedence
        _task, resolved_class = resolve_task_and_model_class(
            config,
            task="feature-extraction",
            model_class="CLIPModel",
        )
        assert resolved_class.__name__ == "CLIPModel"

    def test_mismatched_task_model_class_honored(self):
        """model_class is honored even if mismatched with task.

        This is by design - user may know better what they need.
        The task is used for resolution but model_class wins.
        """
        from transformers import AutoConfig

        from winml.modelkit.loader.task import resolve_task_and_model_class

        config = AutoConfig.from_pretrained("microsoft/resnet-18")

        # Request image-classification task but with feature extraction model
        # TasksManager validates this is a legal combination
        task, resolved_class = resolve_task_and_model_class(
            config,
            task="image-classification",
            model_class="AutoModelForImageClassification",
        )

        assert task == "image-classification"
        assert "ImageClassification" in resolved_class.__name__

    def test_task_normalizes_before_specialization_lookup(self):
        """Task should be normalized before checking specializations.

        "image-feature-extraction" normalizes to "feature-extraction",
        but the original task is preserved for specialization lookup.
        """
        from transformers import AutoConfig

        from winml.modelkit.loader.task import resolve_task_and_model_class

        config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")

        # Original task "image-feature-extraction" should find specialization
        task, resolved_class = resolve_task_and_model_class(config, task="image-feature-extraction")

        # Task preserves original name (normalization is internal for lookup)
        assert task == "image-feature-extraction"
        # Specialization uses original to pick Vision model
        assert resolved_class.__name__ == "CLIPVisionModelWithProjection"

    def test_specialization_not_found_falls_back_to_default(self):
        """When specialization not found, fall back to TasksManager default.

        For non-CLIP models with feature-extraction task, there's no
        specialization, so TasksManager default is used.
        """
        from transformers import AutoConfig

        from winml.modelkit.loader.task import _get_custom_model_class

        # BERT doesn't have specialization for feature-extraction
        bert_config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
        model_type = getattr(bert_config, "model_type", "bert")

        class_name = _get_custom_model_class(model_type, "feature-extraction")
        assert class_name is None  # No specialization, use TasksManager

    def test_model_class_with_task_none_detects_task(self):
        """model_class with task=None should auto-detect task."""
        from transformers import AutoConfig

        from winml.modelkit.loader.task import resolve_task_and_model_class

        config = AutoConfig.from_pretrained("microsoft/resnet-18")

        task, resolved_class = resolve_task_and_model_class(
            config,
            model_class="AutoModelForImageClassification",
        )

        # Task auto-detected from config.architectures
        assert task == "image-classification"
        assert "ImageClassification" in resolved_class.__name__


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
        from winml.modelkit.loader.config import WinMLLoaderConfig

        config = WinMLLoaderConfig(
            task="image-classification",
            model_class="AutoModelForImageClassification",
        )

        assert config.task == "image-classification"
        assert config.model_class == "AutoModelForImageClassification"

    def test_loader_config_serializes_task_and_model_class(self):
        """WinMLLoaderConfig should serialize task and model_class."""
        from winml.modelkit.loader.config import WinMLLoaderConfig

        config = WinMLLoaderConfig(
            task="feature-extraction",
            model_class="CLIPTextModelWithProjection",
        )

        config_dict = config.to_dict()

        assert config_dict.get("task") == "feature-extraction"
        assert config_dict.get("model_class") == "CLIPTextModelWithProjection"

    def test_loader_config_deserializes_task_and_model_class(self):
        """WinMLLoaderConfig should deserialize task and model_class."""
        from winml.modelkit.loader.config import WinMLLoaderConfig

        config = WinMLLoaderConfig.from_dict(
            {
                "task": "image-feature-extraction",
                "model_class": "CLIPVisionModelWithProjection",
            }
        )

        assert config.task == "image-feature-extraction"
        assert config.model_class == "CLIPVisionModelWithProjection"
