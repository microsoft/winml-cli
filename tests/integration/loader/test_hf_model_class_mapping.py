# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Integration tests for HF model class mapping that download real models.

Extracted from tests/unit/loader/test_hf_model_class_mapping.py.
These tests require network access. Use `pytest -m "not slow"` to skip them.
"""

import pytest

from winml.modelkit.loader import load_hf_model


@pytest.mark.slow
class TestHFModelClassMappingE2E:
    """End-to-end tests that actually load models."""

    def test_clip_image_feature_extraction_e2e(self):
        """E2E: Load CLIP with image-feature-extraction task."""
        model, _config, _task = load_hf_model(
            "openai/clip-vit-base-patch32",
            task="image-feature-extraction",
        )

        assert model.__class__.__name__ == "CLIPVisionModelWithProjection"

    def test_clip_feature_extraction_e2e(self):
        """E2E: Load CLIP with feature-extraction resolves to text model."""
        model, _config, task = load_hf_model(
            "openai/clip-vit-base-patch32",
            task="feature-extraction",
        )
        assert task == "feature-extraction"
        assert model.__class__.__name__ == "CLIPTextModelWithProjection"

    def test_nsp_e2e(self):
        """E2E: Load BERT with next-sentence-prediction task."""
        model, _config, _task = load_hf_model(
            "prajjwal1/bert-tiny",
            task="next-sentence-prediction",
        )

        # Should be BertForNextSentencePrediction (loaded via AutoModelForNextSentencePrediction)
        assert "NextSentencePrediction" in model.__class__.__name__


class TestResolveTaskAndModelClass:
    """Tests for the resolve_task() function.

    Moved from tests/unit/ — these tests call AutoConfig.from_pretrained()
    which downloads model configs from HuggingFace Hub.
    """

    def test_case1_auto_detect_both(self):
        """Case 1: task=None, model_class=None → auto-detect both."""
        from transformers import AutoConfig

        from winml.modelkit.loader import resolve_task

        config = AutoConfig.from_pretrained("microsoft/resnet-18")
        r = resolve_task(config)

        assert r.task == "image-classification"
        assert "ImageClassification" in r.model_class.__name__

    def test_case2_task_only_standard(self):
        """Case 2: task specified, model_class=None → resolve for task."""
        from transformers import AutoConfig

        from winml.modelkit.loader import resolve_task

        config = AutoConfig.from_pretrained("microsoft/resnet-18")
        r = resolve_task(config, task="image-classification")

        assert r.task == "image-classification"
        assert "ImageClassification" in r.model_class.__name__

    def test_case2_task_only_with_specialization(self):
        """Case 2: task specified triggers specialization (CLIP)."""
        from transformers import AutoConfig

        from winml.modelkit.loader import resolve_task

        config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")

        # image-feature-extraction should return CLIPVisionModelWithProjection
        r = resolve_task(config, task="image-feature-extraction")
        assert r.task == "image-feature-extraction"  # preserves original task name
        assert r.model_class.__name__ == "CLIPVisionModelWithProjection"

        # feature-extraction should return CLIPTextModelWithProjection directly
        r = resolve_task(config, task="feature-extraction")
        assert r.task == "feature-extraction"
        assert r.model_class.__name__ == "CLIPTextModelWithProjection"

    def test_case2_task_only_with_segformer_specialization(self):
        """Case 2: task specified triggers specialization (Segformer)."""
        from transformers import SegformerConfig

        from winml.modelkit.loader import resolve_task

        config = SegformerConfig()

        r = resolve_task(config, task="image-segmentation")
        assert r.task == "image-segmentation"
        assert r.model_class.__name__ == "AutoModelForSemanticSegmentation"

    def test_case2_task_only_nsp(self):
        """Case 2: NSP task uses HF_TASK_DEFAULTS."""
        from transformers import AutoConfig

        from winml.modelkit.loader import resolve_task

        config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
        r = resolve_task(config, task="next-sentence-prediction")

        assert r.task == "next-sentence-prediction"
        # AutoModelForNextSentencePrediction resolves to concrete class
        assert "NextSentencePrediction" in r.model_class.__name__

    def test_case3_model_class_override(self):
        """Case 3: model_class specified → honor it."""
        from transformers import AutoConfig

        from winml.modelkit.loader import resolve_task

        config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")

        # Explicitly request CLIPModel even though specialization would pick something else
        r = resolve_task(
            config,
            task="feature-extraction",
            model_class="CLIPModel",
        )

        assert r.task == "feature-extraction"
        assert r.model_class.__name__ == "CLIPModel"

    def test_case3_model_class_with_auto_task(self):
        """Case 3: model_class with task=None → detect task."""
        from transformers import AutoConfig

        from winml.modelkit.loader import resolve_task

        config = AutoConfig.from_pretrained("microsoft/resnet-18")

        # Specify model_class but let task be auto-detected
        r = resolve_task(
            config,
            model_class="AutoModelForImageClassification",
        )

        assert r.task == "image-classification"
        assert "ImageClassification" in r.model_class.__name__

    def test_invalid_task_raises_error(self):
        """Invalid task should raise ValueError."""
        from transformers import AutoConfig

        from winml.modelkit.loader import resolve_task

        config = AutoConfig.from_pretrained("microsoft/resnet-18")

        with pytest.raises(ValueError, match="not supported"):
            resolve_task(config, task="invalid-nonexistent-task")

    def test_invalid_model_class_raises_error(self):
        """Invalid model_class should raise ValueError."""
        from transformers import AutoConfig

        from winml.modelkit.loader import resolve_task

        config = AutoConfig.from_pretrained("microsoft/resnet-18")

        with pytest.raises(ValueError, match="not found"):
            resolve_task(
                config,
                task="image-classification",
                model_class="NonExistentModel",
            )


class TestConflictScenarios:
    """Tests for conflict scenarios between task and model_class.

    Moved from tests/unit/ — these tests call AutoConfig.from_pretrained()
    which downloads model configs from HuggingFace Hub.

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

        from winml.modelkit.loader import resolve_task

        config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")

        # Without model_class: specialization picks CLIPTextModelWithProjection
        r = resolve_task(config, task="feature-extraction")
        assert r.model_class.__name__ == "CLIPTextModelWithProjection"

        # With model_class: user override takes precedence
        r = resolve_task(
            config,
            task="feature-extraction",
            model_class="CLIPModel",
        )
        assert r.model_class.__name__ == "CLIPModel"

    def test_mismatched_task_model_class_honored(self):
        """model_class is honored even if mismatched with task.

        This is by design - user may know better what they need.
        The task is used for resolution but model_class wins.
        """
        from transformers import AutoConfig

        from winml.modelkit.loader import resolve_task

        config = AutoConfig.from_pretrained("microsoft/resnet-18")

        # Request image-classification task but with feature extraction model
        # TasksManager validates this is a legal combination
        r = resolve_task(
            config,
            task="image-classification",
            model_class="AutoModelForImageClassification",
        )

        assert r.task == "image-classification"
        assert "ImageClassification" in r.model_class.__name__

    def test_task_normalizes_before_specialization_lookup(self):
        """Task should be normalized before checking specializations.

        "image-feature-extraction" normalizes to "feature-extraction",
        but the original task is preserved for specialization lookup.
        """
        from transformers import AutoConfig

        from winml.modelkit.loader import resolve_task

        config = AutoConfig.from_pretrained("openai/clip-vit-base-patch32")

        # Original task "image-feature-extraction" should find specialization
        r = resolve_task(config, task="image-feature-extraction")

        # Task preserves original name (normalization is internal for lookup)
        assert r.task == "image-feature-extraction"
        # Specialization uses original to pick Vision model
        assert r.model_class.__name__ == "CLIPVisionModelWithProjection"

    def test_specialization_not_found_falls_back_to_default(self):
        """When specialization not found, fall back to TasksManager default.

        For non-CLIP models with feature-extraction task, there's no
        specialization, so TasksManager default is used.
        """
        from transformers import AutoConfig

        from winml.modelkit.loader.resolution import _get_custom_model_class

        # BERT doesn't have specialization for feature-extraction
        bert_config = AutoConfig.from_pretrained("prajjwal1/bert-tiny")
        model_type = getattr(bert_config, "model_type", "bert")

        class_name = _get_custom_model_class(model_type, "feature-extraction")
        assert class_name is None  # No specialization, use TasksManager

    def test_model_class_with_task_none_detects_task(self):
        """model_class with task=None should auto-detect task."""
        from transformers import AutoConfig

        from winml.modelkit.loader import resolve_task

        config = AutoConfig.from_pretrained("microsoft/resnet-18")

        r = resolve_task(
            config,
            model_class="AutoModelForImageClassification",
        )

        # Task auto-detected from config.architectures
        assert r.task == "image-classification"
        assert "ImageClassification" in r.model_class.__name__
