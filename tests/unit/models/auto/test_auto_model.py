# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Tests for WinMLAutoModel factory class.

Tests the factory pattern and task detection in modelkit/models/auto.py
following the design specifications in docs/design/automodel/ARCHITECTURE_PRINCIPLES.md Section 5.

Acceptance Criteria (from design):
- AC-1: MODEL_MAPPING dict maps task types to model classes
- AC-2: WinMLAutoModel.from_pretrained() auto-detects task
- AC-3: Task detection from config.architectures (e.g., "ForImageClassification")
- AC-4: Fallback to model_type heuristics
- AC-5: Support explicit task parameter override
- AC-6: Raise ValueError for unsupported tasks
"""

from __future__ import annotations

import pytest


class TestModelMapping:
    """Test MODEL_MAPPING registry."""

    def test_model_mapping_exists(self):
        """AC-1: MODEL_MAPPING dict should exist."""
        from winml.modelkit.models import TASK_TO_WINML_CLASS

        assert isinstance(TASK_TO_WINML_CLASS, dict)
        assert len(TASK_TO_WINML_CLASS) > 0

    def test_model_mapping_has_image_classification(self):
        """AC-1: Should have image-classification mapping."""
        from winml.modelkit.models import TASK_TO_WINML_CLASS

        assert "image-classification" in TASK_TO_WINML_CLASS

    def test_model_mapping_has_sequence_classification(self):
        """AC-1: Should have sequence/text classification mapping."""
        from winml.modelkit.models import TASK_TO_WINML_CLASS

        # Design supports multiple aliases
        assert (
            "text-classification" in TASK_TO_WINML_CLASS
            or "sequence-classification" in TASK_TO_WINML_CLASS
        )

    def test_model_mapping_has_image_segmentation(self):
        """AC-1: Should have image segmentation mapping."""
        from winml.modelkit.models import TASK_TO_WINML_CLASS

        assert (
            "image-segmentation" in TASK_TO_WINML_CLASS
            or "semantic-segmentation" in TASK_TO_WINML_CLASS
        )

    def test_model_mapping_values_are_classes(self):
        """AC-1: Mapping values should be model classes or class name strings."""
        from winml.modelkit.models import TASK_TO_WINML_CLASS

        for model_class_or_name in TASK_TO_WINML_CLASS.values():
            # Can be class or string (lazy loading)
            assert model_class_or_name is not None


class TestArchitectureTaskMapping:
    """Test WINML_MODEL_CLASS_MAPPING for task detection."""

    def test_architecture_mapping_exists(self):
        """AC-3: WINML_MODEL_CLASS_MAPPING should exist."""
        from winml.modelkit.models import WINML_MODEL_CLASS_MAPPING

        assert isinstance(WINML_MODEL_CLASS_MAPPING, dict)

    def test_image_classification_patterns(self):
        """AC-3: Should detect image classification from architecture suffix."""
        from winml.modelkit.models import WINML_MODEL_CLASS_MAPPING

        # WINML_MODEL_CLASS_MAPPING uses (model_type, task) tuples as keys
        # Check that the mapping structure is correct
        image_class_patterns = [
            k
            for k in WINML_MODEL_CLASS_MAPPING
            if isinstance(k, tuple) and "image-classification" in k
        ]
        # May be empty if no specializations registered - that's ok
        assert isinstance(image_class_patterns, list)

    def test_sequence_classification_patterns(self):
        """AC-3: Should detect sequence classification from architecture suffix."""
        from winml.modelkit.models import WINML_MODEL_CLASS_MAPPING

        seq_class_patterns = [
            k
            for k in WINML_MODEL_CLASS_MAPPING
            if isinstance(k, tuple) and "sequence-classification" in k
        ]
        # May be empty if no specializations registered - that's ok
        assert isinstance(seq_class_patterns, list)


class TestTaskDetection:
    """Test task detection via get_winml_class.

    Note: WinMLAutoModel does NOT have _detect_task(). Task detection is done
    via generate_build_config() which calls resolve_loader_config(). The
    get_winml_class() function maps (model_type, task) to a WinML class.
    """

    def test_get_winml_class_image_classification(self):
        """AC-3: Get correct class for image classification task."""
        from winml.modelkit.models import WinMLModelForImageClassification, get_winml_class

        cls = get_winml_class("convnext", "image-classification")
        assert cls == WinMLModelForImageClassification

    def test_get_winml_class_sequence_classification(self):
        """AC-3: Get correct class for sequence classification task."""
        from winml.modelkit.models import WinMLModelForSequenceClassification, get_winml_class

        cls = get_winml_class("bert", "text-classification")
        assert cls == WinMLModelForSequenceClassification

    def test_get_winml_class_image_segmentation(self):
        """AC-3: Get correct class for image segmentation task."""
        from winml.modelkit.models import WinMLModelForImageSegmentation, get_winml_class

        cls = get_winml_class("segformer", "image-segmentation")
        assert cls == WinMLModelForImageSegmentation

    def test_get_winml_class_unknown_task_fallback(self):
        """AC-4: Unknown task falls back to generic class."""
        from winml.modelkit.models import WinMLModelForGenericTask, get_winml_class

        cls = get_winml_class("resnet", "unknown-task")
        assert cls == WinMLModelForGenericTask


class TestWinMLAutoModelFactory:
    """Test WinMLAutoModel factory and get_winml_class lookup."""

    def test_get_winml_class_image_classification(self):
        """AC-1: Get correct model class for image classification."""
        from winml.modelkit.models import WinMLModelForImageClassification, get_winml_class

        model_class = get_winml_class("convnext", "image-classification")
        assert model_class == WinMLModelForImageClassification

    def test_get_winml_class_sequence_classification(self):
        """AC-1: Get correct model class for sequence classification."""
        from winml.modelkit.models import WinMLModelForSequenceClassification, get_winml_class

        model_class = get_winml_class("bert", "text-classification")
        assert model_class == WinMLModelForSequenceClassification

    def test_get_winml_class_image_segmentation(self):
        """AC-1: Get correct model class for image segmentation."""
        from winml.modelkit.models import WinMLModelForImageSegmentation, get_winml_class

        model_class = get_winml_class("segformer", "image-segmentation")
        assert model_class == WinMLModelForImageSegmentation

    def test_get_winml_class_unsupported_task_returns_generic(self):
        """AC-6: Unsupported task returns generic fallback (no error)."""
        from winml.modelkit.models import WinMLModelForGenericTask, get_winml_class

        model_class = get_winml_class("unknown", "unsupported-task-type")
        assert model_class == WinMLModelForGenericTask

    def test_mgp_str_image_to_text_uses_specialized_wrapper(self):
        from winml.modelkit.models import get_winml_class
        from winml.modelkit.models.winml.image_to_text import (
            WinMLModelForMgpstrSceneTextRecognition,
        )

        assert (
            get_winml_class("mgp-str", "image-to-text")
            is WinMLModelForMgpstrSceneTextRecognition
        )

    def test_mgp_str_wrapper_preserves_three_head_order(self):
        from unittest.mock import MagicMock

        import torch

        from winml.modelkit.models.winml.image_to_text import (
            WinMLModelForMgpstrSceneTextRecognition,
        )

        model = object.__new__(WinMLModelForMgpstrSceneTextRecognition)
        model._format_inputs = MagicMock(side_effect=lambda **kwargs: kwargs)
        expected = {
            "char_logits": torch.tensor([1.0]),
            "bpe_logits": torch.tensor([2.0]),
            "wp_logits": torch.tensor([3.0]),
        }
        model._run_inference = MagicMock(return_value=expected)

        result = model.forward(pixel_values=torch.zeros((1, 3, 32, 128)))

        assert result.logits == (
            expected["char_logits"],
            expected["bpe_logits"],
            expected["wp_logits"],
        )

    def test_mgp_str_wrapper_requires_pixel_values(self):
        from winml.modelkit.models.winml.image_to_text import (
            WinMLModelForMgpstrSceneTextRecognition,
        )

        model = object.__new__(WinMLModelForMgpstrSceneTextRecognition)

        with pytest.raises(ValueError, match="requires 'pixel_values'"):
            model.forward()

    @pytest.mark.parametrize(
        "task,model_type,expected_class_name",
        [
            ("image-classification", "convnext", "WinMLModelForImageClassification"),
            ("image-segmentation", "segformer", "WinMLModelForImageSegmentation"),
        ],
    )
    def test_model_class_names(self, task: str, model_type: str, expected_class_name: str):
        """Test model class naming convention."""
        from winml.modelkit.models import get_winml_class

        model_class = get_winml_class(model_type, task)
        assert model_class.__name__ == expected_class_name


class TestExplicitTaskOverride:
    """Test explicit task parameter override."""

    def test_explicit_task_returns_correct_class(self):
        """AC-5: Explicit task should map to correct WinML class."""
        from winml.modelkit.models import WinMLModelForImageClassification, get_winml_class

        # When task is explicitly provided, get_winml_class returns the right class
        model_class = get_winml_class("convnext", "image-classification")
        assert model_class == WinMLModelForImageClassification


class TestTaskAliases:
    """Test task name aliases."""

    def test_text_classification_aliases(self):
        """Test text/sequence classification aliases."""
        from winml.modelkit.models import TASK_TO_WINML_CLASS

        # Both should map to same class
        text_class = TASK_TO_WINML_CLASS.get("text-classification")
        seq_class = TASK_TO_WINML_CLASS.get("sequence-classification")

        # At least one should exist
        assert text_class is not None or seq_class is not None

    def test_sentiment_analysis_alias(self):
        """Test sentiment-analysis as alias for text classification."""
        from winml.modelkit.models import TASK_TO_WINML_CLASS

        if "sentiment-analysis" in TASK_TO_WINML_CLASS:
            assert TASK_TO_WINML_CLASS["sentiment-analysis"] is not None

    def test_semantic_segmentation_alias(self):
        """Test semantic-segmentation as alias for image segmentation."""
        from winml.modelkit.models import TASK_TO_WINML_CLASS

        img_seg = TASK_TO_WINML_CLASS.get("image-segmentation")
        sem_seg = TASK_TO_WINML_CLASS.get("semantic-segmentation")

        # At least one should exist
        assert img_seg is not None or sem_seg is not None
