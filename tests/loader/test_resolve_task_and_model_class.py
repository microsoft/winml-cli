# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for resolve_task_and_model_class edge cases.

Tests edge cases not covered by test_detect_task_and_class.py:
- Case 2 with model_type=None in config (specialization silently skipped)
- Case 3 with incompatible task+architecture (currently unvalidated)
"""

from unittest.mock import MagicMock

import pytest

from winml.modelkit.loader.task import resolve_task_and_model_class


class TestCase2ModelTypeNone:
    """Case 2: User specified task, but config has model_type=None.

    When model_type is None, specialization lookup is silently skipped
    and TasksManager is used directly.
    """

    def test_task_resolved_via_tasksmanager_when_model_type_none(self):
        """With model_type=None, TasksManager resolves model class directly."""
        config = MagicMock()
        config.model_type = None
        config.architectures = ["BertForSequenceClassification"]

        task, resolved_class = resolve_task_and_model_class(
            config, task="text-classification"
        )

        assert task == "text-classification"
        # TasksManager should still resolve the class even without model_type
        assert "Classification" in resolved_class.__name__

    def test_specialization_skipped_when_model_type_none(self):
        """CLIP specialization is NOT applied when model_type=None."""
        config = MagicMock()
        config.model_type = None
        config.architectures = ["CLIPModel"]

        # feature-extraction without model_type should NOT trigger CLIP specialization
        task, resolved_class = resolve_task_and_model_class(
            config, task="feature-extraction"
        )

        assert task == "feature-extraction"
        # Should be TasksManager default, not CLIPTextModelWithProjection
        assert resolved_class.__name__ != "CLIPTextModelWithProjection"


class TestCase2OriginalTaskPreserved:
    """Case 2: Original task name is preserved in return value."""

    def test_alias_task_returns_original(self):
        """Task aliases are normalized internally but original is returned."""
        config = MagicMock()
        config.model_type = "bert"
        config.architectures = ["BertForMaskedLM"]

        # "masked-lm" normalizes to "fill-mask" internally
        task, resolved_class = resolve_task_and_model_class(
            config, task="masked-lm"
        )

        # Returns original task name, not normalized
        assert task == "masked-lm"


class TestCase3EdgeCases:
    """Case 3: model_class specified edge cases."""

    def test_model_class_with_task_auto_detected(self):
        """model_class with task=None auto-detects task."""
        config = MagicMock()
        config.model_type = "resnet"
        config.architectures = ["ResNetForImageClassification"]

        task, resolved_class = resolve_task_and_model_class(
            config, model_class="AutoModelForImageClassification"
        )

        assert task == "image-classification"
        assert "ImageClassification" in resolved_class.__name__

    def test_invalid_model_class_raises_error(self):
        """Non-existent model_class raises ValueError."""
        config = MagicMock()
        config.model_type = "bert"
        config.architectures = ["BertForSequenceClassification"]

        with pytest.raises(ValueError, match="not found"):
            resolve_task_and_model_class(
                config,
                task="text-classification",
                model_class="NonExistentModelClass",
            )

    def test_task_normalized_in_case3(self):
        """Task is normalized in Case 3 when provided."""
        config = MagicMock()
        config.model_type = "bert"
        config.architectures = ["BertForMaskedLM"]

        # "masked-lm" should normalize to "fill-mask" for TasksManager lookup
        task, resolved_class = resolve_task_and_model_class(
            config,
            task="masked-lm",
            model_class="AutoModelForMaskedLM",
        )

        # Task is normalized (unlike Case 2 which preserves original)
        assert task == "fill-mask"
