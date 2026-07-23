# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for resolve_task edge cases with explicit task / model_class.

Tests edge cases not covered by test_detect_task_and_class.py:
- User task with model_type=None in config (specialization silently skipped)
- User model_class with incompatible task+architecture (currently unvalidated)
"""

from unittest.mock import MagicMock

import pytest

from winml.modelkit.loader.resolution import TaskSource, resolve_task


class TestUserTaskModelTypeNone:
    """User specified task, but config has model_type=None.

    When model_type is None, specialization lookup is silently skipped
    and TasksManager is used directly.
    """

    def test_task_resolved_via_tasksmanager_when_model_type_none(self):
        """With model_type=None, TasksManager resolves model class directly."""
        config = MagicMock()
        config.model_type = None
        config.architectures = ["BertForSequenceClassification"]
        config._name_or_path = ""

        r = resolve_task(config, task="text-classification")

        assert r.task == "text-classification"
        assert r.source == TaskSource.USER_TASK
        # TasksManager should still resolve the class even without model_type
        assert "Classification" in r.model_class.__name__

    def test_specialization_skipped_when_model_type_none(self):
        """CLIP specialization is NOT applied when model_type=None."""
        config = MagicMock()
        config.model_type = None
        config.architectures = ["CLIPModel"]
        config._name_or_path = ""

        # feature-extraction without model_type should NOT trigger CLIP specialization
        r = resolve_task(config, task="feature-extraction")

        assert r.task == "feature-extraction"
        # Should be TasksManager default, not CLIPTextModelWithProjection
        assert r.model_class.__name__ != "CLIPTextModelWithProjection"


class TestUserTaskOriginalPreserved:
    """User task name is preserved verbatim in the resolution."""

    def test_alias_task_returns_original(self):
        """Task aliases are normalized internally but original is returned."""
        config = MagicMock()
        config.model_type = "bert"
        config.architectures = ["BertForMaskedLM"]
        config._name_or_path = ""

        # "masked-lm" normalizes to "fill-mask" internally
        r = resolve_task(config, task="masked-lm")

        # Returns original task name, not normalized
        assert r.task == "masked-lm"
        assert r.source == TaskSource.USER_TASK


class TestUserModelClassEdgeCases:
    """model_class specified edge cases."""

    def test_model_class_with_task_auto_detected(self):
        """model_class with task=None auto-detects task."""
        config = MagicMock()
        config.model_type = "resnet"
        config.architectures = ["ResNetForImageClassification"]
        config._name_or_path = ""

        r = resolve_task(config, model_class="AutoModelForImageClassification")

        assert r.task == "image-classification"
        assert r.source == TaskSource.USER_CLASS
        assert "ImageClassification" in r.model_class.__name__

    def test_invalid_model_class_raises_error(self):
        """Non-existent model_class raises ValueError."""
        config = MagicMock()
        config.model_type = "bert"
        config.architectures = ["BertForSequenceClassification"]
        config._name_or_path = ""

        with pytest.raises(ValueError, match="not found"):
            resolve_task(
                config,
                task="text-classification",
                model_class="NonExistentModelClass",
            )

    def test_task_normalized_with_model_class(self):
        """Task is normalized when model_class is provided."""
        config = MagicMock()
        config.model_type = "bert"
        config.architectures = ["BertForMaskedLM"]
        config._name_or_path = ""

        # "masked-lm" should normalize to "fill-mask" for TasksManager lookup
        r = resolve_task(
            config,
            task="masked-lm",
            model_class="AutoModelForMaskedLM",
        )

        # Task is normalized (unlike the user-task path which preserves the original)
        assert r.task == "fill-mask"
        assert r.source == TaskSource.USER_CLASS


class TestUnderscoreModelTypePassedToTasksManager:
    """Regression: model_type with underscores must reach TasksManager un-normalized.

    Optimum registers some model types with underscores (e.g. speech_to_text).
    Converting to hyphens (speech-to-text) prevents Optimum from matching the
    correct AutoModel class. See PR review comment on explicit ASR resolution.
    """

    def test_speech_to_text_resolves_seq2seq_class(self):
        """speech_to_text (underscore) resolves AutoModelForSpeechSeq2Seq via TasksManager."""
        config = MagicMock()
        config.model_type = "speech_to_text"
        config.architectures = ["Speech2TextForConditionalGeneration"]
        config._name_or_path = ""

        r = resolve_task(config, task="automatic-speech-recognition")

        assert r.task == "automatic-speech-recognition"
        assert r.source == TaskSource.USER_TASK
        # TasksManager should resolve to SpeechSeq2Seq for speech_to_text
        assert "Seq2Seq" in r.model_class.__name__ or "Speech" in r.model_class.__name__
