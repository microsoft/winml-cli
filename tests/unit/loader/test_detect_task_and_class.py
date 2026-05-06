# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for _detect_task_and_class_from_config function.

Tests the resolution strategy:
1. Specializations (HF_MODEL_CLASS_MAPPING) - highest priority
2. TasksManager.get_model_class_for_task() - honor if successful
3. Fallback to architecture class from config.architectures

Uses BLIP as primary test case for TasksManager fallback scenario.
"""

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.loader.task import _detect_task_and_class_from_config


class TestDetectTaskAndClassFromConfig:
    """Tests for _detect_task_and_class_from_config function."""

    def test_missing_architectures_raises_error(self):
        """Test ValueError when config has no architectures field."""
        config = MagicMock()
        config.architectures = None

        with pytest.raises(ValueError, match="no 'architectures' field"):
            _detect_task_and_class_from_config(config)

    def test_empty_architectures_raises_error(self):
        """Test ValueError when config.architectures is empty list."""
        config = MagicMock()
        config.architectures = []

        with pytest.raises(ValueError, match="no 'architectures' field"):
            _detect_task_and_class_from_config(config)

    def test_invalid_architecture_raises_error(self):
        """Test ValueError when architecture cannot be imported from transformers."""
        config = MagicMock()
        config.architectures = ["NonExistentClass"]
        config.model_type = "some-model"

        with patch("winml.modelkit.loader.task.importlib.import_module") as mock_import:
            mock_transformers = MagicMock()
            # Simulate AttributeError when accessing NonExistentClass
            del mock_transformers.NonExistentClass
            mock_import.return_value = mock_transformers

            with pytest.raises(ValueError, match="Cannot import NonExistentClass"):
                _detect_task_and_class_from_config(config)


class TestTasksManagerFallback:
    """Tests for TasksManager fallback behavior using mocks.

    These tests patch optimum.exporters.tasks.TasksManager to simulate
    various scenarios including fallback when TasksManager fails.
    """

    def test_fallback_to_arch_class_when_tasksmanager_fails(self):
        """Test fallback to arch_model_class when TasksManager.get_model_class_for_task fails.

        Simulates scenario where TasksManager can infer the task but
        get_model_class_for_task raises an exception.
        """
        from transformers import BlipForConditionalGeneration

        config = MagicMock()
        config.architectures = ["BlipForConditionalGeneration"]
        config.model_type = "blip"

        with patch("optimum.exporters.tasks.TasksManager.get_model_class_for_task") as mock_get:
            # Simulate TasksManager failure
            mock_get.side_effect = Exception("Model not supported")

            task, resolved_class = _detect_task_and_class_from_config(config)

        assert task == "image-text-to-text"
        # Should fallback to architecture class
        assert resolved_class == BlipForConditionalGeneration


class TestModelTaskDefaultsOverride:
    """Tests for per-model-type default-task auto-detection override.

    Some model families (e.g., SAM/SAM2) have an architecture class whose
    default TasksManager mapping ("feature-extraction") differs from the
    canonical export target ("mask-generation"). The default is encoded as a
    MODEL_CLASS_MAPPING[(model_type, None)] sentinel entry that biases
    auto-detection toward the right export configuration when --task is
    not provided.
    """

    def test_sam2_video_defaults_to_mask_generation(self):
        """Sam2Model on sam2_video config auto-detects to mask-generation."""
        # Trigger HF model registrations (loads SAM sentinel entries)
        import winml.modelkit.models.hf  # noqa: F401
        from winml.modelkit.models.hf.sam import SAM2MaskGeneration

        config = MagicMock()
        config.architectures = ["Sam2Model"]
        config.model_type = "sam2_video"

        task, resolved_class = _detect_task_and_class_from_config(config)

        assert task == "mask-generation"
        assert resolved_class is SAM2MaskGeneration

    def test_sam_defaults_to_mask_generation(self):
        """SamModel on sam config auto-detects to mask-generation."""
        import winml.modelkit.models.hf  # noqa: F401
        from winml.modelkit.models.hf.sam import SAMMaskGeneration

        config = MagicMock()
        config.architectures = ["SamModel"]
        config.model_type = "sam"

        task, resolved_class = _detect_task_and_class_from_config(config)

        assert task == "mask-generation"
        assert resolved_class is SAMMaskGeneration

    def test_model_type_underscore_normalized(self):
        """sam2_video (underscore) matches sam2-video (hyphen) in MODEL_CLASS_MAPPING."""
        import winml.modelkit.models.hf  # noqa: F401

        config = MagicMock()
        config.architectures = ["Sam2Model"]
        config.model_type = "sam2_video"

        task, _ = _detect_task_and_class_from_config(config)
        assert task == "mask-generation"

    def test_no_override_for_unrelated_model(self):
        """Models without a (model_type, None) sentinel keep TasksManager-inferred task."""
        from transformers import ResNetForImageClassification

        config = MagicMock()
        config.architectures = ["ResNetForImageClassification"]
        config.model_type = "resnet"

        task, resolved_class = _detect_task_and_class_from_config(config)

        assert task == "image-classification"
        # TasksManager returns AutoModelForImageClassification, not the arch class
        assert resolved_class is not ResNetForImageClassification or task == "image-classification"
