"""Tests for _detect_task_and_class_from_config function.

Tests the resolution strategy:
1. Specializations (HF_MODEL_CLASS_MAPPING) - highest priority
2. TasksManager.get_model_class_for_task() - honor if successful
3. Fallback to architecture class from config.architectures

Uses BLIP as primary test case for TasksManager fallback scenario.
"""

from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoConfig

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


@pytest.mark.slow
class TestBlipIntegration:
    """Integration tests with real BLIP model.

    These tests download model configs and may be slow.
    """

    def test_blip_model_type_is_blip(self):
        """Test BLIP config has expected model_type."""
        config = AutoConfig.from_pretrained("Salesforce/blip-image-captioning-base")

        assert config.model_type == "blip"
        assert config.architectures == ["BlipForConditionalGeneration"]

    def test_blip_task_detection(self):
        """Test BLIP task is detected as image-text-to-text."""
        config = AutoConfig.from_pretrained("Salesforce/blip-image-captioning-base")

        task, resolved_class = _detect_task_and_class_from_config(config)

        assert task == "image-text-to-text"
        # TasksManager may return AutoModelForImageTextToText or fallback
        # to BlipForConditionalGeneration
        assert resolved_class is not None

    def test_blip_fallback_when_tasksmanager_fails(self):
        """Test BLIP falls back to architecture class when TasksManager fails.

        When TasksManager.get_model_class_for_task raises an exception,
        we should fallback to BlipForConditionalGeneration from config.architectures.
        """
        from transformers import BlipForConditionalGeneration

        config = AutoConfig.from_pretrained("Salesforce/blip-image-captioning-base")

        with patch("optimum.exporters.tasks.TasksManager.get_model_class_for_task") as mock_get:
            mock_get.side_effect = Exception("No OnnxConfig registered")

            task, resolved_class = _detect_task_and_class_from_config(config)

        assert task == "image-text-to-text"
        assert resolved_class == BlipForConditionalGeneration

    def test_blip_warning_when_different_class_returned(self, caplog):
        """Test warning is logged when TasksManager returns different class than architecture.

        BLIP config specifies BlipForConditionalGeneration but TasksManager
        may return AutoModelForImageTextToText.
        """
        import logging

        config = AutoConfig.from_pretrained("Salesforce/blip-image-captioning-base")

        with caplog.at_level(logging.WARNING):
            task, resolved_class = _detect_task_and_class_from_config(config)

        # If TasksManager returns different class, warning should be logged
        if resolved_class.__name__ != "BlipForConditionalGeneration":
            assert "TasksManager returned" in caplog.text
            assert "BlipForConditionalGeneration" in caplog.text


@pytest.mark.slow
class TestSupportedModelsIntegration:
    """Integration tests with models supported by TasksManager."""

    def test_resnet_uses_tasksmanager(self):
        """Test ResNet model uses TasksManager successfully."""
        config = AutoConfig.from_pretrained("microsoft/resnet-18")

        task, resolved_class = _detect_task_and_class_from_config(config)

        assert task == "image-classification"
        # TasksManager should succeed for ResNet
        assert "ImageClassification" in resolved_class.__name__

    def test_convnext_uses_tasksmanager(self):
        """Test ConvNeXt model uses TasksManager successfully."""
        config = AutoConfig.from_pretrained("facebook/convnext-tiny-224")

        task, resolved_class = _detect_task_and_class_from_config(config)

        assert task == "image-classification"
        assert "ImageClassification" in resolved_class.__name__
