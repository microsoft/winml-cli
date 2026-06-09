# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Integration tests for task/class detection that download model configs.

Extracted from tests/unit/loader/test_detect_task_and_class.py.
These tests require network access. Use `pytest -m "not slow"` to skip them.
"""

from unittest.mock import patch

import pytest
from transformers import AutoConfig

from winml.modelkit.loader.task import _detect_task_and_class_from_config


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

        with caplog.at_level(logging.INFO):
            _task, resolved_class = _detect_task_and_class_from_config(config)

        # If TasksManager returns different class, info should be logged
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


@pytest.mark.slow
class TestSeq2SeqFillMaskCorrection:
    """Encoder-decoder generation heads that Optimum mislabels as fill-mask must
    resolve to text2text-generation, not the encoder-only masked-LM task."""

    def test_bart_conditional_generation_is_text2text(self):
        """BartForConditionalGeneration -> text2text-generation / BartDecoderWrapper."""
        config = AutoConfig.from_pretrained("facebook/bart-large-cnn")

        task, resolved_class = _detect_task_and_class_from_config(config)

        assert task == "text2text-generation"
        assert resolved_class.__name__ == "BartDecoderWrapper"

    def test_bert_masked_lm_stays_fill_mask(self):
        """Encoder-only masked-LM is untouched (is_encoder_decoder is False)."""
        config = AutoConfig.from_pretrained("bert-base-uncased")

        task, _ = _detect_task_and_class_from_config(config)

        assert task == "fill-mask"
