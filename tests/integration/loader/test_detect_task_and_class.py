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

from winml.modelkit.loader.resolution import resolve_task


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

        r = resolve_task(config)

        assert r.task == "image-text-to-text"
        # TasksManager may return AutoModelForImageTextToText or fallback
        # to BlipForConditionalGeneration
        assert r.model_class is not None

    def test_blip_fallback_when_tasksmanager_fails(self):
        """Test BLIP falls back to architecture class when TasksManager fails.

        When TasksManager.get_model_class_for_task raises an exception,
        we should fallback to BlipForConditionalGeneration from config.architectures.
        """
        from transformers import BlipForConditionalGeneration

        config = AutoConfig.from_pretrained("Salesforce/blip-image-captioning-base")

        with patch("optimum.exporters.tasks.TasksManager.get_model_class_for_task") as mock_get:
            mock_get.side_effect = Exception("No OnnxConfig registered")

            r = resolve_task(config)

        assert r.task == "image-text-to-text"
        assert r.model_class == BlipForConditionalGeneration

    def test_blip_resolves_a_valid_class(self):
        """BLIP resolves to a real model class (TasksManager's choice or the arch class).

        BLIP config specifies BlipForConditionalGeneration but TasksManager may return
        a generic AutoModelForImageTextToText; either is a valid importable class.
        """
        config = AutoConfig.from_pretrained("Salesforce/blip-image-captioning-base")

        r = resolve_task(config)

        assert r.task == "image-text-to-text"
        assert hasattr(r.model_class, "__name__")
        assert r.model_class.__name__ in (
            "BlipForConditionalGeneration",
            "AutoModelForImageTextToText",
        ), r.model_class.__name__


@pytest.mark.slow
class TestSupportedModelsIntegration:
    """Integration tests with models supported by TasksManager."""

    def test_resnet_uses_tasksmanager(self):
        """Test ResNet model uses TasksManager successfully."""
        config = AutoConfig.from_pretrained("microsoft/resnet-18")

        r = resolve_task(config)

        assert r.task == "image-classification"
        # TasksManager should succeed for ResNet
        assert "ImageClassification" in r.model_class.__name__

    def test_convnext_uses_tasksmanager(self):
        """Test ConvNeXt model uses TasksManager successfully."""
        config = AutoConfig.from_pretrained("facebook/convnext-tiny-224")

        r = resolve_task(config)

        assert r.task == "image-classification"
        assert "ImageClassification" in r.model_class.__name__


@pytest.mark.slow
class TestSeq2SeqFillMaskCorrection:
    """Encoder-decoder generation heads that Optimum mislabels as fill-mask must
    resolve to text2text-generation, not the encoder-only masked-LM task."""

    def test_bart_conditional_generation_is_text2text(self):
        """BartForConditionalGeneration -> text2text-generation / BartDecoderWrapper."""
        config = AutoConfig.from_pretrained("facebook/bart-large-cnn")

        r = resolve_task(config)

        assert r.task == "text2text-generation"
        # BartDecoderWrapper is WinML's encoder-decoder decoder sub-component class
        # registered for (bart, text2text-generation) — i.e. the seq2seq route,
        # not the AutoModelForMaskedLM that the fill-mask mislabel would produce.
        assert r.model_class.__name__ == "BartDecoderWrapper"

    def test_bert_masked_lm_stays_fill_mask(self):
        """Encoder-only masked-LM is untouched (is_encoder_decoder is False)."""
        config = AutoConfig.from_pretrained("bert-base-uncased")

        r = resolve_task(config)

        assert r.task == "fill-mask"
