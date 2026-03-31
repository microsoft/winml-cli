# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Integration tests for load_hf_model that download real models.

Extracted from tests/unit/loader/test_load_hf_model.py.
These tests require network access. Use `pytest -m "not slow"` to skip them.
"""

import pytest

from winml.modelkit.loader import load_hf_model


@pytest.mark.slow
class TestUserScriptLoading:
    """Tests for user_script model loading.

    These tests download models and may be slow.
    """

    def test_user_script_loads_custom_class(self, tmp_path):
        """Test loading model class from user script.

        Note: The custom class wraps from_pretrained to return itself,
        verifying the custom class path is used.
        """
        script = tmp_path / "custom_model.py"
        script.write_text(
            '''
from transformers import ResNetForImageClassification

class CustomResNet(ResNetForImageClassification):
    """Custom ResNet with modifications."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override to ensure we get CustomResNet instance."""
        # Load the base model
        model = super().from_pretrained(*args, **kwargs)
        # Copy state to our custom class
        custom = cls.__new__(cls)
        custom.__dict__.update(model.__dict__)
        return custom
'''
        )

        model, _config, _task = load_hf_model(
            "microsoft/resnet-18",  # Use smaller model for faster test
            task="image-classification",
            model_class="CustomResNet",
            user_script=str(script),
            trust_remote_code=True,
        )
        assert model.__class__.__name__ == "CustomResNet"

    def test_user_script_class_not_found(self, tmp_path):
        """Test error when class not found in script."""
        script = tmp_path / "empty.py"
        script.write_text("# Empty script")

        with pytest.raises(AttributeError, match="WrongClassName"):
            load_hf_model(
                "microsoft/resnet-50",
                task="image-classification",
                model_class="WrongClassName",
                user_script=str(script),
                trust_remote_code=True,
            )


@pytest.mark.slow
class TestModelArchitectureOverride:
    """Tests for model_class override.

    These tests download models and may be slow.
    """

    def test_model_class_override_basic(self):
        """Test explicit model_class overrides auto-detection."""
        # Use a simple model for faster testing
        model, _config, _task = load_hf_model(
            "microsoft/resnet-18",
            task="image-classification",
            model_class="AutoModelForImageClassification",
        )
        # Model should be loaded with specified architecture
        assert "ImageClassification" in model.__class__.__name__

    def test_model_class_invalid(self):
        """Test error handling for invalid model_class."""
        # resolve_task_and_model_class wraps AttributeError as ValueError
        with pytest.raises(ValueError, match="not found"):
            load_hf_model(
                "microsoft/resnet-50",
                task="image-classification",
                model_class="NonExistentModelClass",
            )


@pytest.mark.slow
class TestCLIPModelArchitectureOverride:
    """Tests for CLIP model with explicit model_class override.

    These tests download models and may be slow.
    """

    def test_load_clip_vision_with_projection(self):
        """Test loading CLIPVisionModelWithProjection with model_class override.

        Uses image-feature-extraction task with explicit model_class
        to get CLIPVisionModelWithProjection instead of default CLIPModel.
        """
        model, _config, task = load_hf_model(
            "openai/clip-vit-base-patch32",
            task="feature-extraction",
            model_class="CLIPVisionModelWithProjection",
        )

        assert model.__class__.__name__ == "CLIPVisionModelWithProjection"
        assert task == "feature-extraction"

    def test_load_clip_text_with_projection(self):
        """Test loading CLIPTextModelWithProjection with model_class override."""
        model, _config, task = load_hf_model(
            "openai/clip-vit-base-patch32",
            task="feature-extraction",
            model_class="CLIPTextModelWithProjection",
        )

        assert model.__class__.__name__ == "CLIPTextModelWithProjection"
        assert task == "feature-extraction"


@pytest.mark.slow
class TestNextSentencePredictionUserScript:
    """Tests for NextSentencePrediction using user_script workaround.

    Since TasksManager doesn't support NSP, users can use user_script
    to load AutoModelForNextSentencePrediction.
    """

    def test_nsp_via_user_script(self, tmp_path):
        """Test loading NextSentencePrediction model via user_script.

        This demonstrates the workaround for tasks not supported by TasksManager.
        """
        script = tmp_path / "nsp_model.py"
        script.write_text(
            '''
from transformers import BertForNextSentencePrediction

class NSPModel(BertForNextSentencePrediction):
    """Wrapper for NextSentencePrediction."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override to return NSPModel instance."""
        model = super().from_pretrained(*args, **kwargs)
        custom = cls.__new__(cls)
        custom.__dict__.update(model.__dict__)
        return custom
'''
        )

        # Use a tiny BERT model for fast testing
        model, _config, _task = load_hf_model(
            "prajjwal1/bert-tiny",
            task="fill-mask",  # Use a supported task for detection
            model_class="NSPModel",
            user_script=str(script),
            trust_remote_code=True,
        )

        assert model.__class__.__name__ == "NSPModel"
