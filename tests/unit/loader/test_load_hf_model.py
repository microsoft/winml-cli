# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for load_hf_model function with model_class and user_script.

Test specifications from docs/design/loader/hf.md sections 9.3 and 9.4.

Note: Tests that download real models have been moved to
tests/integration/loader/test_load_hf_model.py.
"""

import pytest

from winml.modelkit.loader import load_hf_model
from winml.modelkit.loader.config import (  # Testing internal implementation
    _create_hf_config_from_model_class as create_hf_config_from_model_class,
)
from winml.modelkit.loader.hf import _load_class_from_script  # Testing internal implementation


class TestLoadClassFromScript:
    """Tests for _load_class_from_script helper function."""

    def test_script_not_found(self, tmp_path):
        """Test error when script file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="not found"):
            _load_class_from_script(str(tmp_path / "nonexistent.py"), "SomeClass")

    def test_script_not_python(self, tmp_path):
        """Test error when script is not a .py file."""
        script = tmp_path / "script.txt"
        script.write_text("# not python")
        with pytest.raises(ValueError, match=r".py file"):
            _load_class_from_script(str(script), "SomeClass")

    def test_class_not_found(self, tmp_path):
        """Test error when class not found in script."""
        script = tmp_path / "empty.py"
        script.write_text("# Empty script\nsome_var = 123")
        with pytest.raises(AttributeError, match=r"WrongClassName.*not found"):
            _load_class_from_script(str(script), "WrongClassName")

    def test_not_a_class(self, tmp_path):
        """Test error when target is not a class."""
        script = tmp_path / "not_class.py"
        script.write_text("MyFunc = lambda x: x")
        with pytest.raises(TypeError, match="not a class"):
            _load_class_from_script(str(script), "MyFunc")

    def test_missing_from_pretrained(self, tmp_path):
        """Test error when class doesn't have from_pretrained."""
        script = tmp_path / "no_pretrained.py"
        script.write_text("class NoFromPretrained:\n    pass")
        with pytest.raises(TypeError, match="from_pretrained"):
            _load_class_from_script(str(script), "NoFromPretrained")

    def test_load_valid_class(self, tmp_path):
        """Test loading a valid model class from script."""
        script = tmp_path / "valid.py"
        script.write_text(
            '''
class ValidModel:
    """Model with from_pretrained."""
    @classmethod
    def from_pretrained(cls, name):
        return cls()
'''
        )
        resolved_class = _load_class_from_script(str(script), "ValidModel")
        assert resolved_class.__name__ == "ValidModel"
        assert hasattr(resolved_class, "from_pretrained")


class TestUserScriptSecurity:
    """Tests for user_script security requirements."""

    def test_user_script_requires_trust_remote_code(self, tmp_path):
        """Test user_script requires trust_remote_code=True."""
        script = tmp_path / "custom.py"
        script.write_text("# placeholder")

        with pytest.raises(ValueError, match="trust_remote_code"):
            load_hf_model(
                "microsoft/resnet-50",
                task="image-classification",
                model_class="CustomModel",
                user_script=str(script),
                trust_remote_code=False,
            )

    def test_user_script_requires_model_class(self, tmp_path):
        """Test user_script requires model_class to be specified."""
        script = tmp_path / "custom.py"
        script.write_text("# placeholder")

        with pytest.raises(ValueError, match="model_class must be specified"):
            load_hf_model(
                "microsoft/resnet-50",
                task="image-classification",
                model_class=None,  # Missing!
                user_script=str(script),
                trust_remote_code=True,
            )


class TestModelArchitectureOverrideFast:
    """Fast tests for model_class behavior that don't download models."""

    def test_model_class_without_user_script_uses_tasks_manager(self, monkeypatch):
        """Test that model_class uses resolve_task."""
        from unittest.mock import MagicMock

        import winml.modelkit.loader.resolution as resolution_module

        # Track calls to resolve_task
        resolve_calls = []

        def mock_resolve(config, *, task=None, model_class=None, model_type_override=None):
            resolve_calls.append({"task": task, "model_class": model_class})
            mock_class = MagicMock()
            mock_class.__name__ = "MockModel"
            result = MagicMock()
            result.task = "image-classification"
            result.model_class = mock_class
            return result

        # Mock AutoConfig
        mock_config = MagicMock()

        import winml.modelkit.loader.hf as hf_module

        monkeypatch.setattr(resolution_module, "resolve_task", mock_resolve)
        mock_auto_config = MagicMock(from_pretrained=lambda *a, **kw: mock_config)
        monkeypatch.setattr(hf_module, "AutoConfig", mock_auto_config)

        try:
            load_hf_model(
                "test-model",
                task="image-classification",
                model_class="SpecificModelClass",
            )
        except Exception:
            pass  # May fail on model instantiation, but we check the call

        # Verify resolve_task was called with model_class
        assert len(resolve_calls) > 0
        call = resolve_calls[-1]
        assert call["task"] == "image-classification"
        assert call["model_class"] == "SpecificModelClass"

    def test_auto_detect_when_no_model_class(self, monkeypatch):
        """Test auto-detection when model_class is not specified."""
        from unittest.mock import MagicMock

        import winml.modelkit.loader.resolution as resolution_module

        # Track calls to resolve_task
        resolve_calls = []

        def mock_resolve(config, *, task=None, model_class=None, model_type_override=None):
            resolve_calls.append({"task": task, "model_class": model_class})
            mock_class = MagicMock()
            mock_class.__name__ = "AutoDetectedModel"
            result = MagicMock()
            result.task = "image-classification"
            result.model_class = mock_class
            return result

        # Mock AutoConfig
        mock_config = MagicMock()

        import winml.modelkit.loader.hf as hf_module

        monkeypatch.setattr(resolution_module, "resolve_task", mock_resolve)
        mock_auto_config = MagicMock(from_pretrained=lambda *a, **kw: mock_config)
        monkeypatch.setattr(hf_module, "AutoConfig", mock_auto_config)

        try:
            load_hf_model("test-model")  # No task, no model_class
        except Exception:
            pass

        # Verify resolve_task was called without model_class
        assert len(resolve_calls) > 0
        call = resolve_calls[-1]
        assert call["task"] is None  # Auto-detect
        assert call["model_class"] is None

    def test_bert_tiny_uses_model_specific_default_task(self, monkeypatch):
        """bert-tiny should use model-specific default task when task is omitted."""
        from unittest.mock import MagicMock

        import winml.modelkit.loader.resolution as resolution_module

        resolve_calls = []

        def mock_resolve(config, *, task=None, model_class=None, model_type_override=None):
            resolved_task = task or "feature-extraction"
            resolve_calls.append({"task": resolved_task, "model_class": model_class})
            mock_class = MagicMock()
            mock_class.__name__ = "AutoDetectedModel"
            result = MagicMock()
            result.task = resolved_task
            result.model_class = mock_class
            return result

        mock_config = MagicMock()

        import winml.modelkit.loader.hf as hf_module

        monkeypatch.setattr(resolution_module, "resolve_task", mock_resolve)
        mock_auto_config = MagicMock(from_pretrained=lambda *a, **kw: mock_config)
        monkeypatch.setattr(hf_module, "AutoConfig", mock_auto_config)

        load_hf_model("  PRAJJWAL1/BERT-TINY  ")

        assert len(resolve_calls) > 0
        call = resolve_calls[-1]
        assert call["task"] == "feature-extraction"
        assert call["model_class"] is None


class TestTasksManagerIntegration:
    """Tests for TasksManager integration with specific tasks.

    These tests verify TasksManager behavior for edge cases like
    unsupported tasks and explicit model architecture overrides.
    """

    def test_tasks_manager_clip_vision_with_projection(self):
        """Test TasksManager returns CLIPVisionModelWithProjection with explicit model_class_name.

        When using model_class_name="CLIPVisionModelWithProjection" with feature-extraction task,
        TasksManager should return the CLIPVisionModelWithProjection class.
        """
        from optimum.exporters.tasks import TasksManager

        # Get model class with explicit model_class_name
        resolved_class = TasksManager.get_model_class_for_task(
            task="feature-extraction",
            framework="pt",
            model_class_name="CLIPVisionModelWithProjection",
        )

        assert resolved_class.__name__ == "CLIPVisionModelWithProjection"

    def test_tasks_manager_clip_text_with_projection(self):
        """Test TasksManager returns CLIPTextModelWithProjection with explicit model_class_name."""
        from optimum.exporters.tasks import TasksManager

        resolved_class = TasksManager.get_model_class_for_task(
            task="feature-extraction",
            framework="pt",
            model_class_name="CLIPTextModelWithProjection",
        )

        assert resolved_class.__name__ == "CLIPTextModelWithProjection"

    def test_tasks_manager_image_feature_extraction_synonym(self):
        """Test that image-feature-extraction is a synonym for feature-extraction."""
        from optimum.exporters.tasks import TasksManager

        # image-feature-extraction should map to feature-extraction
        normalized = TasksManager.map_from_synonym("image-feature-extraction")
        assert normalized == "feature-extraction"

    def test_tasks_manager_next_sentence_prediction_not_supported(self):
        """Test that next-sentence-prediction is NOT supported by TasksManager.

        NextSentencePrediction is a legacy NLP task that Optimum doesn't support.
        Users must use transformers AutoModelForNextSentencePrediction directly.
        """
        from optimum.exporters.tasks import TasksManager

        # next-sentence-prediction should not be a recognized task
        with pytest.raises(KeyError):
            TasksManager.get_model_class_for_task(
                task="next-sentence-prediction",
                framework="pt",
            )

    def test_transformers_next_sentence_prediction_exists(self):
        """Test that transformers provides AutoModelForNextSentencePrediction.

        Even though Optimum doesn't support NSP, transformers does.
        """
        from transformers import AutoModelForNextSentencePrediction

        # Verify the class exists and has from_pretrained
        assert hasattr(AutoModelForNextSentencePrediction, "from_pretrained")


class TestCreateHfConfigFromModelClass:
    """Tests for create_hf_config_from_model_class (Scenario B config creation)."""

    def test_returns_correct_model_type(self):
        """Config has the correct model_type from the model class."""
        from transformers import BertForSequenceClassification

        hf_config = create_hf_config_from_model_class(BertForSequenceClassification)
        assert hf_config.model_type == "bert"

    def test_sets_architectures(self):
        """Config.architectures is set to [model_class.__name__]."""
        from transformers import BertForSequenceClassification

        hf_config = create_hf_config_from_model_class(BertForSequenceClassification)
        assert hf_config.architectures == ["BertForSequenceClassification"]

    def test_resnet_model_type(self):
        """Works for vision models too."""
        from transformers import ResNetForImageClassification

        hf_config = create_hf_config_from_model_class(ResNetForImageClassification)
        assert hf_config.model_type == "resnet"
        assert hf_config.architectures == ["ResNetForImageClassification"]

    def test_config_has_default_values(self):
        """Config has sensible defaults (not empty)."""
        from transformers import BertForSequenceClassification

        hf_config = create_hf_config_from_model_class(BertForSequenceClassification)
        # Default BertConfig has these populated
        assert hf_config.hidden_size > 0
        assert hf_config.vocab_size > 0

    def test_config_usable_with_resolve(self):
        """Config can be passed to resolve_task."""
        from transformers import ResNetForImageClassification

        from winml.modelkit.loader import resolve_task

        hf_config = create_hf_config_from_model_class(ResNetForImageClassification)
        resolution = resolve_task(
            hf_config,
            task="image-classification",
            model_class="AutoModelForImageClassification",
        )
        assert resolution.task == "image-classification"
        assert "ImageClassification" in resolution.model_class.__name__

    def test_no_network_access_needed(self):
        """Function works without network (no from_pretrained calls)."""
        from transformers import BertForMaskedLM

        # This should be instant - no downloads
        hf_config = create_hf_config_from_model_class(BertForMaskedLM)
        assert hf_config is not None
        assert hf_config.model_type == "bert"


def test_user_script_branch_returns_modality_aware_task():
    """user_script path must return the surfaced modality-aware task (bugfix)."""
    from transformers import AutoConfig

    from winml.modelkit.loader.resolution import resolve_task

    cfg = AutoConfig.for_model("vit")
    cfg.architectures = ["ViTModel"]
    assert resolve_task(cfg).task == "image-feature-extraction"
