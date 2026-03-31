# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for WinMLLoaderConfig.

Test specifications from docs/design/loader/hf.md section 9.5.
"""

from winml.modelkit.config import WinMLBuildConfig
from winml.modelkit.loader.config import WinMLLoaderConfig


class TestWinMLLoaderConfigDefaults:
    """Test WinMLLoaderConfig default values."""

    def test_defaults(self):
        """Test all default values are set correctly."""
        config = WinMLLoaderConfig()
        assert config.task is None
        assert config.model_class is None
        assert config.model_type is None
        assert config.user_script is None
        assert config.trust_remote_code is False


class TestWinMLLoaderConfigSerialization:
    """Test WinMLLoaderConfig serialization/deserialization."""

    def test_to_dict_with_values(self):
        """Test config serialization with values."""
        config = WinMLLoaderConfig(
            task="image-classification",
            model_class="AutoModelForImageClassification",
        )
        d = config.to_dict()
        assert d["task"] == "image-classification"
        assert d["model_class"] == "AutoModelForImageClassification"

    def test_to_dict_with_model_type(self):
        """Test model_type is included in serialization."""
        config = WinMLLoaderConfig(
            task="fill-mask",
            model_class="BertForMaskedLM",
            model_type="bert",
        )
        d = config.to_dict()
        assert d["model_type"] == "bert"

    def test_to_dict_empty_when_defaults(self):
        """Test config serialization returns empty dict for defaults."""
        config = WinMLLoaderConfig()
        d = config.to_dict()
        # Should only include non-default values
        assert "task" not in d
        assert "model_class" not in d
        assert "model_type" not in d
        assert "user_script" not in d
        assert "trust_remote_code" not in d

    def test_to_dict_includes_trust_remote_code_when_true(self):
        """Test trust_remote_code is included when True."""
        config = WinMLLoaderConfig(trust_remote_code=True)
        d = config.to_dict()
        assert d["trust_remote_code"] is True

    def test_from_dict_with_all_fields(self):
        """Test config deserialization with all fields."""
        config = WinMLLoaderConfig.from_dict(
            {
                "task": "feature-extraction",
                "model_class": "CLIPTextModelWithProjection",
                "model_type": "clip",
                "user_script": "scripts/custom.py",
                "trust_remote_code": True,
            }
        )
        assert config.task == "feature-extraction"
        assert config.model_class == "CLIPTextModelWithProjection"
        assert config.model_type == "clip"
        assert config.user_script == "scripts/custom.py"
        assert config.trust_remote_code is True

    def test_from_dict_with_partial_fields(self):
        """Test config deserialization with partial fields."""
        config = WinMLLoaderConfig.from_dict({"task": "text-generation"})
        assert config.task == "text-generation"
        assert config.model_class is None
        assert config.user_script is None
        assert config.trust_remote_code is False

    def test_from_dict_empty(self):
        """Test config deserialization from empty dict."""
        config = WinMLLoaderConfig.from_dict({})
        assert config.task is None
        assert config.model_class is None
        assert config.user_script is None
        assert config.trust_remote_code is False

    def test_roundtrip(self):
        """Test serialization followed by deserialization preserves values."""
        original = WinMLLoaderConfig(
            task="image-classification",
            model_class="ResNetForImageClassification",
            model_type="resnet",
            user_script="custom.py",
            trust_remote_code=True,
        )
        d = original.to_dict()
        restored = WinMLLoaderConfig.from_dict(d)
        assert restored.task == original.task
        assert restored.model_class == original.model_class
        assert restored.model_type == original.model_type
        assert restored.user_script == original.user_script
        assert restored.trust_remote_code == original.trust_remote_code


class TestWinMLBuildConfigIncludesLoader:
    """Test WinMLBuildConfig integration with loader config."""

    def test_model_config_has_loader_attribute(self):
        """Test WinMLBuildConfig has loader attribute."""
        config = WinMLBuildConfig()
        assert hasattr(config, "loader")

    def test_model_config_loader_is_config_instance(self):
        """Test loader attribute is WinMLLoaderConfig instance."""
        config = WinMLBuildConfig()
        assert isinstance(config.loader, WinMLLoaderConfig)

    def test_model_config_loader_defaults(self):
        """Test loader in model config has default values."""
        config = WinMLBuildConfig()
        assert config.loader.task is None
        assert config.loader.model_class is None
        assert config.loader.user_script is None
        assert config.loader.trust_remote_code is False

    def test_model_config_from_dict_with_loader(self):
        """Test WinMLBuildConfig.from_dict includes loader config."""
        config = WinMLBuildConfig.from_dict(
            {
                "loader": {
                    "task": "feature-extraction",
                    "model_class": "CLIPTextModelWithProjection",
                }
            }
        )
        assert config.loader.task == "feature-extraction"
        assert config.loader.model_class == "CLIPTextModelWithProjection"

    def test_model_config_to_dict_excludes_empty_loader(self):
        """Test WinMLBuildConfig.to_dict excludes empty loader."""
        config = WinMLBuildConfig()
        d = config.to_dict()
        # Empty loader should not be included
        assert "loader" not in d or d.get("loader") == {}

    def test_model_config_to_dict_includes_loader_with_values(self):
        """Test WinMLBuildConfig.to_dict includes loader with values."""
        config = WinMLBuildConfig.from_dict({"loader": {"task": "image-classification"}})
        d = config.to_dict()
        assert "loader" in d
        assert d["loader"]["task"] == "image-classification"
