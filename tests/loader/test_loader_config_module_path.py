"""Tests for WinMLLoaderConfig.module_path field.

Validates that module_path can identify specific submodule instances
for per-module build/perf support.
"""

from winml.modelkit.loader.config import WinMLLoaderConfig


class TestModulePathField:
    """Test module_path field on WinMLLoaderConfig."""

    def test_module_path_default_is_none(self):
        """Default module_path should be None."""
        config = WinMLLoaderConfig()
        assert config.module_path is None

    def test_module_path_set_on_init(self):
        """module_path can be set via constructor."""
        config = WinMLLoaderConfig(module_path="encoder.layer.0.attention")
        assert config.module_path == "encoder.layer.0.attention"

    def test_module_path_to_dict(self):
        """to_dict() includes module_path when set."""
        config = WinMLLoaderConfig(module_path="decoder.block.3")
        d = config.to_dict()
        assert d["module_path"] == "decoder.block.3"

    def test_module_path_none_omitted_from_dict(self):
        """to_dict() omits module_path when None."""
        config = WinMLLoaderConfig()
        d = config.to_dict()
        assert "module_path" not in d

    def test_module_path_from_dict(self):
        """from_dict() restores module_path."""
        config = WinMLLoaderConfig.from_dict({"module_path": "vision_model.encoder"})
        assert config.module_path == "vision_model.encoder"

    def test_module_path_missing_from_dict(self):
        """from_dict({}) leaves module_path as None."""
        config = WinMLLoaderConfig.from_dict({})
        assert config.module_path is None

    def test_roundtrip(self):
        """Full roundtrip: construct -> to_dict -> from_dict preserves module_path."""
        original = WinMLLoaderConfig(
            task="image-classification",
            model_class="ResNetForImageClassification",
            model_type="resnet",
            module_path="encoder.layer.0.attention",
            trust_remote_code=False,
        )
        d = original.to_dict()
        restored = WinMLLoaderConfig.from_dict(d)
        assert restored.module_path == original.module_path
        assert restored.task == original.task
        assert restored.model_class == original.model_class
        assert restored.model_type == original.model_type
        assert restored.trust_remote_code == original.trust_remote_code
