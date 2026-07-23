# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Integration tests for config.build that download real models.

Extracted from tests/unit/config/test_build.py.
These tests require network access. Use `pytest -m "not slow"` to skip them.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.config import WinMLBuildConfig, generate_build_config
from winml.modelkit.export import InputTensorSpec, OutputTensorSpec, WinMLExportConfig
from winml.modelkit.loader import WinMLLoaderConfig
from winml.modelkit.session import EPDeviceTarget


@pytest.fixture(autouse=True)
def mock_device_resolution() -> None:
    """Keep build-config integration tests independent of host EP discovery."""
    with (
        patch(
            "winml.modelkit.session.resolve_device",
            return_value=EPDeviceTarget(ep="auto", device="cpu"),
        ),
    ):
        yield


@pytest.mark.slow
class TestGenerateBuildConfigSlow:
    """Slow tests for generate_build_config (downloads models)."""

    def test_bert_tiny_auto_detect(self) -> None:
        """Full test with prajjwal1/bert-tiny (needs explicit task).

        Note: bert-tiny requires explicit task because it's a base model
        without a task-specific head.
        """
        config = generate_build_config(
            "prajjwal1/bert-tiny",
            task="fill-mask",  # Explicit task required for base models
        )

        assert isinstance(config, WinMLBuildConfig)
        assert config.loader is not None
        assert config.loader.task == "fill-mask"
        assert config.export is not None

    def test_resnet_auto_detect(self) -> None:
        """Full test with microsoft/resnet-18 (auto-detects image-classification)."""
        config = generate_build_config("microsoft/resnet-18")

        assert isinstance(config, WinMLBuildConfig)
        assert config.loader is not None
        assert config.loader.task == "image-classification"
        assert config.export is not None


@pytest.mark.slow
class TestSubmoduleDiscovery:
    """Tests for submodule discovery with module parameter."""

    def test_find_submodules_returns_list(
        self,
    ) -> None:
        """module parameter returns list of WinMLBuildConfig.

        Uses mocking to test the submodule discovery flow.
        """
        from transformers import ResNetConfig, ResNetForImageClassification

        # Create a small real ResNet config and model
        resnet_config = ResNetConfig(
            num_channels=3,
            hidden_sizes=[32, 64],
            depths=[1, 1],
            layer_type="basic",
            num_labels=10,
        )
        mock_hf_config = MagicMock()
        mock_hf_config.model_type = "resnet"

        # Create the model class that returns a real model
        def create_model(config: MagicMock) -> ResNetForImageClassification:
            return ResNetForImageClassification(resnet_config)

        mock_model_class = MagicMock()
        mock_model_class.__name__ = "ResNetForImageClassification"
        mock_model_class.side_effect = create_model

        mock_loader_config = WinMLLoaderConfig(
            task="image-classification",
            model_class="ResNetForImageClassification",
            model_type="resnet",
        )

        # Export config with input_tensors that have shapes for submodule discovery
        mock_export_config = WinMLExportConfig(
            input_tensors=[
                InputTensorSpec(
                    name="pixel_values",
                    shape=(1, 3, 224, 224),
                    dtype="float32",
                ),
            ],
            output_tensors=[OutputTensorSpec(name="logits")],
        )

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class, MagicMock()),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            configs = generate_build_config("microsoft/resnet-18", module="ResNetConvLayer")

        assert isinstance(configs, list)
        assert len(configs) > 0
        for cfg in configs:
            assert isinstance(cfg, WinMLBuildConfig)

    def test_empty_input_shapes_raises(
        self,
    ) -> None:
        """Empty input_shapes from export config raises ValueError."""
        mock_hf_config = MagicMock()
        mock_model_class = MagicMock()
        mock_loader_config = WinMLLoaderConfig(
            task="fill-mask",
            model_class="BertForMaskedLM",
            model_type="bert",
        )

        # Export config with no input_tensors (None) - should trigger error
        mock_export_config = WinMLExportConfig(
            input_tensors=None,
            output_tensors=[OutputTensorSpec(name="logits")],
        )

        # Create a mock model instance that the model_class returns
        mock_model_instance = MagicMock()

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class, MagicMock()),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            # Configure the mock model_class to return a model when called
            mock_model_class.return_value = mock_model_instance

            with pytest.raises(ValueError, match="Cannot extract input shapes"):
                generate_build_config("bert-base-uncased", module="SomeModule")


@pytest.mark.slow
class TestGenerateBuildConfigOverride:
    """Tests for generate_build_config with override parameter.

    Uses prajjwal1/bert-tiny for speed. These tests verify that the
    three-tier override system works correctly end-to-end.
    """

    def test_override_opset_version(self) -> None:
        """Override with opset_version=18 produces config with opset 18."""
        override = WinMLBuildConfig(
            export=WinMLExportConfig(opset_version=18),
        )

        config = generate_build_config(
            "prajjwal1/bert-tiny",
            task="fill-mask",
            override=override,
        )

        assert config.export.opset_version == 18

    def test_override_does_not_clobber_auto_detected_task(self) -> None:
        """Override with only export settings preserves auto-detected task.

        When override has a default WinMLLoaderConfig (task=None),
        the auto-detected loader.task should not be clobbered.
        """
        override = WinMLBuildConfig(
            export=WinMLExportConfig(opset_version=18),
        )

        config = generate_build_config(
            "prajjwal1/bert-tiny",
            task="fill-mask",
            override=override,
        )

        # loader.task should still be the auto-detected value, not None
        assert config.loader.task == "fill-mask"
        assert config.loader.model_class is not None

    def test_override_none_is_noop(self) -> None:
        """override=None behaves the same as no override."""
        config_no_override = generate_build_config(
            "prajjwal1/bert-tiny",
            task="fill-mask",
        )

        config_none_override = generate_build_config(
            "prajjwal1/bert-tiny",
            task="fill-mask",
            override=None,
        )

        # Both configs should be equivalent
        assert config_no_override.loader.task == config_none_override.loader.task
        assert config_no_override.loader.model_class == config_none_override.loader.model_class
        assert config_no_override.export.opset_version == config_none_override.export.opset_version
        assert config_no_override.export.batch_size == config_none_override.export.batch_size
