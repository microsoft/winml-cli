# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.config.build module.

Tests the config generation orchestration:
- resolve_io_specs: Get I/O specs without loading model weights
- generate_build_config: Generate WinMLBuildConfig from model ID
- _build_submodule_config: Build config for discovered submodules
- Submodule discovery with torchinfo
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# Import models package to trigger ONNX config registration with TasksManager
import winml.modelkit.models  # noqa: F401
from winml.modelkit.commands.config import config as config_command
from winml.modelkit.compiler import EPConfig, WinMLCompileConfig
from winml.modelkit.config import (
    WinMLBuildConfig,
    generate_build_config,
    generate_onnx_build_config,
)
from winml.modelkit.config.build import (
    SubmoduleInfo,
    _build_submodule_config,
    resolve_quant_compile_config,
)
from winml.modelkit.export import (
    InputTensorSpec,
    OutputTensorSpec,
    WinMLExportConfig,
    resolve_io_specs,
)
from winml.modelkit.loader import WinMLLoaderConfig
from winml.modelkit.optim import WinMLOptimizationConfig
from winml.modelkit.quant import WinMLQuantizationConfig
from winml.modelkit.utils.config_utils import merge_config


# =============================================================================
# Test Constants
# =============================================================================

# Text model constants
TEXT_VOCAB_SIZE = 1000
TEXT_HIDDEN_SIZE = 64
TEXT_NUM_HIDDEN_LAYERS = 2
TEXT_NUM_ATTENTION_HEADS = 2
TEXT_MAX_POSITION_EMBEDDINGS = 32
TEXT_INTERMEDIATE_SIZE = TEXT_HIDDEN_SIZE * 4


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_hf_config() -> MagicMock:
    """Create a mock HuggingFace config with model_type."""
    config = MagicMock(spec=["model_type", "architectures"])
    config.model_type = "bert"
    config.architectures = ["BertForMaskedLM"]
    return config


@pytest.fixture
def mock_model_class() -> MagicMock:
    """Create a mock model class."""
    model_class = MagicMock()
    model_class.__name__ = "BertForMaskedLM"
    return model_class


@pytest.fixture
def mock_loader_config() -> WinMLLoaderConfig:
    """Create a mock WinMLLoaderConfig for BERT fill-mask."""
    return WinMLLoaderConfig(
        task="fill-mask",
        model_class="BertForMaskedLM",
        model_type="bert",
    )


@pytest.fixture
def mock_export_config() -> WinMLExportConfig:
    """Create a mock WinMLExportConfig matching BERT structure."""
    return WinMLExportConfig(
        input_tensors=[
            InputTensorSpec(name="input_ids", shape=(2, 16), dtype="int64"),
            InputTensorSpec(name="attention_mask", shape=(2, 16), dtype="int64"),
        ],
        output_tensors=[OutputTensorSpec(name="logits")],
    )


@pytest.fixture
def mock_io_specs() -> dict:
    """Create mock I/O specs matching BERT structure."""
    return {
        "inputs": {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        },
        "outputs": {
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        "input_shapes": [(2, 16), (2, 16)],
    }


# =============================================================================
# TestGetIoSpecsFromConfig - Unit tests for resolve_io_specs()
# =============================================================================


class TestGetIoSpecsFromConfig:
    """Unit tests for resolve_io_specs function."""

    def test_bert_returns_text_inputs(self) -> None:
        """BERT returns input_ids, attention_mask as expected inputs."""
        from transformers import BertConfig

        hf_config = BertConfig(
            vocab_size=TEXT_VOCAB_SIZE,
            hidden_size=TEXT_HIDDEN_SIZE,
            num_hidden_layers=TEXT_NUM_HIDDEN_LAYERS,
            num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
            intermediate_size=TEXT_INTERMEDIATE_SIZE,
            max_position_embeddings=TEXT_MAX_POSITION_EMBEDDINGS,
        )

        specs = resolve_io_specs(
            model_type="bert",
            task="fill-mask",
            hf_config=hf_config,
        )

        assert "input_ids" in specs["input_names"]
        assert "attention_mask" in specs["input_names"]

    def test_returns_input_shapes(self) -> None:
        """Verifies input_shapes is populated from dummy input generation."""
        from transformers import BertConfig

        hf_config = BertConfig(
            vocab_size=TEXT_VOCAB_SIZE,
            hidden_size=TEXT_HIDDEN_SIZE,
            num_hidden_layers=TEXT_NUM_HIDDEN_LAYERS,
            num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
            intermediate_size=TEXT_INTERMEDIATE_SIZE,
            max_position_embeddings=TEXT_MAX_POSITION_EMBEDDINGS,
        )

        specs = resolve_io_specs(
            model_type="bert",
            task="fill-mask",
            hf_config=hf_config,
        )

        assert "input_shapes" in specs
        assert len(specs["input_shapes"]) > 0
        # Each shape should be a tuple
        for shape in specs["input_shapes"]:
            assert isinstance(shape, tuple)
            assert len(shape) >= 2  # At least (batch, sequence)

    def test_invalid_model_type_raises(self) -> None:
        """Invalid model_type raises ValueError."""
        from transformers import BertConfig

        hf_config = BertConfig(
            vocab_size=TEXT_VOCAB_SIZE,
            hidden_size=TEXT_HIDDEN_SIZE,
            num_hidden_layers=TEXT_NUM_HIDDEN_LAYERS,
            num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
            intermediate_size=TEXT_INTERMEDIATE_SIZE,
            max_position_embeddings=TEXT_MAX_POSITION_EMBEDDINGS,
        )

        with pytest.raises(ValueError, match="No OnnxConfig registered"):
            resolve_io_specs(
                model_type="invalid_model_type_that_does_not_exist",
                task="fill-mask",
                hf_config=hf_config,
            )


# =============================================================================
# TestGenerateBuildConfigFast - Fast tests (no network, use mocks)
# =============================================================================


class TestGenerateBuildConfigFast:
    """Fast tests for generate_build_config using mocks (no network)."""

    def test_returns_winml_build_config(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """generate_build_config returns WinMLBuildConfig instance."""
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            result = generate_build_config("bert-base-uncased")

        assert isinstance(result, WinMLBuildConfig)

    def test_model_type_none_raises(self) -> None:
        """model_type=None in HF config raises ValueError."""
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                side_effect=ValueError("does not have 'model_type' attribute"),
            ),
            pytest.raises(ValueError, match="does not have 'model_type' attribute"),
        ):
            generate_build_config("some-model")

    def test_task_override_used(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """Explicit task parameter is passed to resolve_loader_config."""
        tc_loader_config = WinMLLoaderConfig(
            task="text-classification",
            model_class="BertForSequenceClassification",
            model_type="bert",
        )
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(tc_loader_config, mock_hf_config, mock_model_class),
            ) as mock_resolve,
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            generate_build_config("bert-base-uncased", task="text-classification")

        # Verify task was passed to resolve_loader_config
        mock_resolve.assert_called_once()
        call_kwargs = mock_resolve.call_args
        assert call_kwargs.kwargs.get("task") == "text-classification"

    def test_model_class_override_used(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """Explicit model_class parameter is passed to resolve_loader_config."""
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ) as mock_resolve,
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            generate_build_config("bert-base-uncased", model_class="BertForMaskedLM")

        # Verify model_class was passed
        mock_resolve.assert_called_once()
        call_kwargs = mock_resolve.call_args
        assert call_kwargs.kwargs.get("model_class") == "BertForMaskedLM"

    def test_merge_config_called_with_override(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """Override config is merged via merge_config."""
        override = WinMLBuildConfig()

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
            patch(
                "winml.modelkit.config.build.merge_config", return_value=WinMLBuildConfig()
            ) as mock_merge,
        ):
            generate_build_config("bert-base-uncased", override=override)

        # Verify merge_config was called with the override
        mock_merge.assert_called_once()
        call_args = mock_merge.call_args
        assert call_args[0][1] is override


# =============================================================================
# TestRegistryShortCircuit - Registry-before-Optimum export config resolution
# =============================================================================


class TestRegistryShortCircuit:
    """Tests for the registry short-circuit path in generate_build_config.

    When MODEL_BUILD_CONFIGS has a registered config with input_tensors,
    the Optimum _resolve_export_config_from_specs() call is skipped.
    """

    def test_registry_with_input_tensors_skips_optimum(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
    ) -> None:
        """Registry config with input_tensors skips Optimum lookup."""
        blip_like_export = WinMLExportConfig(
            input_tensors=[
                InputTensorSpec(name="pixel_values", dtype="float32", shape=(1, 3, 384, 384)),
                InputTensorSpec(name="input_ids", dtype="int64", shape=(1, 64)),
            ],
            output_tensors=[OutputTensorSpec(name="logits")],
        )
        blip_like_config = WinMLBuildConfig(export=blip_like_export)
        loader_config = WinMLLoaderConfig(
            task="image-text-to-text",
            model_class="BlipForConditionalGeneration",
            model_type="blip",
        )
        mock_hf_config.model_type = "blip"

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
            ) as mock_optimum,
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {"blip": blip_like_config}),
        ):
            result = generate_build_config("Salesforce/blip-image-captioning-base")

        # Optimum should NOT have been called
        mock_optimum.assert_not_called()
        # Result should have the registered input_tensors
        assert result.export.input_tensors is not None
        assert len(result.export.input_tensors) == 2
        assert result.export.input_tensors[0].name == "pixel_values"

    def test_registry_without_export_falls_through_to_optimum(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """Registry config without export falls through to Optimum."""
        # BERT_CONFIG has optim only, no export
        bert_like_config = WinMLBuildConfig(
            optim=WinMLOptimizationConfig(gelu_fusion=True),
        )

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ) as mock_optimum,
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {"bert": bert_like_config}),
        ):
            generate_build_config("bert-base-uncased")

        # Optimum SHOULD have been called
        mock_optimum.assert_called_once()

    def test_registry_with_none_input_tensors_falls_through(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """Registry config with export but input_tensors=None falls through."""
        config_with_empty_export = WinMLBuildConfig(
            export=WinMLExportConfig(),  # input_tensors defaults to None
        )

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ) as mock_optimum,
            patch(
                "winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {"bert": config_with_empty_export}
            ),
        ):
            generate_build_config("bert-base-uncased")

        # Optimum SHOULD have been called (input_tensors is None)
        mock_optimum.assert_called_once()

    def test_registry_deepcopy_prevents_mutation(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
    ) -> None:
        """Registry export config is deepcopied, preventing singleton mutation."""
        original_export = WinMLExportConfig(
            input_tensors=[
                InputTensorSpec(name="pixel_values", dtype="float32", shape=(1, 3, 224, 224)),
            ],
            output_tensors=[OutputTensorSpec(name="logits")],
        )
        registry_config = WinMLBuildConfig(export=original_export)
        loader_config = WinMLLoaderConfig(
            task="image-classification",
            model_class="SomeVisionModel",
            model_type="some-vision",
        )
        mock_hf_config.model_type = "some-vision"

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {"some-vision": registry_config}),
        ):
            result = generate_build_config("some/vision-model")

        # Result export should NOT be the same object as registry export
        assert result.export is not original_export
        assert result.export.input_tensors is not original_export.input_tensors
        # Content should be preserved (deepcopy correctness)
        assert len(result.export.input_tensors) == 1
        assert result.export.input_tensors[0].name == "pixel_values"
        assert result.export.input_tensors[0].shape == (1, 3, 224, 224)
        assert result.export.input_tensors[0].dtype == "float32"

    def test_registry_underscore_normalization(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
    ) -> None:
        """Registry lookup normalizes underscores to hyphens (e.g., clip_text_model)."""
        clip_export = WinMLExportConfig(
            input_tensors=[
                InputTensorSpec(name="input_ids", dtype="int64", shape=(1, 77)),
            ],
            output_tensors=[OutputTensorSpec(name="text_embeds")],
        )
        clip_config = WinMLBuildConfig(export=clip_export)
        loader_config = WinMLLoaderConfig(
            task="feature-extraction",
            model_class="CLIPTextModel",
            model_type="clip_text_model",  # underscores from HF config
        )
        mock_hf_config.model_type = "clip_text_model"

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
            ) as mock_optimum,
            # Registry uses hyphens
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {"clip-text-model": clip_config}),
        ):
            result = generate_build_config("openai/clip-vit-base-patch32")

        # Underscore model_type should match hyphenated registry key
        mock_optimum.assert_not_called()
        assert result.export.input_tensors[0].name == "input_ids"

    def test_registry_empty_list_input_tensors_skips_optimum(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
    ) -> None:
        """Registry config with input_tensors=[] skips Optimum (is not None)."""
        config_with_empty_list = WinMLBuildConfig(
            export=WinMLExportConfig(input_tensors=[]),
        )

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
            ) as mock_optimum,
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {"bert": config_with_empty_list}),
        ):
            result = generate_build_config("bert-base-uncased")

        # [] is not None, so short-circuit fires
        mock_optimum.assert_not_called()
        assert result.export.input_tensors == []

    def test_registry_miss_falls_through_to_optimum(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """Model not in registry at all falls through to Optimum."""
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ) as mock_optimum,
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),  # empty registry
        ):
            result = generate_build_config("some/unknown-model")

        mock_optimum.assert_called_once()
        assert result.export is mock_export_config


# =============================================================================
# TestBuildSubmoduleConfig - Unit tests for _build_submodule_config()
# =============================================================================


class TestBuildSubmoduleConfig:
    """Tests for _build_submodule_config with multi-input/output modules."""

    @pytest.fixture
    def parent_config(self) -> WinMLBuildConfig:
        """Create a parent config with non-default optim/compile for inheritance tests."""
        from winml.modelkit.compiler import WinMLCompileConfig
        from winml.modelkit.optim import WinMLOptimizationConfig

        return WinMLBuildConfig(
            optim=WinMLOptimizationConfig(gelu_fusion=True, matmul_add_fusion=True),
            compile=WinMLCompileConfig(),
        )

    def test_single_input_single_output(self, parent_config: WinMLBuildConfig) -> None:
        """Basic case: 1 input tensor, 1 output tensor."""
        sub_info = SubmoduleInfo(
            class_name="Conv2d",
            module_path="encoder.layer.0.conv",
            input_shapes=[[1, 64, 32, 32]],
            output_shapes=[[1, 128, 16, 16]],
            input_dtypes=["float32"],
            output_dtypes=["float32"],
        )

        result = _build_submodule_config(sub_info, parent_config)

        assert isinstance(result, WinMLBuildConfig)
        assert result.export.input_tensors is not None
        assert len(result.export.input_tensors) == 1
        assert result.export.input_tensors[0].name == "input_0"
        assert result.export.input_tensors[0].shape == (1, 64, 32, 32)

        assert result.export.output_tensors is not None
        assert len(result.export.output_tensors) == 1
        assert result.export.output_tensors[0].name == "output_0"

    def test_multi_input(self, parent_config: WinMLBuildConfig) -> None:
        """SubmoduleInfo with 2 input_shapes creates 2 InputTensorSpec."""
        sub_info = SubmoduleInfo(
            class_name="CrossAttention",
            module_path="decoder.cross_attn",
            input_shapes=[[1, 16, 64], [1, 16, 64]],
            output_shapes=[[1, 16, 64]],
            input_dtypes=["float32", "float32"],
            output_dtypes=["float32"],
        )

        result = _build_submodule_config(sub_info, parent_config)

        assert result.export.input_tensors is not None
        assert len(result.export.input_tensors) == 2
        assert result.export.input_tensors[0].name == "input_0"
        assert result.export.input_tensors[0].shape == (1, 16, 64)
        assert result.export.input_tensors[1].name == "input_1"
        assert result.export.input_tensors[1].shape == (1, 16, 64)

    def test_multi_output(self, parent_config: WinMLBuildConfig) -> None:
        """SubmoduleInfo with 2 output_shapes creates 2 OutputTensorSpec."""
        sub_info = SubmoduleInfo(
            class_name="SplitHead",
            module_path="encoder.split",
            input_shapes=[[1, 32, 128]],
            output_shapes=[[1, 32, 64], [1, 32, 64]],
            input_dtypes=["float32"],
            output_dtypes=["float32", "float32"],
        )

        result = _build_submodule_config(sub_info, parent_config)

        assert result.export.output_tensors is not None
        assert len(result.export.output_tensors) == 2
        assert result.export.output_tensors[0].name == "output_0"
        assert result.export.output_tensors[1].name == "output_1"

    def test_dtype_propagated(self, parent_config: WinMLBuildConfig) -> None:
        """Verify input_dtypes flow to InputTensorSpec.dtype."""
        sub_info = SubmoduleInfo(
            class_name="Embedding",
            module_path="encoder.embed",
            input_shapes=[[1, 128]],
            output_shapes=[[1, 128, 64]],
            input_dtypes=["int64"],
            output_dtypes=["float32"],
        )

        result = _build_submodule_config(sub_info, parent_config)

        assert result.export.input_tensors is not None
        assert result.export.input_tensors[0].dtype == "int64"

    def test_inherits_optim_from_parent(self, parent_config: WinMLBuildConfig) -> None:
        """Verify optim/compile are deep-copied from parent config."""
        sub_info = SubmoduleInfo(
            class_name="Linear",
            module_path="classifier.dense",
            input_shapes=[[1, 64]],
            output_shapes=[[1, 10]],
            input_dtypes=["float32"],
            output_dtypes=["float32"],
        )

        result = _build_submodule_config(sub_info, parent_config)

        # optim should have same values as parent
        assert result.optim == parent_config.optim
        # But must be a deep copy, not the same object
        assert result.optim is not parent_config.optim

        # compile should also be inherited and deep-copied
        assert result.compile is not parent_config.compile

    def test_submodule_omits_task(self, parent_config: WinMLBuildConfig) -> None:
        """Submodule config omits task (submodules don't have tasks), keeps model_type."""
        parent_config.loader.task = "fill-mask"
        parent_config.loader.model_type = "bert"

        sub_info = SubmoduleInfo(
            class_name="BertAttention",
            module_path="encoder.layer.0.attention",
            input_shapes=[[1, 16, 64]],
            output_shapes=[[1, 16, 64]],
            input_dtypes=["float32"],
            output_dtypes=["float32"],
        )

        result = _build_submodule_config(sub_info, parent_config)

        assert result.loader.task is None
        assert result.loader.model_type == "bert"
        assert result.loader.model_class == "BertAttention"
        assert result.loader.module_path == "encoder.layer.0.attention"

    def test_quant_uses_random_for_submodules(self, parent_config: WinMLBuildConfig) -> None:
        """Submodule quant uses random dataset (task=None, model_name=None)."""
        parent_config.loader.task = "fill-mask"
        parent_config.loader.model_type = "bert"

        sub_info = SubmoduleInfo(
            class_name="BertAttention",
            module_path="encoder.layer.0.attention",
            input_shapes=[[1, 16, 64]],
            output_shapes=[[1, 16, 64]],
            input_dtypes=["float32"],
            output_dtypes=["float32"],
        )

        result = _build_submodule_config(sub_info, parent_config)

        # Submodule quant should exist with task=None (random dataset fallback)
        assert result.quant is not None
        assert result.quant.task is None
        assert result.quant.model_name is None
        assert result.quant.samples == 1

    def test_submodule_config_with_quant_passes_validate(
        self,
        parent_config: WinMLBuildConfig,
    ) -> None:
        """Submodule config with quant(task=None) passes validate()."""
        parent_config.loader.task = "image-classification"
        parent_config.loader.model_type = "resnet"

        sub_info = SubmoduleInfo(
            class_name="ResNetConvLayer",
            module_path="encoder.stages.0.layers.0.layer.0",
            input_shapes=[[1, 64, 32, 32]],
            output_shapes=[[1, 128, 16, 16]],
            input_dtypes=["float32"],
            output_dtypes=["float32"],
        )

        result = _build_submodule_config(sub_info, parent_config)

        # Should NOT raise — validation relaxed for submodules
        result.validate()

    def test_submodule_quant_omits_task_in_json(
        self,
        parent_config: WinMLBuildConfig,
    ) -> None:
        """Submodule quant serialization omits task, model_name, dataset_name when None."""
        parent_config.loader.task = "fill-mask"
        parent_config.loader.model_type = "bert"

        sub_info = SubmoduleInfo(
            class_name="BertAttention",
            module_path="encoder.layer.0.attention",
            input_shapes=[[1, 16, 64]],
            output_shapes=[[1, 16, 64]],
            input_dtypes=["float32"],
            output_dtypes=["float32"],
        )

        result = _build_submodule_config(sub_info, parent_config)

        quant_dict = result.quant.to_dict()
        assert "task" not in quant_dict
        assert "model_name" not in quant_dict
        assert "dataset_name" not in quant_dict

    def test_empty_inputs(self, parent_config: WinMLBuildConfig) -> None:
        """SubmoduleInfo with empty input_shapes results in input_tensors=None."""
        sub_info = SubmoduleInfo(
            class_name="Constant",
            module_path="encoder.constant",
            input_shapes=[],
            output_shapes=[[1, 64]],
            input_dtypes=[],
            output_dtypes=["float32"],
        )

        result = _build_submodule_config(sub_info, parent_config)

        # Empty list is falsy, so input_tensors should be set to None
        assert result.export.input_tensors is None


# =============================================================================
# TestConfigCliOverride - CLI tests for --config flag
# =============================================================================


class TestConfigCliOverride:
    """Tests for the --config CLI override option."""

    def test_config_cli_with_override_file(
        self,
        tmp_path,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """--config flag loads JSON and passes as override to generate_build_config."""
        # Create override JSON file
        override_data = {"export": {"opset_version": 18, "batch_size": 4}}
        override_file = tmp_path / "override.json"
        override_file.write_text(json.dumps(override_data))

        # Write output to file to avoid stdout/stderr mixing
        output_file = tmp_path / "result.json"

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                ["-m", "bert-base-uncased", "-c", str(override_file), "-o", str(output_file)],
                obj={"verbose": 0},
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        output_data = json.loads(output_file.read_text())
        assert output_data["export"]["opset_version"] == 18
        assert output_data["export"]["batch_size"] == 4

    def test_config_cli_without_override(
        self,
        tmp_path,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """Without --config flag, generate_build_config uses defaults."""
        output_file = tmp_path / "result.json"

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                ["-m", "bert-base-uncased", "-o", str(output_file)],
                obj={"verbose": 0},
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        output_data = json.loads(output_file.read_text())
        # Default opset_version from WinMLExportConfig is 17
        assert output_data["export"]["opset_version"] == 17

    def test_config_cli_nonexistent_file_fails(self) -> None:
        """--config with nonexistent file fails with click error."""
        runner = CliRunner()
        result = runner.invoke(
            config_command,
            ["-m", "bert-base-uncased", "-c", "/nonexistent/path/override.json"],
        )

        assert result.exit_code != 0


# =============================================================================
# TestModelTypeOverride - Tests for model_type parameter
# =============================================================================


class TestModelTypeOverride:
    """Tests for generate_build_config with model_type override."""

    def test_model_type_with_task(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """model_type + task: overrides hf_config.model_type, uses given task."""
        gpt2_loader_config = WinMLLoaderConfig(
            task="text-generation",
            model_class="GPT2LMHeadModel",
            model_type="gpt2",
        )

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(gpt2_loader_config, mock_hf_config, mock_model_class),
            ) as mock_resolve,
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ) as mock_gen_export,
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            result = generate_build_config(
                "some-model",
                model_type="gpt2",
                task="text-generation",
            )

        assert isinstance(result, WinMLBuildConfig)
        # model_type should be passed to resolve_loader_config
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.kwargs["model_type"] == "gpt2"
        # _resolve_export_config_from_specs receives model_type from loader_config
        mock_gen_export.assert_called_once()
        assert mock_gen_export.call_args.kwargs["model_type"] == "gpt2"

    def test_model_type_without_task_auto_detects(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """model_type only: resolve_loader_config handles auto-detection."""
        # resolve_loader_config handles auto-detection internally now
        auto_loader_config = WinMLLoaderConfig(
            task="feature-extraction",
            model_class="BertModel",
            model_type="bert",
        )

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(auto_loader_config, mock_hf_config, mock_model_class),
            ) as mock_resolve,
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            result = generate_build_config("some-model", model_type="bert")

        assert isinstance(result, WinMLBuildConfig)
        # resolve_loader_config should be called with model_type but no task
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.kwargs["model_type"] == "bert"
        assert mock_resolve.call_args.kwargs["task"] is None

    def test_model_type_no_supported_tasks_raises(
        self,
        mock_hf_config: MagicMock,
    ) -> None:
        """model_type with no supported tasks raises ValueError."""
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                side_effect=ValueError("No supported tasks found"),
            ),
            pytest.raises(ValueError, match="No supported tasks found"),
        ):
            generate_build_config("some-model", model_type="nonexistent_type")

    def test_model_type_standalone_no_model_id(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """model_type without model_id: resolve_loader_config handles default config."""
        standalone_loader_config = WinMLLoaderConfig(
            task="feature-extraction",
            model_class="BertModel",
            model_type="bert",
        )

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(standalone_loader_config, mock_hf_config, mock_model_class),
            ) as mock_resolve,
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            result = generate_build_config(model_type="bert")

        assert isinstance(result, WinMLBuildConfig)
        # resolve_loader_config should be called with model_id=None and model_type="bert"
        mock_resolve.assert_called_once()
        assert mock_resolve.call_args[0][0] is None  # model_id positional arg
        assert mock_resolve.call_args.kwargs["model_type"] == "bert"

    def test_neither_model_id_nor_model_type_raises(self) -> None:
        """Neither model_id nor model_type raises ValueError."""
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                side_effect=ValueError("Either model_id or model_type"),
            ),
            pytest.raises(ValueError, match="Either model_id or model_type"),
        ):
            generate_build_config()

    def test_model_type_none_uses_hf_config(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """model_type=None (default) uses hf_config.model_type as before."""
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ) as mock_gen_export,
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            generate_build_config("bert-base-uncased")

        # _resolve_export_config_from_specs receives model_type from loader_config
        mock_gen_export.assert_called_once()
        assert mock_gen_export.call_args.kwargs["model_type"] == "bert"


class TestModelTypeCliOverride:
    """Tests for --model-type CLI option."""

    def test_model_type_cli_with_task(
        self,
        tmp_path,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """--model-type + --task both passed through to generate_build_config."""
        output_file = tmp_path / "result.json"

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                [
                    "-m",
                    "some-model",
                    "--model-type",
                    "bert",
                    "--task",
                    "fill-mask",
                    "-o",
                    str(output_file),
                ],
                obj={"verbose": 0},
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        output_data = json.loads(output_file.read_text())
        assert output_data["export"]["opset_version"] == 17

    def test_model_type_cli_auto_task(
        self,
        tmp_path,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """--model-type without --task auto-detects task from resolve_loader_config."""
        output_file = tmp_path / "result.json"

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                ["-m", "some-model", "--model-type", "bert", "-o", str(output_file)],
                obj={"verbose": 0},
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        output_data = json.loads(output_file.read_text())
        assert "export" in output_data

    def test_model_type_cli_standalone_no_model_id(
        self,
        tmp_path,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """--model-type without -m works (resolve_loader_config handles it)."""
        output_file = tmp_path / "result.json"

        standalone_loader_config = WinMLLoaderConfig(
            task="feature-extraction",
            model_class="BertModel",
            model_type="bert",
        )

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(standalone_loader_config, mock_hf_config, mock_model_class),
            ) as mock_resolve,
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                ["--model-type", "bert", "-o", str(output_file)],
                obj={"verbose": 0},
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        mock_resolve.assert_called_once()
        output_data = json.loads(output_file.read_text())
        assert "export" in output_data

    def test_cli_neither_model_nor_model_type_fails(self) -> None:
        """Without both -m and --model-type, CLI exits with error."""
        runner = CliRunner()
        result = runner.invoke(config_command, ["--task", "fill-mask"], obj={"verbose": 0})

        assert result.exit_code == 2
        assert "at least one" in result.output.lower()


# =============================================================================
# TestEdgeCases - Missing edge case tests (T1-T6)
# =============================================================================


class TestEdgeCases:
    """Edge case tests identified during code review."""

    def test_config_cli_malformed_json_fails(self, tmp_path) -> None:
        """T1: --config with malformed JSON shows helpful error."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text('{"export": {invalid json')

        runner = CliRunner()
        result = runner.invoke(
            config_command,
            ["--model-type", "bert", "-c", str(bad_json)],
            obj={"verbose": 0},
        )

        assert result.exit_code == 2
        assert "Invalid JSON" in result.output

    def test_config_cli_empty_file_fails(self, tmp_path) -> None:
        """T6: --config with empty file shows helpful error."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("")

        runner = CliRunner()
        result = runner.invoke(
            config_command,
            ["--model-type", "bert", "-c", str(empty_file)],
            obj={"verbose": 0},
        )

        assert result.exit_code == 2
        assert "empty" in result.output.lower()

    def test_config_cli_non_dict_json_fails(self, tmp_path) -> None:
        """T1b: --config with non-dict JSON (array) shows helpful error."""
        array_json = tmp_path / "array.json"
        array_json.write_text("[1, 2, 3]")

        runner = CliRunner()
        result = runner.invoke(
            config_command,
            ["--model-type", "bert", "-c", str(array_json)],
            obj={"verbose": 0},
        )

        assert result.exit_code == 2
        assert "JSON object" in result.output

    def test_invalid_model_type_fails_gracefully(self) -> None:
        """T2: Invalid model_type gives helpful ValueError, not KeyError."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            generate_build_config(model_type="invalid_model_type_xyz_123")

    def test_config_and_model_type_combined(
        self,
        tmp_path,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """T3: --config + --model-type combined works correctly."""
        standalone_loader_config = WinMLLoaderConfig(
            task="feature-extraction",
            model_class="BertModel",
            model_type="bert",
        )

        override_data = {"export": {"opset_version": 18}}
        override_file = tmp_path / "override.json"
        override_file.write_text(json.dumps(override_data))
        output_file = tmp_path / "result.json"

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(standalone_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                [
                    "--model-type",
                    "bert",
                    "-c",
                    str(override_file),
                    "-o",
                    str(output_file),
                ],
                obj={"verbose": 0},
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        output_data = json.loads(output_file.read_text())
        assert output_data["export"]["opset_version"] == 18

    def test_empty_override_json_is_noop(
        self,
        tmp_path,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """Empty {} override behaves like no override."""
        empty_override = tmp_path / "empty.json"
        empty_override.write_text("{}")
        output_file = tmp_path / "result.json"

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                ["-m", "bert-base-uncased", "-c", str(empty_override), "-o", str(output_file)],
                obj={"verbose": 0},
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        output_data = json.loads(output_file.read_text())
        # Default opset preserved
        assert output_data["export"]["opset_version"] == 17


# =============================================================================
# TestShapeConfig - Tests for shape_config / --shape-config
# =============================================================================


class TestShapeConfig:
    """Tests for shape_config parameter and --shape-config CLI flag."""

    def test_shape_config_passed_as_shape_kwargs(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """shape_config dict is unpacked as **shape_kwargs to _resolve_export_config_from_specs."""
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ) as mock_gen_export,
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            generate_build_config(
                "bert-base-uncased",
                shape_config={"sequence_length": 128},
            )

        mock_gen_export.assert_called_once()
        call_kwargs = mock_gen_export.call_args
        assert call_kwargs.kwargs.get("sequence_length") == 128

    def test_shape_config_none_no_extra_kwargs(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """shape_config=None passes no extra shape_kwargs."""
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ) as mock_gen_export,
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            generate_build_config("bert-base-uncased", shape_config=None)

        mock_gen_export.assert_called_once()
        # Should NOT have image_size or sequence_length in kwargs
        call_kwargs = mock_gen_export.call_args.kwargs
        assert "image_size" not in call_kwargs
        assert "sequence_length" not in call_kwargs

    def test_resnet_shape_config_height_width(self) -> None:
        """Real test: resnet with height/width=128 produces (1,3,128,128) shapes."""
        from transformers import ResNetConfig

        from winml.modelkit.export import resolve_io_specs

        hf_config = ResNetConfig(
            num_channels=3,
            hidden_sizes=[64, 128],
            depths=[1, 1],
            layer_type="basic",
        )

        specs = resolve_io_specs(
            model_type="resnet",
            task="image-classification",
            hf_config=hf_config,
            height=128,
            width=128,
        )

        assert "input_shapes" in specs
        pixel_shape = specs["input_shapes"][0]
        assert pixel_shape == (1, 3, 128, 128), f"Expected (1,3,128,128), got {pixel_shape}"

    def test_resnet_default_vs_shape_config(self) -> None:
        """Real test: resnet without shape_config gets default, with gets 128."""
        from transformers import ResNetConfig

        from winml.modelkit.export import resolve_io_specs

        hf_config = ResNetConfig(
            num_channels=3,
            hidden_sizes=[64, 128],
            depths=[1, 1],
            layer_type="basic",
        )

        # Without shape_config -- Optimum default (64x64 for ResNet without preprocessor)
        specs_default = resolve_io_specs(
            model_type="resnet",
            task="image-classification",
            hf_config=hf_config,
        )

        # With shape_config -- explicit 128x128 via height/width kwargs
        specs_override = resolve_io_specs(
            model_type="resnet",
            task="image-classification",
            hf_config=hf_config,
            height=128,
            width=128,
        )

        default_h = specs_default["input_shapes"][0][2]
        override_h = specs_override["input_shapes"][0][2]

        assert override_h == 128
        assert default_h != 128, f"Default height {default_h} should differ from override 128"


class TestShapeConfigCli:
    """CLI tests for --shape-config flag."""

    def test_shape_config_cli_resnet(self, tmp_path) -> None:
        """E2E: --model-type resnet --shape-config with height/width=128."""
        shape_file = tmp_path / "shapes.json"
        shape_file.write_text('{"height": 128, "width": 128}')
        output_file = tmp_path / "result.json"

        runner = CliRunner()
        result = runner.invoke(
            config_command,
            [
                "--model-type",
                "resnet",
                "--task",
                "image-classification",
                "--shape-config",
                str(shape_file),
                "-o",
                str(output_file),
            ],
            obj={"verbose": 0},
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        output_data = json.loads(output_file.read_text())
        # Check pixel_values shape has 128x128
        input_tensors = output_data["export"]["input_tensors"]
        pixel_values = next(t for t in input_tensors if t["name"] == "pixel_values")
        assert pixel_values["shape"] == [1, 3, 128, 128]

    def test_shape_config_cli_with_config_combined(
        self,
        tmp_path,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """--config + --shape-config can be used together."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"export": {"opset_version": 18}}')

        shape_file = tmp_path / "shapes.json"
        shape_file.write_text('{"sequence_length": 64}')

        output_file = tmp_path / "result.json"

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ) as mock_gen_export,
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                [
                    "-m",
                    "bert-base-uncased",
                    "-c",
                    str(config_file),
                    "--shape-config",
                    str(shape_file),
                    "-o",
                    str(output_file),
                ],
                obj={"verbose": 0},
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        # shape_config passed as shape_kwargs to _resolve_export_config_from_specs
        mock_gen_export.assert_called_once()
        assert mock_gen_export.call_args.kwargs.get("sequence_length") == 64
        # config override also applied (opset_version=18)
        output_data = json.loads(output_file.read_text())
        assert output_data["export"]["opset_version"] == 18

    def test_shape_config_cli_malformed_json(self, tmp_path) -> None:
        """--shape-config with malformed JSON shows helpful error."""
        bad_file = tmp_path / "bad_shapes.json"
        bad_file.write_text("{invalid json")

        runner = CliRunner()
        result = runner.invoke(
            config_command,
            ["--model-type", "bert", "--shape-config", str(bad_file)],
            obj={"verbose": 0},
        )

        assert result.exit_code == 2
        assert "Invalid JSON" in result.output


# =============================================================================
# TestValidate - Tests for WinMLBuildConfig.validate()
# =============================================================================


class TestValidate:
    """Tests for WinMLBuildConfig.validate() method."""

    def test_valid_config_passes(self) -> None:
        """A fully populated config passes validation without error."""
        config = WinMLBuildConfig(
            loader=WinMLLoaderConfig(task="image-classification"),
            export=WinMLExportConfig(),
            optim=WinMLOptimizationConfig(),
            quant=WinMLQuantizationConfig(
                task="image-classification",
                model_name="microsoft/resnet-50",
            ),
            compile=WinMLCompileConfig(
                ep_config=EPConfig(provider="qnn"),
            ),
        )
        # Should not raise
        config.validate()

    def test_valid_config_no_quant_no_compile(self) -> None:
        """Config with quant=None and compile=None passes if required fields set."""
        config = WinMLBuildConfig(
            loader=WinMLLoaderConfig(task="fill-mask"),
            export=WinMLExportConfig(),
            optim=WinMLOptimizationConfig(),
            quant=None,
            compile=None,
        )
        # Should not raise
        config.validate()

    def test_missing_task_raises(self) -> None:
        """Missing loader.task raises ValueError for HF builds."""
        config = WinMLBuildConfig(
            loader=WinMLLoaderConfig(task=None),
            export=WinMLExportConfig(),  # HF build (export present)
            optim=WinMLOptimizationConfig(),
            quant=None,
            compile=None,
        )
        with pytest.raises(ValueError, match=r"loader\.task is required"):
            config.validate()

    def test_empty_task_raises(self) -> None:
        """Empty string loader.task raises ValueError for HF builds."""
        config = WinMLBuildConfig(
            loader=WinMLLoaderConfig(task=""),
            export=WinMLExportConfig(),  # HF build (export present)
            optim=WinMLOptimizationConfig(),
            quant=None,
            compile=None,
        )
        with pytest.raises(ValueError, match=r"loader\.task is required"):
            config.validate()

    def test_valid_onnx_build_no_loader_task(self) -> None:
        """ONNX build (export=None) passes validation without loader.task."""
        config = WinMLBuildConfig(
            loader=WinMLLoaderConfig(task=None),
            export=None,  # ONNX build
            optim=WinMLOptimizationConfig(),
            quant=None,
            compile=None,
        )
        config.validate()  # Should not raise

    def test_valid_onnx_build_with_quant_no_task(self) -> None:
        """ONNX build with quant doesn't require quant.task or quant.model_name."""
        config = WinMLBuildConfig(
            loader=WinMLLoaderConfig(task=None),
            export=None,  # ONNX build
            optim=WinMLOptimizationConfig(),
            quant=WinMLQuantizationConfig(task=None, model_name=None),
            compile=WinMLCompileConfig(),
        )
        config.validate()  # Should not raise

    def test_missing_optim_raises(self) -> None:
        """optim=None raises ValueError."""
        config = WinMLBuildConfig(
            loader=WinMLLoaderConfig(task="fill-mask"),
            export=WinMLExportConfig(),
            optim=None,
            quant=None,
            compile=None,
        )
        with pytest.raises(ValueError, match="optim config is required"):
            config.validate()

    def test_quant_missing_task_raises(self) -> None:
        """quant enabled but task=None raises ValueError for HF builds."""
        config = WinMLBuildConfig(
            loader=WinMLLoaderConfig(task="fill-mask"),
            export=WinMLExportConfig(),  # HF build
            optim=WinMLOptimizationConfig(),
            quant=WinMLQuantizationConfig(task=None, model_name="test-model"),
            compile=None,
        )
        with pytest.raises(ValueError, match=r"quant\.task is required"):
            config.validate()

    def test_quant_missing_model_name_raises(self) -> None:
        """quant enabled but model_name=None raises ValueError for HF builds."""
        config = WinMLBuildConfig(
            loader=WinMLLoaderConfig(task="fill-mask"),
            export=WinMLExportConfig(),  # HF build
            optim=WinMLOptimizationConfig(),
            quant=WinMLQuantizationConfig(task="fill-mask", model_name=None),
            compile=None,
        )
        with pytest.raises(ValueError, match=r"quant\.model_name is required"):
            config.validate()

    def test_compile_missing_provider_raises(self) -> None:
        """compile enabled but ep_config.provider empty raises ValueError."""
        config = WinMLBuildConfig(
            loader=WinMLLoaderConfig(task="fill-mask"),
            export=WinMLExportConfig(),
            optim=WinMLOptimizationConfig(),
            quant=None,
            compile=WinMLCompileConfig(
                ep_config=EPConfig(provider=""),
            ),
        )
        with pytest.raises(ValueError, match=r"compile\.ep_config\.provider is required"):
            config.validate()

    def test_multiple_errors_collected_hf_build(self) -> None:
        """Multiple validation failures are all reported in a single ValueError (HF build)."""
        config = WinMLBuildConfig(
            loader=WinMLLoaderConfig(task=None),
            export=WinMLExportConfig(),  # HF build (export present)
            optim=None,
            quant=WinMLQuantizationConfig(task=None, model_name=None),
            compile=WinMLCompileConfig(ep_config=EPConfig(provider="")),
        )
        with pytest.raises(ValueError, match="Invalid WinMLBuildConfig") as exc_info:
            config.validate()

        error_msg = str(exc_info.value)
        assert "loader.task is required for full model builds" in error_msg
        assert "optim config is required" in error_msg
        assert "quant.task is required when quant is enabled for HF builds" in error_msg
        assert "quant.model_name is required when quant is enabled for HF builds" in error_msg
        assert "compile.ep_config.provider is required" in error_msg

    def test_multiple_errors_collected_onnx_build(self) -> None:
        """ONNX build collects only applicable errors (optim, compile provider)."""
        config = WinMLBuildConfig(
            loader=WinMLLoaderConfig(task=None),
            export=None,  # ONNX build
            optim=None,
            quant=WinMLQuantizationConfig(task=None, model_name=None),
            compile=WinMLCompileConfig(ep_config=EPConfig(provider="")),
        )
        with pytest.raises(ValueError, match="Invalid WinMLBuildConfig") as exc_info:
            config.validate()

        error_msg = str(exc_info.value)
        # ONNX build: loader.task NOT required, quant.task/model_name NOT required
        assert "loader.task" not in error_msg
        assert "quant.task" not in error_msg
        assert "quant.model_name" not in error_msg
        # These still apply
        assert "optim config is required" in error_msg
        assert "compile.ep_config.provider is required" in error_msg


# =============================================================================
# TestInt16QuantTypes - Tests for int16/uint16 quantization type support
# =============================================================================


class TestInt16QuantTypes:
    """Tests for int16/uint16 activation and weight data type support."""

    def test_config_accepts_int16_activation(self) -> None:
        """WinMLQuantizationConfig accepts activation_type='int16'."""
        config = WinMLQuantizationConfig(activation_type="int16")
        assert config.activation_type == "int16"

    def test_config_accepts_uint16_activation(self) -> None:
        """WinMLQuantizationConfig accepts activation_type='uint16'."""
        config = WinMLQuantizationConfig(activation_type="uint16")
        assert config.activation_type == "uint16"

    def test_config_accepts_int16_weight(self) -> None:
        """WinMLQuantizationConfig accepts weight_type='int16'."""
        config = WinMLQuantizationConfig(weight_type="int16")
        assert config.weight_type == "int16"

    def test_config_accepts_uint16_weight(self) -> None:
        """WinMLQuantizationConfig accepts weight_type='uint16'."""
        config = WinMLQuantizationConfig(weight_type="uint16")
        assert config.weight_type == "uint16"

    def test_to_dict_round_trip_int16(self) -> None:
        """Create config with int16 types, to_dict(), from_dict() preserves values."""
        original = WinMLQuantizationConfig(
            weight_type="int16",
            activation_type="uint16",
        )
        data = original.to_dict()
        restored = WinMLQuantizationConfig.from_dict(data)

        assert restored.weight_type == "int16"
        assert restored.activation_type == "uint16"

    def test_default_unchanged(self) -> None:
        """Default weight_type and activation_type are still 'uint8'."""
        config = WinMLQuantizationConfig()
        assert config.weight_type == "uint8"
        assert config.activation_type == "uint8"


# =============================================================================
# TestDevicePrecisionIntegration - device/precision in generate_build_config()
# =============================================================================


class TestDevicePrecisionIntegration:
    """Test generate_build_config() with --device/--precision params.

    All tests mock resolve_device, resolve_loader_config, and
    _resolve_export_config_from_specs for fast execution (no network, no hardware).
    """

    @pytest.fixture(autouse=True)
    def _mock_deps(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """Set up common mocks for loader, export, and device resolution."""
        self._mock_hf_config = mock_hf_config
        self._mock_model_class = mock_model_class
        self._mock_loader_config = mock_loader_config
        self._mock_export_config = mock_export_config

    @pytest.mark.parametrize(
        "device,precision,expect_quant,expect_weight,expect_act,expect_compile_provider",
        [
            ("npu", "auto", True, "uint8", "uint8", "qnn"),
            ("npu", "fp16", False, None, None, "qnn"),
            ("npu", "int8", True, "uint8", "uint8", "qnn"),
            ("gpu", "auto", False, None, None, "dml"),
            ("gpu", "int8", True, "uint8", "uint8", "dml"),
            ("gpu", "fp16", False, None, None, "dml"),
            ("cpu", "auto", False, None, None, None),
            ("cpu", "int8", True, "uint8", "uint8", None),
            ("cpu", "int16", True, "int16", "uint16", None),
            ("cpu", "fp16", False, None, None, None),
            # auto device + explicit precision → picks NPU (mock returns npu first)
            ("auto", "fp16", False, None, None, "qnn"),
            ("auto", "int8", True, "uint8", "uint8", "qnn"),
            ("auto", "int16", True, "int16", "uint16", "qnn"),
        ],
    )
    def test_config_gen_device_precision(
        self,
        device: str,
        precision: str,
        expect_quant: bool,
        expect_weight: str | None,
        expect_act: str | None,
        expect_compile_provider: str | None,
    ) -> None:
        """generate_build_config applies precision policy to quant + compile."""
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(
                    self._mock_loader_config,
                    self._mock_hf_config,
                    self._mock_model_class,
                ),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=self._mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=(
                    "npu" if device == "auto" else device,
                    ["npu", "gpu", "cpu"],
                ),
            ),
        ):
            result = generate_build_config(
                "bert-base-uncased",
                device=device,
                precision=precision,
            )

        # Verify quant config
        if expect_quant:
            assert result.quant is not None, (
                f"Expected quant config for device={device}, precision={precision}"
            )
            assert result.quant.weight_type == expect_weight
            assert result.quant.activation_type == expect_act
        else:
            assert result.quant is None, (
                f"Expected no quant for device={device}, precision={precision}"
            )

        # Verify compile config
        if expect_compile_provider is not None:
            assert result.compile is not None
            assert result.compile.ep_config.provider == expect_compile_provider
            # TODO(#241): assert qdq_config alignment with quant policy
            # Currently for_qnn() creates qdq_config even for fp16.
            # Issue #241 will pass quantize= to for_provider().
        else:
            assert result.compile is None

    def test_auto_auto_is_noop(self) -> None:
        """device='auto' + precision='auto' leaves quant/compile at defaults."""
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(
                    self._mock_loader_config,
                    self._mock_hf_config,
                    self._mock_model_class,
                ),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=self._mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            result = generate_build_config(
                "bert-base-uncased",
                device="auto",
                precision="auto",
            )

        # Both auto: quant and compile should be at dataclass defaults
        # (not None -- they come from _assemble_config defaults)
        assert result.quant is not None, "auto+auto should preserve default quant"
        assert result.compile is not None, "auto+auto should preserve default compile"
        # Default quant weight_type is "uint8" (from WinMLQuantizationConfig)
        assert result.quant.weight_type == "uint8"
        assert result.quant.activation_type == "uint8"
        # Default compile provider is "qnn" (from WinMLCompileConfig -> EPConfig)
        assert result.compile.ep_config.provider == "qnn"

    def test_auto_auto_still_calls_resolve_device(self) -> None:
        """device='auto' + precision='auto' DOES call resolve_device (#412).

        Previously this was skipped, causing EPConfig to default to 'qnn'
        on machines without an NPU. Now we always detect hardware.
        """
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(
                    self._mock_loader_config,
                    self._mock_hf_config,
                    self._mock_model_class,
                ),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=self._mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "gpu", "cpu"]),
            ) as mock_rd,
        ):
            generate_build_config(
                "bert-base-uncased",
                device="auto",
                precision="auto",
            )

        mock_rd.assert_called_once_with(device="auto")

    def test_explicit_precision_triggers_resolve_device(self) -> None:
        """device='auto' + precision='int8' DOES call resolve_device."""
        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(
                    self._mock_loader_config,
                    self._mock_hf_config,
                    self._mock_model_class,
                ),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=self._mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "gpu", "cpu"]),
            ) as mock_rd,
        ):
            generate_build_config(
                "bert-base-uncased",
                device="auto",
                precision="int8",
            )

        assert mock_rd.call_count >= 1, "resolve_device should be called at least once"


# =============================================================================
# TestDevicePrecisionCli - CLI tests for --device/--precision on winml config
# =============================================================================


class TestDevicePrecisionCli:
    """CLI tests for --device and --precision flags on config command.

    All tests mock resolve_device, resolve_loader_config, and
    _resolve_export_config_from_specs for fast execution (no network, no hardware).
    """

    @pytest.fixture(autouse=True)
    def _mock_deps(
        self,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """Set up common mocks."""
        self._patches = {
            "loader": patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            "export": patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            "registry": patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
            "device": patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "gpu", "cpu"]),
            ),
        }

    def _invoke(self, tmp_path, extra_args: list[str] | None = None):
        """Helper: invoke winml config with standard mocks."""
        output_file = tmp_path / "result.json"
        args = ["-m", "bert-base-uncased", "-o", str(output_file)]
        if extra_args:
            args.extend(extra_args)

        with (
            self._patches["loader"],
            self._patches["export"],
            self._patches["registry"],
            self._patches["device"],
        ):
            runner = CliRunner()
            result = runner.invoke(config_command, args, obj={"verbose": 0})

        return result, output_file

    def test_device_npu_produces_qnn(self, tmp_path) -> None:
        """--device npu → compile.provider=qnn, quant with uint8."""
        result, output_file = self._invoke(tmp_path, ["--device", "npu"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        data = json.loads(output_file.read_text())
        assert data["compile"] is not None
        assert data["compile"]["execution_provider"] == "qnn"
        assert data["quant"] is not None
        assert data["quant"]["weight_type"] == "uint8"
        assert data["quant"]["activation_type"] == "uint8"

    def test_device_gpu_precision_fp16(self, tmp_path) -> None:
        """--device gpu --precision fp16 → no quant, compile.provider=dml."""
        self._patches["device"] = patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("gpu", ["gpu", "cpu"]),
        )
        result, output_file = self._invoke(
            tmp_path,
            ["--device", "gpu", "--precision", "fp16"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        data = json.loads(output_file.read_text())
        assert data["quant"] is None
        assert data["compile"] is not None
        assert data["compile"]["execution_provider"] == "dml"

    def test_device_cpu_precision_fp32(self, tmp_path) -> None:
        """--device cpu --precision fp32 → no quant, no compile."""
        self._patches["device"] = patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("cpu", ["cpu"]),
        )
        result, output_file = self._invoke(
            tmp_path,
            ["--device", "cpu", "--precision", "fp32"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        data = json.loads(output_file.read_text())
        assert data["quant"] is None
        assert data["compile"] is None

    def test_default_no_flags_preserves_defaults(self, tmp_path) -> None:
        """No --device/--precision flags preserves default config."""
        result, output_file = self._invoke(tmp_path)

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        data = json.loads(output_file.read_text())
        # Default: quant and compile both present (backward compat)
        assert data["quant"] is not None
        assert data["compile"] is not None

    def test_auto_precision_int8_triggers_detection(self, tmp_path) -> None:
        """--device auto --precision int8 → triggers device detection."""
        result, output_file = self._invoke(
            tmp_path,
            ["--device", "auto", "--precision", "int8"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        data = json.loads(output_file.read_text())
        # Mock resolve_device returns "npu" → qnn
        assert data["compile"]["execution_provider"] == "qnn"
        assert data["quant"] is not None


# =============================================================================
# TestConfigOnnxAutoDetect - ONNX file auto-detection in config command
# =============================================================================


class TestConfigOnnxAutoDetect:
    """Test ONNX file auto-detection in winml config command."""

    def test_config_auto_detect_onnx(self, tmp_path) -> None:
        """When -m points to an existing .onnx file, generates config with export=None."""
        # Create a fake .onnx file
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake-onnx-data")
        output_file = tmp_path / "result.json"

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                ["-m", str(onnx_file), "-o", str(output_file)],
                obj={"verbose": 0},
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        output_data = json.loads(output_file.read_text())
        # ONNX build: export should be None
        assert output_data["export"] is None
        # optim should be present (default)
        assert output_data["optim"] is not None

    def test_config_onnx_with_device_precision(self, tmp_path) -> None:
        """ONNX config with --device npu applies quant/compile policy."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake-onnx-data")
        output_file = tmp_path / "result.json"

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "gpu", "cpu"]),
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                ["-m", str(onnx_file), "--device", "npu", "-o", str(output_file)],
                obj={"verbose": 0},
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        output_data = json.loads(output_file.read_text())
        assert output_data["export"] is None
        assert output_data["quant"] is not None
        assert output_data["compile"] is not None
        assert output_data["compile"]["execution_provider"] == "qnn"

    def test_config_onnx_suffix_not_exists_uses_hf(
        self,
        tmp_path,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """An .onnx path that doesn't exist falls through to HF config generation."""
        output_file = tmp_path / "result.json"

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                ["-m", "nonexistent.onnx", "-o", str(output_file)],
                obj={"verbose": 0},
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        output_data = json.loads(output_file.read_text())
        # Should be HF config (export present)
        assert output_data["export"] is not None


# =============================================================================
# TestGenerateBuildConfigOnnxPath - Comprehensive tests for generate_onnx_build_config
# =============================================================================


class TestGenerateBuildConfigOnnxPath:
    """Comprehensive tests for generate_onnx_build_config() covering all branches.

    Tests call generate_onnx_build_config directly (not the dispatcher) and mock:
    - modelkit.onnx.is_compiled_onnx
    - modelkit.onnx.is_quantized_onnx
    - modelkit.sysinfo.resolve_device
    """

    # -----------------------------------------------------------------
    # Model state detection
    # -----------------------------------------------------------------

    def test_raw_onnx_full_pipeline(self, tmp_path) -> None:
        """Raw ONNX + device=npu resolves quant=w8a8 and compile=qnn."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file), device="npu")

        assert config.export is None
        assert config.quant is not None
        assert config.quant.weight_type == "uint8"
        assert config.quant.activation_type == "uint8"
        assert config.compile is not None
        assert config.compile.ep_config.provider == "qnn"

    def test_raw_onnx_cpu(self, tmp_path) -> None:
        """Raw ONNX + device=cpu resolves quant=None and compile=None."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file), device="cpu")

        assert config.export is None
        assert config.quant is None
        assert config.compile is None

    def test_quantized_onnx_skips_quant(self, tmp_path) -> None:
        """Quantized ONNX + device=npu sets quant=None, compile=qnn."""
        onnx_file = tmp_path / "quantized.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=True),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file), device="npu")

        assert config.quant is None
        assert config.compile is not None
        assert config.compile.ep_config.provider == "qnn"

    def test_quantized_onnx_cpu(self, tmp_path) -> None:
        """Quantized ONNX + device=cpu sets quant=None, compile=None."""
        onnx_file = tmp_path / "quantized.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=True),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file), device="cpu")

        assert config.quant is None
        assert config.compile is None

    def test_compiled_onnx_skips_all(self, tmp_path) -> None:
        """Compiled ONNX (EPContext) sets quant=None and compile=None."""
        onnx_file = tmp_path / "compiled.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=True),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            config = generate_onnx_build_config(str(onnx_file))

        assert config.quant is None
        assert config.compile is None

    def test_compiled_onnx_with_device_npu(self, tmp_path) -> None:
        """Compiled ONNX + device=npu still sets quant=None and compile=None.

        The compiled detection short-circuits before resolve_quant_compile_config
        is called, so device parameter has no effect.
        """
        onnx_file = tmp_path / "compiled.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=True),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            config = generate_onnx_build_config(str(onnx_file), device="npu")

        assert config.quant is None
        assert config.compile is None

    # -----------------------------------------------------------------
    # Config structure invariants
    # -----------------------------------------------------------------

    def test_export_always_none(self, tmp_path) -> None:
        """All ONNX model states produce export=None."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        states = [
            (False, False, "raw"),
            (False, True, "quantized"),
            (True, False, "compiled"),
        ]
        for is_compiled, is_quantized, label in states:
            with (
                patch("winml.modelkit.onnx.is_compiled_onnx", return_value=is_compiled),
                patch("winml.modelkit.onnx.is_quantized_onnx", return_value=is_quantized),
                patch(
                    "winml.modelkit.sysinfo.resolve_device",
                    return_value=("cpu", ["cpu"]),
                ),
            ):
                config = generate_onnx_build_config(str(onnx_file))

            assert config.export is None, f"export should be None for {label} model"

    def test_optim_always_present(self, tmp_path) -> None:
        """All ONNX model states produce a non-None optim config."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        states = [
            (False, False, "raw"),
            (False, True, "quantized"),
            (True, False, "compiled"),
        ]
        for is_compiled, is_quantized, label in states:
            with (
                patch("winml.modelkit.onnx.is_compiled_onnx", return_value=is_compiled),
                patch("winml.modelkit.onnx.is_quantized_onnx", return_value=is_quantized),
                patch(
                    "winml.modelkit.sysinfo.resolve_device",
                    return_value=("cpu", ["cpu"]),
                ),
            ):
                config = generate_onnx_build_config(str(onnx_file))

            assert isinstance(config.optim, WinMLOptimizationConfig), (
                f"optim should be WinMLOptimizationConfig for {label} model"
            )

    def test_task_stored_in_loader(self, tmp_path) -> None:
        """task='image-classification' is stored in config.loader.task."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
        ):
            config = generate_onnx_build_config(
                str(onnx_file),
                task="image-classification",
            )

        assert config.loader.task == "image-classification"

    def test_task_none_by_default(self, tmp_path) -> None:
        """When no task is provided, config.loader.task is None."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file))

        assert config.loader.task is None

    # -----------------------------------------------------------------
    # Override behavior
    # -----------------------------------------------------------------

    def test_override_applied_last(self, tmp_path) -> None:
        """Override with a specific optim flag is present after device resolution.

        WinMLOptimizationConfig is a dict subclass, so flags are dict keys.
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        override = WinMLBuildConfig(
            optim=WinMLOptimizationConfig(gelu_fusion=True),
        )

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(
                str(onnx_file),
                device="npu",
                override=override,
            )

        assert config.optim["gelu_fusion"] is True

    def test_override_quant_none_on_raw(self, tmp_path) -> None:
        """Raw ONNX + device=npu would resolve quant, but override sets quant=None.

        Override is applied last via merge_config, and explicit None in the
        override replaces the resolved quant config.
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        # merge_config uses to_dict() which produces {"quant": None, ...}
        # only when quant is explicitly None. Build from dict to control this.
        override_dict = {"quant": None}

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "cpu"]),
            ),
            patch(
                "winml.modelkit.config.build.merge_config",
                wraps=merge_config,
            ) as mock_merge,
        ):
            # Call with a real override that sets quant=None
            override_cfg = WinMLBuildConfig.from_dict(override_dict)
            config = generate_onnx_build_config(
                str(onnx_file),
                device="npu",
                override=override_cfg,
            )

        mock_merge.assert_called_once()
        # merge_config with quant=None override should set quant to None
        assert config.quant is None

    def test_override_on_compiled_model(self, tmp_path) -> None:
        """Compiled model + override with quant set: override is applied AFTER
        compiled detection, so override CAN set quant on a compiled model.

        This tests that merge_config runs after the compiled branch.
        merge_config reconstructs the quant field from the override dict when
        the base quant is None.
        """
        onnx_file = tmp_path / "compiled.onnx"
        onnx_file.write_bytes(b"fake")

        override = WinMLBuildConfig(
            quant=WinMLQuantizationConfig(weight_type="uint8"),
        )

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=True),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            config = generate_onnx_build_config(
                str(onnx_file),
                override=override,
            )

        # Override is applied after compiled detection, so quant is non-None.
        # merge_config stores it as a dict (base quant is None, so from_dict
        # reconstruction depends on type annotation resolution).
        assert config.quant is not None
        if isinstance(config.quant, dict):
            assert config.quant["weight_type"] == "uint8"
        else:
            assert config.quant.weight_type == "uint8"

    def test_override_none_is_noop(self, tmp_path) -> None:
        """override=None does not change the config."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(
                str(onnx_file),
                device="npu",
                override=None,
            )

        # Without override, raw+npu should have quant and compile
        assert config.quant is not None
        assert config.compile is not None

    # -----------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------

    def test_onnx_path_as_string(self, tmp_path) -> None:
        """String path is accepted and works correctly."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file))

        assert config.export is None

    def test_onnx_path_as_pathlib(self, tmp_path) -> None:
        """pathlib.Path object is accepted and works correctly."""
        from pathlib import Path

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
        ):
            config = generate_onnx_build_config(Path(onnx_file))

        assert config.export is None

    def test_auto_device_auto_precision_defaults(self, tmp_path) -> None:
        """device=auto + precision=auto (defaults) keeps config defaults.

        resolve_quant_compile_config returns (None, None) when both are auto,
        so raw ONNX gets quant=None, compile=None.
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("auto", ["npu", "gpu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file))

        # Both auto -> resolve_precision returns device="auto" -> (None, None)
        assert config.quant is None
        assert config.compile is None

    def test_compiled_does_not_call_resolve_quant_compile(self, tmp_path) -> None:
        """Compiled model short-circuits before resolve_quant_compile_config."""
        onnx_file = tmp_path / "compiled.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=True),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.config.build.resolve_quant_compile_config",
            ) as mock_resolve,
        ):
            generate_onnx_build_config(str(onnx_file), device="npu")

        mock_resolve.assert_not_called()

    def test_raw_onnx_with_gpu(self, tmp_path) -> None:
        """Raw ONNX + device=gpu resolves quant=None, compile=dml."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("gpu", ["gpu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file), device="gpu")

        # GPU auto-precision is fp16 -> no quantization, compile=dml
        assert config.quant is None
        assert config.compile is not None
        assert config.compile.ep_config.provider == "dml"

    def test_ep_override_forwarded(self, tmp_path) -> None:
        """Explicit ep parameter is forwarded to resolve_quant_compile_config."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("gpu", ["gpu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(
                str(onnx_file),
                device="gpu",
                ep="migraphx",
            )

        assert config.compile is not None
        assert config.compile.ep_config.provider == "migraphx"


# =============================================================================
# TestResolveQuantCompileConfig - Tests for the standalone resolver
# =============================================================================


class TestResolveQuantCompileConfig:
    """Tests for resolve_quant_compile_config() standalone function.

    This tests the shared device/precision resolution logic used by both
    the HF and ONNX build config paths.
    """

    def test_auto_auto_returns_none_none(self) -> None:
        """device=auto + precision=auto returns (None, None)."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("auto", ["npu", "gpu", "cpu"]),
        ):
            quant, compile_cfg = resolve_quant_compile_config()

        assert quant is None
        assert compile_cfg is None

    def test_npu_returns_quant_and_compile(self) -> None:
        """device=npu returns (WinMLQuantizationConfig, WinMLCompileConfig)."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("npu", ["npu", "cpu"]),
        ):
            quant, compile_cfg = resolve_quant_compile_config(device="npu")

        assert isinstance(quant, WinMLQuantizationConfig)
        assert quant.weight_type == "uint8"
        assert quant.activation_type == "uint8"
        assert isinstance(compile_cfg, WinMLCompileConfig)
        assert compile_cfg.ep_config.provider == "qnn"

    def test_gpu_returns_none_quant_and_dml_compile(self) -> None:
        """device=gpu returns (None, WinMLCompileConfig(dml))."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("gpu", ["gpu", "cpu"]),
        ):
            quant, compile_cfg = resolve_quant_compile_config(device="gpu")

        assert quant is None
        assert isinstance(compile_cfg, WinMLCompileConfig)
        assert compile_cfg.ep_config.provider == "dml"

    def test_cpu_returns_none_none(self) -> None:
        """device=cpu returns (None, None) since CPU has no compile provider."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("cpu", ["cpu"]),
        ):
            quant, compile_cfg = resolve_quant_compile_config(device="cpu")

        assert quant is None
        assert compile_cfg is None

    def test_ep_override_changes_provider(self) -> None:
        """Explicit ep overrides the default device-to-provider mapping."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("gpu", ["gpu", "cpu"]),
        ):
            _quant, compile_cfg = resolve_quant_compile_config(
                device="gpu",
                ep="tensorrt",
            )

        assert compile_cfg is not None
        assert compile_cfg.ep_config.provider == "tensorrt"

    def test_task_forwarded_to_resolve_precision(self) -> None:
        """task parameter is forwarded to resolve_precision.

        Patch at the source module since it is imported locally inside
        resolve_quant_compile_config.
        """
        with (
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("gpu", ["gpu", "cpu"]),
            ),
            patch(
                "winml.modelkit.config.precision.resolve_precision",
                wraps=__import__(
                    "winml.modelkit.config.precision", fromlist=["resolve_precision"]
                ).resolve_precision,
            ) as mock_prec,
        ):
            resolve_quant_compile_config(device="gpu", task="text-generation")

        mock_prec.assert_called_once()
        assert mock_prec.call_args.kwargs.get("task") == "text-generation"

    def test_explicit_int8_precision_on_npu(self) -> None:
        """Explicit precision=int8 on npu produces uint8 quant."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("npu", ["npu", "cpu"]),
        ):
            quant, _compile_cfg = resolve_quant_compile_config(
                device="npu",
                precision="int8",
            )

        assert quant is not None
        assert quant.weight_type == "uint8"
        assert quant.activation_type == "uint8"

    def test_explicit_fp32_precision_no_quant(self) -> None:
        """Explicit precision=fp32 produces no quantization."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("gpu", ["gpu", "cpu"]),
        ):
            quant, _compile_cfg = resolve_quant_compile_config(
                device="gpu",
                precision="fp32",
            )

        assert quant is None
