# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for modelkit.export.io module.

Tests the model-centric I/O utilities that fully leverage Optimum:
- _get_onnx_config: Get OnnxConfig from model instance (internal)
- generate_dummy_inputs: Generate inputs using OnnxConfig
- resolve_io_specs: Get I/O specs without model weights
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import (
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    ResNetConfig,
    ResNetForImageClassification,
    ViTConfig,
    ViTForImageClassification,
)

# Import models package to trigger ONNX config registration with TasksManager
# This is required for BERT/CLIP to use max_position_embeddings as sequence_length
import winml.modelkit.models  # noqa: F401
from winml.modelkit.export import generate_dummy_inputs, resolve_io_specs
from winml.modelkit.export.io import (  # Testing internal implementation
    _get_onnx_config,
    _populate_image_size_from_preprocessor,
)
from winml.modelkit.models.winml.kv_cache import PastKeyValueInputGenerator


# =============================================================================
# Test Constants - Minimal configs for fast testing
# =============================================================================

# Vision model constants
VISION_HIDDEN_SIZE = 64
VISION_NUM_HIDDEN_LAYERS = 2
VISION_NUM_ATTENTION_HEADS = 2
VISION_IMAGE_SIZE = 32
VISION_PATCH_SIZE = 8
VISION_NUM_CHANNELS = 3
VISION_INTERMEDIATE_SIZE = VISION_HIDDEN_SIZE * 4

# Text model constants
TEXT_VOCAB_SIZE = 1000
TEXT_HIDDEN_SIZE = 64
TEXT_NUM_HIDDEN_LAYERS = 2
TEXT_NUM_ATTENTION_HEADS = 2
TEXT_MAX_POSITION_EMBEDDINGS = 32
TEXT_INTERMEDIATE_SIZE = TEXT_HIDDEN_SIZE * 4

# Shared constants
PROJECTION_DIM = 32
BATCH_SIZE = 2


# =============================================================================
# Fixtures - Minimal model configs and instances
# =============================================================================


@pytest.fixture(scope="module")
def resnet_model():
    """Create minimal ResNet model for testing."""
    config = ResNetConfig(
        num_channels=VISION_NUM_CHANNELS,
        hidden_sizes=[32, 64],
        depths=[1, 1],
        layer_type="basic",
        num_labels=10,
    )
    return ResNetForImageClassification(config)


@pytest.fixture(scope="module")
def vit_model():
    """Create minimal ViT model for testing."""
    config = ViTConfig(
        hidden_size=VISION_HIDDEN_SIZE,
        num_hidden_layers=VISION_NUM_HIDDEN_LAYERS,
        num_attention_heads=VISION_NUM_ATTENTION_HEADS,
        intermediate_size=VISION_INTERMEDIATE_SIZE,
        image_size=VISION_IMAGE_SIZE,
        patch_size=VISION_PATCH_SIZE,
        num_channels=VISION_NUM_CHANNELS,
        num_labels=10,
    )
    return ViTForImageClassification(config)


@pytest.fixture(scope="module")
def clip_vision_model():
    """Create minimal CLIP vision model for testing."""
    config = CLIPVisionConfig(
        hidden_size=VISION_HIDDEN_SIZE,
        projection_dim=PROJECTION_DIM,
        num_hidden_layers=VISION_NUM_HIDDEN_LAYERS,
        num_attention_heads=VISION_NUM_ATTENTION_HEADS,
        image_size=VISION_IMAGE_SIZE,
        patch_size=VISION_PATCH_SIZE,
        num_channels=VISION_NUM_CHANNELS,
        intermediate_size=VISION_INTERMEDIATE_SIZE,
    )
    return CLIPVisionModelWithProjection(config)


@pytest.fixture(scope="module")
def clip_text_model():
    """Create minimal CLIP text model for testing."""
    config = CLIPTextConfig(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN_SIZE,
        projection_dim=PROJECTION_DIM,
        num_hidden_layers=TEXT_NUM_HIDDEN_LAYERS,
        num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
        max_position_embeddings=TEXT_MAX_POSITION_EMBEDDINGS,
        intermediate_size=TEXT_INTERMEDIATE_SIZE,
    )
    return CLIPTextModelWithProjection(config)


@pytest.fixture(scope="module")
def bert_model():
    """Create minimal BERT model for testing."""
    from transformers import BertConfig, BertModel

    config = BertConfig(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN_SIZE,
        num_hidden_layers=TEXT_NUM_HIDDEN_LAYERS,
        num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
        intermediate_size=TEXT_INTERMEDIATE_SIZE,
        max_position_embeddings=TEXT_MAX_POSITION_EMBEDDINGS,
    )
    return BertModel(config)


@pytest.fixture(scope="module")
def segformer_model():
    """Create minimal Segformer model for testing."""
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    config = SegformerConfig(
        image_size=VISION_IMAGE_SIZE,
        num_channels=VISION_NUM_CHANNELS,
        num_labels=10,
        hidden_sizes=[32, 64],
        num_attention_heads=[1, 2],
        num_encoder_blocks=2,
        depths=[1, 1],
        sr_ratios=[8, 4],
        decoder_hidden_size=32,
    )
    return SegformerForSemanticSegmentation(config)


# =============================================================================
# Parametrized Test Cases
# =============================================================================


class TestGetOnnxConfig:
    """Tests for _get_onnx_config internal function."""

    @pytest.mark.parametrize(
        "model_fixture,task,expected_model_type,expected_inputs,expected_outputs",
        [
            ("resnet_model", "image-classification", "resnet", ["pixel_values"], ["logits"]),
            ("vit_model", "image-classification", "vit", ["pixel_values"], ["logits"]),
            (
                "clip_vision_model",
                "feature-extraction",
                "clip_vision_model",
                ["pixel_values"],
                ["last_hidden_state"],
            ),
            (
                "clip_text_model",
                "feature-extraction",
                "clip_text_model",
                ["input_ids", "attention_mask"],
                ["text_embeds", "last_hidden_state"],
            ),
            (
                "bert_model",
                "feature-extraction",
                "bert",
                ["input_ids", "attention_mask", "token_type_ids"],
                ["last_hidden_state"],
            ),
            (
                "segformer_model",
                "image-segmentation",
                "segformer",
                ["pixel_values"],
                ["logits"],
            ),
        ],
        ids=["resnet", "vit", "clip-vision", "clip-text", "bert", "segformer"],
    )
    def test_get_onnx_config_returns_config(
        self, model_fixture, task, expected_model_type, expected_inputs, expected_outputs, request
    ):
        """_get_onnx_config returns valid OnnxConfig with correct I/O specs."""
        model = request.getfixturevalue(model_fixture)

        io_config = _get_onnx_config(model.config.model_type, task, model.config)

        # Basic structure checks
        assert io_config is not None
        assert hasattr(io_config, "inputs")
        assert hasattr(io_config, "outputs")
        assert hasattr(io_config, "generate_dummy_inputs")

        # Verify model type matches
        assert model.config.model_type == expected_model_type

        # Verify exact input names
        actual_inputs = set(io_config.inputs.keys())
        for expected_input in expected_inputs:
            assert expected_input in actual_inputs, (
                f"Missing expected input '{expected_input}'. Got: {actual_inputs}"
            )

        # Verify expected outputs are present
        actual_outputs = set(io_config.outputs.keys())
        for expected_output in expected_outputs:
            assert expected_output in actual_outputs, (
                f"Missing expected output '{expected_output}'. Got: {actual_outputs}"
            )

    @pytest.mark.parametrize(
        "model_fixture,task",
        [
            ("clip_vision_model", "feature-extraction"),
            ("clip_text_model", "feature-extraction"),
        ],
        ids=["clip-vision-image-feat", "clip-text-feat"],
    )
    def test_get_onnx_config_handles_task_synonyms(self, model_fixture, task, request):
        """_get_onnx_config handles task synonyms like image-feature-extraction."""
        model = request.getfixturevalue(model_fixture)

        # Should not raise - task synonyms should be handled
        io_config = _get_onnx_config(model.config.model_type, task, model.config)

        assert io_config is not None


class TestGenerateDummyInputs:
    """Tests for generate_dummy_inputs function."""

    @pytest.mark.parametrize(
        "model_fixture,task,expected_inputs",
        [
            ("resnet_model", "image-classification", ["pixel_values"]),
            ("vit_model", "image-classification", ["pixel_values"]),
            ("clip_vision_model", "feature-extraction", ["pixel_values"]),
            ("clip_text_model", "feature-extraction", ["input_ids", "attention_mask"]),
            ("bert_model", "feature-extraction", ["input_ids", "attention_mask"]),
            ("segformer_model", "image-segmentation", ["pixel_values"]),
        ],
        ids=["resnet", "vit", "clip-vision", "clip-text", "bert", "segformer"],
    )
    def test_generate_dummy_inputs_returns_expected_keys(
        self, model_fixture, task, expected_inputs, request
    ):
        """generate_dummy_inputs returns tensors with expected input names."""
        model = request.getfixturevalue(model_fixture)

        inputs = generate_dummy_inputs(model.config.model_type, task, model.config)

        assert isinstance(inputs, dict)
        for expected_key in expected_inputs:
            assert expected_key in inputs, f"Missing expected input: {expected_key}"
            assert isinstance(inputs[expected_key], torch.Tensor)

    @pytest.mark.parametrize(
        "model_fixture,task,input_name,expected_shape",
        [
            # Vision models: [batch, channels, height, width] - uses model's image_size
            ("resnet_model", "image-classification", "pixel_values", (BATCH_SIZE, 3, 64, 64)),
            (
                "vit_model",
                "image-classification",
                "pixel_values",
                (BATCH_SIZE, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE),
            ),
            (
                "clip_vision_model",
                "feature-extraction",
                "pixel_values",
                (BATCH_SIZE, 3, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE),
            ),
            # Segformer: uses image_size from config
            (
                "segformer_model",
                "image-segmentation",
                "pixel_values",
                (BATCH_SIZE, VISION_NUM_CHANNELS, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE),
            ),
            # Text models: uses max_position_embeddings via MaxLengthTextInputGenerator
            (
                "clip_text_model",
                "feature-extraction",
                "input_ids",
                (BATCH_SIZE, TEXT_MAX_POSITION_EMBEDDINGS),
            ),
            (
                "bert_model",
                "feature-extraction",
                "input_ids",
                (BATCH_SIZE, TEXT_MAX_POSITION_EMBEDDINGS),
            ),
        ],
        ids=["resnet", "vit", "clip-vision", "segformer", "clip-text", "bert"],
    )
    def test_generate_dummy_inputs_default_shape(
        self, model_fixture, task, input_name, expected_shape, request
    ):
        """generate_dummy_inputs generates tensors with correct default shapes.

        Verifies that Optimum uses the model's config for image dimensions
        and applies our specified batch_size.
        """
        model = request.getfixturevalue(model_fixture)

        inputs = generate_dummy_inputs(
            model.config.model_type, task, model.config, batch_size=BATCH_SIZE
        )

        assert input_name in inputs, f"Missing input: {input_name}"
        actual_shape = tuple(inputs[input_name].shape)
        assert actual_shape == expected_shape, (
            f"Input '{input_name}' has shape {actual_shape}, expected {expected_shape}"
        )

    @pytest.mark.parametrize(
        "model_fixture,task",
        [
            ("resnet_model", "image-classification"),
            ("vit_model", "image-classification"),
            ("clip_vision_model", "feature-extraction"),
            ("clip_text_model", "feature-extraction"),
            ("bert_model", "feature-extraction"),
            ("segformer_model", "image-segmentation"),
        ],
        ids=["resnet", "vit", "clip-vision", "clip-text", "bert", "segformer"],
    )
    def test_generate_dummy_inputs_with_custom_batch_size(self, model_fixture, task, request):
        """generate_dummy_inputs respects custom batch_size."""
        model = request.getfixturevalue(model_fixture)
        custom_batch_size = 4

        inputs = generate_dummy_inputs(
            model.config.model_type, task, model.config, batch_size=custom_batch_size
        )

        # All inputs should have the custom batch size
        for name, tensor in inputs.items():
            assert tensor.shape[0] == custom_batch_size, (
                f"Input '{name}' has batch size {tensor.shape[0]}, expected {custom_batch_size}"
            )

    @pytest.mark.parametrize(
        "model_fixture,task",
        [
            ("resnet_model", "image-classification"),
            ("vit_model", "image-classification"),
            ("clip_vision_model", "feature-extraction"),
            ("clip_text_model", "feature-extraction"),
            ("bert_model", "feature-extraction"),
            ("segformer_model", "image-segmentation"),
        ],
        ids=["resnet", "vit", "clip-vision", "clip-text", "bert", "segformer"],
    )
    def test_generate_dummy_inputs_can_forward(self, model_fixture, task, request):
        """Generated inputs can be passed to model.forward() without error."""
        model = request.getfixturevalue(model_fixture)
        model.eval()

        inputs = generate_dummy_inputs(model.config.model_type, task, model.config)

        # Should not raise
        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs is not None


class TestErrorHandling:
    """Tests for error handling in io module."""

    def test_get_onnx_config_invalid_task_raises(self, resnet_model):
        """_get_onnx_config raises ValueError for invalid task."""
        with pytest.raises(ValueError, match="doesn't support task"):
            _get_onnx_config(
                resnet_model.config.model_type, "invalid-task-name", resnet_model.config
            )

    def test_generate_dummy_inputs_invalid_task_raises(self, resnet_model):
        """generate_dummy_inputs raises ValueError for invalid task."""
        with pytest.raises(ValueError):
            generate_dummy_inputs(
                resnet_model.config.model_type, "invalid-task-name", resnet_model.config
            )


class TestTaskMapping:
    """Tests for task mapping in io module.

    Some tasks are not registered in Optimum's TasksManager but have equivalent
    I/O signatures to other tasks. These tests verify that task mapping works.
    """

    def test_next_sentence_prediction_maps_to_text_classification(self, bert_model):
        """next-sentence-prediction should map to text-classification for BERT.

        BertForNextSentencePrediction has the same I/O as text-classification:
        - Inputs: input_ids, attention_mask, token_type_ids
        - Outputs: logits
        """
        # This should NOT raise - task mapping should handle it
        io_config = _get_onnx_config(
            bert_model.config.model_type, "next-sentence-prediction", bert_model.config
        )

        assert io_config is not None
        # Should have text inputs
        assert "input_ids" in io_config.inputs
        assert "attention_mask" in io_config.inputs

    def test_next_sentence_prediction_generates_inputs(self, bert_model):
        """next-sentence-prediction should generate valid dummy inputs."""
        inputs = generate_dummy_inputs(
            bert_model.config.model_type, "next-sentence-prediction", bert_model.config
        )

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert isinstance(inputs["input_ids"], torch.Tensor)

    def test_image_feature_extraction_maps_to_feature_extraction(self, vit_model):
        """image-feature-extraction should map to feature-extraction for vision models.

        This is an Optimum synonym but we test it explicitly.
        """
        io_config = _get_onnx_config(
            vit_model.config.model_type, "image-feature-extraction", vit_model.config
        )

        assert io_config is not None
        assert "pixel_values" in io_config.inputs


# =============================================================================
# TestShapeKwargs - Tests for **shape_kwargs passthrough
# =============================================================================


class TestShapeKwargs:
    """Tests for **shape_kwargs passthrough.

    Covers generate_dummy_inputs and resolve_io_specs.
    """

    def test_generate_dummy_inputs_custom_sequence_length(self) -> None:
        """Pass sequence_length=128 for BERT, verify shape[1]==128."""
        from transformers import BertConfig

        hf_config = BertConfig(
            vocab_size=TEXT_VOCAB_SIZE,
            hidden_size=TEXT_HIDDEN_SIZE,
            num_hidden_layers=TEXT_NUM_HIDDEN_LAYERS,
            num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
            intermediate_size=TEXT_INTERMEDIATE_SIZE,
            max_position_embeddings=TEXT_MAX_POSITION_EMBEDDINGS,
        )

        inputs = generate_dummy_inputs("bert", "fill-mask", hf_config, sequence_length=128)

        assert "input_ids" in inputs
        assert inputs["input_ids"].shape[1] == 128, (
            f"Expected sequence_length=128, got shape {inputs['input_ids'].shape}"
        )
        # attention_mask should also have the same sequence dimension
        assert inputs["attention_mask"].shape[1] == 128

    def test_get_io_specs_custom_sequence_length(self) -> None:
        """Pass sequence_length=128, verify input_shapes[0][1]==128."""
        from transformers import BertConfig

        hf_config = BertConfig(
            vocab_size=TEXT_VOCAB_SIZE,
            hidden_size=TEXT_HIDDEN_SIZE,
            num_hidden_layers=TEXT_NUM_HIDDEN_LAYERS,
            num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
            intermediate_size=TEXT_INTERMEDIATE_SIZE,
            max_position_embeddings=TEXT_MAX_POSITION_EMBEDDINGS,
        )

        specs = resolve_io_specs("bert", "fill-mask", hf_config, sequence_length=128)

        assert "input_shapes" in specs
        assert len(specs["input_shapes"]) > 0
        # First input (input_ids) should have sequence_length=128
        assert specs["input_shapes"][0][1] == 128, (
            f"Expected input_shapes[0][1]==128, got {specs['input_shapes'][0]}"
        )

    def test_generate_dummy_inputs_custom_height_width(self) -> None:
        """Pass height=128, width=128 for ResNet (no model_id), verify shape."""
        resnet_config = ResNetConfig(
            num_channels=VISION_NUM_CHANNELS,
            hidden_sizes=[32, 64],
            depths=[1, 1],
            layer_type="basic",
            num_labels=10,
        )

        inputs = generate_dummy_inputs(
            "resnet",
            "image-classification",
            resnet_config,
            height=128,
            width=128,
        )

        assert "pixel_values" in inputs
        pixel_values = inputs["pixel_values"]
        # Shape should be [batch, channels, height, width]
        assert pixel_values.shape[2] == 128, f"Expected height=128, got shape {pixel_values.shape}"
        assert pixel_values.shape[3] == 128, f"Expected width=128, got shape {pixel_values.shape}"

    def test_generate_dummy_inputs_config_only_no_model(self) -> None:
        """Create BertConfig directly (no model), call generate_dummy_inputs.

        Verifies that generate_dummy_inputs works with just hf_config
        and never requires instantiating a model.
        """
        from transformers import BertConfig

        hf_config = BertConfig(
            vocab_size=TEXT_VOCAB_SIZE,
            hidden_size=TEXT_HIDDEN_SIZE,
            num_hidden_layers=TEXT_NUM_HIDDEN_LAYERS,
            num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
            intermediate_size=TEXT_INTERMEDIATE_SIZE,
            max_position_embeddings=TEXT_MAX_POSITION_EMBEDDINGS,
        )

        # Should work without ever creating a model instance
        inputs = generate_dummy_inputs("bert", "fill-mask", hf_config)

        assert isinstance(inputs, dict)
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert isinstance(inputs["input_ids"], torch.Tensor)
        # Default batch_size=1
        assert inputs["input_ids"].shape[0] == 1


# =============================================================================
# TestPopulateImageSize - Tests for _populate_image_size_from_preprocessor
# =============================================================================


class TestPopulateImageSize:
    """Tests for _populate_image_size_from_preprocessor with all size formats."""

    def test_int_format(self) -> None:
        """Int size format (e.g., 224) populates both height and width."""
        mock_config = {"size": 224}
        shape_kwargs: dict = {}

        with patch(
            "transformers.image_processing_utils.ImageProcessingMixin.get_image_processor_dict",
            return_value=(mock_config, {}),
        ):
            _populate_image_size_from_preprocessor("some-model/id", shape_kwargs)

        assert shape_kwargs["height"] == 224
        assert shape_kwargs["width"] == 224

    def test_dict_height_width_format(self) -> None:
        """Dict with height/width keys populates both values."""
        mock_config = {"size": {"height": 384, "width": 384}}
        shape_kwargs: dict = {}

        with patch(
            "transformers.image_processing_utils.ImageProcessingMixin.get_image_processor_dict",
            return_value=(mock_config, {}),
        ):
            _populate_image_size_from_preprocessor("some-model/id", shape_kwargs)

        assert shape_kwargs["height"] == 384
        assert shape_kwargs["width"] == 384

    def test_dict_shortest_edge_format(self) -> None:
        """Dict with shortest_edge key populates both height and width."""
        mock_config = {"size": {"shortest_edge": 256}}
        shape_kwargs: dict = {}

        with patch(
            "transformers.image_processing_utils.ImageProcessingMixin.get_image_processor_dict",
            return_value=(mock_config, {}),
        ):
            _populate_image_size_from_preprocessor("some-model/id", shape_kwargs)

        assert shape_kwargs["height"] == 256
        assert shape_kwargs["width"] == 256

    def test_model_id_none_skips(self) -> None:
        """model_id=None returns early, shape_kwargs unchanged."""
        shape_kwargs: dict = {}

        _populate_image_size_from_preprocessor(None, shape_kwargs)

        assert "height" not in shape_kwargs
        assert "width" not in shape_kwargs

    def test_existing_height_not_overridden(self) -> None:
        """Existing height in shape_kwargs is not overridden."""
        mock_config = {"size": 224}
        shape_kwargs = {"height": 512}

        with patch(
            "transformers.image_processing_utils.ImageProcessingMixin.get_image_processor_dict",
            return_value=(mock_config, {}),
        ):
            _populate_image_size_from_preprocessor("some-model/id", shape_kwargs)

        # height should remain 512, not be overwritten to 224
        assert shape_kwargs["height"] == 512
        # width should not be added since early return triggered
        assert "width" not in shape_kwargs

    def test_oserror_handled_gracefully(self) -> None:
        """OSError from get_image_processor_dict does not crash."""
        shape_kwargs: dict = {}

        with patch(
            "transformers.image_processing_utils.ImageProcessingMixin.get_image_processor_dict",
            side_effect=OSError("Not found"),
        ):
            # Should not raise
            _populate_image_size_from_preprocessor("nonexistent/model", shape_kwargs)

        assert "height" not in shape_kwargs
        assert "width" not in shape_kwargs

    def test_no_size_key_in_config(self) -> None:
        """Config dict without 'size' key leaves shape_kwargs unchanged."""
        mock_config: dict = {}  # No "size" key
        shape_kwargs: dict = {}

        with patch(
            "transformers.image_processing_utils.ImageProcessingMixin.get_image_processor_dict",
            return_value=(mock_config, {}),
        ):
            _populate_image_size_from_preprocessor("some-model/id", shape_kwargs)

        assert "height" not in shape_kwargs
        assert "width" not in shape_kwargs

    def test_partial_preprocessor_without_size_falls_back_to_synthesis(self) -> None:
        """A partial preprocessor_config.json (no ``size``) synthesizes from hf_config.

        Without the fall-through, a hub dict carrying only mean/std would leave
        ``size`` unresolved and Optimum would default to 64x64.
        """
        mock_config = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}  # no "size"
        hf_config = SimpleNamespace(pretrained_cfg={"input_size": [3, 224, 224]})
        shape_kwargs: dict = {}

        with patch(
            "transformers.image_processing_utils.ImageProcessingMixin.get_image_processor_dict",
            return_value=(mock_config, {}),
        ):
            _populate_image_size_from_preprocessor(
                "timm/some-model",
                shape_kwargs,
                hf_config,
            )

        assert shape_kwargs["height"] == 224
        assert shape_kwargs["width"] == 224

    def test_nested_dict_input_size_chw(self) -> None:
        """``pretrained_cfg.input_size = [C, H, W]`` (timm) synthesizes a size dict."""
        hf_config = SimpleNamespace(
            pretrained_cfg={"input_size": [3, 224, 224], "mean": [0.485, 0.456, 0.406]},
        )
        shape_kwargs: dict = {}

        # No preprocessor_config.json on the hub -> synthesize from hf_config.
        with patch(
            "transformers.image_processing_utils.ImageProcessingMixin.get_image_processor_dict",
            side_effect=OSError("404"),
        ):
            _populate_image_size_from_preprocessor(
                "timm/some-model",
                shape_kwargs,
                hf_config,
            )

        assert shape_kwargs["height"] == 224
        assert shape_kwargs["width"] == 224

    def test_preprocessor_takes_precedence_over_nested_dict(self) -> None:
        """When preprocessor_config.json resolves, nested dict is not consulted."""
        hf_config = SimpleNamespace(pretrained_cfg={"input_size": [3, 320, 320]})
        shape_kwargs: dict = {}

        with patch(
            "transformers.image_processing_utils.ImageProcessingMixin.get_image_processor_dict",
            return_value=({"size": 384}, {}),
        ):
            _populate_image_size_from_preprocessor(
                "some-model/id",
                shape_kwargs,
                hf_config,
            )

        assert shape_kwargs["height"] == 384
        assert shape_kwargs["width"] == 384

    def test_nested_dict_input_size_scalar(self) -> None:
        """``pretrained_cfg.input_size = [side]`` (length-1) maps to a square size."""
        hf_config = SimpleNamespace(pretrained_cfg={"input_size": [320]})
        shape_kwargs: dict = {}

        with patch(
            "transformers.image_processing_utils.ImageProcessingMixin.get_image_processor_dict",
            side_effect=OSError("404"),
        ):
            _populate_image_size_from_preprocessor(
                "some/model",
                shape_kwargs,
                hf_config,
            )

        assert shape_kwargs["height"] == 320
        assert shape_kwargs["width"] == 320

    def test_pretrained_cfg_without_input_size_ignored(self) -> None:
        """``pretrained_cfg`` without ``input_size`` (e.g. only mean/std) is skipped."""
        hf_config = SimpleNamespace(
            pretrained_cfg={"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        )
        shape_kwargs: dict = {}

        with patch(
            "transformers.image_processing_utils.ImageProcessingMixin.get_image_processor_dict",
            side_effect=OSError("404"),
        ):
            _populate_image_size_from_preprocessor(
                "some/model",
                shape_kwargs,
                hf_config,
            )

        assert shape_kwargs == {}

    def test_existing_height_blocks_nested_dict_too(self) -> None:
        """If height/width already set, nested-dict path must also be skipped."""
        hf_config = SimpleNamespace(pretrained_cfg={"input_size": [3, 224, 224]})
        shape_kwargs = {"height": 128}

        with patch(
            "transformers.image_processing_utils.ImageProcessingMixin.get_image_processor_dict",
            side_effect=OSError("404"),
        ):
            _populate_image_size_from_preprocessor(
                "some/model",
                shape_kwargs,
                hf_config,
            )

        assert shape_kwargs == {"height": 128}


# =============================================================================
# PastKeyValueInputGenerator — shared KV cache dummy input generation
# =============================================================================


def _make_normalized_config(
    num_layers: int = 4,
    num_attention_heads: int = 2,
    head_dim: int = 32,
    max_cache_len: int = 16,
) -> SimpleNamespace:
    """Create a lightweight object that quacks like NormalizedConfig."""
    return SimpleNamespace(
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        max_cache_len=max_cache_len,
    )


@pytest.fixture(scope="module")
def t5_config():
    """Synthetic T5Config — small dims, no network.

    ``n_positions`` maps to ``max_cache_len`` (decoder static buffer size) via
    the T5 NormalizedConfig, so it fixes the KV cache length at 32.
    """
    from transformers import T5Config

    return T5Config(
        d_model=32,
        num_layers=2,
        num_heads=2,
        d_kv=16,
        vocab_size=100,
        n_positions=32,
    )


@pytest.fixture(scope="module")
def qwen_config():
    """Synthetic Qwen3Config — small dims, no network.

    ``max_position_embeddings`` maps to ``max_cache_len`` via the Qwen
    NormalizedConfig, so it fixes the KV cache length at 256.
    """
    from transformers import Qwen3Config

    return Qwen3Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=100,
        intermediate_size=64,
        max_position_embeddings=256,
    )


class TestPastKeyValueInputGenerator:
    """Direct tests for PastKeyValueInputGenerator."""

    def test_supported_input_names(self) -> None:
        nc = _make_normalized_config(num_layers=3)
        gen = PastKeyValueInputGenerator("text-generation", nc)
        expected = (
            "past_0_key",
            "past_0_value",
            "past_1_key",
            "past_1_value",
            "past_2_key",
            "past_2_value",
        )
        assert expected == gen.SUPPORTED_INPUT_NAMES

    def test_generate_key_shape(self) -> None:
        nc = _make_normalized_config(
            num_layers=2,
            num_attention_heads=4,
            head_dim=16,
            max_cache_len=64,
        )
        gen = PastKeyValueInputGenerator("text-generation", nc, batch_size=2)
        tensor = gen.generate("past_0_key")
        assert tensor.shape == (2, 4, 64, 16)

    def test_generate_value_shape(self) -> None:
        nc = _make_normalized_config(
            num_layers=2,
            num_attention_heads=4,
            head_dim=16,
            max_cache_len=64,
        )
        gen = PastKeyValueInputGenerator("text-generation", nc, batch_size=1)
        tensor = gen.generate("past_1_value")
        assert tensor.shape == (1, 4, 64, 16)

    def test_generate_returns_float_tensor(self) -> None:
        nc = _make_normalized_config()
        gen = PastKeyValueInputGenerator("text-generation", nc)
        tensor = gen.generate("past_0_key")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32

    def test_single_layer(self) -> None:
        nc = _make_normalized_config(num_layers=1)
        gen = PastKeyValueInputGenerator("text-generation", nc)
        assert gen.SUPPORTED_INPUT_NAMES == ("past_0_key", "past_0_value")

    def test_batch_size_propagated(self) -> None:
        nc = _make_normalized_config()
        gen = PastKeyValueInputGenerator("text-generation", nc, batch_size=8)
        assert gen.batch_size == 8
        tensor = gen.generate("past_0_key")
        assert tensor.shape[0] == 8


class TestT5DecoderKVInputs:
    """T5 decoder dummy inputs use PastKeyValueInputGenerator."""

    def test_kv_input_names(self, t5_config) -> None:
        inputs = generate_dummy_inputs("t5", "text2text-generation", t5_config)
        num_layers = t5_config.num_layers  # 2 (synthetic)
        for i in range(num_layers):
            assert f"past_{i}_key" in inputs
            assert f"past_{i}_value" in inputs

    def test_kv_shape(self, t5_config) -> None:
        inputs = generate_dummy_inputs("t5", "text2text-generation", t5_config)
        kv = inputs["past_0_key"]
        # [batch=1, heads=num_heads, max_cache_len=32 (n_positions), d_kv]
        assert kv.shape == (1, t5_config.num_heads, 32, t5_config.d_kv)

    def test_decoder_attention_mask_matches_cache_len(self, t5_config) -> None:
        inputs = generate_dummy_inputs("t5", "text2text-generation", t5_config)
        assert inputs["decoder_attention_mask"].shape[1] == 32

    def test_all_kv_layers_present(self, t5_config) -> None:
        inputs = generate_dummy_inputs("t5", "text2text-generation", t5_config)
        kv_names = [n for n in inputs if n.startswith("past_")]
        assert len(kv_names) == t5_config.num_layers * 2


class TestQwenPrefillKVInputs:
    """Qwen3 prefill dummy inputs use PastKeyValueInputGenerator."""

    def test_kv_input_names(self, qwen_config) -> None:
        inputs = generate_dummy_inputs("qwen3", "feature-extraction", qwen_config)
        num_layers = qwen_config.num_hidden_layers  # 2 (synthetic)
        for i in range(num_layers):
            assert f"past_{i}_key" in inputs
            assert f"past_{i}_value" in inputs

    def test_kv_shape(self, qwen_config) -> None:
        inputs = generate_dummy_inputs("qwen3", "feature-extraction", qwen_config)
        kv = inputs["past_0_key"]
        # [batch=1, kv_heads, max_cache_len=256 (max_position_embeddings), head_dim]
        assert kv.shape == (1, qwen_config.num_key_value_heads, 256, qwen_config.head_dim)

    def test_attention_mask_matches_cache_len(self, qwen_config) -> None:
        inputs = generate_dummy_inputs("qwen3", "feature-extraction", qwen_config)
        assert inputs["attention_mask"].shape[1] == 256


class TestQwenGenKVInputs:
    """Qwen3 generation dummy inputs use PastKeyValueInputGenerator."""

    def test_kv_shape_matches_prefill(self, qwen_config) -> None:
        inputs = generate_dummy_inputs("qwen3", "text2text-generation", qwen_config)
        kv = inputs["past_0_key"]
        assert kv.shape == (1, qwen_config.num_key_value_heads, 256, qwen_config.head_dim)

    def test_input_ids_single_token(self, qwen_config) -> None:
        inputs = generate_dummy_inputs("qwen3", "text2text-generation", qwen_config)
        assert inputs["input_ids"].shape == (1, 1)


# =============================================================================
# WinMLCache — build_decoder_mask and prepare_prefill_chunk
# =============================================================================


def _make_cache(cls, num_layers=2, num_heads=2, max_cache_len=16, head_dim=8):
    """Create a WinMLCache instance with minimal config.

    Uses a real PretrainedConfig subclass because HF StaticCache.__init__
    calls config.get_text_config().
    """
    from transformers import PretrainedConfig

    config = PretrainedConfig(num_hidden_layers=num_layers)
    cache = cls.create(config, [1, num_heads, max_cache_len, head_dim], torch.float32)
    cache.reset()
    return cache


class TestStaticCacheBuildDecoderMask:
    """WinMLStaticCache.build_decoder_mask — left-aligned mask."""

    def test_default_single_token(self) -> None:
        from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

        cache = _make_cache(WinMLStaticCache)
        cache.step = 3
        mask = cache.build_decoder_mask(16)
        assert mask.shape == (1, 16)
        assert mask[0, :4].tolist() == [1, 1, 1, 1]
        assert mask[0, 4:].sum().item() == 0

    def test_num_new_tokens(self) -> None:
        from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

        cache = _make_cache(WinMLStaticCache)
        cache.step = 2
        mask = cache.build_decoder_mask(16, num_new_tokens=4)
        assert mask[0, :6].tolist() == [1, 1, 1, 1, 1, 1]
        assert mask[0, 6:].sum().item() == 0


class TestSlidingWindowCacheBuildDecoderMask:
    """WinMLSlidingWindowCache.build_decoder_mask — right-aligned mask."""

    def test_default_single_token(self) -> None:
        from winml.modelkit.models.winml.kv_cache import WinMLSlidingWindowCache

        cache = _make_cache(WinMLSlidingWindowCache)
        cache.step = 3
        mask = cache.build_decoder_mask(16)
        # rightmost 4 positions should be 1
        assert mask[0, -4:].tolist() == [1, 1, 1, 1]
        assert mask[0, :-4].sum().item() == 0

    def test_num_new_tokens(self) -> None:
        from winml.modelkit.models.winml.kv_cache import WinMLSlidingWindowCache

        cache = _make_cache(WinMLSlidingWindowCache)
        cache.step = 2
        mask = cache.build_decoder_mask(16, num_new_tokens=4)
        # rightmost 6 positions
        assert mask[0, -6:].tolist() == [1, 1, 1, 1, 1, 1]
        assert mask[0, :-6].sum().item() == 0

    def test_saturates_at_max_len(self) -> None:
        from winml.modelkit.models.winml.kv_cache import WinMLSlidingWindowCache

        cache = _make_cache(WinMLSlidingWindowCache, max_cache_len=8)
        cache.step = 10
        mask = cache.build_decoder_mask(8, num_new_tokens=4)
        # min(10+4, 8)=8 → all 1s
        assert mask[0].sum().item() == 8


class TestStaticCachePreparePrefillChunk:
    """WinMLStaticCache.prepare_prefill_chunk — right-pad."""

    def test_full_chunk_no_padding(self) -> None:
        from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

        cache = _make_cache(WinMLStaticCache)
        chunk = torch.tensor([[10, 20, 30, 40]])
        padded_ids, pos_ids, pad_len = cache.prepare_prefill_chunk(
            chunk, start=0, prefill_seq_len=4
        )
        assert pad_len == 0
        assert padded_ids[0].tolist() == [10, 20, 30, 40]
        assert pos_ids[0].tolist() == [0, 1, 2, 3]

    def test_partial_chunk_right_padded(self) -> None:
        from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

        cache = _make_cache(WinMLStaticCache)
        chunk = torch.tensor([[10, 20]])
        padded_ids, pos_ids, pad_len = cache.prepare_prefill_chunk(
            chunk, start=4, prefill_seq_len=4
        )
        assert pad_len == 0
        assert padded_ids[0, :2].tolist() == [10, 20]
        assert padded_ids[0, 2:].tolist() == [0, 0]
        assert pos_ids[0].tolist() == [4, 5, 6, 7]


class TestSlidingWindowCachePreparePrefillChunk:
    """WinMLSlidingWindowCache.prepare_prefill_chunk — left-pad."""

    def test_full_chunk_no_padding(self) -> None:
        from winml.modelkit.models.winml.kv_cache import WinMLSlidingWindowCache

        cache = _make_cache(WinMLSlidingWindowCache)
        chunk = torch.tensor([[10, 20, 30, 40]])
        padded_ids, pos_ids, pad_len = cache.prepare_prefill_chunk(
            chunk, start=0, prefill_seq_len=4
        )
        assert pad_len == 0
        assert padded_ids[0].tolist() == [10, 20, 30, 40]
        assert pos_ids[0].tolist() == [0, 1, 2, 3]

    def test_partial_chunk_left_padded(self) -> None:
        from winml.modelkit.models.winml.kv_cache import WinMLSlidingWindowCache

        cache = _make_cache(WinMLSlidingWindowCache)
        chunk = torch.tensor([[10, 20]])
        padded_ids, pos_ids, pad_len = cache.prepare_prefill_chunk(
            chunk, start=4, prefill_seq_len=4
        )
        assert pad_len == 2
        assert padded_ids[0].tolist() == [0, 0, 10, 20]
        assert pos_ids[0].tolist() == [0, 0, 4, 5]


# =============================================================================
# WinMLStaticCache.update — multi-dim index_put_ writes correct positions
# =============================================================================


class TestWinMLStaticCacheUpdate:
    """WinMLStaticCache.update writes new KV at the requested cache_position.

    The PR replaced ``index_copy_`` with multi-dim ``index_put_`` so the ONNX
    exporter emits ScatterND instead of ScatterElements.  These tests verify
    the runtime semantics: the buffer slot at ``cache_position`` matches the
    new KV, untouched slots stay zero, and ``captured`` is populated for the
    ONNX present-output path.
    """

    def test_single_token_writes_at_cache_position(self) -> None:
        from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

        cache = _make_cache(
            WinMLStaticCache, num_layers=2, num_heads=2, max_cache_len=16, head_dim=4
        )
        key_states = torch.randn(1, 2, 1, 4)
        value_states = torch.randn(1, 2, 1, 4)
        cache_position = torch.tensor([5], dtype=torch.int64)

        cache.update(
            key_states, value_states, layer_idx=0, cache_kwargs={"cache_position": cache_position}
        )

        # Slot 5 holds the new KV for every (batch, head)
        assert torch.allclose(cache.layers[0].keys[0, 0, 5], key_states[0, 0, 0])
        assert torch.allclose(cache.layers[0].keys[0, 1, 5], key_states[0, 1, 0])
        assert torch.allclose(cache.layers[0].values[0, 0, 5], value_states[0, 0, 0])

    def test_other_positions_remain_zero(self) -> None:
        from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

        cache = _make_cache(WinMLStaticCache, num_heads=2, max_cache_len=16, head_dim=4)
        key_states = torch.randn(1, 2, 1, 4)
        value_states = torch.randn(1, 2, 1, 4)
        cache_position = torch.tensor([5], dtype=torch.int64)

        cache.update(
            key_states, value_states, layer_idx=0, cache_kwargs={"cache_position": cache_position}
        )

        # Every slot != 5 must still be zero (cache was reset at construction)
        zero_slots = [i for i in range(16) if i != 5]
        for s in zero_slots:
            assert torch.all(cache.layers[0].keys[0, :, s] == 0)
            assert torch.all(cache.layers[0].values[0, :, s] == 0)

    def test_multi_token_writes_at_each_position(self) -> None:
        """Multi-token write (e.g. prefill chunk) lands at every listed position."""
        from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

        cache = _make_cache(WinMLStaticCache, num_heads=2, max_cache_len=16, head_dim=4)
        key_states = torch.randn(1, 2, 3, 4)  # 3 new tokens
        value_states = torch.randn(1, 2, 3, 4)
        cache_position = torch.tensor([3, 4, 5], dtype=torch.int64)

        cache.update(
            key_states, value_states, layer_idx=1, cache_kwargs={"cache_position": cache_position}
        )

        for i, pos in enumerate([3, 4, 5]):
            assert torch.allclose(cache.layers[1].keys[0, 0, pos], key_states[0, 0, i])
            assert torch.allclose(cache.layers[1].keys[0, 1, pos], key_states[0, 1, i])
            assert torch.allclose(cache.layers[1].values[0, 0, pos], value_states[0, 0, i])

    def test_captured_kv_for_onnx_present_output(self) -> None:
        """``captured[layer_idx]`` holds the new-token KV (export reads this)."""
        from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

        cache = _make_cache(WinMLStaticCache, num_heads=2, max_cache_len=16, head_dim=4)
        key_states = torch.randn(1, 2, 1, 4)
        value_states = torch.randn(1, 2, 1, 4)
        cache.update(
            key_states,
            value_states,
            layer_idx=0,
            cache_kwargs={"cache_position": torch.tensor([0], dtype=torch.int64)},
        )

        captured_k, captured_v = cache.captured[0]
        assert captured_k is key_states
        assert captured_v is value_states


# =============================================================================
# WinMLCache.num_layers — asymmetric encoder-decoder fix
# =============================================================================


class TestWinMLCacheAsymmetricNumLayers:
    """Distilbart-style fix: cache must use decoder_layers, not encoder_layers.

    For ``BartConfig(encoder_layers=12, decoder_layers=6, ...)`` the outer
    ``config.num_hidden_layers`` is 12 (encoder count), but HF's
    ``StaticCache.__init__`` builds 6 layer buffers via
    ``config.get_text_config(decoder=True).num_hidden_layers``.  Reading the
    outer attribute caused ``WinMLCache.reset()`` to walk past the end of
    ``self.layers`` and raise ``IndexError`` for distilbart-cnn-12-6.
    """

    @staticmethod
    def _make_asymmetric_bart_config():
        from transformers import BartConfig

        return BartConfig(
            d_model=32,
            decoder_layers=6,
            decoder_attention_heads=2,
            encoder_layers=12,  # asymmetric
            encoder_attention_heads=2,
            vocab_size=100,
            max_position_embeddings=16,
        )

    def test_num_layers_uses_decoder_count(self) -> None:
        from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

        config = self._make_asymmetric_bart_config()
        # Sanity: outer config still reports the encoder count (12).
        assert config.num_hidden_layers == 12

        cache = WinMLStaticCache.create(config, [1, 2, 16, 16], torch.float32)

        # Cache must follow the decoder count (6), matching HF StaticCache.layers.
        assert cache.num_layers == 6
        assert len(cache.layers) == 6

    def test_reset_does_not_index_error(self) -> None:
        from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

        config = self._make_asymmetric_bart_config()
        cache = WinMLStaticCache.create(config, [1, 2, 16, 16], torch.float32)

        # Pre-fix this raised IndexError because reset iterated 0..11 over a 6-layer list.
        cache.reset()
        assert cache.step == 0

    def test_symmetric_unchanged(self) -> None:
        """Symmetric encoder-decoder still gets the right layer count (regression guard)."""
        from transformers import BartConfig

        from winml.modelkit.models.winml.kv_cache import WinMLStaticCache

        config = BartConfig(
            d_model=32,
            decoder_layers=4,
            decoder_attention_heads=2,
            encoder_layers=4,
            encoder_attention_heads=2,
            vocab_size=100,
            max_position_embeddings=16,
        )
        cache = WinMLStaticCache.create(config, [1, 2, 16, 16], torch.float32)
        assert cache.num_layers == 4
        assert len(cache.layers) == 4


# =============================================================================
# Marian / BART decoder dummy inputs use PastKeyValueInputGenerator
# =============================================================================


@pytest.fixture(scope="module")
def marian_config():
    """Synthetic MarianConfig — small dims, no network."""
    from transformers import MarianConfig

    return MarianConfig(
        d_model=32,
        decoder_layers=2,
        decoder_attention_heads=2,
        encoder_layers=2,
        encoder_attention_heads=2,
        vocab_size=100,
        max_position_embeddings=16,
    )


@pytest.fixture(scope="module")
def bart_config_symmetric():
    """Synthetic symmetric BartConfig."""
    from transformers import BartConfig

    return BartConfig(
        d_model=32,
        decoder_layers=4,
        decoder_attention_heads=2,
        encoder_layers=4,
        encoder_attention_heads=2,
        vocab_size=100,
        max_position_embeddings=16,
    )


@pytest.fixture(scope="module")
def bart_config_asymmetric():
    """Synthetic asymmetric BartConfig — distilbart-cnn-12-6 shape (encoder>decoder)."""
    from transformers import BartConfig

    return BartConfig(
        d_model=32,
        decoder_layers=6,
        decoder_attention_heads=2,
        encoder_layers=12,
        encoder_attention_heads=2,
        vocab_size=100,
        max_position_embeddings=24,
    )


class TestMarianDecoderKVInputs:
    """Marian decoder dummy inputs use PastKeyValueInputGenerator."""

    def test_kv_input_names(self, marian_config) -> None:
        inputs = generate_dummy_inputs("marian", "text2text-generation", marian_config)
        for i in range(marian_config.decoder_layers):
            assert f"past_{i}_key" in inputs
            assert f"past_{i}_value" in inputs

    def test_kv_shape(self, marian_config) -> None:
        inputs = generate_dummy_inputs("marian", "text2text-generation", marian_config)
        kv = inputs["past_0_key"]
        head_dim = marian_config.d_model // marian_config.decoder_attention_heads
        assert kv.shape == (
            1,
            marian_config.decoder_attention_heads,
            marian_config.max_position_embeddings,
            head_dim,
        )

    def test_decoder_attention_mask_matches_cache_len(self, marian_config) -> None:
        inputs = generate_dummy_inputs("marian", "text2text-generation", marian_config)
        assert inputs["decoder_attention_mask"].shape[1] == marian_config.max_position_embeddings

    def test_all_kv_layers_present(self, marian_config) -> None:
        inputs = generate_dummy_inputs("marian", "text2text-generation", marian_config)
        kv_names = [n for n in inputs if n.startswith("past_")]
        assert len(kv_names) == marian_config.decoder_layers * 2


class TestBartDecoderKVInputs:
    """BART decoder dummy inputs use PastKeyValueInputGenerator."""

    def test_kv_input_names(self, bart_config_symmetric) -> None:
        inputs = generate_dummy_inputs("bart", "text2text-generation", bart_config_symmetric)
        for i in range(bart_config_symmetric.decoder_layers):
            assert f"past_{i}_key" in inputs
            assert f"past_{i}_value" in inputs

    def test_kv_shape(self, bart_config_symmetric) -> None:
        inputs = generate_dummy_inputs("bart", "text2text-generation", bart_config_symmetric)
        kv = inputs["past_0_key"]
        head_dim = bart_config_symmetric.d_model // bart_config_symmetric.decoder_attention_heads
        assert kv.shape == (
            1,
            bart_config_symmetric.decoder_attention_heads,
            bart_config_symmetric.max_position_embeddings,
            head_dim,
        )

    def test_decoder_attention_mask_matches_cache_len(self, bart_config_symmetric) -> None:
        inputs = generate_dummy_inputs("bart", "text2text-generation", bart_config_symmetric)
        assert (
            inputs["decoder_attention_mask"].shape[1]
            == bart_config_symmetric.max_position_embeddings
        )

    def test_asymmetric_uses_decoder_layers_not_encoder(self, bart_config_asymmetric) -> None:
        """Distilbart-style asymmetric config — KV layer count must follow ``decoder_layers``.

        Pre-fix, ``_BartDecoderNormalizedConfig.num_layers`` (then via
        ``NormalizedConfig.with_args(num_layers="decoder_layers")``) was already
        correct here.  This test pins the contract so any future refactor of
        the NormalizedConfig keeps reading ``decoder_layers`` (not the outer
        ``num_hidden_layers``, which on BART is the encoder count).
        """
        inputs = generate_dummy_inputs("bart", "text2text-generation", bart_config_asymmetric)
        kv_names = [n for n in inputs if n.startswith("past_")]
        # decoder_layers=6 → 12 KV tensors, NOT encoder_layers=12 → 24
        assert len(kv_names) == bart_config_asymmetric.decoder_layers * 2
        assert "past_5_key" in inputs
        assert "past_6_key" not in inputs
