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
