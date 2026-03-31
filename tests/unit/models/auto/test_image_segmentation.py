# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Tests for WinMLModelForImageSegmentation and WinMLModelForSemanticSegmentation.

Tests the image segmentation models in modelkit/models/winml/image_segmentation.py
following the design specifications in docs/design/automodel/.

Acceptance Criteria (from design):
- AC-1: Classes exist and inherit from WinMLPreTrainedModel
- AC-2: forward() accepts pixel_values + pixel_mask (ImageSeg) or pixel_values only (SemanticSeg)
- AC-3: ImageSegmentation returns ImageSegmentationOutput with logits, pred_masks, pred_boxes
- AC-4: SemanticSegmentation returns SemanticSegmenterOutput with logits
- AC-5: Two classes are distinct (not aliases)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch


# =============================================================================
# WinMLModelForImageSegmentation (panoptic/instance, DETR-style)
# =============================================================================


def create_mock_model(
    num_queries: int = 100,
    num_classes: int = 150,
    output_h: int = 128,
    output_w: int = 128,
):
    """Create a WinMLModelForImageSegmentation with mocked DETR-style session.

    DETR outputs: logits [B, num_queries, num_classes+1],
                  pred_boxes [B, num_queries, 4],
                  pred_masks [B, num_queries, H, W]
    """
    from winml.modelkit.models import (
        WinMLModelForImageSegmentation,
    )

    model = WinMLModelForImageSegmentation.__new__(WinMLModelForImageSegmentation)
    mock_session = MagicMock()
    mock_session.run.return_value = {
        "logits": np.random.randn(1, num_queries, num_classes + 1).astype(np.float32),
        "pred_boxes": np.random.randn(1, num_queries, 4).astype(np.float32),
        "pred_masks": np.random.randn(1, num_queries, output_h, output_w).astype(np.float32),
    }
    mock_session.io_config = {
        "input_names": ["pixel_values"],
        "output_names": ["logits", "pred_boxes", "pred_masks"],
    }
    model._session = mock_session
    model.config = MagicMock()
    model.config.num_labels = num_classes
    model._onnx_path = "mock.onnx"
    model._device = "cpu"
    return model


class TestWinMLModelForImageSegmentationBasic:
    """Basic functionality tests."""

    def test_class_exists(self):
        """Test that the class exists and is importable."""
        from winml.modelkit.models import (
            WinMLModelForImageSegmentation,
        )

        assert WinMLModelForImageSegmentation is not None

    def test_inherits_from_base(self):
        """Test class inherits from WinMLPreTrainedModel."""
        from winml.modelkit.models import (
            WinMLModelForImageSegmentation,
            WinMLPreTrainedModel,
        )

        assert issubclass(WinMLModelForImageSegmentation, WinMLPreTrainedModel)

    def test_semantic_segmentation_class_separate(self):
        """Test WinMLModelForSemanticSegmentation is a separate class (not an alias).

        In HuggingFace, AutoModelForImageSegmentation (panoptic/DETR) and
        AutoModelForSemanticSegmentation (pixel-level/SegFormer) are distinct
        classes with zero model overlap. WinML should mirror this distinction.
        """
        from winml.modelkit.models import (
            WinMLModelForImageSegmentation,
            WinMLModelForSemanticSegmentation,
        )

        # Must NOT be the same class
        assert WinMLModelForSemanticSegmentation is not WinMLModelForImageSegmentation


class TestForwardMethod:
    """Test forward() method for ImageSegmentation (DETR-style)."""

    def test_forward_accepts_pixel_values(self):
        """AC-2: forward() accepts pixel_values."""
        model = create_mock_model()

        pixel_values = torch.randn(1, 3, 512, 512)
        model.forward(pixel_values=pixel_values)

        model._session.run.assert_called_once()

    def test_forward_accepts_pixel_mask(self):
        """AC-2: forward() accepts optional pixel_mask."""
        model = create_mock_model()
        # Add pixel_mask to io_config input_names
        model._session.io_config["input_names"] = ["pixel_values", "pixel_mask"]

        pixel_values = torch.randn(1, 3, 512, 512)
        pixel_mask = torch.ones((1, 512, 512), dtype=torch.long)

        model.forward(pixel_values=pixel_values, pixel_mask=pixel_mask)

        model._session.run.assert_called()

    def test_forward_returns_image_segmentation_output(self):
        """AC-3: forward() returns ImageSegmentationOutput."""
        from winml.modelkit.models import (
            ImageSegmentationOutput,
        )

        model = create_mock_model()

        pixel_values = torch.randn(1, 3, 512, 512)
        output = model.forward(pixel_values=pixel_values)

        assert isinstance(output, ImageSegmentationOutput)

    def test_forward_has_logits(self):
        """AC-3: Output has logits field."""
        model = create_mock_model()

        pixel_values = torch.randn(1, 3, 512, 512)
        output = model.forward(pixel_values=pixel_values)

        assert output.logits is not None
        assert len(output.logits.shape) == 3  # [B, num_queries, num_classes+1]

    def test_forward_has_pred_masks(self):
        """AC-3: Output has pred_masks for panoptic post-processing."""
        model = create_mock_model()

        pixel_values = torch.randn(1, 3, 512, 512)
        output = model.forward(pixel_values=pixel_values)

        assert output.pred_masks is not None
        assert len(output.pred_masks.shape) == 4  # [B, num_queries, H, W]

    def test_forward_has_pred_boxes(self):
        """AC-3: Output has pred_boxes for detection post-processing."""
        model = create_mock_model()

        pixel_values = torch.randn(1, 3, 512, 512)
        output = model.forward(pixel_values=pixel_values)

        assert output.pred_boxes is not None
        assert output.pred_boxes.shape[-1] == 4  # [B, num_queries, 4]

    def test_forward_loss_is_none(self):
        """forward() does not compute loss (thin wrapper)."""
        model = create_mock_model()

        pixel_values = torch.randn(1, 3, 512, 512)
        output = model.forward(pixel_values=pixel_values)

        assert output.loss is None

    def test_forward_missing_outputs_are_none(self):
        """When ONNX model lacks pred_masks/pred_boxes, those fields are None."""
        from winml.modelkit.models import (
            WinMLModelForImageSegmentation,
        )

        model = WinMLModelForImageSegmentation.__new__(WinMLModelForImageSegmentation)
        mock_session = MagicMock()
        # Only logits output (no pred_masks or pred_boxes)
        mock_session.run.return_value = {
            "logits": np.random.randn(1, 100, 151).astype(np.float32),
        }
        mock_session.io_config = {
            "input_names": ["pixel_values"],
            "output_names": ["logits"],
        }
        model._session = mock_session
        model.config = MagicMock()
        model._onnx_path = "mock.onnx"
        model._device = "cpu"

        pixel_values = torch.randn(1, 3, 512, 512)
        output = model.forward(pixel_values=pixel_values)

        assert output.logits is not None
        assert output.pred_masks is None
        assert output.pred_boxes is None


class TestImageSegmentationOutputType:
    """Test ImageSegmentationOutput is a proper ModelOutput."""

    def test_output_is_model_output(self):
        """ImageSegmentationOutput inherits from ModelOutput."""
        from transformers.utils import ModelOutput

        from winml.modelkit.models import (
            ImageSegmentationOutput,
        )

        assert issubclass(ImageSegmentationOutput, ModelOutput)

    def test_output_supports_dict_access(self):
        """ImageSegmentationOutput supports dict-style access (pipeline compat)."""
        from winml.modelkit.models import (
            ImageSegmentationOutput,
        )

        logits = torch.randn(1, 100, 151)
        pred_masks = torch.randn(1, 100, 128, 128)
        output = ImageSegmentationOutput(logits=logits, pred_masks=pred_masks)

        # Dict-style access (used by pipelines)
        assert output["logits"] is logits
        assert output["pred_masks"] is pred_masks

    def test_output_supports_attribute_access(self):
        """ImageSegmentationOutput supports attribute access (used by post-processors)."""
        from winml.modelkit.models import (
            ImageSegmentationOutput,
        )

        logits = torch.randn(1, 100, 151)
        pred_masks = torch.randn(1, 100, 128, 128)
        pred_boxes = torch.randn(1, 100, 4)
        output = ImageSegmentationOutput(
            logits=logits, pred_masks=pred_masks, pred_boxes=pred_boxes
        )

        # Attribute access (used by image_processor.post_process_*)
        assert output.logits is logits
        assert output.pred_masks is pred_masks
        assert output.pred_boxes is pred_boxes


class TestProperties:
    """Test model properties."""

    def test_num_labels_from_config(self):
        """num_labels property reads from config."""
        model = create_mock_model(num_classes=150)
        model.config.num_labels = 150

        assert model.num_labels == 150

    def test_device_property(self):
        """device property returns current device."""
        model = create_mock_model()

        assert model.device == "cpu"

    def test_dtype_property(self):
        """dtype property returns float32."""
        model = create_mock_model()

        assert model.dtype == torch.float32


class TestSupportedModels:
    """Test supported segmentation model types."""

    def test_registered_in_task_mapping(self):
        """Test image-segmentation is registered in TASK_TO_WINML_CLASS."""
        from winml.modelkit.models import TASK_TO_WINML_CLASS

        assert "image-segmentation" in TASK_TO_WINML_CLASS


# =============================================================================
# WinMLModelForSemanticSegmentation (pixel-level, SegFormer/BEiT/DPT)
# =============================================================================


def create_mock_semantic_model(num_labels: int = 150, output_h: int = 128, output_w: int = 128):
    """Create a WinMLModelForSemanticSegmentation with mocked session.

    Semantic segmentation outputs: logits [B, num_labels, H, W]
    """
    from winml.modelkit.models import (
        WinMLModelForSemanticSegmentation,
    )

    model = WinMLModelForSemanticSegmentation.__new__(WinMLModelForSemanticSegmentation)
    mock_session = MagicMock()
    mock_session.run.return_value = {
        "logits": np.random.randn(1, num_labels, output_h, output_w).astype(np.float32)
    }
    mock_session.io_config = {
        "input_names": ["pixel_values"],
        "output_names": ["logits"],
    }
    model._session = mock_session
    model.config = MagicMock()
    model.config.num_labels = num_labels
    model._onnx_path = "mock.onnx"
    model._device = "cpu"
    return model


class TestWinMLModelForSemanticSegmentationBasic:
    """Basic functionality tests for WinMLModelForSemanticSegmentation."""

    def test_class_exists(self):
        """Test that the class exists and is importable."""
        from winml.modelkit.models import (
            WinMLModelForSemanticSegmentation,
        )

        assert WinMLModelForSemanticSegmentation is not None

    def test_inherits_from_base(self):
        """Test class inherits from WinMLPreTrainedModel."""
        from winml.modelkit.models import (
            WinMLModelForSemanticSegmentation,
            WinMLPreTrainedModel,
        )

        assert issubclass(WinMLModelForSemanticSegmentation, WinMLPreTrainedModel)

    def test_forward_accepts_pixel_values(self):
        """forward() accepts pixel_values and runs inference."""
        model = create_mock_semantic_model()

        pixel_values = torch.randn(1, 3, 512, 512)
        model.forward(pixel_values=pixel_values)

        model._session.run.assert_called_once()

    def test_forward_returns_semantic_segmenter_output(self):
        """AC-4: forward() returns SemanticSegmenterOutput (not ImageSegmentationOutput)."""
        from transformers.modeling_outputs import SemanticSegmenterOutput

        model = create_mock_semantic_model()

        pixel_values = torch.randn(1, 3, 512, 512)
        output = model.forward(pixel_values=pixel_values)

        assert isinstance(output, SemanticSegmenterOutput)
        assert hasattr(output, "logits")
        assert output.logits is not None

    def test_forward_logits_shape(self):
        """Semantic segmentation logits are [B, num_labels, H, W]."""
        model = create_mock_semantic_model(num_labels=150, output_h=128, output_w=128)

        pixel_values = torch.randn(1, 3, 512, 512)
        output = model.forward(pixel_values=pixel_values)

        assert len(output.logits.shape) == 4
        assert output.logits.shape[0] == 1  # batch
        assert output.logits.shape[1] == 150  # num_labels

    def test_forward_loss_is_none(self):
        """forward() does not compute loss (thin wrapper)."""
        model = create_mock_semantic_model()

        pixel_values = torch.randn(1, 3, 512, 512)
        output = model.forward(pixel_values=pixel_values)

        assert output.loss is None

    def test_registered_in_task_mapping(self):
        """semantic-segmentation is registered in TASK_TO_WINML_CLASS."""
        from winml.modelkit.models import TASK_TO_WINML_CLASS

        assert "semantic-segmentation" in TASK_TO_WINML_CLASS
        assert TASK_TO_WINML_CLASS["semantic-segmentation"] == "WinMLModelForSemanticSegmentation"


class TestOutputTypeDistinction:
    """Verify the two classes return different output types."""

    def test_image_seg_returns_image_segmentation_output(self):
        """ImageSegmentation returns ImageSegmentationOutput."""
        from winml.modelkit.models import (
            ImageSegmentationOutput,
        )

        model = create_mock_model()
        output = model.forward(pixel_values=torch.randn(1, 3, 512, 512))
        assert isinstance(output, ImageSegmentationOutput)

    def test_semantic_seg_returns_semantic_segmenter_output(self):
        """SemanticSegmentation returns SemanticSegmenterOutput."""
        from transformers.modeling_outputs import SemanticSegmenterOutput

        model = create_mock_semantic_model()
        output = model.forward(pixel_values=torch.randn(1, 3, 512, 512))
        assert isinstance(output, SemanticSegmenterOutput)

    def test_different_output_types(self):
        """The two classes return different output types."""
        from transformers.modeling_outputs import SemanticSegmenterOutput

        from winml.modelkit.models import (
            ImageSegmentationOutput,
        )

        assert ImageSegmentationOutput is not SemanticSegmenterOutput
