# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLImageSegmentationEvaluator schema validation,
prepare_prediction, prepare_reference, and MeanIoUMetric."""

import numpy as np
import pytest
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

from winml.modelkit.eval.image_segmentation_evaluator import WinMLImageSegmentationEvaluator
from winml.modelkit.eval.metrics.mean_iou import IGNORE_INDEX, MeanIoUMetric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockConfig:
    def __init__(self, label2id=None, id2label=None, num_labels=5):
        self.label2id = label2id or {}
        self.id2label = id2label or {}
        self.num_labels = num_labels


class MockModel:
    def __init__(self, label2id=None, num_labels=5):
        id2label = {v: k for k, v in label2id.items()} if label2id else {}
        self.config = MockConfig(label2id, id2label, num_labels)

    @property
    def io_config(self):
        return {"input_shapes": [[1, 3, 224, 224]]}


def make_evaluator(label2id=None, columns_mapping=None, num_labels=5):
    """Create evaluator without triggering __init__ data loading."""
    ev = object.__new__(WinMLImageSegmentationEvaluator)
    ev.model = MockModel(label2id, num_labels)
    ev._annotation_col = "annotation"
    if columns_mapping:
        ev._annotation_col = columns_mapping.get("annotation_column", "annotation")
    return ev


def make_seg_dataset(images, annotations):
    """Build a dataset with segmentation structure (image + annotation)."""
    features = Features(
        {
            "image": Image(mode="RGB"),
            "annotation": Image(mode="L"),
        }
    )
    return Dataset.from_dict(
        {
            "image": images,
            "annotation": annotations,
        },
        features=features,
    )


def create_dummy_image(width, height, color=(128, 128, 128)):
    """Create a dummy PIL RGB image."""
    return PILImage.new("RGB", (width, height), color)


def create_annotation_image(label_map):
    """Create an annotation PIL Image from a 2D numpy array."""
    return PILImage.fromarray(label_map.astype(np.uint8), mode="L")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestValidateSchema:
    def test_valid_schema_passes(self):
        ev = make_evaluator()
        img = create_dummy_image(4, 3)
        ann = create_annotation_image(np.array([[1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 3, 3]]))
        ds = make_seg_dataset([img], [ann])
        ev._validate_schema(ds)  # should not raise

    def test_missing_image_column_raises(self):
        ev = make_evaluator()
        ds = Dataset.from_dict({"text": ["hello"], "annotation": ["a"]})
        with pytest.raises(ValueError, match="missing 'image' column"):
            ev._validate_schema(ds)

    def test_missing_annotation_column_raises(self):
        ev = make_evaluator()
        features = Features({"image": Image(mode="RGB"), "label": Value("int64")})
        img = create_dummy_image(4, 3)
        ds = Dataset.from_dict({"image": [img], "label": [0]}, features=features)
        with pytest.raises(ValueError, match="missing annotation column"):
            ev._validate_schema(ds)

    def test_custom_annotation_column(self):
        ev = make_evaluator(columns_mapping={"annotation_column": "segmap"})
        img = create_dummy_image(4, 3)
        ann = create_annotation_image(np.zeros((3, 4), dtype=np.uint8))
        features = Features({"image": Image(mode="RGB"), "segmap": Image(mode="L")})
        ds = Dataset.from_dict({"image": [img], "segmap": [ann]}, features=features)
        ev._validate_schema(ds)  # should not raise


# ---------------------------------------------------------------------------
# MeanIoUMetric
# ---------------------------------------------------------------------------


class TestMeanIoUMetric:
    def test_perfect_prediction(self):
        """Identical prediction and reference should give mIoU = 1.0."""
        metric = MeanIoUMetric(num_classes=3, ignore_index=IGNORE_INDEX)
        pred = np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 2, 2]])
        ref = pred.copy()
        metric.update(pred, ref)
        result = metric.compute()
        assert result["mean_iou"] == pytest.approx(1.0)
        assert result["overall_accuracy"] == pytest.approx(1.0)

    def test_imperfect_prediction(self):
        """Partially wrong prediction should give mIoU < 1.0."""
        metric = MeanIoUMetric(num_classes=3, ignore_index=IGNORE_INDEX)
        pred = np.array(
            [
                [0, 0, 0, 0],  # row: all class 0
                [1, 1, 1, 1],  # wrong: should be [1,1,2,2]
                [2, 2, 2, 2],
            ]
        )
        ref = np.array([[0, 0, 0, 0], [1, 1, 2, 2], [2, 2, 2, 2]])
        metric.update(pred, ref)
        result = metric.compute()
        assert 0.0 < result["mean_iou"] < 1.0
        assert 0.0 < result["overall_accuracy"] < 1.0

    def test_pixel_label_mapping_shift(self):
        """With a 1-indexed→0-indexed mapping, metric should work correctly."""
        # Simulate ADE20K: GT is 1-indexed, model is 0-indexed
        # Mapping: {1: 0, 2: 1, 3: 2}. GT pixel 0 is not in mapping → ignored.
        mapping = {1: 0, 2: 1, 3: 2}
        metric = MeanIoUMetric(num_classes=3, ignore_index=IGNORE_INDEX)
        # GT: 1-indexed (0=background, 1=class0, 2=class1, 3=class2)
        ref_raw = np.array([[0, 1, 1, 2], [2, 2, 3, 3], [3, 3, 3, 3]])
        # Apply mapping (same logic as evaluator)
        ref = np.full_like(ref_raw, IGNORE_INDEX, dtype=np.int64)
        for src, dst in mapping.items():
            ref[ref_raw == src] = dst

        # Prediction: 0-indexed
        pred = np.array([[0, 0, 0, 1], [1, 1, 2, 2], [2, 2, 2, 2]])
        metric.update(pred, ref)
        result = metric.compute()
        assert result["mean_iou"] == pytest.approx(1.0)
        assert result["overall_accuracy"] == pytest.approx(1.0)

    def test_ignore_index(self):
        """Pixels with ignore_index in GT should not affect metric."""
        metric = MeanIoUMetric(num_classes=2, ignore_index=IGNORE_INDEX)
        # ignore predictions at ignore pixels
        pred = np.array(
            [
                [0, 0, 1, 1],
                [IGNORE_INDEX, IGNORE_INDEX, 1, 1],
            ]
        )
        ref = np.array([[0, 0, 1, 1], [IGNORE_INDEX, IGNORE_INDEX, 1, 1]])
        metric.update(pred, ref)
        result = metric.compute()
        assert result["mean_iou"] == pytest.approx(1.0)

    def test_incremental_updates(self):
        """Multiple update calls should accumulate correctly."""
        metric = MeanIoUMetric(num_classes=2, ignore_index=IGNORE_INDEX)

        # Image 1: perfect
        pred1 = np.array([[0, 0], [1, 1]])
        ref1 = np.array([[0, 0], [1, 1]])
        metric.update(pred1, ref1)

        # Image 2: perfect
        pred2 = np.array([[1, 1], [0, 0]])
        ref2 = np.array([[1, 1], [0, 0]])
        metric.update(pred2, ref2)

        result = metric.compute()
        assert result["mean_iou"] == pytest.approx(1.0)

    def test_per_category_iou_length(self):
        """per_category_iou should have length = num_classes."""
        metric = MeanIoUMetric(num_classes=5, ignore_index=IGNORE_INDEX)
        pred = np.array([[0, 1], [2, 3]])
        ref = np.array([[0, 1], [2, 3]])
        metric.update(pred, ref)
        result = metric.compute()
        assert len(result["per_category_iou"]) == 5


# ---------------------------------------------------------------------------
# prepare_prediction
# ---------------------------------------------------------------------------


class TestPreparePrediction:
    """Test stacking pipeline binary masks into a single label map."""

    def test_single_class(self):
        mask = PILImage.fromarray(
            np.array([[255, 255], [0, 0]], dtype=np.uint8),
            mode="L",
        )
        result = [{"label": "sky", "mask": mask}]
        label2id = {"sky": 2}
        pred = WinMLImageSegmentationEvaluator.prepare_prediction(
            result,
            label2id,
            (2, 2),
        )
        expected = np.array([[2, 2], [IGNORE_INDEX, IGNORE_INDEX]])
        np.testing.assert_array_equal(pred, expected)

    def test_multiple_classes(self):
        sky_mask = PILImage.fromarray(
            np.array([[255, 255, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
            mode="L",
        )
        building_mask = PILImage.fromarray(
            np.array([[0, 0, 255, 255], [0, 0, 0, 0]], dtype=np.uint8),
            mode="L",
        )
        tree_mask = PILImage.fromarray(
            np.array([[0, 0, 0, 0], [255, 255, 255, 255]], dtype=np.uint8),
            mode="L",
        )
        result = [
            {"label": "sky", "mask": sky_mask},
            {"label": "building", "mask": building_mask},
            {"label": "tree", "mask": tree_mask},
        ]
        label2id = {"sky": 2, "building": 1, "tree": 4}
        pred = WinMLImageSegmentationEvaluator.prepare_prediction(
            result,
            label2id,
            (4, 2),
        )
        expected = np.array([[2, 2, 1, 1], [4, 4, 4, 4]])
        np.testing.assert_array_equal(pred, expected)

    def test_unknown_label_skipped(self):
        mask = PILImage.fromarray(
            np.array([[255, 0], [0, 255]], dtype=np.uint8),
            mode="L",
        )
        result = [{"label": "unknown_class", "mask": mask}]
        label2id = {"sky": 0}
        pred = WinMLImageSegmentationEvaluator.prepare_prediction(
            result,
            label2id,
            (2, 2),
        )
        # Unknown label is skipped, all pixels stay IGNORE_INDEX
        expected = np.full((2, 2), IGNORE_INDEX, dtype=np.int64)
        np.testing.assert_array_equal(pred, expected)

    def test_empty_result(self):
        pred = WinMLImageSegmentationEvaluator.prepare_prediction(
            [],
            {"sky": 0},
            (4, 3),
        )
        assert pred.shape == (3, 4)
        assert np.all(pred == IGNORE_INDEX)

    def test_mask_nonbinary_values(self):
        """Mask with values other than 0/255 (e.g., 128) should still work."""
        mask = PILImage.fromarray(
            np.array([[128, 0], [1, 0]], dtype=np.uint8),
            mode="L",
        )
        result = [{"label": "wall", "mask": mask}]
        label2id = {"wall": 5}
        pred = WinMLImageSegmentationEvaluator.prepare_prediction(
            result,
            label2id,
            (2, 2),
        )
        expected = np.array([[5, IGNORE_INDEX], [5, IGNORE_INDEX]])
        np.testing.assert_array_equal(pred, expected)


# ---------------------------------------------------------------------------
# prepare_reference
# ---------------------------------------------------------------------------


class TestPrepareReference:
    """Test annotation image to label map conversion with remapping."""

    def test_no_mapping(self):
        ann = PILImage.fromarray(
            np.array([[0, 1], [2, 3]], dtype=np.uint8),
            mode="L",
        )
        ref = WinMLImageSegmentationEvaluator.prepare_reference(ann, None)
        expected = np.array([[0, 1], [2, 3]])
        np.testing.assert_array_equal(ref, expected)

    def test_identity_mapping(self):
        ann = PILImage.fromarray(
            np.array([[0, 1], [2, 3]], dtype=np.uint8),
            mode="L",
        )
        mapping = {"0": 0, "1": 1, "2": 2, "3": 3}
        ref = WinMLImageSegmentationEvaluator.prepare_reference(ann, mapping)
        expected = np.array([[0, 1], [2, 3]])
        np.testing.assert_array_equal(ref, expected)

    def test_ade20k_style_mapping(self):
        """1-indexed GT with background=0 → 0-indexed, background ignored."""
        ann = PILImage.fromarray(
            np.array([[0, 1, 2], [3, 0, 1]], dtype=np.uint8),
            mode="L",
        )
        mapping = {"1": 0, "2": 1, "3": 2}
        ref = WinMLImageSegmentationEvaluator.prepare_reference(ann, mapping)
        expected = np.array(
            [
                [IGNORE_INDEX, 0, 1],
                [2, IGNORE_INDEX, 0],
            ]
        )
        np.testing.assert_array_equal(ref, expected)

    def test_cityscapes_style_mapping(self):
        """Sparse label IDs with void classes → mapped + ignored."""
        ann = PILImage.fromarray(
            np.array([[0, 7, 8], [11, 1, 23]], dtype=np.uint8),
            mode="L",
        )
        mapping = {"7": 0, "8": 1, "11": 2, "23": 10}
        ref = WinMLImageSegmentationEvaluator.prepare_reference(ann, mapping)
        expected = np.array(
            [
                [IGNORE_INDEX, 0, 1],
                [2, IGNORE_INDEX, 10],
            ]
        )
        np.testing.assert_array_equal(ref, expected)

    def test_rgb_annotation(self):
        """RGB annotation where R=G=B=label_id → extract first channel."""
        arr = np.zeros((2, 3, 3), dtype=np.uint8)
        arr[:, :, 0] = [[5, 10, 15], [20, 25, 30]]
        arr[:, :, 1] = [[5, 10, 15], [20, 25, 30]]
        arr[:, :, 2] = [[5, 10, 15], [20, 25, 30]]
        ann = PILImage.fromarray(arr, mode="RGB")
        ref = WinMLImageSegmentationEvaluator.prepare_reference(ann, None)
        expected = np.array([[5, 10, 15], [20, 25, 30]])
        np.testing.assert_array_equal(ref, expected)
        assert ref.ndim == 2

    def test_rgb_annotation_with_mapping(self):
        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        arr[:, :, 0] = [[7, 8], [0, 11]]
        arr[:, :, 1] = arr[:, :, 0]
        arr[:, :, 2] = arr[:, :, 0]
        ann = PILImage.fromarray(arr, mode="RGB")
        mapping = {"7": 0, "8": 1, "11": 2}
        ref = WinMLImageSegmentationEvaluator.prepare_reference(ann, mapping)
        expected = np.array([[0, 1], [IGNORE_INDEX, 2]])
        np.testing.assert_array_equal(ref, expected)


# ---------------------------------------------------------------------------
# prepare_pipeline
# ---------------------------------------------------------------------------


class TestPreparePipeline:
    """Test image processor size override from ONNX input shape."""

    def test_size_override_from_onnx_input(self):
        """prepare_pipeline should set image processor size to ONNX input HxW."""
        from unittest.mock import MagicMock, patch

        ev = make_evaluator(label2id={"sky": 0}, num_labels=1)

        mock_pipe = MagicMock()
        mock_pipe.image_processor = MagicMock()
        mock_pipe.image_processor.size = {"height": 512, "width": 512}

        with patch.object(
            WinMLImageSegmentationEvaluator.__bases__[0],
            "prepare_pipeline",
            return_value=mock_pipe,
        ):
            pipe = ev.prepare_pipeline()

        assert pipe.image_processor.size == {"height": 224, "width": 224}

    def test_no_override_without_input_shapes(self):
        """When io_config has no input_shapes, size should remain unchanged."""
        from unittest.mock import MagicMock, patch

        ev = make_evaluator(label2id={"sky": 0}, num_labels=1)
        ev.model = MagicMock()
        ev.model.io_config = {}

        mock_pipe = MagicMock()
        mock_pipe.image_processor = MagicMock()
        mock_pipe.image_processor.size = {"height": 512, "width": 512}

        with patch.object(
            WinMLImageSegmentationEvaluator.__bases__[0],
            "prepare_pipeline",
            return_value=mock_pipe,
        ):
            pipe = ev.prepare_pipeline()

        assert pipe.image_processor.size == {"height": 512, "width": 512}


# ---------------------------------------------------------------------------
# Overlapping masks in prepare_prediction
# ---------------------------------------------------------------------------


class TestPreparePredictionOverlap:
    """Test overlapping mask behavior in prepare_prediction."""

    def test_overlapping_masks_last_write_wins(self):
        """When two masks claim the same pixel, the last one in the list wins."""
        mask_a = PILImage.fromarray(
            np.array([[255, 255], [255, 0]], dtype=np.uint8),
            mode="L",
        )
        mask_b = PILImage.fromarray(
            np.array([[255, 0], [255, 255]], dtype=np.uint8),
            mode="L",
        )
        result = [
            {"label": "sky", "mask": mask_a},
            {"label": "tree", "mask": mask_b},
        ]
        label2id = {"sky": 1, "tree": 2}
        pred = WinMLImageSegmentationEvaluator.prepare_prediction(
            result,
            label2id,
            (2, 2),
        )
        # Pixel (0,0) and (1,0) are claimed by both; tree (last) wins
        expected = np.array([[2, 1], [2, 2]])
        np.testing.assert_array_equal(pred, expected)


# ---------------------------------------------------------------------------
# compute() integration
# ---------------------------------------------------------------------------


class TestCompute:
    """Integration test for compute() with a mocked pipeline."""

    def test_compute_perfect_prediction(self):
        """compute() should return mIoU=1.0 when pipeline output matches GT."""
        from unittest.mock import MagicMock

        label2id = {"cat": 0, "dog": 1}
        ev = make_evaluator(label2id=label2id, num_labels=2)

        # Build a small dataset with 2 images
        gt1 = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        gt2 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        img1 = create_dummy_image(2, 2)
        img2 = create_dummy_image(2, 2)

        ev.data = [
            {"image": img1, "annotation": create_annotation_image(gt1)},
            {"image": img2, "annotation": create_annotation_image(gt2)},
        ]

        # Mock pipeline to return masks matching GT exactly
        def mock_pipe(image):
            # Determine which image this is by identity
            gt = gt1 if image is img1 else gt2
            masks = []
            for label_name, class_id in label2id.items():
                mask_arr = np.where(gt == class_id, 255, 0).astype(np.uint8)
                masks.append(
                    {
                        "label": label_name,
                        "mask": PILImage.fromarray(mask_arr, mode="L"),
                    }
                )
            return masks

        ev.pipe = mock_pipe

        # Mock config with no label_mapping
        ev.config = MagicMock()
        ev.config.dataset.label_mapping = None

        result = ev.compute()
        assert result["mean_iou"] == pytest.approx(1.0)
        assert result["overall_accuracy"] == pytest.approx(1.0)
        assert len(result["per_category_iou"]) == 2
