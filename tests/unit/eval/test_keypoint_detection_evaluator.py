# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLKeypointDetectionEvaluator.

The end-to-end pose pipeline is covered by integration runs; these tests
pin the box-format handling, prediction flattening, and the ``compute()``
loop wiring with a mocked image processor and model.
"""

from __future__ import annotations

import pytest
import torch

from winml.modelkit.eval import WinMLKeypointDetectionEvaluator


def _make_evaluator(box_format: str = "xywh") -> WinMLKeypointDetectionEvaluator:
    """Create an evaluator instance without triggering data/model loading."""
    ev = object.__new__(WinMLKeypointDetectionEvaluator)
    ev._image_col = "image"
    ev._annotation_col = "objects"
    ev._keypoints_key = "keypoints"
    ev._bbox_key = "bbox"
    ev._area_key = "area"
    ev._box_format = box_format
    return ev


class TestBoxFormat:
    def test_xywh_passthrough(self):
        ev = _make_evaluator("xywh")
        assert ev._to_xywh([10.0, 20.0, 30.0, 40.0]) == [10.0, 20.0, 30.0, 40.0]

    def test_xyxy_converted_to_xywh(self):
        ev = _make_evaluator("xyxy")
        assert ev._to_xywh([10.0, 20.0, 40.0, 60.0]) == [10.0, 20.0, 30.0, 40.0]


class TestPredictionFlattening:
    def test_flatten_interleaves_xy_and_score(self):
        pose = {
            "keypoints": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "scores": torch.tensor([0.5, 0.9]),
        }
        flat = WinMLKeypointDetectionEvaluator._flatten_prediction(pose)
        assert flat == pytest.approx([1.0, 2.0, 0.5, 3.0, 4.0, 0.9])

    def test_person_score_is_mean(self):
        pose = {"scores": torch.tensor([0.4, 0.6, 0.8])}
        assert WinMLKeypointDetectionEvaluator._person_score(pose) == pytest.approx(0.6)


class _MockProcessor:
    """Mock image processor returning fixed pixel values and poses."""

    def __init__(self, num_keypoints: int = 17) -> None:
        self._num_keypoints = num_keypoints

    def preprocess(self, images, boxes, return_tensors="pt"):
        num_persons = len(boxes[0])
        return {"pixel_values": torch.zeros(num_persons, 3, 256, 192)}

    def post_process_pose_estimation(self, outputs, boxes):
        num_persons = outputs.heatmaps.shape[0]
        poses = [
            {
                "keypoints": torch.ones(self._num_keypoints, 2),
                "scores": torch.full((self._num_keypoints,), 0.8),
            }
            for _ in range(num_persons)
        ]
        return [poses]


class _MockModel:
    """Mock model returning a single-person heatmap per call."""

    def __init__(self, num_keypoints: int = 17) -> None:
        self._num_keypoints = num_keypoints

    def __call__(self, pixel_values):
        batch = pixel_values.shape[0]
        return {"heatmaps": torch.zeros(batch, self._num_keypoints, 64, 48)}


class TestComputeLoop:
    def test_compute_returns_ap_metrics(self):
        ev = _make_evaluator("xywh")
        ev.pipe = _MockProcessor()
        ev.model = _MockModel()
        # Two images: one with 2 persons, one with 1.
        ev.data = [
            {
                "image": object(),
                "objects": {
                    "keypoints": [[1.0, 1.0, 2.0] * 17, [2.0, 2.0, 2.0] * 17],
                    "bbox": [[0.0, 0.0, 50.0, 80.0], [10.0, 10.0, 40.0, 70.0]],
                    "area": [4000.0, 2800.0],
                },
            },
            {
                "image": object(),
                "objects": {
                    "keypoints": [[3.0, 3.0, 2.0] * 17],
                    "bbox": [[5.0, 5.0, 30.0, 60.0]],
                    "area": [1800.0],
                },
            },
        ]

        result = ev.compute()

        assert "AP" in result
        assert result["num_images"] == 2
        assert result["num_predictions"] == 3
        assert result["num_ground_truths"] == 3
