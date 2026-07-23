# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for the keypoint-detection inference adapter."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from winml.modelkit.inference.pipeline import KeypointDetectionPipeline


class _Processor:
    def __init__(self) -> None:
        self.size = {"height": 256, "width": 192}
        self.boxes_seen = None

    def preprocess(self, images, boxes, return_tensors="pt"):
        self.boxes_seen = boxes
        return {"pixel_values": torch.zeros(len(boxes[0]), 3, 256, 192)}

    def post_process_pose_estimation(self, outputs, boxes):
        num_persons = outputs.heatmaps.shape[0]
        poses = [
            {
                "keypoints": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "scores": torch.tensor([0.5, 0.75]),
            }
            for _ in range(num_persons)
        ]
        return [poses]


class _Model:
    io_config = {"input_shapes": [[1, 3, 384, 288]], "input_names": ["pixel_values"]}

    def __call__(self, pixel_values, **kwargs):
        return {"heatmaps": torch.zeros(pixel_values.shape[0], 2, 64, 48)}


def test_adapter_runs_preprocess_model_and_postprocess() -> None:
    processor = _Processor()
    with patch(
        "transformers.AutoImageProcessor.from_pretrained",
        return_value=processor,
    ):
        pipe = KeypointDetectionPipeline(_Model(), "test/model")

    result = pipe(object(), boxes=[[10, 20, 30, 40]])

    assert processor.size == {"height": 384, "width": 288}
    assert processor.boxes_seen == [[[10.0, 20.0, 30.0, 40.0]]]
    assert result == [
        {
            "keypoints": [[1.0, 2.0], [3.0, 4.0]],
            "scores": [0.5, 0.75],
            "score": 0.625,
        }
    ]


def test_xyxy_boxes_are_converted_to_xywh() -> None:
    processor = _Processor()
    with patch(
        "transformers.AutoImageProcessor.from_pretrained",
        return_value=processor,
    ):
        pipe = KeypointDetectionPipeline(_Model(), "test/model")

    pipe(object(), boxes=[[10, 20, 40, 70]], box_format="xyxy")

    assert processor.boxes_seen == [[[10.0, 20.0, 30.0, 50.0]]]


def test_dataset_index_is_forwarded_when_model_declares_it() -> None:
    class ModelWithDatasetIndex(_Model):
        io_config = {
            "input_shapes": [[1, 3, 256, 192]],
            "input_names": ["pixel_values", "dataset_index"],
        }

        def __init__(self) -> None:
            self.dataset_index = None

        def __call__(self, pixel_values, **kwargs):
            self.dataset_index = kwargs.get("dataset_index")
            return super().__call__(pixel_values, **kwargs)

    model = ModelWithDatasetIndex()
    with patch(
        "transformers.AutoImageProcessor.from_pretrained",
        return_value=_Processor(),
    ):
        pipe = KeypointDetectionPipeline(model, "test/model")

    pipe(object(), boxes=[[0, 0, 10, 10]], dataset_index=3)

    assert model.dataset_index.tolist() == [3]


def test_sanitize_parameters_exposes_kwargs_for_engine_filtering() -> None:
    processor = _Processor()
    with patch(
        "transformers.AutoImageProcessor.from_pretrained",
        return_value=processor,
    ):
        pipe = KeypointDetectionPipeline(_Model(), "test/model")

    _, forward_kwargs, _ = pipe._sanitize_parameters()

    assert set(forward_kwargs) == {"boxes", "box_format", "dataset_index"}


def test_empty_boxes_raise_clear_error() -> None:
    with patch(
        "transformers.AutoImageProcessor.from_pretrained",
        return_value=_Processor(),
    ):
        pipe = KeypointDetectionPipeline(_Model(), "test/model")

    with pytest.raises(ValueError, match="boxes must be a non-empty list"):
        pipe(object(), boxes=[])
