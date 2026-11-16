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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from winml.modelkit.eval import WinMLEvaluationConfig, WinMLKeypointDetectionEvaluator


def _make_evaluator(box_format: str = "xywh") -> WinMLKeypointDetectionEvaluator:
    """Create an evaluator instance without triggering data/model loading."""
    ev = object.__new__(WinMLKeypointDetectionEvaluator)
    ev._image_col = "image"
    ev._annotation_col = "objects"
    ev._keypoints_key = "keypoints"
    ev._bbox_key = "bbox"
    ev._area_key = "area"
    ev._box_format = box_format
    ev._sigmas = None
    ev._keypoint_names = None
    ev._dataset_index = 0
    ev.config = WinMLEvaluationConfig(task="keypoint-detection")
    return ev


class TestBoxFormat:
    def test_xywh_passthrough(self):
        ev = _make_evaluator("xywh")
        assert ev._to_xywh([10.0, 20.0, 30.0, 40.0]) == [10.0, 20.0, 30.0, 40.0]

    def test_xyxy_converted_to_xywh(self):
        ev = _make_evaluator("xyxy")
        assert ev._to_xywh([10.0, 20.0, 40.0, 60.0]) == [10.0, 20.0, 30.0, 40.0]

    def test_processor_forwards_trust_remote_code(self):
        ev = _make_evaluator()
        ev.config = WinMLEvaluationConfig(
            model_id="test/model",
            task="keypoint-detection",
            trust_remote_code=True,
        )
        ev.model = MagicMock(io_config={})
        processor = MagicMock()

        with patch(
            "transformers.AutoImageProcessor.from_pretrained",
            return_value=processor,
        ) as load_processor:
            assert ev.prepare_pipeline() is processor

        load_processor.assert_called_once_with(
            "test/model",
            trust_remote_code=True,
        )

    @pytest.mark.parametrize(
        ("model_type", "architectures"),
        [
            ("vitpose", []),
            ("custom", ["ViTPoseForPoseEstimation"]),
        ],
    )
    def test_vitpose_alias_mismatch_uses_gated_fallback(self, model_type, architectures):
        ev = _make_evaluator()
        ev.config = WinMLEvaluationConfig(
            model_id="nielsr/vitpose-base-simple",
            task="keypoint-detection",
            trust_remote_code=True,
        )
        ev.model = MagicMock(
            io_config={},
            config=SimpleNamespace(
                model_type=model_type,
                architectures=architectures,
            ),
        )
        fallback_processor = MagicMock()

        with (
            patch(
                "transformers.AutoImageProcessor.from_pretrained",
                side_effect=ValueError(
                    "Unrecognized image processor in nielsr/vitpose-base-simple"
                ),
            ) as load_auto,
            patch(
                "transformers.VitPoseImageProcessor.from_pretrained",
                return_value=fallback_processor,
            ) as load_vitpose,
        ):
            assert ev.prepare_pipeline() is fallback_processor

        load_auto.assert_called_once_with(
            "nielsr/vitpose-base-simple",
            trust_remote_code=True,
        )
        load_vitpose.assert_called_once_with(
            "nielsr/vitpose-base-simple",
            trust_remote_code=True,
        )

    def test_known_error_for_non_vitpose_model_is_not_swallowed(self):
        ev = _make_evaluator()
        ev.config = WinMLEvaluationConfig(
            model_id="microsoft/resnet-50",
            task="keypoint-detection",
            trust_remote_code=False,
        )
        ev.model = MagicMock(
            io_config={},
            config=SimpleNamespace(
                model_type="resnet", architectures=["ResNetForImageClassification"]
            ),
        )

        with (
            patch(
                "transformers.AutoImageProcessor.from_pretrained",
                side_effect=ValueError("Unrecognized image processor for resnet"),
            ) as load_auto,
            patch("transformers.VitPoseImageProcessor.from_pretrained") as load_vitpose,
            pytest.raises(ValueError, match="Unrecognized image processor"),
        ):
            ev.prepare_pipeline()

        load_auto.assert_called_once_with(
            "microsoft/resnet-50",
            trust_remote_code=False,
        )
        load_vitpose.assert_not_called()

    def test_non_vitpose_success_keeps_strict_auto_path(self):
        ev = _make_evaluator()
        ev.config = WinMLEvaluationConfig(
            model_id="microsoft/resnet-50",
            task="keypoint-detection",
            trust_remote_code=False,
        )
        ev.model = MagicMock(
            io_config={},
            config=SimpleNamespace(
                model_type="resnet", architectures=["ResNetForImageClassification"]
            ),
        )
        processor = MagicMock()

        with (
            patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=processor,
            ) as load_auto,
            patch("transformers.VitPoseImageProcessor.from_pretrained") as load_vitpose,
        ):
            assert ev.prepare_pipeline() is processor

        load_auto.assert_called_once_with(
            "microsoft/resnet-50",
            trust_remote_code=False,
        )
        load_vitpose.assert_not_called()

    def test_unrelated_value_error_is_not_swallowed(self):
        ev = _make_evaluator()
        ev.config = WinMLEvaluationConfig(
            model_id="nielsr/vitpose-base-simple",
            task="keypoint-detection",
            trust_remote_code=False,
        )
        ev.model = MagicMock(
            io_config={},
            config=SimpleNamespace(
                model_type="vitpose",
                architectures=["ViTPoseForPoseEstimation"],
            ),
        )

        with (
            patch(
                "transformers.AutoImageProcessor.from_pretrained",
                side_effect=ValueError("network timeout while loading processor"),
            ) as load_auto,
            patch("transformers.VitPoseImageProcessor.from_pretrained") as load_vitpose,
            pytest.raises(ValueError, match="network timeout"),
        ):
            ev.prepare_pipeline()

        load_auto.assert_called_once_with(
            "nielsr/vitpose-base-simple",
            trust_remote_code=False,
        )
        load_vitpose.assert_not_called()

    def test_missing_model_id_raises_before_processor_load(self):
        ev = _make_evaluator()
        ev.config = WinMLEvaluationConfig(task="keypoint-detection")
        ev.model = MagicMock(io_config={})

        with (
            patch("transformers.AutoImageProcessor.from_pretrained") as load_auto,
            patch("transformers.VitPoseImageProcessor.from_pretrained") as load_vitpose,
            pytest.raises(ValueError, match="model_id is required"),
        ):
            ev.prepare_pipeline()

        load_auto.assert_not_called()
        load_vitpose.assert_not_called()


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

        assert "map" in result
        assert result["num_images"] == 2
        assert result["num_predictions"] == 3
        assert result["num_ground_truths"] == 3


class _RecordingModel:
    """Mock model that records call kwargs and declares an input contract."""

    def __init__(self, input_names, num_keypoints: int = 17) -> None:
        self.io_config = {"input_names": list(input_names)}
        self._num_keypoints = num_keypoints
        self.calls: list[dict] = []

    def __call__(self, pixel_values, **kwargs):
        self.calls.append(kwargs)
        batch = pixel_values.shape[0]
        return {"heatmaps": torch.zeros(batch, self._num_keypoints, 64, 48)}


class _ForwardSignatureModel:
    """Mock PyTorch-style model with no ``io_config``; declares ``dataset_index``
    only in its ``forward`` signature (like a native ``VitPoseForPoseEstimation``)."""

    def __init__(self, num_keypoints: int = 17) -> None:
        self._num_keypoints = num_keypoints
        self.calls: list[dict] = []

    def forward(self, pixel_values, dataset_index=None, **kwargs):
        self.calls.append({"dataset_index": dataset_index, **kwargs})
        batch = pixel_values.shape[0]
        return {"heatmaps": torch.zeros(batch, self._num_keypoints, 64, 48)}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TestModelInputContract:
    def test_dataset_index_passed_when_declared(self):
        ev = _make_evaluator()
        ev._dataset_index = 0
        model = _RecordingModel(["pixel_values", "dataset_index"])
        ev.model = model
        ev._predict_poses(_MockProcessor(), object(), [[0.0, 0.0, 10.0, 10.0]])
        assert "dataset_index" in model.calls[0]
        assert model.calls[0]["dataset_index"].tolist() == [0]

    def test_dataset_index_absent_when_not_declared(self):
        ev = _make_evaluator()
        ev._dataset_index = 0
        model = _RecordingModel(["pixel_values"])
        ev.model = model
        ev._predict_poses(_MockProcessor(), object(), [[0.0, 0.0, 10.0, 10.0]])
        assert model.calls[0] == {}

    def test_configured_dataset_index_value_is_used(self):
        ev = _make_evaluator()
        ev._dataset_index = 3
        model = _RecordingModel(["pixel_values", "dataset_index"])
        ev.model = model
        ev._predict_poses(_MockProcessor(), object(), [[0.0, 0.0, 10.0, 10.0]])
        assert model.calls[0]["dataset_index"].tolist() == [3]

    def test_dataset_index_detected_from_forward_signature(self):
        # Native PyTorch modules have no ``io_config``; the input contract is
        # read from the ``forward`` signature instead (e.g. ViTPose+ MoE).
        ev = _make_evaluator()
        ev._dataset_index = 2
        model = _ForwardSignatureModel()
        ev.model = model
        ev._predict_poses(_MockProcessor(), object(), [[0.0, 0.0, 10.0, 10.0]])
        assert model.calls[0]["dataset_index"].tolist() == [2]
