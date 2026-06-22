# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for KeypointAPMetric (COCO OKS-based AP)."""

from __future__ import annotations

import pytest

from winml.modelkit.eval import KeypointAPMetric


def _coco_person_keypoints(cx: float, cy: float) -> list[float]:
    """Build a plausible 17-keypoint COCO layout around a center, all visible."""
    offsets = [
        (0, -40), (-5, -45), (5, -45), (-10, -42), (10, -42),
        (-20, -20), (20, -20), (-25, 0), (25, 0),
        (-28, 20), (28, 20), (-15, 25), (15, 25),
        (-15, 55), (15, 55), (-15, 85), (15, 85),
    ]
    flat: list[float] = []
    for dx, dy in offsets:
        flat.extend([cx + dx, cy + dy, 2.0])  # visibility 2 = labeled + visible
    return flat


class TestKeypointAPMetricPerfectMatch:
    """Predictions identical to ground truth should score AP ~= 1.0."""

    def test_single_person_perfect_match(self) -> None:
        kpts = _coco_person_keypoints(100.0, 100.0)
        pred_kpts = [v if (i % 3) != 2 else 1.0 for i, v in enumerate(kpts)]

        metric = KeypointAPMetric()
        result = metric.compute(
            predictions=[{"image_id": 1, "keypoints": pred_kpts, "score": 0.95}],
            references=[
                {
                    "image_id": 1,
                    "keypoints": kpts,
                    "bbox": [60.0, 50.0, 80.0, 110.0],
                    "area": 80.0 * 110.0,
                }
            ],
        )

        assert result["AP"] == pytest.approx(1.0, abs=0.01)
        assert result["AP50"] == pytest.approx(1.0, abs=0.01)
        assert result["num_predictions"] == 1
        assert result["num_ground_truths"] == 1
        assert result["num_images"] == 1

    def test_two_people_two_images_perfect_match(self) -> None:
        refs = []
        preds = []
        for img_id, (cx, cy) in enumerate([(100.0, 100.0), (300.0, 200.0)], start=1):
            kpts = _coco_person_keypoints(cx, cy)
            pred_kpts = [v if (i % 3) != 2 else 1.0 for i, v in enumerate(kpts)]
            refs.append(
                {
                    "image_id": img_id,
                    "keypoints": kpts,
                    "bbox": [cx - 40, cy - 50, 80.0, 110.0],
                    "area": 80.0 * 110.0,
                }
            )
            preds.append({"image_id": img_id, "keypoints": pred_kpts, "score": 0.9})

        result = KeypointAPMetric().compute(predictions=preds, references=refs)

        assert result["AP"] == pytest.approx(1.0, abs=0.01)
        assert result["num_images"] == 2


class TestKeypointAPMetricImperfect:
    """Offset and empty-input behavior."""

    def test_large_offset_lowers_ap(self) -> None:
        kpts = _coco_person_keypoints(100.0, 100.0)
        # Shift every predicted keypoint far from GT -> low OKS -> low AP.
        pred_kpts: list[float] = []
        for i, v in enumerate(kpts):
            if i % 3 == 0 or i % 3 == 1:
                pred_kpts.append(v + 60.0)
            else:
                pred_kpts.append(1.0)

        result = KeypointAPMetric().compute(
            predictions=[{"image_id": 1, "keypoints": pred_kpts, "score": 0.9}],
            references=[
                {
                    "image_id": 1,
                    "keypoints": kpts,
                    "bbox": [60.0, 50.0, 80.0, 110.0],
                    "area": 80.0 * 110.0,
                }
            ],
        )

        assert result["AP"] < 0.5

    def test_no_predictions_returns_zero(self) -> None:
        kpts = _coco_person_keypoints(100.0, 100.0)
        result = KeypointAPMetric().compute(
            predictions=[],
            references=[
                {
                    "image_id": 1,
                    "keypoints": kpts,
                    "bbox": [60.0, 50.0, 80.0, 110.0],
                    "area": 80.0 * 110.0,
                }
            ],
        )

        assert result["AP"] == 0.0
        assert result["num_predictions"] == 0
        assert result["num_ground_truths"] == 1


class TestKeypointAPMetricMismatch:
    """A non-COCO keypoint layout must fail early with a clear message."""

    def test_mismatched_keypoint_count_raises(self):
        # Model predicts 52 keypoints (e.g. SynthPose) against COCO-17 ground truth.
        pred_kpts = [0.0, 0.0, 1.0] * 52
        gt_kpts = _coco_person_keypoints(100.0, 100.0)

        with pytest.raises(ValueError, match="Keypoint count mismatch"):
            KeypointAPMetric().compute(
                predictions=[{"image_id": 1, "keypoints": pred_kpts, "score": 0.9}],
                references=[
                    {
                        "image_id": 1,
                        "keypoints": gt_kpts,
                        "bbox": [60.0, 50.0, 80.0, 110.0],
                        "area": 80.0 * 110.0,
                    }
                ],
            )

