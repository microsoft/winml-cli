# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for MAPMetric class."""

from __future__ import annotations

import pytest

from winml.modelkit.eval.metrics.mean_average_precision import MAPMetric


class TestMAPMetricPerfectMatch:
    """Test mAP with perfect predictions matching ground truth."""

    def test_single_image_perfect_match(self) -> None:
        """One image, one object, prediction matches GT exactly."""
        metric = MAPMetric()
        result = metric.compute(
            predictions=[
                {
                    "boxes": [[10.0, 10.0, 50.0, 50.0]],
                    "scores": [0.99],
                    "labels": [1],
                }
            ],
            references=[
                {
                    "boxes": [[10.0, 10.0, 50.0, 50.0]],
                    "labels": [1],
                }
            ],
            box_format="xyxy",
        )

        assert result["map"] == pytest.approx(1.0, abs=0.01)
        assert result["map_50"] == pytest.approx(1.0, abs=0.01)
        assert result["num_predictions"] == 1
        assert result["num_ground_truths"] == 1
        assert result["num_images"] == 1

    def test_multiple_objects_perfect_match(self) -> None:
        """One image, multiple objects, all predictions match."""
        metric = MAPMetric()
        result = metric.compute(
            predictions=[
                {
                    "boxes": [[10, 10, 50, 50], [100, 100, 200, 200], [300, 300, 400, 400]],
                    "scores": [0.9, 0.85, 0.8],
                    "labels": [1, 2, 3],
                }
            ],
            references=[
                {
                    "boxes": [[10, 10, 50, 50], [100, 100, 200, 200], [300, 300, 400, 400]],
                    "labels": [1, 2, 3],
                }
            ],
            box_format="xyxy",
        )

        assert result["map"] == pytest.approx(1.0, abs=0.01)
        assert result["num_predictions"] == 3
        assert result["num_ground_truths"] == 3


class TestMAPMetricPartialMatch:
    """Test mAP with partial/imperfect predictions."""

    def test_missed_detection(self) -> None:
        """Two GT objects but model only detects one — recall drops."""
        metric = MAPMetric()
        result = metric.compute(
            predictions=[
                {
                    "boxes": [[10, 10, 50, 50]],
                    "scores": [0.9],
                    "labels": [1],
                }
            ],
            references=[
                {
                    "boxes": [[10, 10, 50, 50], [100, 100, 200, 200]],
                    "labels": [1, 1],
                }
            ],
            box_format="xyxy",
        )

        assert result["map"] < 1.0
        assert result["num_predictions"] == 1
        assert result["num_ground_truths"] == 2

    def test_wrong_label_no_match(self) -> None:
        """Prediction has wrong label — should not match GT."""
        metric = MAPMetric()
        result = metric.compute(
            predictions=[
                {
                    "boxes": [[10, 10, 50, 50]],
                    "scores": [0.9],
                    "labels": [2],
                }
            ],
            references=[
                {
                    "boxes": [[10, 10, 50, 50]],
                    "labels": [1],
                }
            ],
            box_format="xyxy",
        )

        assert result["map"] == pytest.approx(0.0, abs=0.01)

    def test_low_iou_no_match(self) -> None:
        """Prediction box far from GT — IoU too low to match."""
        metric = MAPMetric()
        result = metric.compute(
            predictions=[
                {
                    "boxes": [[500, 500, 600, 600]],
                    "scores": [0.9],
                    "labels": [1],
                }
            ],
            references=[
                {
                    "boxes": [[10, 10, 50, 50]],
                    "labels": [1],
                }
            ],
            box_format="xyxy",
        )

        assert result["map"] == pytest.approx(0.0, abs=0.01)


class TestMAPMetricBoxConversion:
    """Test box format and coordinate conversion."""

    def test_xywh_conversion(self) -> None:
        """GT in xywh format should produce same result as xyxy."""
        metric = MAPMetric()

        result_xywh = metric.compute(
            predictions=[
                {
                    "boxes": [[10, 10, 50, 50]],
                    "scores": [0.99],
                    "labels": [1],
                }
            ],
            references=[
                {
                    "boxes": [
                        [10, 10, 40, 40]
                    ],  # xywh: x=10, y=10, w=40, h=40 → xyxy [10,10,50,50]
                    "labels": [1],
                }
            ],
            box_format="xywh",
        )

        result_xyxy = metric.compute(
            predictions=[
                {
                    "boxes": [[10, 10, 50, 50]],
                    "scores": [0.99],
                    "labels": [1],
                }
            ],
            references=[
                {
                    "boxes": [[10, 10, 50, 50]],
                    "labels": [1],
                }
            ],
            box_format="xyxy",
        )

        assert result_xywh["map"] == pytest.approx(result_xyxy["map"], abs=0.01)

    def test_normalized_coords_denormalized(self) -> None:
        """Normalized GT boxes should be denormalized before computing."""
        metric = MAPMetric()

        result = metric.compute(
            predictions=[
                {
                    "boxes": [[10, 10, 50, 50]],
                    "scores": [0.99],
                    "labels": [1],
                }
            ],
            references=[
                {
                    "boxes": [[0.1, 0.1, 0.5, 0.5]],  # normalized → [10, 10, 50, 50]
                    "labels": [1],
                    "image_size": (100, 100),
                }
            ],
            box_format="xyxy",
            box_coords="normalized",
        )

        assert result["map"] == pytest.approx(1.0, abs=0.01)


class TestMAPMetricEdgeCases:
    """Test edge cases: empty images, unmapped labels."""

    def test_empty_image_no_crash(self) -> None:
        """Image with no GT and no predictions should not crash."""
        metric = MAPMetric()
        result = metric.compute(
            predictions=[{"boxes": [], "scores": [], "labels": []}],
            references=[{"boxes": [], "labels": []}],
            box_format="xyxy",
        )

        assert "map" in result
        assert result["num_images"] == 1

    def test_unmapped_labels_excluded(self) -> None:
        """Labels of -1 should be filtered from GT."""
        metric = MAPMetric()
        result = metric.compute(
            predictions=[
                {
                    "boxes": [[10, 10, 50, 50]],
                    "scores": [0.9],
                    "labels": [1],
                }
            ],
            references=[
                {
                    "boxes": [[10, 10, 50, 50], [100, 100, 200, 200]],
                    "labels": [1, -1],
                }
            ],
            box_format="xyxy",
        )

        assert result["num_ground_truths"] == 1
        assert result["map"] == pytest.approx(1.0, abs=0.01)

    def test_multiple_images(self) -> None:
        """Metric computes across multiple images."""
        metric = MAPMetric()
        preds = [{"boxes": [[10, 10, 50, 50]], "scores": [0.95], "labels": [1]} for _ in range(5)]
        refs = [{"boxes": [[10, 10, 50, 50]], "labels": [1]} for _ in range(5)]

        result = metric.compute(
            predictions=preds,
            references=refs,
            box_format="xyxy",
        )

        assert result["num_images"] == 5
        assert result["num_predictions"] == 5
        assert result["num_ground_truths"] == 5
        assert result["map"] == pytest.approx(1.0, abs=0.01)
