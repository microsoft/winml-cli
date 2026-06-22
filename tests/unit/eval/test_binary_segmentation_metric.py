# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for BinarySegmentationMetric."""

from __future__ import annotations

import numpy as np
import pytest

from winml.modelkit.eval.metrics.binary_segmentation import BinarySegmentationMetric


class TestPerfectAndDisjoint:
    def test_perfect_overlap(self) -> None:
        m = BinarySegmentationMetric()
        gt = np.zeros((10, 10), dtype=bool)
        gt[2:6, 2:6] = True
        m.update(gt.copy(), gt)
        out = m.compute()
        assert out["mIoU"] == 1.0
        assert out["dice"] == 1.0
        assert out["num_samples"] == 1

    def test_disjoint(self) -> None:
        m = BinarySegmentationMetric()
        gt = np.zeros((10, 10), dtype=bool)
        gt[0:4, 0:4] = True
        pred = np.zeros((10, 10), dtype=bool)
        pred[5:9, 5:9] = True
        m.update(pred, gt)
        out = m.compute()
        assert out["mIoU"] == 0.0
        assert out["dice"] == 0.0


class TestKnownValue:
    def test_half_overlap(self) -> None:
        # GT is 4x4 = 16 px, pred covers half of it + 8 px elsewhere ->
        # intersection=8, union=24, IoU=1/3; |pred|+|gt|=32, Dice=16/32=0.5.
        m = BinarySegmentationMetric()
        gt = np.zeros((10, 10), dtype=bool)
        gt[0:4, 0:4] = True
        pred = np.zeros((10, 10), dtype=bool)
        pred[0:4, 0:2] = True  # 8 px overlap with gt
        pred[6:8, 0:4] = True  # 8 px outside gt
        m.update(pred, gt)
        out = m.compute()
        assert out["mIoU"] == pytest.approx(1 / 3, abs=1e-6)
        assert out["dice"] == pytest.approx(0.5, abs=1e-6)


class TestAggregation:
    def test_mean_across_samples(self) -> None:
        m = BinarySegmentationMetric()
        gt = np.zeros((4, 4), dtype=bool)
        gt[0:2, 0:2] = True
        m.update(gt.copy(), gt)  # IoU=1.0
        m.update(np.zeros_like(gt), gt)  # IoU=0.0
        out = m.compute()
        assert out["mIoU"] == 0.5
        assert out["num_samples"] == 2


class TestEdgeCases:
    def test_empty_gt_skipped(self) -> None:
        m = BinarySegmentationMetric()
        empty = np.zeros((5, 5), dtype=bool)
        pred = np.ones((5, 5), dtype=bool)
        m.update(pred, empty)
        out = m.compute()
        assert out["num_samples"] == 0
        assert out["num_skipped"] == 1
        assert out["mIoU"] == 0.0  # documented fallback when nothing scored

    def test_shape_mismatch_raises(self) -> None:
        m = BinarySegmentationMetric()
        with pytest.raises(ValueError, match="shape"):
            m.update(np.zeros((4, 4), dtype=bool), np.zeros((5, 5), dtype=bool))

    def test_nonzero_treated_as_foreground(self) -> None:
        # Pass raw uint8 masks (255) -> should be treated as fg
        m = BinarySegmentationMetric()
        gt = np.zeros((4, 4), dtype=np.uint8)
        gt[0:2, 0:2] = 255
        m.update(gt.copy(), gt)
        assert m.compute()["mIoU"] == 1.0


class TestRegistryLazyAttr:
    def test_registered_in_metrics_package(self) -> None:
        from winml.modelkit.eval import metrics

        cls = metrics.BinarySegmentationMetric
        assert cls is BinarySegmentationMetric
