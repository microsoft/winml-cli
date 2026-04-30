# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for DepthMetric (AbsRel, RMSE, delta1)."""

import numpy as np
import pytest
import torch

from winml.modelkit.eval import DepthMetric


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self):
        m = DepthMetric()
        # Reset state available for compute after at least one update.
        with pytest.raises(ValueError, match="no valid pixels"):
            m.compute()

    def test_invalid_align_raises(self):
        with pytest.raises(ValueError, match="align must be"):
            DepthMetric(align="mean")  # type: ignore[arg-type]

    def test_invalid_delta_threshold_raises(self):
        with pytest.raises(ValueError, match="delta_threshold"):
            DepthMetric(delta_threshold=1.0)


# ---------------------------------------------------------------------------
# Perfect & known-value cases
# ---------------------------------------------------------------------------


class TestKnownValues:
    def test_perfect_prediction_align_none(self):
        gt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        pred = gt.copy()
        m = DepthMetric(align="none", min_depth=0.0, max_depth=None)
        m.update(pred, gt)
        result = m.compute()
        assert result["abs_rel"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["delta1"] == pytest.approx(1.0)
        assert result["num_images"] == 1
        assert result["num_valid_pixels"] == 4

    def test_perfect_prediction_align_median(self):
        """Scaled-by-constant prediction is perfect after median alignment."""
        gt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        pred = gt * 7.0  # arbitrary scale
        m = DepthMetric(align="median", min_depth=0.0, max_depth=None)
        m.update(pred, gt)
        result = m.compute()
        assert result["abs_rel"] == pytest.approx(0.0, abs=1e-6)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert result["delta1"] == pytest.approx(1.0)

    def test_known_abs_rel(self):
        """AbsRel = mean(|pred-gt|/gt) = mean({1, 0.5}) = 0.75."""
        gt = np.array([[1.0, 2.0]], dtype=np.float32)
        pred = np.array([[2.0, 3.0]], dtype=np.float32)
        m = DepthMetric(align="none", min_depth=0.0, max_depth=None)
        m.update(pred, gt)
        result = m.compute()
        assert result["abs_rel"] == pytest.approx(0.75)

    def test_known_rmse(self):
        """RMSE = sqrt(mean((pred-gt)^2)) = sqrt(mean({1, 1})) = 1."""
        gt = np.array([[1.0, 2.0]], dtype=np.float32)
        pred = np.array([[2.0, 3.0]], dtype=np.float32)
        m = DepthMetric(align="none", min_depth=0.0, max_depth=None)
        m.update(pred, gt)
        result = m.compute()
        assert result["rmse"] == pytest.approx(1.0)

    def test_known_delta1(self):
        """ratios = {2, 1.5}; both >= 1.25, so delta1 = 0."""
        gt = np.array([[1.0, 2.0]], dtype=np.float32)
        pred = np.array([[2.0, 3.0]], dtype=np.float32)
        m = DepthMetric(align="none", min_depth=0.0, max_depth=None)
        m.update(pred, gt)
        result = m.compute()
        assert result["delta1"] == pytest.approx(0.0)

    def test_delta1_partial_within_threshold(self):
        """Two pixels: one ratio 1.1 (< 1.25), one 2.0 (>= 1.25). delta1 = 0.5."""
        gt = np.array([1.0, 1.0], dtype=np.float32).reshape(1, 2)
        pred = np.array([1.1, 2.0], dtype=np.float32).reshape(1, 2)
        m = DepthMetric(align="none", min_depth=0.0, max_depth=None)
        m.update(pred, gt)
        result = m.compute()
        assert result["delta1"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Valid mask
# ---------------------------------------------------------------------------


class TestValidMask:
    def test_zero_gt_pixels_excluded(self):
        gt = np.array([[0.0, 1.0]], dtype=np.float32)
        pred = np.array([[5.0, 1.0]], dtype=np.float32)
        m = DepthMetric(align="none", min_depth=1e-3, max_depth=None)
        m.update(pred, gt)
        result = m.compute()
        # Only the second pixel counted.
        assert result["abs_rel"] == pytest.approx(0.0)
        assert result["num_valid_pixels"] == 1

    def test_nan_inf_excluded(self):
        gt = np.array([[np.nan, np.inf, 2.0]], dtype=np.float32)
        pred = np.array([[1.0, 1.0, 2.0]], dtype=np.float32)
        m = DepthMetric(align="none", min_depth=0.0, max_depth=None)
        m.update(pred, gt)
        result = m.compute()
        assert result["num_valid_pixels"] == 1
        assert result["abs_rel"] == pytest.approx(0.0)

    def test_max_depth_clip(self):
        gt = np.array([[5.0, 100.0]], dtype=np.float32)
        pred = np.array([[5.0, 5.0]], dtype=np.float32)
        m = DepthMetric(align="none", min_depth=0.0, max_depth=10.0)
        m.update(pred, gt)
        result = m.compute()
        assert result["num_valid_pixels"] == 1

    def test_max_depth_none_keeps_all(self):
        gt = np.array([[5.0, 100.0]], dtype=np.float32)
        pred = gt.copy()
        m = DepthMetric(align="none", min_depth=0.0, max_depth=None)
        m.update(pred, gt)
        result = m.compute()
        assert result["num_valid_pixels"] == 2

    def test_negative_predictions_excluded(self):
        gt = np.array([[1.0, 1.0]], dtype=np.float32)
        pred = np.array([[-1.0, 1.0]], dtype=np.float32)
        m = DepthMetric(align="none", min_depth=0.0, max_depth=None)
        m.update(pred, gt)
        result = m.compute()
        assert result["num_valid_pixels"] == 1

    def test_all_invalid_image_counted_no_pixels(self):
        gt = np.zeros((2, 2), dtype=np.float32)
        pred = np.ones((2, 2), dtype=np.float32)
        m = DepthMetric(align="none", min_depth=1e-3, max_depth=None)
        m.update(pred, gt)
        with pytest.raises(ValueError, match="no valid pixels"):
            m.compute()


# ---------------------------------------------------------------------------
# Multi-image accumulation
# ---------------------------------------------------------------------------


class TestAccumulation:
    def test_multi_image_pixel_weighted(self):
        gt1 = np.array([[1.0, 1.0]], dtype=np.float32)
        pred1 = gt1.copy()  # perfect
        gt2 = np.array([[1.0]], dtype=np.float32)
        pred2 = np.array([[2.0]], dtype=np.float32)  # error
        m = DepthMetric(align="none", min_depth=0.0, max_depth=None)
        m.update(pred1, gt1)
        m.update(pred2, gt2)
        result = m.compute()
        # AbsRel = (0 + 0 + 1) / 3 = 1/3
        assert result["abs_rel"] == pytest.approx(1.0 / 3.0)
        assert result["num_images"] == 2
        assert result["num_valid_pixels"] == 3

    def test_reset_clears_state(self):
        gt = np.array([[1.0]], dtype=np.float32)
        pred = np.array([[2.0]], dtype=np.float32)
        m = DepthMetric(align="none", min_depth=0.0, max_depth=None)
        m.update(pred, gt)
        m.reset()
        with pytest.raises(ValueError, match="no valid pixels"):
            m.compute()


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


class TestInputTypes:
    def test_torch_tensor_input(self):
        gt = np.array([[1.0, 2.0]], dtype=np.float32)
        pred_t = torch.tensor([[1.0, 2.0]])
        m = DepthMetric(align="none", min_depth=0.0, max_depth=None)
        m.update(pred_t, gt)
        result = m.compute()
        assert result["rmse"] == pytest.approx(0.0)

    def test_extra_singleton_dims_squeezed(self):
        gt = np.ones((1, 2, 2), dtype=np.float32)
        pred = np.ones((1, 1, 2, 2), dtype=np.float32)
        m = DepthMetric(align="none", min_depth=0.0, max_depth=None)
        m.update(pred, gt)
        result = m.compute()
        assert result["num_valid_pixels"] == 4

    def test_shape_mismatch_raises(self):
        gt = np.ones((2, 2), dtype=np.float32)
        pred = np.ones((3, 3), dtype=np.float32)
        m = DepthMetric()
        with pytest.raises(ValueError, match="shape"):
            m.update(pred, gt)


# ---------------------------------------------------------------------------
# Median alignment
# ---------------------------------------------------------------------------


class TestMedianAlignment:
    def test_median_alignment_recovers_perfect(self):
        rng = np.random.default_rng(0)
        gt = rng.uniform(1.0, 10.0, size=(8, 8)).astype(np.float32)
        pred = gt * 0.25  # uniform scale
        m = DepthMetric(align="median", min_depth=0.0, max_depth=None)
        m.update(pred, gt)
        result = m.compute()
        assert result["abs_rel"] == pytest.approx(0.0, abs=1e-5)

    def test_align_none_keeps_scale_error(self):
        gt = np.array([[1.0, 2.0]], dtype=np.float32)
        pred = gt * 0.5
        m = DepthMetric(align="none", min_depth=0.0, max_depth=None)
        m.update(pred, gt)
        result = m.compute()
        assert result["abs_rel"] > 0.4
