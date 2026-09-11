# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for :class:`~winml.modelkit.eval.metrics.MeanReciprocalRankMetric`."""

from __future__ import annotations

import numpy as np
import pytest

from winml.modelkit.eval.metrics import MeanReciprocalRankMetric


# =============================================================================
# Single-relevant semantics
# =============================================================================


class TestSingleRelevant:
    def test_hit_at_rank_1(self) -> None:
        metric = MeanReciprocalRankMetric()
        metric.update(np.array([42, 1, 2, 3]), 42)
        assert metric.compute()["mrr"] == 1.0

    def test_hit_at_rank_2(self) -> None:
        metric = MeanReciprocalRankMetric()
        metric.update(np.array([1, 42, 2, 3]), 42)
        assert metric.compute()["mrr"] == 0.5

    def test_hit_at_rank_3(self) -> None:
        metric = MeanReciprocalRankMetric()
        metric.update(np.array([1, 2, 42, 3]), 42)
        # 1/3 rounded to 4 dp
        assert metric.compute()["mrr"] == round(1 / 3, 4)

    def test_no_hit_scores_zero(self) -> None:
        metric = MeanReciprocalRankMetric()
        metric.update(np.array([1, 2, 3, 4]), 999)
        assert metric.compute()["mrr"] == 0.0

    def test_np_integer_ground_truth_accepted(self) -> None:
        metric = MeanReciprocalRankMetric()
        metric.update(np.array([1, 2, 3]), np.int64(1))
        assert metric.compute()["mrr"] == 1.0


# =============================================================================
# Multi-relevant semantics (uses first hit)
# =============================================================================


class TestMultiRelevant:
    def test_first_hit_wins(self) -> None:
        # Relevant {2, 3}; ranked [1, 2, 3] -> first hit at rank 2 -> RR = 1/2.
        metric = MeanReciprocalRankMetric()
        metric.update(np.array([1, 2, 3]), [2, 3])
        assert metric.compute()["mrr"] == 0.5

    def test_earliest_position_used(self) -> None:
        # Relevant {5, 2}; ranked [1, 2, 3, 4, 5] -> first hit at rank 2.
        metric = MeanReciprocalRankMetric()
        metric.update(np.array([1, 2, 3, 4, 5]), [5, 2])
        assert metric.compute()["mrr"] == 0.5

    def test_no_relevant_in_ranked_scores_zero(self) -> None:
        metric = MeanReciprocalRankMetric()
        metric.update(np.array([10, 20, 30]), [1, 2, 3])
        assert metric.compute()["mrr"] == 0.0

    def test_set_input(self) -> None:
        metric = MeanReciprocalRankMetric()
        metric.update(np.array([1, 2, 3]), {3})
        # Hit at rank 3 -> 1/3
        assert metric.compute()["mrr"] == round(1 / 3, 4)

    def test_empty_ground_truth_rejected(self) -> None:
        metric = MeanReciprocalRankMetric()
        with pytest.raises(ValueError, match="at least one"):
            metric.update(np.array([1, 2, 3]), [])


# =============================================================================
# Aggregation
# =============================================================================


class TestAggregation:
    def test_empty_state_returns_none(self) -> None:
        metric = MeanReciprocalRankMetric()
        result = metric.compute()
        assert result["mrr"] is None
        assert result["n_samples"] == 0

    def test_batch_mean(self) -> None:
        # RR values: [1.0, 0.5, 0.0] -> mean 0.5
        metric = MeanReciprocalRankMetric()
        metric.update(np.array([1, 2, 3]), 1)  # rank 1 -> 1.0
        metric.update(np.array([1, 2, 3]), 2)  # rank 2 -> 0.5
        metric.update(np.array([1, 2, 3]), 999)  # miss  -> 0.0
        result = metric.compute()
        assert result["mrr"] == 0.5
        assert result["n_samples"] == 3

    def test_reset_clears_state(self) -> None:
        metric = MeanReciprocalRankMetric()
        metric.update(np.array([1, 2, 3]), 1)
        assert metric.compute()["n_samples"] == 1
        metric.reset()
        assert metric.compute()["n_samples"] == 0
        assert metric.compute()["mrr"] is None


# =============================================================================
# Shape handling
# =============================================================================


class TestShapeHandling:
    def test_2d_input_flattened(self) -> None:
        metric = MeanReciprocalRankMetric()
        metric.update(np.array([[1, 42, 2]]), 42)
        assert metric.compute()["mrr"] == 0.5

    def test_empty_ranked_predictions_rejected(self) -> None:
        metric = MeanReciprocalRankMetric()
        with pytest.raises(ValueError, match="cannot be empty"):
            metric.update(np.array([]), 1)
