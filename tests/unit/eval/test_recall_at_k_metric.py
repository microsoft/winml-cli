# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for :class:`~winml.modelkit.eval.metrics.RecallAtKMetric`."""

from __future__ import annotations

import numpy as np
import pytest

from winml.modelkit.eval.metrics import RecallAtKMetric


# =============================================================================
# Construction & validation
# =============================================================================


class TestConstruction:
    def test_default_k_values(self) -> None:
        # Sanity: default (1, 5, 10) surfaces in compute() output keys.
        metric = RecallAtKMetric()
        metric.update(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 1)
        result = metric.compute()
        assert set(result) == {"recall_at_1", "recall_at_5", "recall_at_10", "n_samples"}

    def test_custom_k_values(self) -> None:
        metric = RecallAtKMetric(k_values=(2, 7))
        metric.update(np.array([10, 20, 30]), 20)
        result = metric.compute()
        assert set(result) == {"recall_at_2", "recall_at_7", "n_samples"}

    def test_k_values_sorted_and_deduplicated(self) -> None:
        # Order shouldn't matter; duplicates collapse.
        metric = RecallAtKMetric(k_values=(5, 1, 5, 10))
        metric.update(np.array([1, 2, 3, 4, 5]), 1)
        result = metric.compute()
        # Keys sorted ascending by K.
        assert list(result)[:-1] == ["recall_at_1", "recall_at_5", "recall_at_10"]

    def test_empty_k_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            RecallAtKMetric(k_values=())

    def test_non_positive_k_rejected(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            RecallAtKMetric(k_values=(0, 5))
        with pytest.raises(ValueError, match=">= 1"):
            RecallAtKMetric(k_values=(-1,))


# =============================================================================
# Single-relevant semantics (hit@k)
# =============================================================================


class TestSingleRelevant:
    def test_hit_at_top(self) -> None:
        # gt is rank 1 -> hits all K's.
        metric = RecallAtKMetric(k_values=(1, 5, 10))
        metric.update(np.array([42, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 42)
        result = metric.compute()
        assert result["recall_at_1"] == 1.0
        assert result["recall_at_5"] == 1.0
        assert result["recall_at_10"] == 1.0

    def test_hit_at_5_but_not_1(self) -> None:
        # gt at rank 3 -> hits @5 and @10 but not @1.
        metric = RecallAtKMetric(k_values=(1, 5, 10))
        metric.update(np.array([1, 2, 42, 3, 4, 5, 6, 7, 8, 9]), 42)
        result = metric.compute()
        assert result["recall_at_1"] == 0.0
        assert result["recall_at_5"] == 1.0
        assert result["recall_at_10"] == 1.0

    def test_miss_all(self) -> None:
        # gt not in ranked list -> zeros.
        metric = RecallAtKMetric(k_values=(1, 5, 10))
        metric.update(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 999)
        result = metric.compute()
        assert result["recall_at_1"] == 0.0
        assert result["recall_at_5"] == 0.0
        assert result["recall_at_10"] == 0.0

    def test_np_integer_ground_truth_accepted(self) -> None:
        # numpy integer scalar (from argmax etc.) is a common gt source.
        metric = RecallAtKMetric(k_values=(1,))
        metric.update(np.array([1, 2, 3]), np.int64(1))
        assert metric.compute()["recall_at_1"] == 1.0


# =============================================================================
# Multi-relevant semantics (classical retrieval recall)
# =============================================================================


class TestMultiRelevant:
    def test_all_relevant_retrieved(self) -> None:
        # 3 relevant items, all in top 3 -> recall@3 = 3/3 = 1.0
        metric = RecallAtKMetric(k_values=(3,))
        metric.update(np.array([1, 2, 3, 4, 5]), [1, 2, 3])
        assert metric.compute()["recall_at_3"] == 1.0

    def test_partial_relevant_retrieved(self) -> None:
        # 3 relevant items {1, 2, 3}, top-2 = [1, 2] -> recall@2 = 2/3
        metric = RecallAtKMetric(k_values=(2,))
        metric.update(np.array([1, 2, 4, 3, 5]), [1, 2, 3])
        assert metric.compute()["recall_at_2"] == round(2 / 3, 4)

    def test_no_relevant_retrieved(self) -> None:
        metric = RecallAtKMetric(k_values=(3,))
        metric.update(np.array([10, 20, 30]), [1, 2, 3])
        assert metric.compute()["recall_at_3"] == 0.0

    def test_set_ground_truth_accepted(self) -> None:
        metric = RecallAtKMetric(k_values=(2,))
        metric.update(np.array([5, 3, 1]), {1, 3, 7})
        # top-2 = [5, 3], relevant overlap = {3}, |relevant| = 3 -> 1/3
        assert metric.compute()["recall_at_2"] == round(1 / 3, 4)

    def test_empty_ground_truth_rejected(self) -> None:
        metric = RecallAtKMetric()
        with pytest.raises(ValueError, match="at least one"):
            metric.update(np.array([1, 2, 3]), [])


# =============================================================================
# Aggregation over multiple queries
# =============================================================================


class TestAggregation:
    def test_empty_state_returns_nones(self) -> None:
        metric = RecallAtKMetric(k_values=(1, 5))
        result = metric.compute()
        assert result["recall_at_1"] is None
        assert result["recall_at_5"] is None
        assert result["n_samples"] == 0

    def test_batch_mean(self) -> None:
        # Two queries: one hit@1, one miss@1 -> mean = 0.5
        metric = RecallAtKMetric(k_values=(1, 5))
        metric.update(np.array([1, 2, 3, 4, 5]), 1)  # hit @1 and @5
        metric.update(np.array([2, 3, 4, 5, 6]), 1)  # miss @1, miss @5
        result = metric.compute()
        assert result["recall_at_1"] == 0.5
        assert result["recall_at_5"] == 0.5
        assert result["n_samples"] == 2

    def test_reset_clears_state(self) -> None:
        metric = RecallAtKMetric(k_values=(1,))
        metric.update(np.array([1, 2, 3]), 1)
        assert metric.compute()["n_samples"] == 1
        metric.reset()
        assert metric.compute()["n_samples"] == 0
        assert metric.compute()["recall_at_1"] is None


# =============================================================================
# Shape handling
# =============================================================================


class TestShapeHandling:
    def test_2d_input_flattened(self) -> None:
        # A (1, K) shape from `.reshape(1, -1)` or slicing flattens.
        metric = RecallAtKMetric(k_values=(1,))
        metric.update(np.array([[42, 1, 2]]), 42)
        assert metric.compute()["recall_at_1"] == 1.0

    def test_empty_ranked_predictions_rejected(self) -> None:
        metric = RecallAtKMetric()
        with pytest.raises(ValueError, match="cannot be empty"):
            metric.update(np.array([]), 1)

    def test_k_larger_than_ranked_list(self) -> None:
        # K=10 with only 3 predictions -> treats top-3 == whole list.
        metric = RecallAtKMetric(k_values=(10,))
        metric.update(np.array([1, 2, 3]), 2)
        assert metric.compute()["recall_at_10"] == 1.0
