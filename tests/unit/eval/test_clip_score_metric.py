# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for :class:`~winml.modelkit.eval.metrics.CLIPScoreMetric`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from winml.modelkit.eval.metrics import CLIPScoreMetric


# =============================================================================
# Construction & validation
# =============================================================================


class TestConstruction:
    def test_default_weight_is_2_5(self) -> None:
        metric = CLIPScoreMetric()
        v = np.array([1.0, 0.0])
        metric.update(v, v)
        # identical -> cos=1 -> score = 1 * 2.5
        assert metric.compute()["clip_score_mean"] == 2.5

    def test_custom_weight(self) -> None:
        metric = CLIPScoreMetric(weight=1.0)
        v = np.array([1.0, 0.0])
        metric.update(v, v)
        # identical -> cos=1 -> score = 1 * 1.0
        assert metric.compute()["clip_score_mean"] == 1.0

    def test_zero_weight_valid(self) -> None:
        metric = CLIPScoreMetric(weight=0.0)
        v = np.array([1.0, 0.0])
        metric.update(v, v)
        assert metric.compute()["clip_score_mean"] == 0.0

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            CLIPScoreMetric(weight=-1.0)

    def test_nan_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            CLIPScoreMetric(weight=float("nan"))

    def test_inf_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            CLIPScoreMetric(weight=float("inf"))


# =============================================================================
# Cosine semantics — the actual scoring math
# =============================================================================


class TestCosineSemantics:
    def test_identical_embeddings_max_score(self) -> None:
        metric = CLIPScoreMetric(weight=1.0)
        v = np.array([1.0, 2.0, 3.0])
        metric.update(v, v)
        assert metric.compute()["clip_score_mean"] == 1.0

    def test_orthogonal_embeddings_zero(self) -> None:
        metric = CLIPScoreMetric(weight=1.0)
        metric.update(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        assert metric.compute()["clip_score_mean"] == 0.0

    def test_antiparallel_clipped_to_zero(self) -> None:
        # cos = -1 -> max(0, -1) = 0
        metric = CLIPScoreMetric(weight=1.0)
        v = np.array([1.0, 2.0, 3.0])
        metric.update(v, -v)
        assert metric.compute()["clip_score_mean"] == 0.0

    def test_known_cosine_value(self) -> None:
        # cos([1,0], [1,1]) = 1/sqrt(2), scaled x1.0
        metric = CLIPScoreMetric(weight=1.0)
        metric.update(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
        expected = 1.0 / math.sqrt(2)
        assert metric.compute()["clip_score_mean"] == round(expected, 4)

    def test_weight_scales_positive_cosines(self) -> None:
        # cos = 1/sqrt(2), weight = 2.5 -> 2.5/sqrt(2)
        metric = CLIPScoreMetric(weight=2.5)
        metric.update(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
        expected = 2.5 / math.sqrt(2)
        assert metric.compute()["clip_score_mean"] == round(expected, 4)


# =============================================================================
# Zero-vector handling
# =============================================================================


class TestZeroVectorHandling:
    def test_zero_text_scores_zero(self) -> None:
        metric = CLIPScoreMetric(weight=1.0)
        metric.update(np.zeros(3), np.array([1.0, 2.0, 3.0]))
        assert metric.compute()["clip_score_mean"] == 0.0

    def test_zero_image_scores_zero(self) -> None:
        metric = CLIPScoreMetric(weight=1.0)
        metric.update(np.array([1.0, 2.0, 3.0]), np.zeros(3))
        assert metric.compute()["clip_score_mean"] == 0.0

    def test_both_zero_scores_zero(self) -> None:
        # Both-zero is an undefined angle; treated as no alignment.
        metric = CLIPScoreMetric(weight=1.0)
        metric.update(np.zeros(3), np.zeros(3))
        assert metric.compute()["clip_score_mean"] == 0.0


# =============================================================================
# Shape handling
# =============================================================================


class TestShapeHandling:
    def test_2d_input_flattened(self) -> None:
        # (1, D) and (D,) with the same D flatten to the same 1-D vector.
        metric = CLIPScoreMetric(weight=1.0)
        metric.update(np.array([[1.0, 2.0]]), np.array([1.0, 2.0]))
        assert metric.compute()["clip_score_mean"] == 1.0

    def test_size_mismatch_raises(self) -> None:
        metric = CLIPScoreMetric()
        with pytest.raises(ValueError, match="size mismatch"):
            metric.update(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_empty_embedding_rejected(self) -> None:
        metric = CLIPScoreMetric()
        with pytest.raises(ValueError, match="empty"):
            metric.update(np.array([]), np.array([]))

    def test_integer_arrays_accepted(self) -> None:
        # ``update`` coerces to float64 -- ints should work.
        metric = CLIPScoreMetric(weight=1.0)
        metric.update(np.array([1, 0]), np.array([1, 0]))
        assert metric.compute()["clip_score_mean"] == 1.0


# =============================================================================
# Aggregation over multiple samples
# =============================================================================


class TestAggregation:
    def test_empty_state_returns_nones(self) -> None:
        metric = CLIPScoreMetric()
        result = metric.compute()
        assert result["clip_score_mean"] is None
        assert result["clip_score_std"] is None
        assert result["clip_score_min"] is None
        assert result["clip_score_max"] is None
        assert result["n_samples"] == 0

    def test_batch_statistics(self) -> None:
        # Two orthogonal pairs (score 0) + one identical pair (score 2.5).
        metric = CLIPScoreMetric(weight=2.5)
        metric.update(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        metric.update(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        metric.update(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        result = metric.compute()
        assert result["n_samples"] == 3
        # scores = [0, 0, 2.5] -> mean = 2.5 / 3
        assert result["clip_score_mean"] == round(2.5 / 3, 4)
        assert result["clip_score_min"] == 0.0
        assert result["clip_score_max"] == 2.5

    def test_reset_clears_state(self) -> None:
        metric = CLIPScoreMetric()
        metric.update(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        assert metric.compute()["n_samples"] == 1
        metric.reset()
        assert metric.compute()["n_samples"] == 0
        assert metric.compute()["clip_score_mean"] is None


# =============================================================================
# Real-shaped embeddings (CLIP typically emits 512-D vectors)
# =============================================================================


class TestRealisticShapes:
    def test_512_dim_deterministic(self) -> None:
        rng = np.random.default_rng(0)
        metric = CLIPScoreMetric(weight=2.5)

        text_embeddings = rng.standard_normal(size=(10, 512))
        image_embeddings = rng.standard_normal(size=(10, 512))
        for t, i in zip(text_embeddings, image_embeddings, strict=True):
            metric.update(t, i)

        result = metric.compute()
        assert result["n_samples"] == 10
        # Random Gaussian pairs in 512-D are near-orthogonal on average --
        # the mean score should be modest and clipped positive.
        assert 0.0 <= result["clip_score_mean"] <= 2.5

    def test_matched_pairs_score_higher_than_random(self) -> None:
        # Sanity: identical pairs beat random pairs on average.
        rng = np.random.default_rng(42)

        matched = CLIPScoreMetric(weight=1.0)
        random_pairs = CLIPScoreMetric(weight=1.0)
        for _ in range(20):
            v = rng.standard_normal(size=(512,))
            matched.update(v, v)
            random_pairs.update(v, rng.standard_normal(size=(512,)))

        matched_mean = matched.compute()["clip_score_mean"]
        random_mean = random_pairs.compute()["clip_score_mean"]
        assert matched_mean is not None
        assert random_mean is not None
        assert matched_mean > random_mean
