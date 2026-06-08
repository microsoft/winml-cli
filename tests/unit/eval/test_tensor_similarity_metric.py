# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for TensorSimilarityMetric."""

from __future__ import annotations

import math

import numpy as np
import pytest

from winml.modelkit.eval.metrics.tensor_similarity import (
    _SCALAR_METRICS,
    TensorSimilarityMetric,
)


_STATS = ("mean", "std", "min", "max")


# ---------------------------------------------------------------------------
# Per-metric numerical correctness on a single (pred, ref) pair
# ---------------------------------------------------------------------------

class TestPerSampleMath:
    def test_identical_inputs(self):
        m = TensorSimilarityMetric()
        x = np.array([1.0, -2.0, 3.0, 4.5], dtype=np.float32)
        m.update(x, x)
        r = m.compute()
        # Identical tensors: SQNR/PSNR = +inf; cosine = 1; MSE = 0; max|diff| = 0.
        assert r["sqnr_db_max"] == math.inf
        assert r["psnr_db_max"] == math.inf
        assert r["cosine_similarity_mean"] == pytest.approx(1.0)
        assert r["mse_mean"] == 0.0
        assert r["max_abs_diff_mean"] == 0.0
        # No finite SQNR/PSNR samples -> mean falls back to +inf.
        assert r["sqnr_db_mean"] == math.inf
        assert r["psnr_db_mean"] == math.inf

    def test_known_diff(self):
        # ref=[1,2,3,4], test=ref + 0.1 everywhere.
        ref = np.array([1.0, 2.0, 3.0, 4.0])
        test = ref + 0.1
        m = TensorSimilarityMetric()
        m.update(test, ref)
        r = m.compute()

        # MSE = 0.01, max|diff| = 0.1
        assert r["mse_mean"] == pytest.approx(0.01)
        assert r["max_abs_diff_mean"] == pytest.approx(0.1)

        # SQNR = 10*log10(sum(ref^2) / sum(noise^2)) = 10*log10(30 / 0.04)
        expected_sqnr = 10.0 * math.log10(30.0 / 0.04)
        assert r["sqnr_db_mean"] == pytest.approx(expected_sqnr)

        # PSNR = 10*log10(peak^2 / mse), peak = 4
        expected_psnr = 10.0 * math.log10(16.0 / 0.01)
        assert r["psnr_db_mean"] == pytest.approx(expected_psnr)

        # Cosine: dot(ref,test)/(|ref||test|).
        expected_cos = float(np.dot(ref, test) / (np.linalg.norm(ref) * np.linalg.norm(test)))
        assert r["cosine_similarity_mean"] == pytest.approx(expected_cos)

    def test_multidim_returns_scalar_metrics(self):
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((1, 100, 92)).astype(np.float32)
        test = ref + rng.standard_normal((1, 100, 92)).astype(np.float32) * 0.01
        m = TensorSimilarityMetric()
        m.update(test, ref)
        r = m.compute()
        for metric in _SCALAR_METRICS:
            assert isinstance(r[f"{metric}_mean"], float)
            assert math.isfinite(r[f"{metric}_mean"])

    def test_shape_mismatch_raises(self):
        m = TensorSimilarityMetric()
        with pytest.raises(ValueError, match="shape mismatch"):
            m.update(np.zeros((2, 3)), np.zeros((2, 4)))


# ---------------------------------------------------------------------------
# Cosine zero-vector convention (asymmetric)
# ---------------------------------------------------------------------------

class TestCosineZeroHandling:
    def test_both_zero_returns_one(self):
        z = np.zeros(8)
        m = TensorSimilarityMetric()
        m.update(z, z)
        assert m.compute()["cosine_similarity_mean"] == pytest.approx(1.0)

    def test_one_zero_returns_zero(self):
        z = np.zeros(8)
        nz = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        m = TensorSimilarityMetric()
        m.update(z, nz)
        m.update(nz, z)
        r = m.compute()
        assert r["cosine_similarity_min"] == pytest.approx(0.0)
        assert r["cosine_similarity_max"] == pytest.approx(0.0)
        assert r["cosine_similarity_mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SQNR/PSNR degenerate signal: -inf, not 0
# ---------------------------------------------------------------------------

class TestDegenerateSignal:
    def test_zero_ref_nonzero_test_yields_neg_inf(self):
        ref = np.zeros(4)
        test = np.array([0.1, 0.2, 0.3, 0.4])
        m = TensorSimilarityMetric()
        m.update(test, ref)
        r = m.compute()
        assert r["sqnr_db_min"] == -math.inf
        assert r["psnr_db_min"] == -math.inf


# ---------------------------------------------------------------------------
# All-non-finite aggregation: must NOT misreport as +inf "perfect"
# ---------------------------------------------------------------------------

class TestAllNonFiniteAggregation:
    def test_all_nan_predictions_aggregate_to_nan(self):
        # A model that emits NaN poisons cosine / MSE; the aggregate must
        # surface this as NaN, not as a false "+inf perfect match".
        ref = np.array([1.0, 2.0, 3.0, 4.0])
        nan_pred = np.full(4, math.nan)
        m = TensorSimilarityMetric()
        m.update(nan_pred, ref)
        m.update(nan_pred, ref)
        r = m.compute()
        assert math.isnan(r["cosine_similarity_mean"])
        assert math.isnan(r["cosine_similarity_std"])
        assert math.isnan(r["mse_mean"])

    def test_all_neg_inf_aggregates_to_neg_inf(self):
        # Two samples of (zero ref, non-zero test) -> SQNR/PSNR are all -inf.
        ref = np.zeros(4)
        test = np.array([0.1, 0.2, 0.3, 0.4])
        m = TensorSimilarityMetric()
        m.update(test, ref)
        m.update(test, ref)
        r = m.compute()
        assert r["sqnr_db_mean"] == -math.inf
        assert r["sqnr_db_std"] == 0.0
        assert r["psnr_db_mean"] == -math.inf
        assert r["psnr_db_std"] == 0.0

    def test_mixed_pos_inf_and_neg_inf_aggregates_to_nan(self):
        # One identical sample (+inf SQNR) + one zero-ref/nonzero-test sample
        # (-inf SQNR). No finite values, mixed signs -> NaN.
        ref_a = np.array([1.0, 2.0, 3.0, 4.0])
        ref_b = np.zeros(4)
        test_b = np.array([0.1, 0.2, 0.3, 0.4])
        m = TensorSimilarityMetric()
        m.update(ref_a, ref_a)   # SQNR = +inf
        m.update(test_b, ref_b)  # SQNR = -inf
        r = m.compute()
        assert math.isnan(r["sqnr_db_mean"])
        assert math.isnan(r["sqnr_db_std"])
        # min/max remain well-defined.
        assert r["sqnr_db_min"] == -math.inf
        assert r["sqnr_db_max"] == math.inf


# ---------------------------------------------------------------------------
# Multi-sample aggregation (mean/std/min/max)
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_mean_filters_non_finite(self):
        # Two finite SQNR samples + one +inf sample (identical pair).
        # Mean should ignore +inf so it stays finite.
        m = TensorSimilarityMetric()
        ref = np.array([1.0, 2.0, 3.0])
        m.update(ref + 0.1, ref)
        m.update(ref + 0.2, ref)
        m.update(ref, ref)
        r = m.compute()
        assert math.isfinite(r["sqnr_db_mean"])
        assert r["sqnr_db_max"] == math.inf

    def test_summary_keys(self):
        m = TensorSimilarityMetric()
        m.update(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
        r = m.compute()
        expected = {f"{metric}_{stat}" for metric in _SCALAR_METRICS for stat in _STATS}
        assert set(r) == expected
        assert all(isinstance(v, float) for v in r.values())

    def test_reset_clears_state(self):
        m = TensorSimilarityMetric()
        m.update(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
        m.reset()
        assert m.compute() == {}
