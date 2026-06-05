# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tensor-similarity metrics for compare-mode (ONNX vs HF reference parity).

A :class:`TensorSimilarityMetric` instance accumulates per-sample scalar
metrics across ``(prediction, reference)`` tensor pairs via ``update``
and reports ``{f"{metric}_{stat}": float}`` (4 stats per metric: mean,
std, min, max) via ``compute``. The per-sample math (SQNR, PSNR, cosine,
MSE, max abs diff) mirrors the team-wide ``eval_tensors`` library so
numbers match bit-for-bit on the same ``.npy`` pair.
"""

from __future__ import annotations

import math

import numpy as np


_SCALAR_METRICS = (
    "sqnr_db",
    "psnr_db",
    "cosine_similarity",
    "mse",
    "max_abs_diff",
)


def _sqnr_db(ref: np.ndarray, test: np.ndarray) -> float:
    """``10 * log10(sum(ref^2) / sum((ref-test)^2))``. ``+inf`` if identical."""
    signal = float(np.sum(ref * ref))
    noise = float(np.sum((ref - test) ** 2))
    if noise == 0.0:
        return math.inf
    if signal == 0.0:
        return -math.inf
    return 10.0 * math.log10(signal / noise)


def _mse(ref: np.ndarray, test: np.ndarray) -> float:
    return float(np.mean((ref - test) ** 2))


def _max_abs_diff(ref: np.ndarray, test: np.ndarray) -> float:
    return float(np.max(np.abs(ref - test)))


def _psnr_db(ref: np.ndarray, mse_val: float) -> float:
    """``10 * log10(peak^2 / mse)``, ``peak = max(|ref|)``."""
    if mse_val == 0.0:
        return math.inf
    peak = float(np.max(np.abs(ref)))
    if peak == 0.0:
        return -math.inf
    return 10.0 * math.log10((peak * peak) / mse_val)


def _cosine_similarity(ref: np.ndarray, test: np.ndarray) -> float:
    """``dot(ref, test) / (||ref|| * ||test||)``, asymmetric zero handling.

    Both inputs all-zero -> ``1.0`` (identical zero vectors).
    Exactly one input all-zero -> ``0.0`` (a dead vector against a live
    one is NOT a perfect match, even though the angle is undefined).
    """
    norm_ref = float(np.linalg.norm(ref))
    norm_test = float(np.linalg.norm(test))
    if norm_ref == 0.0 and norm_test == 0.0:
        return 1.0
    if norm_ref == 0.0 or norm_test == 0.0:
        return 0.0
    return float(np.dot(ref, test) / (norm_ref * norm_test))


class TensorSimilarityMetric:
    """Streaming per-sample tensor-parity metrics.

    Each ``update(prediction, reference)`` computes the 5 scalar metrics
    on the pair and appends them to internal per-metric lists. ``compute``
    aggregates each list to ``mean`` / ``std`` / ``min`` / ``max`` and
    returns a flat ``{f"{metric}_{stat}": float}`` dict ready for direct
    consumption by the generic eval report renderer. ``mean`` and ``std``
    are computed over only the finite values so a single bit-identical
    sample (``sqnr_db = +inf``, ``psnr_db = +inf``) does not poison
    the aggregate.
    """

    def __init__(self) -> None:
        self._per_sample: dict[str, list[float]] = {m: [] for m in _SCALAR_METRICS}

    def update(self, prediction: np.ndarray, reference: np.ndarray) -> None:
        """Compute all scalar metrics on one pair and append to per-metric lists."""
        if prediction.shape != reference.shape:
            raise ValueError(
                f"shape mismatch: prediction {prediction.shape} vs "
                f"reference {reference.shape}",
            )
        ref = reference.astype(np.float64).ravel()
        test = prediction.astype(np.float64).ravel()

        mse_val = _mse(ref, test)
        self._per_sample["sqnr_db"].append(_sqnr_db(ref, test))
        self._per_sample["psnr_db"].append(_psnr_db(ref, mse_val))
        self._per_sample["cosine_similarity"].append(_cosine_similarity(ref, test))
        self._per_sample["mse"].append(mse_val)
        self._per_sample["max_abs_diff"].append(_max_abs_diff(ref, test))

    def compute(self) -> dict[str, float]:
        """Return ``{f"{metric}_{stat}": float}`` for stats mean/std/min/max."""
        result: dict[str, float] = {}
        for metric, values in self._per_sample.items():
            if not values:
                continue
            finite = [v for v in values if math.isfinite(v)]
            if finite:
                arr = np.asarray(finite, dtype=np.float64)
                mean_val = float(arr.mean())
                std_val = float(arr.std())
            elif all(v == math.inf for v in values):
                mean_val, std_val = math.inf, 0.0
            elif all(v == -math.inf for v in values):
                mean_val, std_val = -math.inf, 0.0
            else:
                # Any NaN, or a mix of +inf and -inf: un-summarizable.
                mean_val, std_val = math.nan, math.nan
            result[f"{metric}_mean"] = mean_val
            result[f"{metric}_std"] = std_val
            result[f"{metric}_min"] = float(min(values))
            result[f"{metric}_max"] = float(max(values))
        return result

    def reset(self) -> None:
        """Clear all accumulated per-sample values."""
        for k in self._per_sample:
            self._per_sample[k] = []
