# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Accuracy delta computation and per-metric comparison thresholds.

The eval runner records measured facts only (the winml-eval ``metrics`` map and
the ``dataset`` it ran on). Grading a measured value against a PyTorch baseline
— computing the delta and turning it into a PASS/REGRESSION verdict — is done by
the report site (``ModelKitArtifacts-site``), which imports
:data:`METRIC_COMPARE_STRATEGY` and :func:`compute_delta` from here so the
runner and the report stay in lock-step on thresholds and delta math.
"""

from __future__ import annotations


# Per-metric comparison strategy:
# (delta_key, thresh_pass, thresh_at_risk, higher_is_better)
# higher_is_better: True  = larger value is better (accuracy, f1, spearman)
#                   False = smaller value is better (WER, loss, CER)
METRIC_COMPARE_STRATEGY: dict[str, tuple[str, float, float, bool]] = {
    "cosine_spearman": ("delta_absolute", 2.0, 4.0, True),
    # WinML-vs-baseline delta is small — pick a tighter threshold than default.
    "knn_top1_accuracy": ("delta_relative", 0.02, 0.05, True),
    "pseudo_perplexity": ("delta_relative", 0.05, 0.10, False),
    # CER (OCR error rate): lower is better.
    "cer": ("delta_relative", 0.05, 0.10, False),
    "default": ("delta_relative", 0.05, 0.10, True),  # 5% and 10%
}


def compute_delta(
    winml_metric: dict | None,
    baseline_metric: dict | None,
) -> tuple[float | None, float | None]:
    """Return (delta_absolute, delta_relative) from metric dicts.

    Both dicts must have a ``"value"`` key.
    Returns (None, None) if either is missing or baseline value is zero.

    Note: For error-rate metrics (WER/CER — lower is better) a positive delta
    means the WinML pipeline is *worse*. The grader normalizes the sign using
    ``higher_is_better`` from :data:`METRIC_COMPARE_STRATEGY`.
    """
    if winml_metric is None or baseline_metric is None:
        return None, None
    winml_val = winml_metric.get("value")
    base_val = baseline_metric.get("value")
    if winml_val is None or base_val is None:
        return None, None
    delta_abs = winml_val - base_val
    if base_val == 0:
        return round(delta_abs, 6), None
    return round(delta_abs, 6), round(delta_abs / base_val, 6)
