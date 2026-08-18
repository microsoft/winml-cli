# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for average normalized Levenshtein similarity."""

from __future__ import annotations

import pytest

from winml.modelkit.eval.metrics.document_qa import (
    ANLSMetric,
    _normalized_levenshtein_similarity,
)


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("ITC Limited", "ITC Limited", 1.0),
        ("  itc   LIMITED ", "ITC Limited", 1.0),
        ("", "", 1.0),
        ("", "invoice", 0.0),
        ("abc", "xyz", 0.0),
        ("invoicf", "invoice", pytest.approx(6 / 7)),
    ],
)
def test_similarity_normalization_and_threshold(prediction, reference, expected):
    assert _normalized_levenshtein_similarity(prediction, reference) == expected


def test_metric_uses_best_reference_and_aggregates():
    metric = ANLSMetric()
    metric.update("ITC Limited", ["Other Company", "itc limited"])
    metric.update("invoice", "invoicf")

    assert metric.compute() == {
        "anls": pytest.approx((1 + 6 / 7) / 2, abs=1e-4),
        "n_samples": 2,
        "skipped_samples": 0,
    }


def test_metric_reports_empty_and_skipped_samples():
    metric = ANLSMetric()
    metric.update("unused", [])

    assert metric.compute() == {"anls": None, "n_samples": 0, "skipped_samples": 1}
