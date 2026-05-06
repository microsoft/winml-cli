# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for ClassificationMetric class."""

from __future__ import annotations

import pytest

from winml.modelkit.eval import ClassificationMetric


class TestClassificationMetricBasic:
    """Accuracy and macro-F1 over string labels."""

    def test_perfect_predictions(self) -> None:
        metric = ClassificationMetric()
        result = metric.compute(
            predictions=["a", "b", "c"],
            references=["a", "b", "c"],
            labels=["a", "b", "c"],
        )
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        metric = ClassificationMetric()
        result = metric.compute(
            predictions=["a", "a"],
            references=["b", "b"],
            labels=["a", "b"],
        )
        assert result["accuracy"] == pytest.approx(0.0)
        assert result["f1"] == pytest.approx(0.0)

    def test_half_correct(self) -> None:
        metric = ClassificationMetric()
        result = metric.compute(
            predictions=["a", "b", "a", "b"],
            references=["a", "a", "b", "b"],
            labels=["a", "b"],
        )
        assert result["accuracy"] == pytest.approx(0.5)


class TestClassificationMetricLabels:
    """Full class set preserved via ``labels`` argument."""

    def test_unseen_class_in_macro_f1(self) -> None:
        """Classes with no predictions should contribute 0 to macro-F1."""
        metric = ClassificationMetric()
        result = metric.compute(
            predictions=["a", "a"],
            references=["a", "a"],
            labels=["a", "b", "c"],
        )
        # Class 'a' is perfect (F1=1.0); 'b' and 'c' unseen (F1=0 each).
        # Macro-F1 = (1 + 0 + 0) / 3.
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0 / 3)

    def test_labels_order_does_not_affect_result(self) -> None:
        metric = ClassificationMetric()
        r1 = metric.compute(
            predictions=["a", "b"],
            references=["a", "b"],
            labels=["a", "b"],
        )
        r2 = metric.compute(
            predictions=["a", "b"],
            references=["a", "b"],
            labels=["b", "a"],
        )
        assert r1 == r2


class TestClassificationMetricValidation:
    def test_length_mismatch_raises(self) -> None:
        metric = ClassificationMetric()
        with pytest.raises(ValueError, match="same length"):
            metric.compute(
                predictions=["a", "b"],
                references=["a"],
                labels=["a", "b"],
            )

    def test_empty_references_raises(self) -> None:
        metric = ClassificationMetric()
        with pytest.raises(ValueError, match="references"):
            metric.compute(predictions=[], references=[], labels=["a"])

    def test_empty_labels_raises(self) -> None:
        metric = ClassificationMetric()
        with pytest.raises(ValueError, match="labels"):
            metric.compute(predictions=["a"], references=["a"], labels=[])


class TestClassificationMetricZeroDivision:
    """``zero_division=0`` prevents sklearn warnings / crashes."""

    def test_single_class_prediction(self) -> None:
        """All predictions collapse to one class — other classes have no preds."""
        metric = ClassificationMetric()
        result = metric.compute(
            predictions=["a", "a", "a"],
            references=["a", "b", "c"],
            labels=["a", "b", "c"],
        )
        # accuracy = 1/3; classes b and c have precision/recall issues,
        # but zero_division=0 guarantees a finite F1.
        assert result["accuracy"] == pytest.approx(1 / 3)
        assert 0.0 <= result["f1"] <= 1.0
