# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLZeroShotImageClassificationEvaluator and TopKAccuracyMetric."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.eval import TopKAccuracyMetric, WinMLZeroShotImageClassificationEvaluator


# ---------------------------------------------------------------------------
# TopKAccuracyMetric
# ---------------------------------------------------------------------------


class TestTopKAccuracyMetric:
    def test_perfect_predictions(self):
        metric = TopKAccuracyMetric()
        metric.update(["cat", "dog", "bird"], "cat")
        metric.update(["dog", "cat", "bird"], "dog")
        result = metric.compute()
        assert result["top1_accuracy"] == pytest.approx(100.0)
        assert result["top5_accuracy"] == pytest.approx(100.0)

    def test_zero_top1_nonzero_top5(self):
        metric = TopKAccuracyMetric()
        # GT is 2nd predicted — top1 wrong, top5 correct
        metric.update(["dog", "cat", "bird"], "cat")
        result = metric.compute()
        assert result["top1_accuracy"] == pytest.approx(0.0)
        assert result["top5_accuracy"] == pytest.approx(100.0)

    def test_both_zero(self):
        metric = TopKAccuracyMetric()
        # GT not in top-5
        labels = ["a", "b", "c", "d", "e"]
        metric.update(labels, "z")
        result = metric.compute()
        assert result["top1_accuracy"] == pytest.approx(0.0)
        assert result["top5_accuracy"] == pytest.approx(0.0)

    def test_mixed_samples(self):
        metric = TopKAccuracyMetric()
        # Sample 1: top1 correct
        metric.update(["cat", "dog"], "cat")
        # Sample 2: top1 wrong, top5 correct
        metric.update(["dog", "cat"], "cat")
        # Sample 3: both wrong (gt not in list)
        metric.update(["dog", "bird"], "cat")
        result = metric.compute()
        assert result["top1_accuracy"] == pytest.approx(1 / 3 * 100, abs=0.01)
        assert result["top5_accuracy"] == pytest.approx(2 / 3 * 100, abs=0.01)

    def test_empty_raises(self):
        metric = TopKAccuracyMetric()
        with pytest.raises(ValueError, match="No samples"):
            metric.compute()

    def test_empty_predictions(self):
        metric = TopKAccuracyMetric()
        metric.update([], "cat")
        result = metric.compute()
        assert result["top1_accuracy"] == pytest.approx(0.0)
        assert result["top5_accuracy"] == pytest.approx(0.0)

    def test_incremental_updates(self):
        metric = TopKAccuracyMetric()
        for _ in range(10):
            metric.update(["correct", "other"], "correct")
        result = metric.compute()
        assert result["top1_accuracy"] == pytest.approx(100.0)

    def test_top5_boundary(self):
        """GT at exactly position 5 (index 4) should be top-5 correct."""
        metric = TopKAccuracyMetric()
        metric.update(["a", "b", "c", "d", "e", "f"], "e")
        result = metric.compute()
        assert result["top1_accuracy"] == pytest.approx(0.0)
        assert result["top5_accuracy"] == pytest.approx(100.0)

    def test_top5_just_outside(self):
        """GT at position 6 (index 5) should be top-5 wrong."""
        metric = TopKAccuracyMetric()
        metric.update(["a", "b", "c", "d", "e", "f"], "f")
        result = metric.compute()
        assert result["top1_accuracy"] == pytest.approx(0.0)
        assert result["top5_accuracy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Helpers for evaluator tests
# ---------------------------------------------------------------------------


def _make_evaluator(class_names=None):
    """Instantiate evaluator by patching external dependencies."""
    from datasets import ClassLabel, Dataset, Features, Image

    from winml.modelkit.datasets import DatasetConfig
    from winml.modelkit.eval import WinMLEvaluationConfig

    class_names = class_names or ["airplane", "automobile", "bird", "cat", "deer",
                                   "dog", "frog", "horse", "ship", "truck"]

    features = Features({
        "image": Image(),
        "label": ClassLabel(names=class_names),
    })

    # Build a small mock dataset with 4 samples
    from PIL import Image as PILImage
    images = [PILImage.new("RGB", (32, 32), color=(i * 25, i * 25, i * 25))
              for i in range(4)]
    mock_ds = Dataset.from_dict(
        {"image": images, "label": [0, 1, 2, 3]},
        features=features,
    )

    mock_pipe = MagicMock()

    model = MagicMock(spec=["config", "io_config", "eval"])
    model.config.model_type = "clip"
    model.config.label2id = None
    model.io_config = {}

    config = WinMLEvaluationConfig(
        model_id="openai/clip-vit-base-patch32",
        task="zero-shot-image-classification",
        dataset=DatasetConfig(
            path="uoft-cs/cifar10", split="test", samples=4, shuffle=False,
        ),
    )

    with patch("datasets.load_dataset", return_value=mock_ds), \
         patch("transformers.pipeline", return_value=mock_pipe):
        ev = WinMLZeroShotImageClassificationEvaluator(config, model)

    return ev, mock_pipe


# ---------------------------------------------------------------------------
# WinMLZeroShotImageClassificationEvaluator
# ---------------------------------------------------------------------------


class TestEvaluatorSetup:
    def test_candidate_labels_are_raw_class_names(self):
        """Pipeline's default hypothesis_template wraps candidates; we pass raw names."""
        ev, _ = _make_evaluator()
        assert len(ev._candidate_labels) == 10
        assert ev._candidate_labels[0] == "airplane"
        assert ev._candidate_labels[5] == "dog"

    def test_candidate_labels_match_provided_classes(self):
        ev, _ = _make_evaluator(["cat", "dog", "bird", "frog"])
        assert ev._candidate_labels == ["cat", "dog", "bird", "frog"]

    def test_align_labels_is_noop(self):
        ev, _ = _make_evaluator()
        mock_ds = MagicMock()
        result = ev.align_labels(mock_ds, MagicMock())
        assert result is mock_ds

    def test_schema_has_image_and_label(self):
        schema = WinMLZeroShotImageClassificationEvaluator.schema_info()
        names = [col.name for col in schema]
        assert "image" in names
        assert "label" in names


class TestEvaluatorCompute:
    def test_compute_perfect_accuracy(self):
        ev, _ = _make_evaluator()

        # Pipeline returns correct prediction for each sample
        call_count = [0]

        def side_effect(image, candidate_labels=None):
            current = call_count[0]
            call_count[0] += 1
            gt_label = ev._candidate_labels[current]
            others = [c for c in candidate_labels if c != gt_label]
            return [{"label": gt_label, "score": 0.9}] + [
                {"label": c, "score": 0.01} for c in others
            ]

        ev.pipe = MagicMock(side_effect=side_effect)

        result = ev.compute()
        assert result["top1_accuracy"] == pytest.approx(100.0)
        assert result["top5_accuracy"] == pytest.approx(100.0)

    def test_compute_returns_expected_keys(self):
        ev, mock_pipe = _make_evaluator()

        # Pipeline always returns fixed wrong predictions (raw class names, as HF
        # postprocess returns whatever candidate_labels we passed).
        mock_pipe.return_value = [
            {"label": "truck", "score": 0.9},
            {"label": "ship", "score": 0.05},
        ]

        result = ev.compute()
        assert "top1_accuracy" in result
        assert "top5_accuracy" in result
        assert 0.0 <= result["top1_accuracy"] <= 100.0
        assert 0.0 <= result["top5_accuracy"] <= 100.0
