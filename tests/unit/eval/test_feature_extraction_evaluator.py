# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLFeatureExtractionEvaluator and SpearmanCorrelationMetric."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from winml.modelkit.eval import SpearmanCorrelationMetric, WinMLFeatureExtractionEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_evaluator(columns_mapping=None):
    """Instantiate evaluator by patching external dependencies."""
    from winml.modelkit.datasets import DatasetConfig
    from winml.modelkit.eval import WinMLEvaluationConfig

    mapping = columns_mapping or {
        "input_column_1": "sentence1",
        "input_column_2": "sentence2",
        "score_column": "score",
    }

    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: 10
    mock_ds.shuffle.return_value = mock_ds
    mock_ds.select.return_value = mock_ds

    mock_pipe = MagicMock()
    mock_pipe.tokenizer = None
    mock_pipe._preprocess_params = {}

    model = MagicMock()
    model.config.label2id = None
    model.io_config = {}

    config = WinMLEvaluationConfig(
        model_id="test/model",
        task="feature-extraction",
        dataset=DatasetConfig(path="mteb/stsbenchmark-sts", columns_mapping=mapping),
    )

    with patch("datasets.load_dataset", return_value=mock_ds), \
         patch("transformers.pipeline", return_value=mock_pipe):
        return WinMLFeatureExtractionEvaluator(config, model)


# ---------------------------------------------------------------------------
# SpearmanCorrelationMetric
# ---------------------------------------------------------------------------

class TestSpearmanCorrelationMetric:
    def test_perfect_positive_correlation(self):
        metric = SpearmanCorrelationMetric()
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = metric.compute(scores, scores)
        assert result["cosine_spearman"] == pytest.approx(100.0, abs=1e-4)

    def test_perfect_negative_correlation(self):
        metric = SpearmanCorrelationMetric()
        preds = [1.0, 2.0, 3.0, 4.0, 5.0]
        refs = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = metric.compute(preds, refs)
        assert result["cosine_spearman"] == pytest.approx(-100.0, abs=1e-4)

    def test_returns_float(self):
        metric = SpearmanCorrelationMetric()
        result = metric.compute([0.1, 0.5, 0.9], [1.0, 3.0, 5.0])
        assert isinstance(result["cosine_spearman"], float)

    def test_too_few_samples_raises(self):
        metric = SpearmanCorrelationMetric()
        with pytest.raises(ValueError, match="At least 3 samples"):
            metric.compute([0.5, 0.9], [1.0, 5.0])

    def test_result_key_present(self):
        metric = SpearmanCorrelationMetric()
        result = metric.compute([0.1, 0.5, 0.9, 0.3], [1.0, 3.0, 5.0, 2.0])
        assert "cosine_spearman" in result

    def test_score_range_is_0_to_100(self):
        metric = SpearmanCorrelationMetric()
        result = metric.compute([0.1, 0.5, 0.9, 0.3], [1.0, 3.0, 5.0, 2.0])
        assert -100.0 <= result["cosine_spearman"] <= 100.0

    def test_constant_predictions_returns_zero(self):
        """Zero variance in predictions → correlation = 0.0, not NaN."""
        metric = SpearmanCorrelationMetric()
        result = metric.compute([1.0] * 10, [0.5, 1.2, 3.1, 4.0, 2.0, 0.1, 3.5, 4.8, 1.7, 2.9])
        assert result["cosine_spearman"] == 0.0


# ---------------------------------------------------------------------------
# WinMLFeatureExtractionEvaluator._embed
# ---------------------------------------------------------------------------

class TestEmbed:
    def test_masked_mean_pooling_excludes_padding(self):
        """Padding tokens (mask=0) must not contribute to the mean."""
        ev = make_evaluator()
        # 3 tokens: [real, real, padding]
        pipe_output = [[[1.0, 0.0], [3.0, 0.0], [99.0, 99.0]]]  # 3rd is padding
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "attention_mask": np.array([[1, 1, 0]]),  # last token is padding
        }
        ev.pipe = MagicMock(return_value=pipe_output)
        ev.pipe.tokenizer = mock_tokenizer
        ev.pipe._preprocess_params = {}

        embedding = ev._embed("hello world")

        # Mean of only the first two real tokens: (1+3)/2=2, (0+0)/2=0
        np.testing.assert_array_almost_equal(embedding, [2.0, 0.0])

    def test_fallback_simple_mean_without_tokenizer(self):
        """When no tokenizer available, falls back to simple mean over all tokens."""
        ev = make_evaluator()
        pipe_output = [[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
        ev.pipe = MagicMock(return_value=pipe_output)
        ev.pipe.tokenizer = None

        embedding = ev._embed("hello world")

        expected = np.array([2.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(embedding, expected)

    def test_all_real_tokens_same_as_simple_mean(self):
        """When mask is all-ones, masked mean equals simple mean."""
        ev = make_evaluator()
        pipe_output = [[[1.0, 2.0], [3.0, 4.0]]]
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"attention_mask": np.array([[1, 1]])}
        ev.pipe = MagicMock(return_value=pipe_output)
        ev.pipe.tokenizer = mock_tokenizer
        ev.pipe._preprocess_params = {}

        embedding = ev._embed("hello world")
        np.testing.assert_array_almost_equal(embedding, [2.0, 3.0])


# ---------------------------------------------------------------------------
# WinMLFeatureExtractionEvaluator.compute
# ---------------------------------------------------------------------------

class TestComputeSpearman:
    def _make_pipe(self, vectors: list[np.ndarray]):
        """Return a pipeline mock that yields embeddings in sequence."""
        calls = iter([[[v.tolist()]] for v in vectors])
        return MagicMock(side_effect=lambda text: next(calls))

    def test_compute_returns_spearman_key(self):
        ev = make_evaluator()

        # Build a simple 4-row dataset
        ev.data = [
            {"sentence1": "a", "sentence2": "b", "score": 5.0},
            {"sentence1": "c", "sentence2": "d", "score": 1.0},
            {"sentence1": "e", "sentence2": "f", "score": 2.5},
            {"sentence1": "g", "sentence2": "h", "score": 4.0},
        ]

        # Patch _embed to return predictable values
        embeddings = [
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),   # cos=1 -> score=5
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),   # cos=0 -> score=1
            np.array([0.7071, 0.7071]),
            np.array([0.7071, 0.7071]),  # cos=1 -> score=2.5 (middle)
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),   # cos=1 -> score=4
        ]
        emb_iter = iter(embeddings)
        ev._embed = MagicMock(side_effect=lambda _: next(emb_iter))

        result = ev.compute()
        assert "cosine_spearman" in result
        assert -100.0 <= result["cosine_spearman"] <= 100.0

    def test_align_labels_is_noop(self):
        """align_labels returns dataset unchanged (no class labels in STS-B)."""
        ev = make_evaluator()
        mock_ds = MagicMock()
        result = ev.align_labels(mock_ds, MagicMock())
        assert result is mock_ds


# ---------------------------------------------------------------------------
# prepare_pipeline: tokenizer padding
# ---------------------------------------------------------------------------

class TestPreparePipeline:
    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_sets_padding_when_io_config_present(self, mock_load_ds, mock_pipeline):
        from winml.modelkit.datasets import DatasetConfig
        from winml.modelkit.eval import WinMLEvaluationConfig

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 10
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_load_ds.return_value = mock_ds

        mock_pipe = MagicMock()
        mock_pipe.tokenizer = MagicMock()
        mock_pipe._preprocess_params = {}
        mock_pipeline.return_value = mock_pipe

        model = MagicMock()
        model.config.label2id = None
        model.io_config = {"input_shapes": [[1, 256], [1, 256]]}

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="feature-extraction",
            dataset=DatasetConfig(path="mteb/stsbenchmark-sts"),
        )

        WinMLFeatureExtractionEvaluator(config, model)

        assert mock_pipe._preprocess_params["padding"] == "max_length"
        assert mock_pipe._preprocess_params["max_length"] == 256
        assert mock_pipe._preprocess_params["truncation"] is True

    @patch("transformers.pipeline")
    @patch("datasets.load_dataset")
    def test_no_padding_without_tokenizer(self, mock_load_ds, mock_pipeline):
        from winml.modelkit.datasets import DatasetConfig
        from winml.modelkit.eval import WinMLEvaluationConfig

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 10
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_load_ds.return_value = mock_ds

        mock_pipe = MagicMock()
        mock_pipe.tokenizer = None
        mock_pipe._preprocess_params = {}
        mock_pipeline.return_value = mock_pipe

        model = MagicMock()
        model.config.label2id = None
        model.io_config = {"input_shapes": [[1, 256]]}

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="feature-extraction",
            dataset=DatasetConfig(path="mteb/stsbenchmark-sts"),
        )

        WinMLFeatureExtractionEvaluator(config, model)

        assert "padding" not in mock_pipe._preprocess_params
