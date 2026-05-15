# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLImageFeatureExtractionEvaluator and KNNAccuracyMetric."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from winml.modelkit.eval import KNNAccuracyMetric, WinMLImageFeatureExtractionEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_evaluator(columns_mapping=None):
    """Instantiate evaluator by patching external dependencies."""
    from winml.modelkit.datasets import DatasetConfig
    from winml.modelkit.eval import WinMLEvaluationConfig

    mapping = columns_mapping or {}

    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: 10
    mock_ds.shuffle.return_value = mock_ds
    mock_ds.select.return_value = mock_ds
    mock_ds.column_names = ["image", "label"]

    mock_pipe = MagicMock()
    mock_pipe.image_processor = MagicMock()

    model = MagicMock()
    model.config.label2id = None
    model.io_config = {}

    config = WinMLEvaluationConfig(
        model_id="test/model",
        task="image-feature-extraction",
        dataset=DatasetConfig(path="timm/mini-imagenet", columns_mapping=mapping),
    )

    with patch("datasets.load_dataset", return_value=mock_ds), \
         patch("transformers.pipeline", return_value=mock_pipe):
        return WinMLImageFeatureExtractionEvaluator(config, model)


# ---------------------------------------------------------------------------
# KNNAccuracyMetric
# ---------------------------------------------------------------------------

class TestKNNAccuracyMetric:
    def test_perfect_clusters(self):
        """Embeddings from the same class are identical -> 100% accuracy."""
        metric = KNNAccuracyMetric(k=3)
        # 4 samples, 2 classes. Class 0 at origin-ish, class 1 far away.
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 0.0, 1.0],
            [0.01, 0.0, 0.99],
        ])
        labels = np.array([0, 0, 1, 1])
        result = metric.compute(embeddings, labels)
        assert result["knn_top1_accuracy"] == 100.0
        assert result["knn_top5_accuracy"] == 100.0

    def test_random_embeddings_returns_valid_range(self):
        """Random embeddings should still return accuracy in [0, 100]."""
        rng = np.random.RandomState(42)
        metric = KNNAccuracyMetric(k=5)
        embeddings = rng.randn(50, 32)
        labels = rng.randint(0, 5, size=50)
        result = metric.compute(embeddings, labels)
        assert 0.0 <= result["knn_top1_accuracy"] <= 100.0
        assert 0.0 <= result["knn_top5_accuracy"] <= 100.0

    def test_k_capped_to_n_minus_1(self):
        """k should be capped when larger than N-1."""
        metric = KNNAccuracyMetric(k=100)
        embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ])
        labels = np.array([0, 0, 1])
        # Should not raise, k capped to 2
        result = metric.compute(embeddings, labels)
        assert "knn_top1_accuracy" in result

    def test_too_few_samples_raises(self):
        metric = KNNAccuracyMetric(k=5)
        with pytest.raises(ValueError, match="At least 2 samples"):
            metric.compute(np.array([[1.0]]), np.array([0]))

    def test_k_less_than_1_raises(self):
        metric = KNNAccuracyMetric(k=0)
        with pytest.raises(ValueError, match="k must be >= 1"):
            metric.compute(np.array([[1.0], [2.0]]), np.array([0, 1]))

    def test_returns_float(self):
        metric = KNNAccuracyMetric(k=2)
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        labels = np.array([0, 0, 1, 1])
        result = metric.compute(embeddings, labels)
        assert isinstance(result["knn_top1_accuracy"], float)
        assert isinstance(result["knn_top5_accuracy"], float)

    def test_two_samples_minimal(self):
        """Smallest valid case: two samples."""
        metric = KNNAccuracyMetric(k=1)
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        result = metric.compute(embeddings, labels)
        # With only 1 neighbor each, both predict the other's label
        assert result["knn_top1_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# WinMLImageFeatureExtractionEvaluator
# ---------------------------------------------------------------------------

class TestImageFeatureExtractionEvaluatorSchema:
    def test_schema_has_image_and_label(self):
        schema = WinMLImageFeatureExtractionEvaluator.schema_info()
        names = [col.name for col in schema]
        assert "image" in names
        assert "label" in names

    def test_schema_column_types(self):
        schema = WinMLImageFeatureExtractionEvaluator.schema_info()
        type_map = {col.name: col.type for col in schema}
        assert type_map["image"] == "Image"
        assert type_map["label"] == "ClassLabel"


class TestImageFeatureExtractionEvaluatorInit:
    def test_default_label_column(self):
        evaluator = make_evaluator()
        assert evaluator._label_col == "label"

    def test_custom_label_column(self):
        evaluator = make_evaluator(columns_mapping={"label_column": "category"})
        assert evaluator._label_col == "category"


class TestImageFeatureExtractionEvaluatorAlignLabels:
    def test_align_labels_is_noop(self):
        evaluator = make_evaluator()
        mock_dataset = MagicMock()
        mock_ds_config = MagicMock()
        result = evaluator.align_labels(mock_dataset, mock_ds_config)
        assert result is mock_dataset


class TestExtractImageEmbedding:
    """Tests for `_extract_image_embedding` across supported output shapes."""

    def test_tokens_3d_returns_cls(self):
        # [1, num_tokens, hidden] — ViT/DINOv2 default (pool=False).
        raw = np.arange(1 * 4 * 8, dtype=np.float32).reshape(1, 4, 8)
        out = WinMLImageFeatureExtractionEvaluator._extract_image_embedding(raw)
        assert out.shape == (8,)
        np.testing.assert_array_equal(out, raw[0, 0])

    def test_pooled_2d_returns_vector(self):
        # [1, hidden] — pooled / projected output.
        raw = np.arange(16, dtype=np.float32).reshape(1, 16)
        out = WinMLImageFeatureExtractionEvaluator._extract_image_embedding(raw)
        assert out.shape == (16,)
        np.testing.assert_array_equal(out, raw[0])

    def test_nested_list_input_supported(self):
        # HF pipeline typically returns nested Python lists.
        raw = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]  # [1, 2, 3]
        out = WinMLImageFeatureExtractionEvaluator._extract_image_embedding(raw)
        assert out.shape == (3,)
        np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))

    def test_cnn_feature_map_raises(self):
        # [1, C, H, W] — not supported, surface error instead of silent bad output.
        raw = np.zeros((1, 8, 3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported"):
            WinMLImageFeatureExtractionEvaluator._extract_image_embedding(raw)

    def test_scalar_raises(self):
        raw = [np.float32(1.0)]
        with pytest.raises(ValueError, match="Unsupported"):
            WinMLImageFeatureExtractionEvaluator._extract_image_embedding(raw)


class TestImageFeatureExtractionEvaluatorRegistry:
    def test_registered_in_evaluator_registry(self):
        from winml.modelkit.eval.evaluate import _EVALUATOR_REGISTRY

        assert "image-feature-extraction" in _EVALUATOR_REGISTRY
        assert (
            _EVALUATOR_REGISTRY["image-feature-extraction"]
            is WinMLImageFeatureExtractionEvaluator
        )

    def test_default_dataset_registered(self):
        from winml.modelkit.eval.evaluate import _DEFAULT_DATASETS

        assert "image-feature-extraction" in _DEFAULT_DATASETS
        ds = _DEFAULT_DATASETS["image-feature-extraction"]
        assert ds["path"] == "timm/mini-imagenet"


# ---------------------------------------------------------------------------
# WinMLImageFeatureExtractionEvaluator.compute
# ---------------------------------------------------------------------------

class TestCompute:
    """End-to-end: pipeline output -> CLS extraction -> kNN metric."""

    @staticmethod
    def _token_sequence(cls_vec: list[float], num_tokens: int = 3) -> list:
        """Build a [1, num_tokens, hidden] pipeline output with given CLS vec."""
        hidden = len(cls_vec)
        # Non-CLS tokens are arbitrary — only index 0 should be used.
        other = [[0.5] * hidden for _ in range(num_tokens - 1)]
        return [[cls_vec, *other]]

    def test_end_to_end_flow(self):
        """Pipeline tokens -> CLS extraction -> kNN produces valid accuracies."""
        ev = make_evaluator()

        # Two well-separated clusters, two samples each.
        cluster_a = [1.0, 0.0, 0.0]
        cluster_b = [0.0, 1.0, 0.0]
        ev.data = [
            {"image": "img1", "label": 0},
            {"image": "img2", "label": 0},
            {"image": "img3", "label": 1},
            {"image": "img4", "label": 1},
        ]
        outputs = iter([
            self._token_sequence(cluster_a),
            self._token_sequence([0.99, 0.01, 0.0]),
            self._token_sequence(cluster_b),
            self._token_sequence([0.01, 0.99, 0.0]),
        ])
        ev.pipe = MagicMock(side_effect=lambda _img: next(outputs))

        result = ev.compute()

        assert "knn_top1_accuracy" in result
        assert "knn_top5_accuracy" in result
        # Perfectly separable clusters -> 100% top-1.
        assert result["knn_top1_accuracy"] == 100.0

    def test_skips_samples_with_none_image_or_label(self):
        """Samples missing image or label are dropped before embedding."""
        ev = make_evaluator()

        ev.data = [
            {"image": "img1", "label": 0},
            {"image": None, "label": 0},          # skipped
            {"image": "img2", "label": None},     # skipped
            {"image": "img3", "label": 1},
        ]
        outputs = iter([
            self._token_sequence([1.0, 0.0]),
            self._token_sequence([0.0, 1.0]),
        ])
        ev.pipe = MagicMock(side_effect=lambda _img: next(outputs))

        result = ev.compute()

        # Pipe should only be called for valid samples.
        assert ev.pipe.call_count == 2
        assert "knn_top1_accuracy" in result

    def test_raises_when_fewer_than_two_valid_samples(self):
        """ValueError is raised if <2 valid samples remain after filtering."""
        ev = make_evaluator()

        ev.data = [
            {"image": "img1", "label": 0},
            {"image": None, "label": 0},
        ]
        ev.pipe = MagicMock(
            return_value=self._token_sequence([1.0, 0.0])
        )

        with pytest.raises(ValueError, match="at least 2 valid samples"):
            ev.compute()

