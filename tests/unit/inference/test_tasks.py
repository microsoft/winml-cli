# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for inference/tasks.py postprocess callbacks and helpers.

Covers:
  - _masked_mean_pool
  - _postprocess_segmentation
  - _postprocess_sentence_similarity
  - sentence-similarity registry entry
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from winml.modelkit.inference import TASK_REGISTRY

# Private helpers under test
from winml.modelkit.inference.tasks import (
    _masked_mean_pool,
    _postprocess_segmentation,
    _postprocess_sentence_similarity,
)


# ---------------------------------------------------------------------------
# _masked_mean_pool
# ---------------------------------------------------------------------------


class TestMaskedMeanPool:
    def test_with_mask_excludes_padding(self) -> None:
        """Masked mean should ignore zero-masked positions."""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mask = np.array([1, 1, 0])  # third token is padding
        result = _masked_mean_pool(embeddings, mask)
        expected = np.array([2.0, 3.0])  # mean of first two rows
        np.testing.assert_allclose(result, expected)

    def test_without_mask_uses_all_tokens(self) -> None:
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = _masked_mean_pool(embeddings)
        expected = np.array([3.0, 4.0])
        np.testing.assert_allclose(result, expected)

    def test_all_zeros_mask_falls_back_to_mean(self) -> None:
        """Edge case: all-padding mask should fall back to simple mean."""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([0, 0])
        result = _masked_mean_pool(embeddings, mask)
        expected = np.array([2.0, 3.0])
        np.testing.assert_allclose(result, expected)

    def test_1d_input_without_mask(self) -> None:
        """1D input should be returned as-is."""
        vec = np.array([1.0, 2.0, 3.0])
        result = _masked_mean_pool(vec)
        np.testing.assert_allclose(result, vec)


# ---------------------------------------------------------------------------
# _postprocess_segmentation
# ---------------------------------------------------------------------------


class TestPostprocessSegmentation:
    def test_computes_coverage_and_filters_empty(self) -> None:
        """Masks with zero coverage should be filtered; non-zero sorted by coverage."""
        mask_big = np.ones((10, 10), dtype=np.uint8) * 255  # 100% coverage
        mask_small = np.zeros((10, 10), dtype=np.uint8)
        mask_small[:3, :3] = 255  # 9% coverage
        mask_empty = np.zeros((10, 10), dtype=np.uint8)  # 0% — filtered

        raw = [
            {"label": "shirt", "score": None, "mask": mask_big},
            {"label": "pants", "score": None, "mask": mask_small},
            {"label": "hat", "score": None, "mask": mask_empty},
        ]
        result = _postprocess_segmentation(raw)
        assert len(result) == 2  # empty mask filtered
        assert result[0].label == "shirt"
        assert result[0].score == 1.0
        assert result[1].label == "pants"
        assert result[0].mask is not None  # base64 PNG

    def test_missing_label_uses_unknown(self) -> None:
        """Items without 'label' key should get 'unknown'."""
        mask = np.ones((5, 5), dtype=np.uint8)
        raw = [{"mask": mask}]  # no "label" key
        result = _postprocess_segmentation(raw)
        assert len(result) == 1
        assert result[0].label == "unknown"

    def test_missing_mask_skipped(self) -> None:
        raw = [{"label": "x", "mask": None}]
        result = _postprocess_segmentation(raw)
        assert result == []


# ---------------------------------------------------------------------------
# _postprocess_sentence_similarity
# ---------------------------------------------------------------------------


class TestPostprocessSentenceSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        """Cosine similarity of identical vectors should be ~1.0."""
        vec = [[[1.0, 0.0, 0.0]]]
        raw = [vec, vec]
        result = _postprocess_sentence_similarity(raw)
        assert len(result) == 1
        assert result[0].label == "similarity"
        assert abs(result[0].score - 1.0) < 0.001

    def test_orthogonal_vectors_score_zero(self) -> None:
        vec_a = [[[1.0, 0.0]]]
        vec_b = [[[0.0, 1.0]]]
        result = _postprocess_sentence_similarity([vec_a, vec_b])
        assert abs(result[0].score) < 0.001

    def test_with_tokenizer_masking(self) -> None:
        """When pipeline has a tokenizer, attention mask should be used."""
        pipeline = MagicMock()
        pipeline._preprocess_params = {"padding": "max_length", "max_length": 4, "truncation": True}
        # Two tokens real + two padding
        pipeline.tokenizer.return_value = {
            "attention_mask": np.array([[1, 1, 0, 0]]),
        }
        # 4 tokens of embeddings (2 real + 2 padding noise)
        emb = [[[1.0, 0.0], [1.0, 0.0], [99.0, 99.0], [99.0, 99.0]]]
        raw = [emb, emb]
        inputs = {"text_1": "hello", "text_2": "hello"}
        result = _postprocess_sentence_similarity(raw, pipeline=pipeline, inputs=inputs)
        # Identical real embeddings → similarity ≈ 1.0
        assert result[0].score > 0.99

    def test_short_input_returns_dict(self) -> None:
        """Input with < 2 items should return a fallback dict."""
        result = _postprocess_sentence_similarity([[1.0]])
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# sentence-similarity registry entry
# ---------------------------------------------------------------------------


class TestSentenceSimilarityRegistry:
    def test_registered_with_two_text_fields(self) -> None:
        spec = TASK_REGISTRY.get("sentence-similarity")
        assert spec is not None
        names = [f.name for f in spec.user_inputs]
        assert names == ["text_1", "text_2"]

    def test_has_postprocess_callback(self) -> None:
        spec = TASK_REGISTRY["sentence-similarity"]
        assert spec.postprocess is _postprocess_sentence_similarity

    def test_mapping_is_list_mode(self) -> None:
        spec = TASK_REGISTRY["sentence-similarity"]
        assert spec.mapping.pipe_input_as_list is True


# ---------------------------------------------------------------------------
# image-segmentation registry entry
# ---------------------------------------------------------------------------


class TestSegmentationRegistry:
    def test_has_postprocess_callback(self) -> None:
        spec = TASK_REGISTRY["image-segmentation"]
        assert spec.postprocess is _postprocess_segmentation

    def test_semantic_alias(self) -> None:
        assert TASK_REGISTRY["semantic-segmentation"].postprocess is _postprocess_segmentation
