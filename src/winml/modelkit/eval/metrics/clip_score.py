# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""CLIPScore metric for text-image alignment.

Standard evaluation metric for text-to-image and image captioning workflows
(Hessel et al., "CLIPScore: A Reference-free Evaluation Metric for Image
Captioning", EMNLP 2021).  For a text-image pair::

    clip_score(text, image) = weight * max(0, cos(t_emb, i_emb))

where ``t_emb`` and ``i_emb`` are CLIP text and image embeddings, and
``weight`` is a fixed multiplier (2.5 in the original paper, keeping the
reported score in a ``[0, 2.5]`` range for typical positive cosines around
``[0, 1]``).

This metric handles the scoring math only.  Callers are responsible for
running whichever CLIP variant they want to obtain the embeddings -- the
metric stays model-agnostic and works for any embedding pair (image-image,
text-text, or cross-modal).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# Hessel et al. (2021) report scores in a ``[0, 2.5]`` range for typical
# positive cosines in ``[0, 1]``; the ``2.5`` multiplier keeps our output
# comparable to numbers in the CLIPScore literature.
_DEFAULT_WEIGHT = 2.5


class CLIPScoreMetric:
    """Cosine-based text-image alignment score with CLIPScore semantics.

    Typical usage::

        metric = CLIPScoreMetric()
        for text_emb, image_emb in embedding_pairs:
            metric.update(text_emb, image_emb)
        result = metric.compute()
        # {"clip_score_mean": 0.75, "clip_score_std": 0.1, ..., "n_samples": 100}

    Attributes:
        weight: Scaling factor applied to each positive cosine (default 2.5,
            matching Hessel et al. 2021).  Set to ``1.0`` to report raw
            positive-cosine values in ``[0, 1]``.
    """

    def __init__(self, weight: float = _DEFAULT_WEIGHT) -> None:
        """Initialize the metric with a scaling weight.

        Args:
            weight: Non-negative, finite multiplier applied to each positive
                cosine similarity.  Defaults to ``2.5`` (Hessel et al.
                convention).

        Raises:
            ValueError: If ``weight`` is negative, ``NaN``, or ``±inf``.
        """
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"weight must be non-negative and finite, got {weight!r}",
            )
        self._weight = float(weight)
        self._scores: list[float] = []

    def update(
        self,
        text_embedding: np.ndarray,
        image_embedding: np.ndarray,
    ) -> None:
        """Record one text-image pair's alignment score.

        Computes ``weight * max(0, cos(text, image))`` and accumulates it.
        Zero-norm inputs (dead embeddings) score 0 -- a dead vector against
        anything else has an undefined angle, so we treat it as no alignment.

        Args:
            text_embedding: 1-D CLIP text embedding (or any shape that
                flattens to 1-D; typically ``(D,)`` or ``(1, D)``).
            image_embedding: 1-D CLIP image embedding of the same total size
                as ``text_embedding``.

        Raises:
            ValueError: If the two embeddings do not share a total size.
        """
        text = np.asarray(text_embedding, dtype=np.float64).ravel()
        image = np.asarray(image_embedding, dtype=np.float64).ravel()
        if text.shape != image.shape:
            raise ValueError(
                f"text/image embedding size mismatch: {text.shape} vs {image.shape}",
            )
        if text.size == 0:
            raise ValueError("embeddings cannot be empty")

        norm_t = float(np.linalg.norm(text))
        norm_i = float(np.linalg.norm(image))
        if norm_t == 0.0 or norm_i == 0.0:
            score = 0.0
        else:
            cos_sim = float(np.dot(text, image) / (norm_t * norm_i))
            score = max(0.0, cos_sim) * self._weight
        self._scores.append(score)

    def compute(self) -> dict[str, Any]:
        """Return aggregate statistics over all recorded pairs.

        Returns:
            Dictionary with ``clip_score_mean``, ``clip_score_std``,
            ``clip_score_min``, ``clip_score_max`` (each rounded to 4
            decimals) and ``n_samples``.  Every stat is ``None`` when no
            samples have been recorded.
        """
        if not self._scores:
            return {
                "clip_score_mean": None,
                "clip_score_std": None,
                "clip_score_min": None,
                "clip_score_max": None,
                "n_samples": 0,
            }
        arr = np.asarray(self._scores, dtype=np.float64)
        return {
            "clip_score_mean": round(float(arr.mean()), 4),
            "clip_score_std": round(float(arr.std()), 4),
            "clip_score_min": round(float(arr.min()), 4),
            "clip_score_max": round(float(arr.max()), 4),
            "n_samples": len(self._scores),
        }

    def reset(self) -> None:
        """Clear all accumulated scores."""
        self._scores = []
