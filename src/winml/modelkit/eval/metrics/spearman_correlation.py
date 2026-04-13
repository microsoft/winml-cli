# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Spearman correlation metric for feature extraction (sentence similarity).

Measures rank correlation between predicted cosine similarities and
ground-truth human similarity scores (e.g. STS-B scores in [0, 5]).
"""

from __future__ import annotations

import logging
import math
from typing import Any


logger = logging.getLogger(__name__)


class SpearmanCorrelationMetric:
    """Spearman rank correlation between predicted and reference scores.

    Typical usage for sentence-similarity evaluation::

        metric = SpearmanCorrelationMetric()
        result = metric.compute(
            predictions=[0.92, 0.41, 0.78, ...],   # cosine similarities
            references=[4.5, 1.2, 3.8, ...],        # GT scores (e.g. STS-B)
        )
        # {"cosine_spearman": 87.0}  -> matches MTEB reporting convention
    """

    def compute(
        self,
        predictions: list[float],
        references: list[float],
    ) -> dict[str, Any]:
        """Compute Spearman rank correlation.

        Args:
            predictions: Predicted similarity scores (e.g. cosine similarities).
            references: Ground-truth similarity scores.

        Returns:
            Dict with ``cosine_spearman`` (float in [-100, 100], MTEB convention).

        Raises:
            ValueError: If fewer than 3 samples are provided (correlation
                is undefined or unreliable with very small N).
        """
        from scipy.stats import spearmanr

        if len(predictions) < 3:
            raise ValueError(
                f"At least 3 samples required for Spearman correlation, "
                f"got {len(predictions)}."
            )

        corr, _ = spearmanr(predictions, references)

        if math.isnan(corr):
            logger.warning(
                "Spearman correlation is NaN. This typically means the model "
                "produced constant outputs (zero variance). Returning 0.0.",
            )
            corr = 0.0

        return {"cosine_spearman": round(float(corr) * 100, 4)}
