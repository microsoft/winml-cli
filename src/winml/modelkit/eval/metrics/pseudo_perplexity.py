# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Pseudo-perplexity and top-k accuracy metrics for fill-mask evaluation.

Pseudo-perplexity (PPPL) measures how well a masked language model predicts
each token in context.  Unlike standard perplexity (which requires a causal LM),
PPPL masks one token at a time and checks the model's prediction probability:

    PPPL = exp( (1/N) * sum( -log P(t_i | context) ) )

Lower PPPL indicates better predictions. Top-k accuracy reports the fraction
of masked tokens where the correct token appears in the model's top-k.
"""

from __future__ import annotations

import math
from typing import Any


class PseudoPerplexityMetric:
    """Pseudo-perplexity and top-k accuracy for masked language models.

    Typical usage::

        metric = PseudoPerplexityMetric()
        result = metric.compute(
            neg_log_likelihoods=[0.3, 1.2, 0.8, ...],
            top1_hits=[True, False, True, ...],
            top5_hits=[True, True, True, ...],
        )
        # {"pseudo_perplexity": 5.2, "accuracy_at_1": 0.67, "accuracy_at_5": 1.0}
    """

    def compute(
        self,
        neg_log_likelihoods: list[float],
        top1_hits: list[bool],
        top5_hits: list[bool],
    ) -> dict[str, Any]:
        """Compute pseudo-perplexity and top-k accuracy.

        Args:
            neg_log_likelihoods: ``-log P(original_token)`` for each masked position.
            top1_hits: Whether the correct token was the top-1 prediction.
            top5_hits: Whether the correct token was in the top-5 predictions.

        Returns:
            Dict with ``pseudo_perplexity``, ``accuracy_at_1``, ``accuracy_at_5``.

        Raises:
            ValueError: If no measurements are provided.
        """
        n = len(neg_log_likelihoods)
        if n == 0:
            raise ValueError("At least 1 measurement required, got 0.")

        mean_nll = sum(neg_log_likelihoods) / n
        pppl = math.exp(mean_nll)
        acc1 = sum(top1_hits) / len(top1_hits) if top1_hits else 0.0
        acc5 = sum(top5_hits) / len(top5_hits) if top5_hits else 0.0

        return {
            "cross_entropy": round(mean_nll, 4),
            "pseudo_perplexity": round(pppl, 4),
            "accuracy_at_1": round(acc1, 4),
            "accuracy_at_5": round(acc5, 4),
        }
