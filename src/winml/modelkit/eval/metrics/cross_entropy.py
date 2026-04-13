# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Cross-entropy and perplexity metric for masked language model evaluation."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


class CrossEntropyMetric:
    """Incremental cross-entropy and perplexity metric for MLM evaluation.

    Accumulates per-token cross-entropy loss across calls to ``update()``,
    then computes mean cross-entropy and perplexity in ``compute()``.

    Typical usage::

        metric = CrossEntropyMetric()
        for logits, labels in predictions:
            metric.update(logits, labels)
        result = metric.compute()
        # {"cross_entropy": 2.19, "perplexity": 8.94}
    """

    def __init__(self) -> None:
        self._total_loss = 0.0
        self._total_tokens = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate cross-entropy loss for one sample's masked positions.

        Args:
            logits: ``[seq_len, vocab_size]`` model output logits.
            labels: ``[seq_len]`` ground-truth token IDs, with ``-100``
                for positions that should be ignored (non-masked tokens).
        """
        mask = labels != -100
        n_masked = mask.sum().item()
        if n_masked == 0:
            return

        loss = F.cross_entropy(logits[mask], labels[mask], reduction="sum")
        self._total_loss += loss.item()
        self._total_tokens += n_masked

    def compute(self) -> dict[str, Any]:
        """Compute mean cross-entropy and perplexity over all accumulated tokens.

        Returns:
            Dict with ``cross_entropy`` (mean NLL per token) and
            ``perplexity`` (exp of mean cross-entropy).

        Raises:
            ValueError: If no masked tokens have been accumulated.
        """
        if self._total_tokens == 0:
            raise ValueError("No masked tokens accumulated; call update() first.")

        mean_ce = self._total_loss / self._total_tokens
        perplexity = math.exp(mean_ce)

        return {
            "cross_entropy": round(mean_ce, 4),
            "perplexity": round(perplexity, 4),
        }

    def reset(self) -> None:
        """Clear accumulated state."""
        self._total_loss = 0.0
        self._total_tokens = 0
