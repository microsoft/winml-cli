# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Ranking metrics for reranking evaluators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RerankingMetric:
    """Aggregate MRR@K and Recall@K over grouped candidates.

    Ties are broken stably by the candidates' original order. Groups with no
    positive labels are counted separately and excluded from the metric
    denominator so malformed supervision never scores as a silent miss.
    """

    recall_ks: tuple[int, ...] = (1, 10)
    mrr_k: int = 10

    def __post_init__(self) -> None:
        ordered = sorted({int(k) for k in self.recall_ks if int(k) > 0})
        if not ordered:
            raise ValueError("RerankingMetric requires at least one positive Recall@K value.")
        self.recall_ks = tuple(ordered)
        if self.mrr_k <= 0:
            raise ValueError("mrr_k must be positive.")
        self._scored_groups = 0
        self._groups_without_positive = 0
        self._mrr_sum = 0.0
        self._recall_sums = dict.fromkeys(self.recall_ks, 0.0)

    def update(self, scores: list[float], labels: list[bool]) -> None:
        """Update the aggregate with one ranked group."""
        if len(scores) != len(labels):
            raise ValueError("scores and labels must have the same length.")
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        positive_ranks = [rank for rank, index in enumerate(ranked, start=1) if labels[index]]
        if not positive_ranks:
            self._groups_without_positive += 1
            return

        self._scored_groups += 1
        best_rank = positive_ranks[0]
        if best_rank <= self.mrr_k:
            self._mrr_sum += 1.0 / best_rank

        for k in self.recall_ks:
            recalled = sum(rank <= k for rank in positive_ranks)
            self._recall_sums[k] += recalled / len(positive_ranks)

    def compute(self) -> dict[str, float | int]:
        """Return aggregated ranking metrics and accounting."""
        denom = self._scored_groups
        metrics: dict[str, float | int] = {
            f"mrr@{self.mrr_k}": round(self._mrr_sum / denom, 6) if denom else 0.0,
            "scored_groups": denom,
            "groups_without_positive": self._groups_without_positive,
        }
        for k in self.recall_ks:
            metrics[f"recall@{k}"] = round(self._recall_sums[k] / denom, 6) if denom else 0.0
        return metrics
