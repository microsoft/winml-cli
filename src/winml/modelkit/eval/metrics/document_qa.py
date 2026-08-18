# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Average normalized Levenshtein similarity for document QA."""

from __future__ import annotations

import re
from typing import Any

from rapidfuzz.distance import Levenshtein


def _normalize_text(value: str) -> str:
    """Normalize case and whitespace before scoring."""
    return " ".join(re.sub(r"\s+", " ", value.casefold()).strip().split())


def _normalized_levenshtein_similarity(prediction: str, reference: str) -> float:
    """Return thresholded normalized Levenshtein similarity."""
    prediction = _normalize_text(prediction)
    reference = _normalize_text(reference)
    if not prediction and not reference:
        return 1.0
    denominator = max(len(prediction), len(reference))
    similarity = 1.0 - Levenshtein.distance(prediction, reference) / denominator
    return similarity if similarity >= 0.5 else 0.0


class ANLSMetric:
    """Aggregate best-reference ANLS over document QA predictions."""

    def __init__(self) -> None:
        self._scores: list[float] = []
        self._skipped_samples = 0

    def update(self, prediction: str, references: str | list[str]) -> None:
        """Score one prediction against one or more accepted references."""
        if isinstance(references, str):
            references = [references]
        references = [reference for reference in references if isinstance(reference, str)]
        if not references:
            self._skipped_samples += 1
            return
        self._scores.append(
            max(
                _normalized_levenshtein_similarity(prediction or "", reference)
                for reference in references
            )
        )

    def compute(self) -> dict[str, Any]:
        """Return ANLS and explicit processed/skipped sample counts."""
        return {
            "anls": round(sum(self._scores) / len(self._scores), 4) if self._scores else None,
            "n_samples": len(self._scores),
            "skipped_samples": self._skipped_samples,
        }
