# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Corpus-level machine-translation metrics."""

from __future__ import annotations

from typing import Any


class TranslationMetric:
    """Aggregate corpus SacreBLEU and chrF over translated sentences."""

    def __init__(self) -> None:
        self._predictions: list[str] = []
        self._references: list[list[str]] = []

    def update(self, prediction: str, references: str | list[str]) -> None:
        """Record one prediction and one or more non-empty references."""
        refs = [references] if isinstance(references, str) else references
        cleaned = [reference.strip() for reference in refs if reference and reference.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty translation reference is required")
        self._predictions.append((prediction or "").strip())
        self._references.append(cleaned)

    def compute(self) -> dict[str, Any]:
        """Return corpus SacreBLEU and chrF scores on the conventional 0-100 scale."""
        if not self._predictions:
            return {"sacrebleu": None, "chrf": None, "n_samples": 0}

        from torchmetrics.text import CHRFScore, SacreBLEUScore

        sacrebleu = SacreBLEUScore(tokenize="13a")(
            self._predictions,
            self._references,
        )
        # Standard chrF2: character n-grams only (word_order=0), beta=2.
        # TorchMetrics otherwise defaults to word_order=2, which is chrF++.
        chrf = CHRFScore(n_word_order=0, beta=2.0)(self._predictions, self._references)

        return {
            "sacrebleu": round(float(sacrebleu) * 100, 4),
            "chrf": round(float(chrf) * 100, 4),
            "n_samples": len(self._predictions),
        }
