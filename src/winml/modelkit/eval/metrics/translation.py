# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Corpus-level machine-translation metrics."""

from __future__ import annotations

from typing import Any


class TranslationMetric:
    """Aggregate corpus SacreBLEU-13a and chrF2 on a 0-100 scale."""

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
        """Return corpus scores with names that state variant and scale."""
        if not self._predictions:
            return {
                "sacrebleu_13a_0_100": None,
                "chrf2_0_100": None,
                "n_samples": 0,
            }

        from torchmetrics.text import CHRFScore, SacreBLEUScore

        sacrebleu = SacreBLEUScore(tokenize="13a")(
            self._predictions,
            self._references,
        )
        chrf = CHRFScore(n_char_order=6, n_word_order=0, beta=2.0)(
            self._predictions,
            self._references,
        )

        return {
            "sacrebleu_13a_0_100": round(float(sacrebleu) * 100, 4),
            "chrf2_0_100": round(float(chrf) * 100, 4),
            "n_samples": len(self._predictions),
        }
