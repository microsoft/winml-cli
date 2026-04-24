# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Document question answering evaluator (ANLS metric)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


class WinMLDocumentQuestionAnsweringEvaluator(WinMLEvaluator):
    """Evaluator for document question answering tasks.

    Runs inference through the HF ``document-question-answering`` pipeline and
    computes Average Normalized Levenshtein Similarity (ANLS), the standard
    metric for DocVQA-style benchmarks.

    Dataset schema requires: image, question (or nested dict with ``en`` key),
    and answers (list of accepted answer strings).

    When ``question_column`` points to a nested-dict column (e.g. the
    ``query`` field in ``nielsr/docvqa_1200_examples_donut`` which has per-
    language keys), the evaluator automatically extracts the ``en`` value
    before running inference.
    """

    @classmethod
    def schema_info(cls) -> list:
        """Return expected dataset schema for document question answering."""
        from .config import SchemaColumn

        return [
            SchemaColumn(
                "image", "Image", "image_column", description="PIL Image of document page"
            ),
            SchemaColumn(
                "question",
                "Value(string)",
                "question_column",
                description="question text (or dict with language-code keys)",
            ),
            SchemaColumn(
                "answers",
                "List(Value(string))",
                "label_column",
                description="list of accepted answer strings",
            ),
        ]

    def prepare_data(self) -> Dataset:
        """Load dataset and normalise nested question column to a flat string."""
        dataset = super().prepare_data()

        question_col = self.config.dataset.columns_mapping.get("question_column", "question")
        if question_col not in dataset.column_names:
            return dataset

        # Flatten per-language dict (e.g. {"en": "...", "de": "..."}) to string.
        # Use "en" when present; fall back to the first available value.
        sample_val = dataset[question_col][0] if len(dataset) > 0 else None
        if isinstance(sample_val, dict):
            logger.debug("Flattening nested question column '%s' to English string.", question_col)
            dataset = dataset.map(
                lambda row: {
                    question_col: (
                        row[question_col].get("en") or next(iter(row[question_col].values()), "")
                    )
                }
            )

        return dataset

    def compute(self) -> dict[str, Any]:
        """Run DQA pipeline and return ANLS metric."""
        image_col = self.config.dataset.columns_mapping.get("image_column", "image")
        question_col = self.config.dataset.columns_mapping.get("question_column", "question")
        label_col = self.config.dataset.columns_mapping.get("label_column", "answers")

        logger.info("Running document question answering evaluation...")
        anls_scores: list[float] = []

        for row in self.data:
            result = self.pipe(row[image_col], question=row[question_col])
            # Pipeline returns a list of dicts or a single dict depending on model.
            if isinstance(result, list):
                pred = result[0].get("answer", "")
            else:
                pred = result.get("answer", "")

            refs = row[label_col]
            if isinstance(refs, str):
                refs = [refs]

            anls_scores.append(self._max_anls(str(pred), refs))

        anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0
        logger.info("ANLS: %.4f (over %d samples)", anls, len(anls_scores))
        return {"anls": anls}

    # ------------------------------------------------------------------
    # ANLS helpers (no external dependency required)
    # ------------------------------------------------------------------

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Compute case-insensitive Levenshtein edit distance."""
        s1, s2 = s1.lower(), s2.lower()
        if not s1:
            return len(s2)
        if not s2:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1, 1):
            curr = [i] + [0] * len(s2)
            for j, c2 in enumerate(s2, 1):
                curr[j] = min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (c1 != c2),
                )
            prev = curr
        return prev[-1]

    @classmethod
    def _max_anls(cls, prediction: str, references: list[str], threshold: float = 0.5) -> float:
        """Return the highest ANLS score between *prediction* and any reference.

        ANLS (Average Normalized Levenshtein Similarity):
            NLS  = levenshtein(pred, ref) / max(len(pred), len(ref))
            score = (1 - NLS) if NLS < threshold else 0
        """
        best = 0.0
        pred_s = prediction.strip()
        for ref in references:
            ref_s = ref.strip()
            max_len = max(len(pred_s), len(ref_s))
            if max_len == 0:
                best = max(best, 1.0)
                continue
            nls = cls._levenshtein(pred_s, ref_s) / max_len
            score = 1.0 - nls if nls < threshold else 0.0
            best = max(best, score)
        return best
