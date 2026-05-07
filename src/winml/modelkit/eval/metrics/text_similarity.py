# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Text similarity metrics for image-to-text evaluation.

Two metrics, computed once at the end of an eval pass:

- **CER** (Character Error Rate): manual Levenshtein over (pred, ref).
  Standard OCR metric; lower is better.  No external deps.
- **CIDEr**: ``pycocoevalcap.cider.cider.Cider`` — TF-IDF n-gram consensus
  across multi-reference captions.  Standard image-captioning metric;
  higher is better.  IDF is computed across the references provided in
  the current call (self-IDF over the eval set).
"""

from __future__ import annotations

import logging
from typing import Any


logger = logging.getLogger(__name__)


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings (DP, O(len(a)*len(b)))."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


class TextSimilarityMetric:
    """Aggregate CER + CIDEr over a stream of (prediction, references) pairs.

    Usage::

        metric = TextSimilarityMetric()
        for pred, refs in samples:        # refs: str or list[str]
            metric.update(pred, refs)
        result = metric.compute()
        # {"cer": 0.034, "cider": 0.789, "n_samples": 100}
    """

    def __init__(self) -> None:
        self._predictions: list[str] = []
        self._references: list[list[str]] = []

    def update(self, prediction: str, references: str | list[str]) -> None:
        """Record one (prediction, reference(s)) pair."""
        if isinstance(references, str):
            references = [references]
        if not references:
            return
        self._predictions.append((prediction or "").strip())
        self._references.append([r for r in references if r])

    def compute(self) -> dict[str, Any]:
        """Return aggregated metrics."""
        n = len(self._predictions)
        if n == 0:
            return {"cer": None, "cider": None, "n_samples": 0}

        return {
            "cer": _round(self._cer()),
            "cider": _round(self._cider()),
            "n_samples": n,
        }

    # --- per-metric helpers -----------------------------------------------

    def _cer(self) -> float:
        """Aggregate CER: total Levenshtein edit distance / total reference chars.

        Multi-reference samples take the *minimum* edit distance across
        references (most lenient, matches HF Evaluate's ``cer`` semantics).
        """
        edits = 0
        ref_chars = 0
        for pred, refs in zip(self._predictions, self._references, strict=True):
            best = min(_levenshtein(pred, r) for r in refs)
            shortest_ref = min(refs, key=len)
            edits += best
            ref_chars += max(len(shortest_ref), 1)
        return edits / ref_chars

    def _cider(self) -> float | None:
        """CIDEr-D via pycocoevalcap. Returns None if dep missing."""
        try:
            from pycocoevalcap.cider.cider import Cider
        except ImportError:
            logger.warning("pycocoevalcap not installed; CIDEr will be None.")
            return None
        try:
            refs_dict = {str(i): refs for i, refs in enumerate(self._references)}
            preds_dict = {str(i): [pred] for i, pred in enumerate(self._predictions)}
            score, _ = Cider().compute_score(refs_dict, preds_dict)
            return float(score)
        except Exception as e:
            logger.warning("CIDEr computation failed: %s", e)
            return None


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None
