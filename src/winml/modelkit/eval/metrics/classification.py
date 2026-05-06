# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Classification metrics.

Accuracy and macro-F1 over string labels, for classification evaluators
that do not have an HF evaluate wrapper (e.g. zero-shot-classification).
"""

from __future__ import annotations

from typing import Any


class ClassificationMetric:
    """Accuracy and macro-F1 over string labels."""

    def compute(
        self,
        predictions: list[str],
        references: list[str],
        labels: list[str],
    ) -> dict[str, Any]:
        """Compute accuracy and macro-F1.

        Args:
            predictions: Predicted label strings, one per sample.
            references: Ground-truth label strings, one per sample.
            labels: Full set of class labels for macro-F1 averaging.

        Returns:
            Dict with ``accuracy`` and ``f1`` (both floats in [0, 1]).
        """
        from sklearn.metrics import accuracy_score, f1_score

        if len(predictions) != len(references):
            raise ValueError(
                f"predictions and references must have the same length, "
                f"got {len(predictions)} vs {len(references)}.",
            )
        if not references:
            raise ValueError("references must not be empty.")
        if not labels:
            raise ValueError("labels must not be empty.")

        accuracy = accuracy_score(references, predictions)
        macro_f1 = f1_score(
            references,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        return {"accuracy": float(accuracy), "f1": float(macro_f1)}
