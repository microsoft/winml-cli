# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""End-to-end integration test for zero-shot classification evaluation.

Downloads a small NLI checkpoint and runs the evaluator against a handful
of samples from AG News. Skipped by default via `pytest -m "not slow"`.
"""

from __future__ import annotations

import pytest

from winml.modelkit.datasets.config import DatasetConfig
from winml.modelkit.eval import WinMLZeroShotClassificationEvaluator
from winml.modelkit.eval.config import WinMLEvaluationConfig


# Representative NLI checkpoints across the three families listed in issue #325.
_MODEL_IDS = [
    "typeform/distilbert-base-uncased-mnli",
    "cross-encoder/nli-roberta-base",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
]


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.integration
@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_zero_shot_classification_end_to_end(model_id: str) -> None:
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    config = WinMLEvaluationConfig(
        model_id=model_id,
        task="zero-shot-classification",
        dataset=DatasetConfig(
            path="fancyzhx/ag_news",
            split="test",
            samples=5,
            shuffle=False,
        ),
    )

    results = WinMLZeroShotClassificationEvaluator(config, model).compute()

    assert "accuracy" in results
    assert "f1" in results
    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["f1"] <= 1.0
