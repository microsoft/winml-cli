# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from datasets import ClassLabel, Dataset, Features, Value

from winml.modelkit.eval.metrics.classification import ClassificationMetric
from winml.modelkit.eval.zero_shot_classification_evaluator import (
    WinMLZeroShotClassificationEvaluator,
)


class _MockConfig:
    class _Inner:
        id2label = {2: "NEUTRAL", 0: "CONTRADICTION", 1: "ENTAILMENT"}

    config = _Inner()


def _build_nli_dataset() -> Dataset:
    features = Features(
        {
            "premise": Value("string"),
            "hypothesis": Value("string"),
            "label": ClassLabel(names=["entailment", "contradiction", "neutral"]),
        }
    )
    return Dataset.from_dict(
        {
            "premise": ["p1", "p2", "p3"],
            "hypothesis": ["h1", "h2", "h3"],
            "label": [0, 1, 2],
        },
        features=features,
    )


def test_nli_pair_mode_consumes_hypothesis_and_maps_labels() -> None:
    def _fake_compute(
        self: ClassificationMetric,
        predictions: list[str],
        references: list[str],
        label_names: list[str],
    ) -> dict[str, float]:
        del self, label_names
        accuracy = sum(int(p == r) for p, r in zip(predictions, references, strict=False)) / len(
            references
        )
        return {"accuracy": accuracy, "f1": accuracy}

    ClassificationMetric.compute = _fake_compute

    evaluator = WinMLZeroShotClassificationEvaluator.__new__(WinMLZeroShotClassificationEvaluator)
    evaluator.model = _MockConfig()
    evaluator._input_col = "premise"
    evaluator._input_pair_col = "hypothesis"
    evaluator._label_col = "label"
    evaluator._candidate_labels_override = None
    evaluator._hypothesis_template = None
    evaluator.data = _build_nli_dataset()

    calls: list[tuple[str, str]] = []

    def _pipe(text: str, *, text_pair: str, top_k: None = None) -> list[dict[str, float | str]]:
        assert top_k is None
        calls.append((text, text_pair))
        if text_pair == "h1":
            return [{"label": "ENTAILMENT", "score": 0.9}]
        if text_pair == "h2":
            return [{"label": "CONTRADICTION", "score": 0.9}]
        return [{"label": "NEUTRAL", "score": 0.9}]

    evaluator.pipe = _pipe

    result = evaluator.compute()

    assert calls == [("p1", "h1"), ("p2", "h2"), ("p3", "h3")]
    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0


def test_legacy_candidate_labels_mode_unchanged() -> None:
    def _fake_compute(
        self: ClassificationMetric,
        predictions: list[str],
        references: list[str],
        label_names: list[str],
    ) -> dict[str, float]:
        del self, label_names
        accuracy = sum(int(p == r) for p, r in zip(predictions, references, strict=False)) / len(
            references
        )
        return {"accuracy": accuracy, "f1": accuracy}

    ClassificationMetric.compute = _fake_compute

    features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(names=["world", "sports"]),
        }
    )
    evaluator = WinMLZeroShotClassificationEvaluator.__new__(WinMLZeroShotClassificationEvaluator)
    evaluator._input_col = "text"
    evaluator._input_pair_col = None
    evaluator._label_col = "label"
    evaluator._candidate_labels_override = None
    evaluator._hypothesis_template = None
    evaluator.data = Dataset.from_dict(
        {
            "text": ["news1", "news2"],
            "label": [0, 1],
        },
        features=features,
    )

    seen_candidate_labels: list[str] = []

    def _pipe(text: str, **kwargs: object) -> dict[str, list[str]]:
        del text
        seen_candidate_labels.extend(kwargs["candidate_labels"])
        if kwargs["candidate_labels"][0] == "world":
            return {"labels": ["world"]}
        return {"labels": ["sports"]}

    evaluator.pipe = _pipe

    result = evaluator.compute()

    assert seen_candidate_labels[:2] == ["world", "sports"]
    assert result["accuracy"] == 0.5
