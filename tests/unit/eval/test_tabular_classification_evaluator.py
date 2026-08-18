from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from winml.modelkit.eval import WinMLEvaluationConfig, get_evaluator_class
from winml.modelkit.eval.tabular_classification_evaluator import (
    WinMLTabularClassificationEvaluator,
)
from winml.modelkit.utils.eval_utils import TASK_SCHEMAS


class _FakeModel:
    config = SimpleNamespace()

    def __call__(self, *, features: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=features[:, :1] - 0.5)


def _make_evaluator() -> WinMLTabularClassificationEvaluator:
    evaluator = object.__new__(WinMLTabularClassificationEvaluator)
    evaluator.model = _FakeModel()
    evaluator.pipe = evaluator.model
    evaluator._input_col = "features"
    evaluator._label_col = "label"
    return evaluator


def test_tabular_evaluator_registered_with_schema() -> None:
    assert (
        get_evaluator_class(WinMLEvaluationConfig(task="tabular-classification"))
        is WinMLTabularClassificationEvaluator
    )
    schema = TASK_SCHEMAS["tabular-classification"]
    assert [column.default for column in schema.columns] == ["features", "label"]


def test_compute_uses_binary_logit_semantics() -> None:
    evaluator = _make_evaluator()
    evaluator.data = [
        {"features": [0.25, 2.0], "label": 0},
        {"features": [0.75, 3.0], "label": 1},
    ]

    assert evaluator.compute() == {"accuracy": 1.0, "f1": 1.0, "num_samples": 2}


def test_compute_rejects_non_binary_labels() -> None:
    evaluator = _make_evaluator()
    evaluator.data = [{"features": [0.25, 2.0], "label": 2}]

    with pytest.raises(ValueError, match="0 or 1"):
        evaluator.compute()


def test_compute_accepts_two_class_logits() -> None:
    evaluator = _make_evaluator()
    evaluator.model = lambda **_kwargs: SimpleNamespace(logits=torch.tensor([[0.1, 0.9]]))
    evaluator.data = [{"features": [0.25, 2.0], "label": 1}]

    result = evaluator.compute()

    assert result["accuracy"] == 1.0
