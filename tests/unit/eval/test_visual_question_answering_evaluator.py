# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for fixed-vocabulary visual-question-answering evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig
from winml.modelkit.eval.visual_question_answering_evaluator import (
    WinMLVisualQuestionAnsweringEvaluator,
    normalize_vqa_answer,
    vqa_soft_accuracy,
)


def _make_evaluator(columns_mapping=None) -> WinMLVisualQuestionAnsweringEvaluator:
    mapping = columns_mapping or {}
    dataset = MagicMock()
    dataset.__len__ = lambda _self: 0
    dataset.shuffle.return_value = dataset
    dataset.select.return_value = dataset
    dataset.column_names = [
        mapping.get("input_column", "image"),
        mapping.get("question_column", "question"),
        mapping.get("label_column", "answers"),
    ]
    config = WinMLEvaluationConfig(
        model_id="test/model",
        task="visual-question-answering",
        dataset=DatasetConfig(path="test/data", columns_mapping=mapping),
    )
    with (
        patch("datasets.load_dataset", return_value=dataset),
        patch(
            "winml.modelkit.inference.pipeline.create_pipeline",
            return_value=MagicMock(),
        ),
    ):
        return WinMLVisualQuestionAnsweringEvaluator(config, MagicMock())


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("The TWO, cats!", "2 cats"),
        ("dont", "don't"),
        ("3.14", "3.14"),
        ("3,000", "3000"),
    ],
)
def test_official_answer_normalization(answer: str, expected: str) -> None:
    assert normalize_vqa_answer(answer) == expected


@pytest.mark.parametrize(
    ("matches", "expected"),
    [(0, 0.0), (1, 1 / 3), (2, 2 / 3), (3, 1.0), (10, 1.0)],
)
def test_soft_accuracy(matches: int, expected: float) -> None:
    answers = ["two"] * matches + ["other"] * (10 - matches)
    assert vqa_soft_accuracy("2", answers) == pytest.approx(expected)


def test_compute_uses_top_answer_and_reports_rows() -> None:
    evaluator = _make_evaluator()
    evaluator.data = [
        {
            "image": Image.new("RGB", (640, 512)),
            "question": "Where?",
            "answers": [{"answer": "down"}] * 3 + [{"answer": "up"}] * 7,
        }
    ]
    evaluator.pipe = MagicMock(return_value=[{"answer": "down", "score": 0.8}])

    result = evaluator.compute()

    assert result["vqa_accuracy"] == 1.0
    assert result["n_samples"] == 1
    assert result["skipped"] == 0
    assert result["predictions"] == [
        {"sample_index": 0, "prediction": "down", "score": 1.0}
    ]


def test_custom_columns_and_skipped_rows() -> None:
    evaluator = _make_evaluator(
        {"input_column": "img", "question_column": "q", "label_column": "labels"}
    )
    evaluator.data = [
        {"img": None, "q": "Skipped?", "labels": ["yes"] * 10},
        {"img": Image.new("RGB", (4, 4)), "q": "Kept?", "labels": {"answer": ["yes"] * 10}},
    ]
    evaluator.pipe = MagicMock(return_value=[{"answer": "yes"}])

    result = evaluator.compute()

    assert result["n_samples"] == 1
    assert result["skipped"] == 1


def test_zero_usable_rows_is_failure() -> None:
    evaluator = _make_evaluator()
    evaluator.data = [{"image": None, "question": "No image", "answers": []}]
    with pytest.raises(ValueError, match="No usable"):
        evaluator.compute()


def test_default_dataset_is_pinned_deterministic_streaming() -> None:
    from winml.modelkit.eval.evaluate import _DEFAULT_DATASETS

    default = _DEFAULT_DATASETS["visual-question-answering"]
    assert default["path"] == "lmms-lab-encoder/VQAv2"
    assert default["revision"] == "32665d35052eb4a6d4414851c3c829a72754915a"
    assert default["split"] == "validation"
    assert default["streaming"] is True
    assert default["shuffle"] is False
