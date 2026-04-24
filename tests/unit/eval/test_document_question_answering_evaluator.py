# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLDocumentQuestionAnsweringEvaluator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.eval import WinMLDocumentQuestionAnsweringEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_evaluator(columns_mapping=None, dataset_rows=None):
    """Create evaluator without hitting real network or GPU."""
    from winml.modelkit.datasets import DatasetConfig
    from winml.modelkit.eval import WinMLEvaluationConfig

    mapping = columns_mapping or {
        "image_column": "image",
        "question_column": "question",
        "label_column": "answers",
    }

    rows = dataset_rows or [
        {"image": MagicMock(), "question": "What is shown?", "answers": ["a cat"]},
        {"image": MagicMock(), "question": "What color?", "answers": ["red", "crimson"]},
    ]

    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: len(rows)
    mock_ds.__iter__ = lambda self: iter(rows)
    mock_ds.shuffle.return_value = mock_ds
    mock_ds.select.return_value = mock_ds
    # make column access work for prepare_data flattening check
    mock_ds.column_names = list(rows[0].keys())
    mock_ds.__getitem__ = lambda self, key: [r[key] for r in rows]
    mock_ds.map = lambda fn, **kw: mock_ds  # no-op for flat string questions

    model = MagicMock()
    model.config.label2id = None

    config = WinMLEvaluationConfig(
        model_id="test/model",
        task="document-question-answering",
        dataset=DatasetConfig(path="fake/dataset", columns_mapping=mapping),
    )

    with (
        patch("datasets.load_dataset", return_value=mock_ds),
        patch("transformers.pipeline", return_value=MagicMock()),
    ):
        return WinMLDocumentQuestionAnsweringEvaluator(config, model)


# ---------------------------------------------------------------------------
# schema_info
# ---------------------------------------------------------------------------


class TestSchemaInfo:
    def test_returns_three_columns(self):
        schema = WinMLDocumentQuestionAnsweringEvaluator.schema_info()
        assert len(schema) == 3

    def test_column_names(self):
        schema = WinMLDocumentQuestionAnsweringEvaluator.schema_info()
        names = [col.name for col in schema]
        assert names == ["image", "question", "answers"]

    def test_column_overrides(self):
        schema = WinMLDocumentQuestionAnsweringEvaluator.schema_info()
        overrides = [col.override for col in schema]
        assert overrides == ["image_column", "question_column", "label_column"]


# ---------------------------------------------------------------------------
# _levenshtein
# ---------------------------------------------------------------------------


class TestLevenshtein:
    def test_identical_strings(self):
        assert WinMLDocumentQuestionAnsweringEvaluator._levenshtein("abc", "abc") == 0

    def test_empty_first(self):
        assert WinMLDocumentQuestionAnsweringEvaluator._levenshtein("", "abc") == 3

    def test_empty_second(self):
        assert WinMLDocumentQuestionAnsweringEvaluator._levenshtein("abc", "") == 3

    def test_both_empty(self):
        assert WinMLDocumentQuestionAnsweringEvaluator._levenshtein("", "") == 0

    def test_single_substitution(self):
        assert WinMLDocumentQuestionAnsweringEvaluator._levenshtein("cat", "bat") == 1

    def test_case_insensitive(self):
        assert WinMLDocumentQuestionAnsweringEvaluator._levenshtein("Cat", "cat") == 0

    def test_insertion(self):
        assert WinMLDocumentQuestionAnsweringEvaluator._levenshtein("ab", "abc") == 1

    def test_deletion(self):
        assert WinMLDocumentQuestionAnsweringEvaluator._levenshtein("abc", "ab") == 1


# ---------------------------------------------------------------------------
# _max_anls
# ---------------------------------------------------------------------------


class TestMaxAnls:
    def test_exact_match_returns_one(self):
        score = WinMLDocumentQuestionAnsweringEvaluator._max_anls("answer", ["answer"])
        assert score == pytest.approx(1.0)

    def test_both_empty_returns_one(self):
        score = WinMLDocumentQuestionAnsweringEvaluator._max_anls("", [""])
        assert score == pytest.approx(1.0)

    def test_completely_wrong_returns_zero(self):
        # NLS >= 0.5 → score = 0
        score = WinMLDocumentQuestionAnsweringEvaluator._max_anls("xyz", ["abcdefgh"])
        assert score == 0.0

    def test_multiple_refs_uses_best(self):
        # "cat" vs ["dog", "cat"] — should pick exact match
        score = WinMLDocumentQuestionAnsweringEvaluator._max_anls("cat", ["dog", "cat"])
        assert score == pytest.approx(1.0)

    def test_partial_match(self):
        # "cats" vs ["cat"] — levenshtein=1, NLS=1/4=0.25 < 0.5 threshold → score=0.75
        score = WinMLDocumentQuestionAnsweringEvaluator._max_anls("cats", ["cat"])
        assert score == pytest.approx(0.75)

    def test_case_insensitive_match(self):
        # NLS computed on lowercased strings
        score = WinMLDocumentQuestionAnsweringEvaluator._max_anls("CAT", ["cat"])
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# prepare_data: nested query dict flattening
# ---------------------------------------------------------------------------


class TestPrepareData:
    def test_flattens_nested_query_dict(self):
        """When question column holds a dict, the English value is extracted."""
        from winml.modelkit.datasets import DatasetConfig
        from winml.modelkit.eval import WinMLEvaluationConfig

        rows = [
            {
                "image": MagicMock(),
                "query": {"en": "What year?", "de": "Welches Jahr?"},
                "answers": ["2020"],
            },
        ]

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 1
        mock_ds.__iter__ = lambda self: iter(rows)
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_ds.column_names = ["image", "query", "answers"]
        mock_ds.__getitem__ = lambda self, key: [r[key] for r in rows]

        flattened_rows = [
            {"image": rows[0]["image"], "query": "What year?", "answers": ["2020"]},
        ]
        flat_ds = MagicMock()
        flat_ds.__len__ = lambda self: 1
        flat_ds.__iter__ = lambda self: iter(flattened_rows)
        flat_ds.shuffle.return_value = flat_ds
        flat_ds.select.return_value = flat_ds
        flat_ds.column_names = ["image", "query", "answers"]
        flat_ds.__getitem__ = lambda self, key: [r[key] for r in flattened_rows]

        mock_ds.map = MagicMock(return_value=flat_ds)

        model = MagicMock()
        model.config.label2id = None

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="document-question-answering",
            dataset=DatasetConfig(
                path="fake/ds",
                columns_mapping={
                    "image_column": "image",
                    "question_column": "query",
                    "label_column": "answers",
                },
            ),
        )

        with (
            patch("datasets.load_dataset", return_value=mock_ds),
            patch("transformers.pipeline", return_value=MagicMock()),
        ):
            WinMLDocumentQuestionAnsweringEvaluator(config, model)

        # map() was called (flattening happened)
        assert mock_ds.map.called

    def test_flat_string_question_not_modified(self):
        """When question is already a string, map() is not called."""
        from winml.modelkit.datasets import DatasetConfig
        from winml.modelkit.eval import WinMLEvaluationConfig

        rows = [
            {"image": MagicMock(), "question": "Plain string question", "answers": ["yes"]},
        ]

        mock_ds = MagicMock()
        mock_ds.__len__ = lambda self: 1
        mock_ds.__iter__ = lambda self: iter(rows)
        mock_ds.shuffle.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mock_ds.column_names = ["image", "question", "answers"]
        mock_ds.__getitem__ = lambda self, key: [r[key] for r in rows]
        mock_ds.map = MagicMock(return_value=mock_ds)

        model = MagicMock()
        model.config.label2id = None

        config = WinMLEvaluationConfig(
            model_id="test/model",
            task="document-question-answering",
            dataset=DatasetConfig(
                path="fake/ds",
                columns_mapping={
                    "image_column": "image",
                    "question_column": "question",
                    "label_column": "answers",
                },
            ),
        )

        with (
            patch("datasets.load_dataset", return_value=mock_ds),
            patch("transformers.pipeline", return_value=MagicMock()),
        ):
            WinMLDocumentQuestionAnsweringEvaluator(config, model)

        mock_ds.map.assert_not_called()


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------


class TestCompute:
    def test_returns_anls_key(self):
        ev = make_evaluator()
        ev.pipe = MagicMock(return_value=[{"answer": "a cat"}])
        result = ev.compute()
        assert "anls" in result

    def test_exact_prediction_anls_is_one(self):
        rows = [{"image": MagicMock(), "question": "Q?", "answers": ["exact answer"]}]
        ev = make_evaluator(dataset_rows=rows)
        ev.pipe = MagicMock(return_value=[{"answer": "exact answer"}])
        result = ev.compute()
        assert result["anls"] == pytest.approx(1.0)

    def test_wrong_prediction_anls_is_zero(self):
        rows = [{"image": MagicMock(), "question": "Q?", "answers": ["completely different text"]}]
        ev = make_evaluator(dataset_rows=rows)
        ev.pipe = MagicMock(return_value=[{"answer": "xyz"}])
        result = ev.compute()
        assert result["anls"] == 0.0

    def test_averages_over_samples(self):
        rows = [
            {"image": MagicMock(), "question": "Q1?", "answers": ["exact"]},
            {"image": MagicMock(), "question": "Q2?", "answers": ["completely different"]},
        ]
        ev = make_evaluator(dataset_rows=rows)

        # First call: exact match (ANLS=1.0), second: wrong (ANLS=0.0)
        ev.pipe = MagicMock(
            side_effect=[
                [{"answer": "exact"}],
                [{"answer": "xyz"}],
            ]
        )
        result = ev.compute()
        assert result["anls"] == pytest.approx(0.5)

    def test_string_label_treated_as_single_ref(self):
        rows = [{"image": MagicMock(), "question": "Q?", "answers": "the answer"}]
        ev = make_evaluator(dataset_rows=rows)
        ev.pipe = MagicMock(return_value=[{"answer": "the answer"}])
        result = ev.compute()
        assert result["anls"] == pytest.approx(1.0)

    def test_pipe_returns_dict_not_list(self):
        rows = [{"image": MagicMock(), "question": "Q?", "answers": ["cat"]}]
        ev = make_evaluator(dataset_rows=rows)
        ev.pipe = MagicMock(return_value={"answer": "cat"})
        result = ev.compute()
        assert result["anls"] == pytest.approx(1.0)

    def test_empty_dataset_returns_zero(self):
        ev = make_evaluator()
        ev.data = []
        ev.pipe = MagicMock()
        result = ev.compute()
        assert result["anls"] == 0.0


# ---------------------------------------------------------------------------
# Registry and default dataset
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_dqa_evaluator_in_registry(self):
        from winml.modelkit.eval.evaluate import _EVALUATOR_REGISTRY

        assert "document-question-answering" in _EVALUATOR_REGISTRY
        assert (
            _EVALUATOR_REGISTRY["document-question-answering"]
            is WinMLDocumentQuestionAnsweringEvaluator
        )

    def test_dqa_default_dataset_in_registry(self):
        from winml.modelkit.eval.evaluate import _DEFAULT_DATASETS

        assert "document-question-answering" in _DEFAULT_DATASETS
        ds = _DEFAULT_DATASETS["document-question-answering"]
        assert ds.path == "nielsr/docvqa_1200_examples_donut"
        assert "image_column" in ds.columns_mapping
        assert "question_column" in ds.columns_mapping
        assert "label_column" in ds.columns_mapping

    def test_dqa_exported_from_eval_package(self):
        from winml.modelkit import eval as eval_pkg

        assert hasattr(eval_pkg, "WinMLDocumentQuestionAnsweringEvaluator")
        assert (
            eval_pkg.WinMLDocumentQuestionAnsweringEvaluator
            is WinMLDocumentQuestionAnsweringEvaluator
        )
