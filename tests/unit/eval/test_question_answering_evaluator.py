# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLQuestionAnsweringEvaluator and WinMLModelForQuestionAnswering."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from winml.modelkit.eval import WinMLQuestionAnsweringEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TokenizerStub:
    model_max_length = 512
    model_input_names = ("input_ids", "attention_mask")
    is_fast = True


_DEFAULT_TOKENIZER = object()


def make_evaluator(
    io_config=None,
    columns_mapping=None,
    tokenizer=_DEFAULT_TOKENIZER,
):
    """Create evaluator without triggering __init__ data loading."""
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    mapping = columns_mapping or {
        "question_column": "question",
        "context_column": "context",
        "id_column": "id",
        "label_column": "answers",
    }

    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: 10
    mock_ds.shuffle.return_value = mock_ds
    mock_ds.select.return_value = mock_ds

    model = MagicMock()
    model.config.label2id = None
    model.io_config = io_config or {}
    resolved_tokenizer = _TokenizerStub() if tokenizer is _DEFAULT_TOKENIZER else tokenizer

    config = WinMLEvaluationConfig(
        model_id="test/model",
        task="question-answering",
        dataset=DatasetConfig(path="squad", columns_mapping=mapping),
    )

    with (
        patch("datasets.load_dataset", return_value=mock_ds),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=resolved_tokenizer),
    ):
        return WinMLQuestionAnsweringEvaluator(config, model)


# ---------------------------------------------------------------------------
# prepare_pipeline: tokenizer padding
# ---------------------------------------------------------------------------


class TestPreparePipeline:
    def test_sets_padding_when_io_config_present(self):
        ev = make_evaluator(io_config={"input_shapes": [[1, 384], [1, 384]]})

        assert ev.pipe.tokenizer.model_max_length == 384
        assert ev.pipe._preprocess_params["padding"] == "max_length"
        assert ev.pipe._preprocess_params["max_length"] == 384
        assert ev.pipe._preprocess_params["max_seq_len"] == 384

    def test_raises_when_tokenizer_is_not_fast(self):
        with pytest.raises(ValueError, match="fast tokenizer with offset mappings"):
            make_evaluator(io_config={"input_shapes": [[1, 384]]}, tokenizer=None)

    def test_no_padding_without_shapes(self):
        ev = make_evaluator()

        assert ev.pipe._preprocess_params == {}

    def test_logs_warning_without_shapes(self, caplog):
        with caplog.at_level(logging.WARNING):
            make_evaluator()

        assert any("Could not determine sequence length" in msg for msg in caplog.messages)

    def test_evaluate_accepts_compatibility_pipeline(self):
        from evaluate.evaluator.question_answering import QuestionAnsweringEvaluator

        from winml.modelkit.eval.base_evaluator import _ensure_evaluate_transformers_compat

        ev = make_evaluator()

        _ensure_evaluate_transformers_compat()
        task_evaluator = QuestionAnsweringEvaluator(task="question-answering")
        assert task_evaluator.prepare_pipeline(ev.pipe) is ev.pipe


# ---------------------------------------------------------------------------
# compute: SQuAD v1 vs v2 detection
# ---------------------------------------------------------------------------


class TestCompute:
    def test_squad_v1_uses_squad_metric(self):
        ev = make_evaluator()

        mock_task_evaluator = MagicMock()
        mock_task_evaluator.is_squad_v2_format.return_value = False
        mock_task_evaluator.compute.return_value = {"exact_match": 80.0, "f1": 85.0}

        with patch(
            "evaluate.evaluator.question_answering.QuestionAnsweringEvaluator",
            return_value=mock_task_evaluator,
        ):
            result = ev.compute()

        call_kwargs = mock_task_evaluator.compute.call_args[1]
        assert call_kwargs["metric"] == "squad"
        assert call_kwargs["squad_v2_format"] is False
        assert result["exact_match"] == 80.0
        assert result["f1"] == 85.0

    def test_squad_v2_uses_squad_v2_metric(self):
        ev = make_evaluator()

        mock_task_evaluator = MagicMock()
        mock_task_evaluator.is_squad_v2_format.return_value = True
        mock_task_evaluator.compute.return_value = {"exact": 70.0, "f1": 75.0}

        with patch(
            "evaluate.evaluator.question_answering.QuestionAnsweringEvaluator",
            return_value=mock_task_evaluator,
        ):
            ev.compute()

        call_kwargs = mock_task_evaluator.compute.call_args[1]
        assert call_kwargs["metric"] == "squad_v2"
        assert call_kwargs["squad_v2_format"] is True

    def test_compute_passes_column_mappings(self):
        mapping = {
            "question_column": "q",
            "context_column": "ctx",
            "id_column": "uid",
            "label_column": "ans",
        }
        ev = make_evaluator(columns_mapping=mapping)

        mock_task_evaluator = MagicMock()
        mock_task_evaluator.is_squad_v2_format.return_value = False
        mock_task_evaluator.compute.return_value = {"exact_match": 80.0, "f1": 85.0}

        with patch(
            "evaluate.evaluator.question_answering.QuestionAnsweringEvaluator",
            return_value=mock_task_evaluator,
        ):
            ev.compute()

        call_kwargs = mock_task_evaluator.compute.call_args[1]
        assert call_kwargs["question_column"] == "q"
        assert call_kwargs["context_column"] == "ctx"
        assert call_kwargs["id_column"] == "uid"
        assert call_kwargs["label_column"] == "ans"

    def test_label_col_default_derived_from_schema(self):
        """When label_column is not in columns_mapping, default is 'answers'."""
        ev = make_evaluator(
            columns_mapping={
                "question_column": "question",
                "context_column": "context",
                "id_column": "id",
            }
        )

        mock_task_evaluator = MagicMock()
        mock_task_evaluator.is_squad_v2_format.return_value = False
        mock_task_evaluator.compute.return_value = {"exact_match": 80.0, "f1": 85.0}

        with patch(
            "evaluate.evaluator.question_answering.QuestionAnsweringEvaluator",
            return_value=mock_task_evaluator,
        ):
            ev.compute()

        # is_squad_v2_format should receive the default "answers"
        v2_call_kwargs = mock_task_evaluator.is_squad_v2_format.call_args
        assert v2_call_kwargs[1]["label_column"] == "answers"

    def test_falls_back_to_v1_when_v2_detection_fails(self, caplog):
        """If is_squad_v2_format raises, default to SQuAD v1 with a warning."""
        ev = make_evaluator()

        mock_task_evaluator = MagicMock()
        mock_task_evaluator.is_squad_v2_format.side_effect = KeyError("bad column")
        mock_task_evaluator.compute.return_value = {"exact_match": 80.0, "f1": 85.0}

        import logging

        with (
            caplog.at_level(logging.WARNING),
            patch(
                "evaluate.evaluator.question_answering.QuestionAnsweringEvaluator",
                return_value=mock_task_evaluator,
            ),
        ):
            result = ev.compute()

        call_kwargs = mock_task_evaluator.compute.call_args[1]
        assert call_kwargs["metric"] == "squad"
        assert call_kwargs["squad_v2_format"] is False
        assert any("defaulting to v1" in msg for msg in caplog.messages)
        assert result["exact_match"] == 80.0


# ---------------------------------------------------------------------------
# WinMLModelForQuestionAnswering.forward
# ---------------------------------------------------------------------------


class TestModelForward:
    def _make_model(self, input_names=None, has_token_type_ids=True):
        """Create a WinMLModelForQuestionAnswering with mocked internals."""
        from winml.modelkit.models.winml.question_answering import (
            WinMLModelForQuestionAnswering,
        )

        names = input_names or ["input_ids", "attention_mask"]
        if has_token_type_ids and "token_type_ids" not in names:
            names.append("token_type_ids")

        model = object.__new__(WinMLModelForQuestionAnswering)
        model._session = MagicMock()
        model._session.io_config = {"input_names": names}
        model._format_inputs = MagicMock(side_effect=lambda **kw: kw)
        model._run_inference = MagicMock(
            return_value={
                "start_logits": torch.tensor([[0.1, 0.9, 0.3]]),
                "end_logits": torch.tensor([[0.2, 0.4, 0.8]]),
            }
        )
        return model

    def test_returns_question_answering_output(self):
        model = self._make_model()
        ids = np.array([[1, 2, 3]])
        mask = np.array([[1, 1, 1]])

        result = model.forward(input_ids=ids, attention_mask=mask)

        assert isinstance(result, QuestionAnsweringModelOutput)
        assert result.start_logits is not None
        assert result.end_logits is not None

    def test_passes_input_ids_and_attention_mask(self):
        model = self._make_model()
        ids = np.array([[1, 2, 3]])
        mask = np.array([[1, 1, 1]])

        model.forward(input_ids=ids, attention_mask=mask)

        call_kwargs = model._format_inputs.call_args[1]
        np.testing.assert_array_equal(call_kwargs["input_ids"], ids)
        np.testing.assert_array_equal(call_kwargs["attention_mask"], mask)

    def test_includes_token_type_ids_when_model_accepts(self):
        model = self._make_model(has_token_type_ids=True)
        ids = np.array([[1, 2, 3]])
        mask = np.array([[1, 1, 1]])
        tids = np.array([[0, 0, 1]])

        model.forward(input_ids=ids, attention_mask=mask, token_type_ids=tids)

        call_kwargs = model._format_inputs.call_args[1]
        np.testing.assert_array_equal(call_kwargs["token_type_ids"], tids)

    def test_excludes_token_type_ids_when_model_lacks_input(self):
        model = self._make_model(
            input_names=["input_ids", "attention_mask"],
            has_token_type_ids=False,
        )
        ids = np.array([[1, 2, 3]])
        mask = np.array([[1, 1, 1]])
        tids = np.array([[0, 0, 1]])

        model.forward(input_ids=ids, attention_mask=mask, token_type_ids=tids)

        call_kwargs = model._format_inputs.call_args[1]
        assert "token_type_ids" not in call_kwargs

    def test_token_type_ids_none_not_passed(self):
        model = self._make_model(has_token_type_ids=True)
        ids = np.array([[1, 2, 3]])
        mask = np.array([[1, 1, 1]])

        model.forward(input_ids=ids, attention_mask=mask, token_type_ids=None)

        call_kwargs = model._format_inputs.call_args[1]
        assert "token_type_ids" not in call_kwargs

    def test_raises_when_input_ids_is_none(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="input_ids must be provided"):
            model.forward(input_ids=None)

    def test_extra_kwargs_ignored(self):
        model = self._make_model()
        ids = np.array([[1, 2, 3]])

        result = model.forward(input_ids=ids, some_extra_arg="ignored")

        assert isinstance(result, QuestionAnsweringModelOutput)
