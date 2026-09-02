# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLQuestionAnsweringEvaluator and WinMLModelForQuestionAnswering."""

from __future__ import annotations

import builtins
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from winml.modelkit.eval import WinMLQuestionAnsweringEvaluator
from winml.modelkit.eval.question_answering_evaluator import (
    _align_token_boxes,
    _decode_document_span,
    _normalize_boxes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TokenizerStub:
    model_max_length = 512
    model_input_names = ("input_ids", "attention_mask")
    is_fast = True


_DEFAULT_TOKENIZER = object()


class _DocumentEncoding(dict):
    def __init__(self):
        super().__init__(
            input_ids=torch.tensor([[0, 10, 2, 2, 20, 21, 22, 2, 1]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]]),
            token_type_ids=torch.zeros((1, 9), dtype=torch.long),
        )
        self._sequence_ids = [None, 0, None, None, 1, 1, 1, None, None]
        self._word_ids = [None, 0, None, None, 0, 1, 1, None, None]

    def sequence_ids(self, feature_index):
        assert feature_index == 0
        return self._sequence_ids

    def word_ids(self, feature_index):
        assert feature_index == 0
        return self._word_ids

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _DocumentEncoding):
            return False
        return (
            self.keys() == other.keys()
            and all(torch.equal(value, other[key]) for key, value in self.items())
            and self._sequence_ids == other._sequence_ids
            and self._word_ids == other._word_ids
        )

    def __ne__(self, other: object) -> bool:
        return not self == other


class _DocumentTokenizer:
    model_max_length = 512
    sep_token_id = 2

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _DocumentEncoding()


def make_evaluator(
    io_config=None,
    columns_mapping=None,
    tokenizer=_DEFAULT_TOKENIZER,
):
    """Create evaluator without triggering __init__ data loading."""
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig
    from winml.modelkit.models.winml.question_answering import (
        WinMLModelForQuestionAnswering,
    )

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

    model = object.__new__(WinMLModelForQuestionAnswering)
    model._session = MagicMock()
    model._session.io_config = io_config or {}
    model._session.device = "cpu"
    model.config = MagicMock()
    model.config.label2id = None
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
        assert ev.pipe._preprocess_params["max_seq_len"] == 384

    def test_raises_when_tokenizer_is_not_fast(self):
        with (
            patch("transformers.__version__", "5.0.0"),
            pytest.raises(ValueError, match="fast tokenizer with offset mappings"),
        ):
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


class TestDocumentHelpers:
    def test_document_encoding_equality_includes_alignment_metadata(self):
        encoding = _DocumentEncoding()
        same = _DocumentEncoding()
        different = _DocumentEncoding()
        different._word_ids[4] = 7

        assert encoding == same
        assert same == encoding
        assert encoding != different
        assert different != encoding

    def test_document_encoding_is_not_equal_to_plain_dict(self):
        encoding = _DocumentEncoding()
        plain = dict(encoding)

        assert encoding != plain
        assert plain != encoding

    def test_normalizes_absolute_boxes_and_clamps(self):
        boxes = _normalize_boxes(
            [[-10, 20, 110, 220], [80, 180, 20, 40]],
            image_size=(100, 200),
            coordinate_system="absolute",
        )

        assert boxes == [[0, 100, 1000, 1000], [200, 200, 800, 900]]

    def test_preserves_normalized_boxes(self):
        assert _normalize_boxes(
            [[10, 20, 30, 40]],
            image_size=None,
            coordinate_system="normalized",
        ) == [[10, 20, 30, 40]]

    def test_auto_scales_sub_1000_pixel_boxes_when_image_size_is_available(self):
        assert _normalize_boxes(
            [[80, 60, 160, 120]],
            image_size=(800, 600),
            coordinate_system="auto",
        ) == [[100, 100, 200, 200]]

    def test_aligns_document_subwords_and_special_tokens(self):
        aligned = _align_token_boxes(
            _DocumentEncoding(),
            0,
            [[10, 20, 30, 40], [50, 60, 70, 80]],
            sep_token_id=2,
        )

        assert aligned[1] == [0, 0, 0, 0]
        assert aligned[2] == [1000, 1000, 1000, 1000]
        assert aligned[4] == [10, 20, 30, 40]
        assert aligned[5:7] == [[50, 60, 70, 80], [50, 60, 70, 80]]

    def test_decodes_highest_scoring_contiguous_word_span(self):
        encoding = _DocumentEncoding()
        starts = np.zeros(9)
        ends = np.zeros(9)
        starts[4] = 8
        ends[6] = 9

        _, answer = _decode_document_span(
            starts,
            ends,
            encoding.sequence_ids(0),
            encoding.word_ids(0),
            ["ITC", "Limited"],
            max_answer_words=2,
        )

        assert answer == "ITC Limited"

    def test_answer_word_limit_excludes_longer_span(self):
        encoding = _DocumentEncoding()
        starts = np.zeros(9)
        ends = np.zeros(9)
        starts[4] = 8
        ends[6] = 9
        starts[5] = 7

        _, answer = _decode_document_span(
            starts,
            ends,
            encoding.sequence_ids(0),
            encoding.word_ids(0),
            ["ITC", "Limited"],
            max_answer_words=1,
        )

        assert answer == "Limited"


class TestDocumentCompute:
    @staticmethod
    def _make_document_evaluator(row):
        from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

        tokenizer = _DocumentTokenizer()
        model = MagicMock()
        model.io_config = {
            "input_names": ["input_ids", "bbox", "attention_mask", "token_type_ids"],
            "input_shapes": [[1, 512], [1, 512, 4], [1, 512], [1, 512]],
        }
        starts = torch.zeros((1, 9))
        ends = torch.zeros((1, 9))
        starts[0, 4] = 8
        ends[0, 6] = 9
        model.return_value = QuestionAnsweringModelOutput(start_logits=starts, end_logits=ends)

        evaluator = object.__new__(WinMLQuestionAnsweringEvaluator)
        evaluator.model = model
        evaluator.data = [row]
        evaluator.pipe = SimpleNamespace(tokenizer=tokenizer, device="cpu")
        evaluator.config = WinMLEvaluationConfig(
            model_id="test/document-model",
            task="question-answering",
            dataset=DatasetConfig(
                path="test/document-dataset",
                columns_mapping={
                    "document_mode": "true",
                    "max_windows": "1",
                    "doc_stride": "128",
                    "max_answer_words": "64",
                    "top_k": "1",
                },
            ),
        )
        return evaluator, tokenizer, model

    def test_precomputed_document_path_forwards_exact_named_inputs(self):
        evaluator, tokenizer, model = self._make_document_evaluator(
            {
                "question": "What is the company?",
                "words": ["ITC", "Limited"],
                "boxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
                "answers": ["ITC Limited"],
            }
        )

        with patch.object(
            WinMLQuestionAnsweringEvaluator,
            "_require_tesseract_runtime",
            side_effect=AssertionError("precomputed data must not require native OCR"),
        ):
            result = evaluator.compute()

        assert result == {
            "anls": 1.0,
            "n_samples": 1,
            "skipped_samples": 0,
            "windows_processed": 1,
            "ocr_samples": 0,
            "precomputed_samples": 1,
        }
        assert set(model.call_args.kwargs) == {
            "input_ids",
            "bbox",
            "attention_mask",
            "token_type_ids",
        }
        assert model.call_args.kwargs["bbox"].shape == (1, 9, 4)
        _, tokenizer_kwargs = tokenizer.calls[0]
        assert tokenizer_kwargs["max_length"] == 512
        assert tokenizer_kwargs["stride"] == 128
        assert tokenizer_kwargs["return_overflowing_tokens"] is True

    def test_missing_token_type_ids_are_forwarded_as_zeros_when_declared(self):
        evaluator, tokenizer, model = self._make_document_evaluator(
            {
                "question": "What is the company?",
                "words": ["ITC", "Limited"],
                "boxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
                "answers": ["ITC Limited"],
            }
        )
        encoding = _DocumentEncoding()
        del encoding["token_type_ids"]
        tokenizer.__call__ = MagicMock(return_value=encoding)

        with patch.object(_DocumentTokenizer, "__call__", return_value=encoding):
            evaluator.compute()

        torch.testing.assert_close(
            model.call_args.kwargs["token_type_ids"],
            torch.zeros((1, 9), dtype=torch.long),
        )

    def test_declared_image_input_is_rejected_before_document_inference(self):
        evaluator, _, model = self._make_document_evaluator({})
        model.io_config["input_names"].append("pixel_values")

        with pytest.raises(
            ValueError,
            match=r"text-and-layout model inputs.*cannot produce.*pixel_values",
        ):
            evaluator.compute()

        model.assert_not_called()

    def test_sample_index_selects_pinned_document_row(self):
        evaluator, _, _ = self._make_document_evaluator({})
        evaluator.config.dataset.columns_mapping["sample_index"] = "2"
        dataset = MagicMock()
        dataset.__len__.return_value = 3
        selected = MagicMock()
        dataset.select.return_value = selected

        with patch(
            "winml.modelkit.eval.base_evaluator.WinMLEvaluator.prepare_data",
            return_value=dataset,
        ):
            result = evaluator.prepare_data()

        assert result is selected
        dataset.select.assert_called_once_with([2])
        assert evaluator.config.dataset.samples == 100
        assert evaluator.config.dataset.shuffle is True

    def test_image_path_uses_one_ocr_pass_and_normalizes_boxes(self):
        evaluator, _, _ = self._make_document_evaluator({})
        evaluator.config.dataset.columns_mapping = {"document_mode": "true"}
        image = SimpleNamespace(size=(200, 100))
        image_to_data = MagicMock(
            return_value={
                "text": ["", "ITC", "Limited"],
                "left": [0, 20, 60],
                "top": [0, 10, 10],
                "width": [0, 30, 40],
                "height": [0, 20, 20],
            }
        )
        pytesseract = SimpleNamespace(
            image_to_data=image_to_data,
            get_tesseract_version=MagicMock(return_value="5.5.3"),
            Output=SimpleNamespace(DICT="dict"),
            TesseractNotFoundError=RuntimeError,
        )

        with patch.dict(sys.modules, {"pytesseract": pytesseract}):
            words, boxes, source = evaluator._document_words_and_boxes({"image": image})

        assert image_to_data.call_count == 1
        assert words == ["ITC", "Limited"]
        assert boxes == [[100, 100, 250, 300], [300, 100, 500, 300]]
        assert source == "tesseract"
        assert evaluator._last_ocr_version == "5.5.3"

    def test_missing_native_tesseract_fails_during_data_validation(self):
        evaluator, _, _ = self._make_document_evaluator({})
        evaluator.config.dataset.columns_mapping = {"document_mode": "true"}
        not_found = type("TesseractNotFoundError", (Exception,), {})
        pytesseract = SimpleNamespace(
            get_tesseract_version=MagicMock(side_effect=not_found()),
            TesseractNotFoundError=not_found,
        )
        dataset = SimpleNamespace(column_names=["question", "answers", "image"])

        with (
            patch.dict(sys.modules, {"pytesseract": pytesseract}),
            pytest.raises(RuntimeError, match=r"tesseract --version.*winget install"),
        ):
            evaluator.validate_data(dataset)

    def test_image_compute_records_native_tesseract_version(self):
        evaluator, _, _ = self._make_document_evaluator(
            {
                "question": "What is the company?",
                "image": SimpleNamespace(size=(200, 100)),
                "answers": ["ITC Limited"],
            }
        )
        evaluator.config.dataset.columns_mapping = {"document_mode": "true"}
        pytesseract = SimpleNamespace(
            image_to_data=MagicMock(
                return_value={
                    "text": ["ITC", "Limited"],
                    "left": [20, 60],
                    "top": [10, 10],
                    "width": [30, 40],
                    "height": [20, 20],
                }
            ),
            get_tesseract_version=MagicMock(return_value="5.5.3"),
            Output=SimpleNamespace(DICT="dict"),
            TesseractNotFoundError=RuntimeError,
        )

        with patch.dict(sys.modules, {"pytesseract": pytesseract}):
            result = evaluator.compute()

        assert result["ocr_engine_version"] == "5.5.3"
        assert result["ocr_samples"] == 1

    def test_unsupported_native_tesseract_version_is_rejected(self):
        evaluator, _, _ = self._make_document_evaluator({})
        pytesseract = SimpleNamespace(
            get_tesseract_version=MagicMock(return_value="4.1.3"),
            TesseractNotFoundError=RuntimeError,
        )

        with (
            patch.dict(sys.modules, {"pytesseract": pytesseract}),
            pytest.raises(RuntimeError, match=r"4\.1\.3 is unsupported.*5\.x is required"),
        ):
            evaluator._require_tesseract_runtime()

    def test_missing_python_ocr_extra_has_actionable_error(self):
        evaluator, _, _ = self._make_document_evaluator({})
        evaluator.config.dataset.columns_mapping = {"document_mode": "true"}
        original_import = builtins.__import__

        def import_without_pytesseract(name, *args, **kwargs):
            if name == "pytesseract":
                raise ImportError(name)
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=import_without_pytesseract),
            pytest.raises(RuntimeError, match=r"Install winml-cli\[document\]"),
        ):
            evaluator._document_words_and_boxes({"image": SimpleNamespace(size=(10, 10))})

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

    def test_includes_bbox_when_model_accepts(self):
        model = self._make_model(input_names=["input_ids", "attention_mask", "bbox"])
        ids = np.array([[1, 2, 3]])
        boxes = np.array([[[0, 0, 0, 0], [10, 20, 30, 40], [1000, 1000, 1000, 1000]]])

        model.forward(input_ids=ids, bbox=boxes)

        call_kwargs = model._format_inputs.call_args[1]
        np.testing.assert_array_equal(call_kwargs["bbox"], boxes)

    def test_excludes_bbox_when_model_lacks_input(self):
        model = self._make_model(
            input_names=["input_ids", "attention_mask"],
            has_token_type_ids=False,
        )
        ids = np.array([[1, 2, 3]])
        boxes = np.zeros((1, 3, 4), dtype=np.int64)

        model.forward(input_ids=ids, bbox=boxes)

        assert "bbox" not in model._format_inputs.call_args[1]

    def test_raises_when_input_ids_is_none(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="input_ids must be provided"):
            model.forward(input_ids=None)

    def test_extra_kwargs_ignored(self):
        model = self._make_model()
        ids = np.array([[1, 2, 3]])

        result = model.forward(input_ids=ids, some_extra_arg="ignored")

        assert isinstance(result, QuestionAnsweringModelOutput)
