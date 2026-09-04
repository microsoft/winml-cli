# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from winml.modelkit.eval.document_question_answering_evaluator import (
    WinMLDocumentQuestionAnsweringEvaluator,
    _decode_document_span,
    _extract_ocr_words_and_boxes,
    _polygon_to_box,
)
from winml.modelkit.eval.metrics.document_qa import (
    ANLSMetric,
    normalized_levenshtein_similarity,
)


class _Encoding(dict):
    def __init__(self):
        super().__init__(
            input_ids=torch.tensor([[0, 10, 2, 2, 20, 21, 22, 2, 1]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]]),
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
        if not isinstance(other, _Encoding):
            return False
        return (
            self.keys() == other.keys()
            and all(torch.equal(value, other[key]) for key, value in self.items())
            and self._sequence_ids == other._sequence_ids
            and self._word_ids == other._word_ids
        )

    def __ne__(self, other: object) -> bool:
        return not self == other


class _Tokenizer:
    model_max_length = 512
    model_input_names = ("input_ids", "attention_mask")
    sep_token_id = 2

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _Encoding()


class _NativeDocumentQAModel:
    def __init__(self):
        self.inputs = None

    def forward(self, input_ids, bbox, attention_mask, token_type_ids):
        self.inputs = {
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        starts = torch.zeros((1, 9))
        ends = torch.zeros((1, 9))
        starts[0, 4] = 8
        ends[0, 6] = 9
        return QuestionAnsweringModelOutput(start_logits=starts, end_logits=ends)

    __call__ = forward


class _NativeDocumentQAModelWithOptionalTrainingInputs(_NativeDocumentQAModel):
    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        return super().forward(input_ids, bbox, attention_mask, token_type_ids)

    __call__ = forward


def _ocr():
    return {
        "page": 1,
        "width": 200,
        "height": 100,
        "lines": [
            {
                "words": [
                    {"text": "ITC", "bounding_box": [20, 10, 50, 10, 50, 30, 20, 30]},
                    {
                        "text": "Limited",
                        "bounding_box": [60, 10, 100, 10, 100, 30, 60, 30],
                    },
                ]
            }
        ],
    }


def test_polygon_normalization_orders_scales_and_clips():
    assert _polygon_to_box([220, 80, -10, 80, -10, 20, 220, 20], 200, 100) == [0, 200, 1000, 800]


def test_nested_ocr_flattens_words_and_polygons_in_source_order():
    words, boxes = _extract_ocr_words_and_boxes(_ocr())
    assert words == ["ITC", "Limited"]
    assert boxes == [[100, 100, 250, 300], [300, 100, 500, 300]]


@pytest.mark.parametrize(
    "ocr,match",
    [
        ({}, "lines"),
        ({"width": 1, "height": 1, "lines": [{"words": []}]}, "aligned non-empty"),
        (
            {
                "width": 100,
                "height": 100,
                "lines": [{"words": [{"text": "x", "bounding_box": [1, 2]}]}],
            },
            "eight polygon",
        ),
    ],
)
def test_nested_ocr_rejects_missing_or_misaligned_data(ocr, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _extract_ocr_words_and_boxes(ocr)


def test_span_decoder_returns_contiguous_ocr_words_and_honors_limit():
    encoding = _Encoding()
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
        2,
    )
    assert answer == "ITC Limited"


def test_anls_normalization_references_threshold_and_accounting():
    assert normalized_levenshtein_similarity("  ITC   limited ", "itc Limited") == 1.0
    assert normalized_levenshtein_similarity("abc", "xyz") == 0.0
    metric = ANLSMetric()
    metric.update("ITC Limited", ["other", "itc limited"])
    metric.update("unused", [])
    assert metric.compute() == {"anls": 1.0, "n_samples": 1, "skipped_samples": 1}


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("abc", "ab", pytest.approx(2 / 3)),
        ("ab", "a", 0.0),
        ("abcd", "a", 0.0),
    ],
)
def test_anls_uses_strict_normalized_distance_threshold(
    prediction, reference, expected
):
    assert normalized_levenshtein_similarity(prediction, reference) == expected


def test_compute_uses_declared_inputs_and_bounded_single_window():
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    tokenizer = _Tokenizer()
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

    evaluator = object.__new__(WinMLDocumentQuestionAnsweringEvaluator)
    evaluator.model = model
    evaluator.pipe = SimpleNamespace(tokenizer=tokenizer, device="cpu")
    evaluator.data = [
        {
            "question": "What company?",
            "answers": ["ITC Limited"],
            "ocr_results": _ocr(),
        }
    ]
    evaluator.config = WinMLEvaluationConfig(
        model_id="test/model",
        task="document-question-answering",
        dataset=DatasetConfig(path="test/data", samples=1, shuffle=False),
    )

    assert evaluator.compute() == {
        "anls": 1.0,
        "n_samples": 1,
        "skipped_samples": 0,
        "requested_samples": 1,
        "processed_samples": 1,
        "windows_processed": 1,
    }
    assert set(model.call_args.kwargs) == {
        "input_ids",
        "bbox",
        "attention_mask",
        "token_type_ids",
    }
    assert torch.count_nonzero(model.call_args.kwargs["token_type_ids"]) == 0
    assert tokenizer.calls[0][1]["max_length"] == 512
    assert tokenizer.calls[0][1]["return_overflowing_tokens"] is True


def test_compute_discovers_native_model_inputs_from_forward_signature():
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    model = _NativeDocumentQAModel()
    evaluator = object.__new__(WinMLDocumentQuestionAnsweringEvaluator)
    evaluator.model = model
    evaluator.pipe = SimpleNamespace(tokenizer=_Tokenizer(), device="cpu")
    evaluator.data = [
        {
            "question": "What company?",
            "answers": ["ITC Limited"],
            "ocr_results": _ocr(),
        }
    ]
    evaluator.config = WinMLEvaluationConfig(
        model_id="test/model",
        task="document-question-answering",
        runtime="pytorch",
        dataset=DatasetConfig(path="test/data", samples=1, shuffle=False),
    )

    result = evaluator.compute()

    assert result["anls"] == 1.0
    assert set(model.inputs) == {
        "input_ids",
        "bbox",
        "attention_mask",
        "token_type_ids",
    }


def test_compute_excludes_optional_native_training_inputs():
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    model = _NativeDocumentQAModelWithOptionalTrainingInputs()
    evaluator = object.__new__(WinMLDocumentQuestionAnsweringEvaluator)
    evaluator.model = model
    evaluator.pipe = SimpleNamespace(tokenizer=_Tokenizer(), device="cpu")
    evaluator.data = [
        {
            "question": "What company?",
            "answers": ["ITC Limited"],
            "ocr_results": _ocr(),
        }
    ]
    evaluator.config = WinMLEvaluationConfig(
        model_id="test/model",
        task="document-question-answering",
        runtime="pytorch",
        dataset=DatasetConfig(path="test/data", samples=1, shuffle=False),
    )

    result = evaluator.compute()

    assert result["anls"] == 1.0
    assert set(model.inputs) == {
        "input_ids",
        "bbox",
        "attention_mask",
        "token_type_ids",
    }


def test_compute_moves_all_inputs_to_pipeline_device():
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    model = _NativeDocumentQAModel()
    evaluator = object.__new__(WinMLDocumentQuestionAnsweringEvaluator)
    evaluator.model = model
    evaluator.pipe = SimpleNamespace(tokenizer=_Tokenizer(), device="meta")
    evaluator.data = [
        {
            "question": "What company?",
            "answers": ["ITC Limited"],
            "ocr_results": _ocr(),
        }
    ]
    evaluator.config = WinMLEvaluationConfig(
        model_id="test/model",
        task="document-question-answering",
        runtime="pytorch",
        dataset=DatasetConfig(path="test/data", samples=1, shuffle=False),
        _pipeline_device_override="meta",
    )

    evaluator.compute()

    assert {tensor.device.type for tensor in model.inputs.values()} == {"meta"}


def test_compute_rejects_unbounded_top_k():
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    evaluator = object.__new__(WinMLDocumentQuestionAnsweringEvaluator)
    evaluator.model = MagicMock(io_config={"input_names": ["bbox"]})
    evaluator.pipe = SimpleNamespace(tokenizer=_Tokenizer(), device="cpu")
    evaluator.data = []
    evaluator.config = WinMLEvaluationConfig(
        model_id="test/model",
        task="document-question-answering",
        dataset=DatasetConfig(path="test/data", columns_mapping={"top_k": "2"}),
    )
    with pytest.raises(ValueError, match="top_k=1"):
        evaluator.compute()
