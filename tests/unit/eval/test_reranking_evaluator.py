# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig
from winml.modelkit.eval.metrics.ranking import RerankingMetric
from winml.modelkit.eval.reranking_evaluator import WinMLRerankingEvaluator
from winml.modelkit.utils.eval_utils import DatasetValidationError


class _FakeTokenizer:
    def __call__(self, query: str, document: str, **_kwargs):
        width = max(len(query), len(document), 1)
        values = torch.ones((1, min(width, 4)), dtype=torch.int64)
        return {
            "input_ids": values,
            "attention_mask": torch.ones_like(values),
        }

    def pad(self, encoding, **_kwargs):
        return encoding


class _FakeModel:
    def __init__(self, scores: list[float]):
        self._scores = list(scores)
        self.io_config = {"input_shapes": [[1, 4]]}

    def __call__(self, **_kwargs):
        score = self._scores.pop(0)
        return SimpleNamespace(logits=torch.tensor([[score]], dtype=torch.float32))


def _make_evaluator(data, scores: list[float]) -> WinMLRerankingEvaluator:
    evaluator = WinMLRerankingEvaluator.__new__(WinMLRerankingEvaluator)
    evaluator.config = WinMLEvaluationConfig(
        model_id="cross-encoder/ms-marco-MiniLM-L6-v2",
        task="reranking",
        dataset=DatasetConfig(
            path="dummy",
            columns_mapping={
                "query_column": "query",
                "document_column": "document",
                "group_column": "group_id",
                "label_column": "label",
                "candidate_id_column": "candidate_id",
                "recall_ks": "1,2,10",
            },
        ),
    )
    evaluator.model = _FakeModel(scores)
    evaluator.data = data
    evaluator._query_col = "query"
    evaluator._expected_output_col = "expected_output"
    evaluator._metadata_col = "metadata"
    evaluator._candidates_col = None
    evaluator._document_col = "document"
    evaluator._group_col = "group_id"
    evaluator._label_col = "label"
    evaluator._candidate_id_col = "candidate_id"
    evaluator._candidate_text_key = "text"
    evaluator._candidate_id_key = "id"
    evaluator._metadata_group_key = "query_id"
    evaluator._recall_ks = (1, 2, 10)
    evaluator._tokenizer = _FakeTokenizer()
    return evaluator


def test_reranking_metric_handles_ties_and_no_positive_groups() -> None:
    metric = RerankingMetric(recall_ks=(1, 2, 10))
    metric.update([0.9, 0.9, 0.1], [False, True, False])
    metric.update([0.2, 0.1], [False, False])

    result = metric.compute()

    assert result["mrr@10"] == 0.5
    assert result["recall@1"] == 0.0
    assert result["recall@2"] == 1.0
    assert result["groups_without_positive"] == 1
    assert result["scored_groups"] == 1


def test_reranking_evaluator_scores_single_logits_and_accounts_for_groups() -> None:
    evaluator = _make_evaluator(
        [
            {
                "query": "what is pcnt",
                "document": "negative passage",
                "group_id": "q1",
                "label": 0,
                "candidate_id": "n1",
            },
            {
                "query": "what is pcnt",
                "document": "positive passage",
                "group_id": "q1",
                "label": 1,
                "candidate_id": "p1",
            },
            {
                "query": "cost of endless pools/swim spa",
                "document": "positive first hit",
                "group_id": "q2",
                "label": 1,
                "candidate_id": "p2",
            },
        ],
        scores=[0.2, 0.8, 0.7],
    )

    result = evaluator.compute()

    assert result["mrr@10"] == 1.0
    assert result["recall@1"] == 1.0
    assert result["processed_groups"] == 2
    assert result["processed_pairs"] == 3
    assert result["expanded_pairs"] == 3
    assert result["skipped_groups"] == 0


def test_reranking_evaluator_rejects_grouped_rows_without_candidates() -> None:
    evaluator = _make_evaluator(
        [
            {
                "query": "what is pcnt",
                "expected_output": '["7187227"]',
                "metadata": '{"query_id": "q1"}',
            }
        ],
        scores=[],
    )
    evaluator._query_col = "query"
    evaluator._document_col = None
    evaluator._group_col = None
    evaluator._label_col = None

    with pytest.raises(DatasetValidationError, match="candidates_column"):
        evaluator.compute()


def test_reranking_evaluator_rejects_multi_logit_classification_outputs() -> None:
    evaluator = _make_evaluator([], scores=[])
    outputs = SimpleNamespace(logits=torch.tensor([[0.1, 0.9]], dtype=torch.float32))

    with pytest.raises(ValueError, match="exactly one logit"):
        evaluator._extract_relevance_score(outputs)