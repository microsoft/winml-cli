# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig
from winml.modelkit.eval.metrics.ranking import RerankingMetric
from winml.modelkit.eval.reranking_evaluator import WinMLRerankingEvaluator
from winml.modelkit.utils.eval_utils import (
    DatasetValidationError,
    detect_reranking_dataset_mode,
)


_FIXTURE_BUILDER_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "e2e_eval"
    / "datasets"
    / "build_msmarco_reranking_fixture.py"
)
_FIXTURE_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_msmarco_reranking_fixture",
    _FIXTURE_BUILDER_PATH,
)
assert _FIXTURE_BUILDER_SPEC is not None
assert _FIXTURE_BUILDER_SPEC.loader is not None
_FIXTURE_BUILDER = importlib.util.module_from_spec(_FIXTURE_BUILDER_SPEC)
sys.modules[_FIXTURE_BUILDER_SPEC.name] = _FIXTURE_BUILDER
_FIXTURE_BUILDER_SPEC.loader.exec_module(_FIXTURE_BUILDER)

CandidateRow = _FIXTURE_BUILDER.CandidateRow
download = _FIXTURE_BUILDER._download
load_top1000 = _FIXTURE_BUILDER._load_top1000
select_rows = _FIXTURE_BUILDER._select_rows


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
    evaluator._positive_col = None
    evaluator._negative_col = None
    evaluator._document_col = "document"
    evaluator._group_col = "group_id"
    evaluator._label_col = "label"
    evaluator._candidate_id_col = "candidate_id"
    evaluator._candidate_text_key = "text"
    evaluator._candidate_id_key = "id"
    evaluator._metadata_group_key = "query_id"
    evaluator._recall_ks = (1, 2, 10)
    evaluator._max_candidates = 10
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


def test_reranking_metric_ties_preserve_authoritative_candidate_order() -> None:
    metric = RerankingMetric(recall_ks=(1, 2, 10))

    metric.update([0.9, 0.9, 0.9], [False, False, True])

    result = metric.compute()

    assert result["mrr@10"] == pytest.approx(1 / 3)
    assert result["recall@1"] == 0.0
    assert result["recall@2"] == 0.0
    assert result["recall@10"] == 1.0


def test_reranking_metric_recall_counts_each_relevant_candidate() -> None:
    metric = RerankingMetric(recall_ks=(1, 2, 3))

    metric.update([0.9, 0.8, 0.1], [True, True, False])

    result = metric.compute()
    assert result["recall@1"] == 0.5
    assert result["recall@2"] == 1.0
    assert result["recall@3"] == 1.0


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


def test_reranking_evaluator_scores_grouped_rows_with_inline_candidates() -> None:
    evaluator = _make_evaluator(
        [
            {
                "query": "what is pcnt",
                "expected_output": ["7187227"],
                "metadata": {"query_id": "1048579", "source_row_index": 1},
                "candidates": [
                    {"id": "n1", "text": "negative passage"},
                    {"id": "7187227", "text": "positive passage"},
                ],
            }
        ],
        scores=[0.1, 0.9],
    )
    evaluator._document_col = None
    evaluator._group_col = None
    evaluator._label_col = None
    evaluator._candidate_id_col = None
    evaluator._candidates_col = "candidates"
    result = evaluator.compute()

    assert result["mrr@10"] == 1.0
    assert result["recall@1"] == 1.0
    assert result["processed_groups"] == 1
    assert result["processed_pairs"] == 2


def test_reranking_evaluator_normalizes_numeric_relevant_ids() -> None:
    evaluator = _make_evaluator(
        [
            {
                "query": "what is pcnt",
                "expected_output": [7187227],
                "metadata": {"query_id": "1048579"},
                "candidates": [{"id": "7187227", "text": "positive passage"}],
            }
        ],
        scores=[0.9],
    )
    evaluator._document_col = None
    evaluator._group_col = None
    evaluator._label_col = None
    evaluator._candidate_id_col = None
    evaluator._candidates_col = "candidates"

    result = evaluator.compute()

    assert result["mrr@10"] == 1.0
    assert result["recall@1"] == 1.0


def test_reranking_evaluator_normalizes_whitespace_in_relevant_ids() -> None:
    evaluator = _make_evaluator(
        [
            {
                "query": "what is pcnt",
                "expected_output": [" 7187227 "],
                "metadata": {"query_id": "1048579"},
                "candidates": [{"id": "7187227", "text": "positive passage"}],
            }
        ],
        scores=[0.9],
    )
    evaluator._document_col = None
    evaluator._group_col = None
    evaluator._label_col = None
    evaluator._candidate_id_col = None
    evaluator._candidates_col = "candidates"

    assert evaluator.compute()["recall@1"] == 1.0


def test_reranking_evaluator_materializes_bounded_positive_and_negative_text() -> None:
    evaluator = _make_evaluator(
        [
            {
                "query": "economic dispatch",
                "positive": ["relevant one", "relevant two"],
                "negative": ["negative one", "negative two", "negative three"],
            }
        ],
        scores=[0.9, 0.8, 0.1],
    )
    evaluator._positive_col = "positive"
    evaluator._negative_col = "negative"
    evaluator._document_col = None
    evaluator._group_col = None
    evaluator._label_col = None
    evaluator._max_candidates = 3
    evaluator.config.dataset.columns_mapping = {
        "query_column": "query",
        "positive_column": "positive",
        "negative_column": "negative",
    }

    groups = evaluator._materialize_groups()
    result = evaluator.compute()

    assert [candidate.candidate_id for candidate in groups[0].candidates] == [
        "0:negative:0",
        "0:positive:0",
        "0:positive:1",
    ]
    assert [candidate.relevant for candidate in groups[0].candidates] == [False, True, True]
    assert result["processed_pairs"] == 3
    assert result["recall@1"] == 0.0


def test_reranking_evaluator_filters_empty_text_before_candidate_cap() -> None:
    evaluator = _make_evaluator(
        [
            {
                "query": "economic dispatch",
                "positive": ["", "relevant"],
                "negative": ["", "negative one", "negative two"],
            }
        ],
        scores=[],
    )
    evaluator._positive_col = "positive"
    evaluator._negative_col = "negative"
    evaluator._document_col = None
    evaluator._group_col = None
    evaluator._label_col = None
    evaluator._max_candidates = 3
    evaluator.config.dataset.columns_mapping = {
        "query_column": "query",
        "positive_column": "positive",
        "negative_column": "negative",
    }

    candidates = evaluator._materialize_groups()[0].candidates

    assert [candidate.text for candidate in candidates] == [
        "negative one",
        "relevant",
        "negative two",
    ]


def test_reranking_dataset_mode_accepts_positive_and_negative_text_lists() -> None:
    mode = detect_reranking_dataset_mode(
        ["query", "positive", "negative"],
        {
            "query_column": "query",
            "positive_column": "positive",
            "negative_column": "negative",
        },
    )

    assert mode == "grouped-text"


def test_reranking_evaluator_does_not_cap_materialized_candidate_column() -> None:
    evaluator = _make_evaluator(
        [
            {
                "query": "what is pcnt",
                "expected_output": ["p1"],
                "metadata": {"query_id": "q1"},
                "candidates": [
                    {"id": "n1", "text": "negative one"},
                    {"id": "n2", "text": "negative two"},
                    {"id": "p1", "text": "positive"},
                ],
            }
        ],
        scores=[0.1, 0.2, 0.9],
    )
    evaluator._document_col = None
    evaluator._group_col = None
    evaluator._label_col = None
    evaluator._candidates_col = "candidates"
    evaluator._max_candidates = 1

    result = evaluator.compute()

    assert result["processed_pairs"] == 3
    assert result["recall@1"] == 1.0


def test_reranking_evaluator_grouped_inline_ties_keep_original_candidate_order() -> None:
    evaluator = _make_evaluator(
        [
            {
                "query": "what is pcnt",
                "expected_output": ["7187227"],
                "metadata": {"query_id": "1048579", "source_row_index": 1},
                "candidates": [
                    {"id": "n1", "text": "negative one"},
                    {"id": "n2", "text": "negative two"},
                    {"id": "7187227", "text": "positive passage"},
                ],
            }
        ],
        scores=[0.5, 0.5, 0.5],
    )
    evaluator._document_col = None
    evaluator._group_col = None
    evaluator._label_col = None
    evaluator._candidate_id_col = None
    evaluator._candidates_col = "candidates"

    result = evaluator.compute()

    assert result["mrr@10"] == pytest.approx(1 / 3)
    assert result["recall@1"] == 0.0
    assert result["recall@2"] == 0.0
    assert result["recall@10"] == 1.0
    assert result["processed_groups"] == 1
    assert result["processed_pairs"] == 3


def test_fixture_builder_preserves_authoritative_order_when_negative_precedes_positive() -> None:
    hf_rows = [
        {
            "input": "what is pcnt",
            "expected_output": ["p1"],
            "metadata": {"query_id": "q1"},
        }
    ]
    queries = {"q1": "what is pcnt"}
    qrels = {"q1": {"p1"}}
    top1000 = {
        "q1": [
            CandidateRow(pid="n1", query="what is pcnt", passage="negative one", rank=1),
            CandidateRow(pid="n2", query="what is pcnt", passage="negative two", rank=2),
            CandidateRow(pid="p1", query="what is pcnt", passage="positive", rank=3),
            CandidateRow(pid="n3", query="what is pcnt", passage="negative three", rank=4),
        ]
    }

    selected_rows, provenance = select_rows(
        hf_rows,
        queries,
        qrels,
        top1000,
        max_queries=1,
        max_negatives=2,
    )

    assert [candidate["id"] for candidate in selected_rows[0]["candidates"]] == ["n1", "n2", "p1"]
    assert [candidate["relevant"] for candidate in selected_rows[0]["candidates"]] == [
        False,
        False,
        True,
    ]
    assert selected_rows[0]["metadata"]["selected_candidate_ids"] == ["n1", "n2", "p1"]
    assert selected_rows[0]["metadata"]["positive_candidate_ids"] == ["p1"]
    assert selected_rows[0]["metadata"]["negative_candidate_ids"] == ["n1", "n2"]
    assert provenance[0]["selected_candidate_ids"] == ["n1", "n2", "p1"]
    assert provenance[0]["candidate_ranks"] == {"n1": 1, "n2": 2, "p1": 3}


def test_fixture_builder_filters_top1000_while_streaming(tmp_path: Path) -> None:
    archive_path = tmp_path / "top1000.dev.tar.gz"
    payload = (
        b"q1\tp1\tquery one\tselected passage\n"
        b"q2\tp2\tquery two\tunselected passage\n"
        b"q1\tp3\tquery one\tsecond selected passage\n"
    )
    info = tarfile.TarInfo("top1000.dev")
    info.size = len(payload)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))

    grouped = load_top1000(archive_path, {"q1"})

    assert list(grouped) == ["q1"]
    assert [candidate.pid for candidate in grouped["q1"]] == ["p1", "p3"]


def test_fixture_download_replaces_invalid_cache_atomically(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "source.bin"
    destination.write_bytes(b"truncated")
    response = io.BytesIO(b"complete")
    monkeypatch.setattr(_FIXTURE_BUILDER.urllib.request, "urlopen", lambda _url: response)

    def validate_complete(path: Path) -> None:
        if path.read_bytes() != b"complete":
            raise ValueError("incomplete source")

    result = download(
        "https://example.test/source.bin",
        destination,
        validator=validate_complete,
    )

    assert result.read_bytes() == b"complete"
    assert not destination.with_name("source.bin.part").exists()


def test_fixture_download_reuses_valid_cache_without_network(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "source.bin"
    destination.write_bytes(b"complete")
    urlopen = MagicMock(side_effect=AssertionError("network should not be used"))
    monkeypatch.setattr(_FIXTURE_BUILDER.urllib.request, "urlopen", urlopen)

    result = download(
        "https://example.test/source.bin",
        destination,
        validator=lambda _path: None,
    )

    assert result == destination
    urlopen.assert_not_called()


def test_fixture_download_failure_publishes_no_partial_file(tmp_path: Path, monkeypatch) -> None:
    class FailingResponse(io.BytesIO):
        def read(self, *_args, **_kwargs):
            raise OSError("connection lost")

    destination = tmp_path / "source.bin"
    destination.write_bytes(b"truncated")
    monkeypatch.setattr(
        _FIXTURE_BUILDER.urllib.request,
        "urlopen",
        lambda _url: FailingResponse(b"partial"),
    )

    def reject_truncated(path: Path) -> None:
        if path.read_bytes() == b"truncated":
            raise ValueError("truncated source")

    with pytest.raises(OSError, match="connection lost"):
        download(
            "https://example.test/source.bin",
            destination,
            validator=reject_truncated,
        )

    assert not destination.exists()
    assert not destination.with_name("source.bin.part").exists()


def test_reranking_dataset_mode_prefers_grouped_inline_candidates() -> None:
    mode = detect_reranking_dataset_mode(
        ["input", "expected_output", "metadata", "candidates"],
        {
            "query_column": "input",
            "expected_output_column": "expected_output",
            "metadata_column": "metadata",
            "candidates_column": "candidates",
        },
    )

    assert mode == "grouped-inline"


def test_reranking_dataset_mode_accepts_pairwise_rows_without_grouped_columns() -> None:
    mode = detect_reranking_dataset_mode(
        ["query", "document", "group_id", "label"],
        {
            "query_column": "query",
            "document_column": "document",
            "group_column": "group_id",
            "label_column": "label",
        },
    )

    assert mode == "pairwise"


def test_reranking_evaluator_rejects_multi_logit_classification_outputs() -> None:
    evaluator = _make_evaluator([], scores=[])
    outputs = SimpleNamespace(logits=torch.tensor([[0.1, 0.9]], dtype=torch.float32))

    with pytest.raises(ValueError, match="exactly one logit"):
        evaluator._extract_relevance_score(outputs)
