# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for WinMLTextGenerationEvaluator (disjoint fixed-length perplexity).

The dataset load and the model's ``encode`` / ``forward`` are stubbed so no
weights or corpora are downloaded.  The tests verify the corpus-blocking
protocol (``num_tokens`` / ``seqlen`` from ``columns_mapping``), the NLL/PPL
math (uniform logits over a vocab of ``V`` give perplexity ``V``), and the
module-level ``_block_nll`` numerator.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from winml.modelkit.eval import WinMLTextGenerationEvaluator
from winml.modelkit.eval.text_generation_evaluator import _block_nll


class _FakeDataset:
    """Minimal stand-in for a HF ``Dataset`` exposing one text column."""

    def __init__(self, rows, column="text") -> None:
        self.column_names = [column]
        self._rows = {column: rows}

    def __getitem__(self, column):
        return self._rows[column]


def _uniform_forward(vocab):
    """Forward returning all-zero (uniform) logits of shape (1, len-1, vocab)."""

    def forward(block):
        out = MagicMock()
        out.logits = np.zeros((1, len(block) - 1, vocab), dtype=np.float32)
        return out

    return forward


def _make_evaluator(
    *,
    corpus_tokens,
    columns_mapping=None,
    corpus_rows=None,
    vocab=50,
    forward=None,
):
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    model = MagicMock()
    model.encode.return_value = corpus_tokens
    model.forward.side_effect = forward or _uniform_forward(vocab)

    fake_ds = _FakeDataset(["dummy"] if corpus_rows is None else corpus_rows)
    config = WinMLEvaluationConfig(
        model_id="test/mock-lm",
        task="text-generation",
        dataset=DatasetConfig(
            path="wikitext",
            name="wikitext-2-raw-v1",
            split="test",
            columns_mapping=columns_mapping or {},
        ),
    )
    with patch("datasets.load_dataset", return_value=fake_ds):
        return WinMLTextGenerationEvaluator(config, model)


class TestPreparePipeline:
    def test_returns_none(self) -> None:
        evaluator = _make_evaluator(corpus_tokens=list(range(10)))
        assert evaluator.pipe is None


class TestPrepareData:
    def test_defaults_from_schema(self) -> None:
        """Empty mapping -> num_tokens=8192, seqlen=2048 (single small block)."""
        evaluator = _make_evaluator(corpus_tokens=list(range(20)))
        assert evaluator._seqlen == 2048
        assert evaluator.data == [list(range(20))]

    def test_blocks_by_seqlen(self) -> None:
        evaluator = _make_evaluator(
            corpus_tokens=list(range(10)),
            columns_mapping={"num_tokens": "10", "seqlen": "4"},
        )
        assert evaluator.data == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]

    def test_truncates_to_num_tokens(self) -> None:
        evaluator = _make_evaluator(
            corpus_tokens=list(range(100)),
            columns_mapping={"num_tokens": "6", "seqlen": "3"},
        )
        assert evaluator.data == [[0, 1, 2], [3, 4, 5]]

    def test_drops_trailing_single_token_block(self) -> None:
        """A trailing block with < 2 tokens can't be scored and is dropped."""
        evaluator = _make_evaluator(
            corpus_tokens=list(range(9)),
            columns_mapping={"num_tokens": "9", "seqlen": "4"},
        )
        assert evaluator.data == [[0, 1, 2, 3], [4, 5, 6, 7]]

    def test_seqlen_below_two_raises(self) -> None:
        with pytest.raises(ValueError, match="seqlen must be at least 2"):
            _make_evaluator(
                corpus_tokens=list(range(10)),
                columns_mapping={"seqlen": "1"},
            )

    def test_no_scorable_blocks_raises(self) -> None:
        with pytest.raises(ValueError, match="no scorable blocks"):
            _make_evaluator(
                corpus_tokens=[0],
                columns_mapping={"num_tokens": "1", "seqlen": "4"},
            )


class TestLoadCorpusTokens:
    def test_missing_input_column_raises(self) -> None:
        with pytest.raises(ValueError, match="no column 'body'"):
            _make_evaluator(
                corpus_tokens=list(range(10)),
                corpus_rows=["a", "b"],
                columns_mapping={"input_column": "body"},
            )

    def test_joins_nonblank_rows_with_double_newline(self) -> None:
        evaluator = _make_evaluator(
            corpus_tokens=list(range(10)),
            corpus_rows=["a", "", "  ", "b"],
        )
        evaluator.model.encode.assert_called_once_with("a\n\nb")


class TestCompute:
    def test_uniform_logits_give_perplexity_equal_vocab(self) -> None:
        vocab = 50
        evaluator = _make_evaluator(
            corpus_tokens=list(range(10)),
            columns_mapping={"num_tokens": "10", "seqlen": "5"},
            vocab=vocab,
        )
        result = evaluator.compute()
        assert result["perplexity"] == pytest.approx(vocab, rel=1e-6)
        assert result["num_scored_positions"] == 8  # (5-1) * 2 blocks
        assert result["num_blocks"] == 2
        assert result["seqlen"] == 5

    def test_one_forward_per_block(self) -> None:
        evaluator = _make_evaluator(
            corpus_tokens=list(range(12)),
            columns_mapping={"num_tokens": "12", "seqlen": "4"},
        )
        evaluator.compute()
        assert evaluator.model.forward.call_count == 3

    def test_zero_scored_positions_raises(self) -> None:
        evaluator = _make_evaluator(corpus_tokens=list(range(10)))
        evaluator.data = []
        with pytest.raises(RuntimeError, match="scored 0 positions"):
            evaluator.compute()


class TestBlockNLL:
    def test_uniform_logits_give_log_vocab_per_position(self) -> None:
        vocab, positions = 20, 3
        logits = np.zeros((positions, vocab), dtype=np.float32)
        targets = np.array([0, 1, 2], dtype=np.int64)
        assert _block_nll(logits, targets) == pytest.approx(positions * math.log(vocab))

    def test_matches_reference_cross_entropy(self) -> None:
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((4, 7)).astype(np.float32)
        targets = np.array([1, 3, 0, 6], dtype=np.int64)

        x = logits.astype(np.float64)
        logsumexp = np.log(np.exp(x - x.max(axis=-1, keepdims=True)).sum(axis=-1))
        logsumexp += x.max(axis=-1)
        expected = float((logsumexp - x[np.arange(len(targets)), targets]).sum())

        assert _block_nll(logits, targets) == pytest.approx(expected)
