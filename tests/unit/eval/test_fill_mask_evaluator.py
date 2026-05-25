# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLFillMaskEvaluator (pseudo-perplexity)."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from winml.modelkit.eval import WinMLFillMaskEvaluator


_SPECIAL_IDS = {101, 102, 0}  # CLS, SEP, PAD for the mock tokenizer


def _make_tokenizer(vocab_size=50, pad_token_id=0, mask_token_id=103):
    tok = MagicMock()
    tok.pad_token_id = pad_token_id
    tok.mask_token_id = mask_token_id
    tok.mask_token = "[MASK]"  # noqa: S105
    tok.pad_token = "[PAD]"  # noqa: S105
    tok.eos_token = None
    tok.get_special_tokens_mask = lambda ids, already_has_special_tokens=True: [
        1 if tid in _SPECIAL_IDS else 0 for tid in ids
    ]
    return tok


def _make_evaluator(model=None, max_length=None):
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig

    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: 5
    mock_ds.shuffle.return_value = mock_ds
    mock_ds.select.return_value = mock_ds

    if model is None:
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {"input_shapes": [[1, max_length]]} if max_length else {}

    config = WinMLEvaluationConfig(
        model_id="test/mock-bert",
        task="fill-mask",
        dataset=DatasetConfig(
            path="Salesforce/wikitext",
            name="wikitext-2-raw-v1",
            columns_mapping={"input_column": "text"},
        ),
    )

    with patch("datasets.load_dataset", return_value=mock_ds), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=_make_tokenizer()):
        return WinMLFillMaskEvaluator(config, model)


class TestLifecycle:
    def test_prepare_pipeline_returns_none(self) -> None:
        assert _make_evaluator().pipe is None

    def test_tokenizer_loaded_in_init(self) -> None:
        evaluator = _make_evaluator()
        assert evaluator._tokenizer.mask_token_id == 103

    def test_align_labels_is_noop(self) -> None:
        evaluator = _make_evaluator()
        ds = MagicMock()
        assert evaluator.align_labels(ds, MagicMock()) is ds


class TestMaxLength:
    def test_fixed(self) -> None:
        assert _make_evaluator(max_length=128)._max_length() == 128

    def test_dynamic_string_dims(self) -> None:
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {"input_shapes": [["batch", "seq"]]}
        assert _make_evaluator(model=model)._max_length() is None

    def test_missing_io_config(self) -> None:
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {}
        assert _make_evaluator(model=model)._max_length() is None


class TestLogits:
    def test_dict_with_logits_key(self) -> None:
        logits = torch.randn(1, 5, 50)
        assert torch.equal(
            _make_evaluator()._logits({"logits": logits, "aux": None}), logits,
        )

    def test_dict_without_logits_key_raises(self) -> None:
        logits = torch.randn(1, 5, 50)
        with pytest.raises(KeyError, match="no 'logits' key"):
            _make_evaluator()._logits({"output": logits})

    def test_dataclass(self) -> None:
        logits = torch.randn(1, 5, 50)
        out = MagicMock()
        out.logits = logits
        assert torch.equal(_make_evaluator()._logits(out), logits)


class TestScore:
    def test_one_forward_per_position(self) -> None:
        evaluator = _make_evaluator(max_length=5)
        evaluator.model = MagicMock(return_value={"logits": torch.zeros(1, 5, 50)})
        encoding = {"input_ids": torch.tensor([[101, 10, 20, 30, 102]])}
        scores = evaluator._score(encoding, [1, 2, 3])
        assert scores.shape == (3,)
        assert evaluator.model.call_count == 3

    def test_masks_each_position_in_turn(self) -> None:
        evaluator = _make_evaluator(max_length=5)
        seen_masks: list[list[int]] = []

        def forward(**kwargs):
            ids = kwargs["input_ids"][0].tolist()
            mask_id = evaluator._tokenizer.mask_token_id
            seen_masks.append([i for i, tid in enumerate(ids) if tid == mask_id])
            return {"logits": torch.zeros(1, 5, 50)}

        evaluator.model = MagicMock(side_effect=forward)
        encoding = {"input_ids": torch.tensor([[101, 10, 20, 30, 102]])}
        evaluator._score(encoding, [1, 2, 3])
        assert seen_masks == [[1], [2], [3]]

    def test_restores_input_ids(self) -> None:
        evaluator = _make_evaluator(max_length=5)
        evaluator.model = MagicMock(return_value={"logits": torch.zeros(1, 5, 50)})
        encoding = {"input_ids": torch.tensor([[101, 10, 20, 30, 102]])}
        before = encoding["input_ids"].clone()
        evaluator._score(encoding, [1, 2, 3])
        assert torch.equal(encoding["input_ids"], before)

    def test_uniform_logits_yield_minus_log_vocab(self) -> None:
        vocab = 50
        evaluator = _make_evaluator(max_length=5)
        evaluator.model = MagicMock(return_value={"logits": torch.zeros(1, 5, vocab)})
        encoding = {"input_ids": torch.tensor([[101, 10, 20, 30, 102]])}
        scores = evaluator._score(encoding, [1, 2, 3])
        assert torch.allclose(scores, torch.full((3,), -math.log(vocab)))

    def test_empty_positions_no_forward(self) -> None:
        evaluator = _make_evaluator(max_length=4)
        evaluator.model = MagicMock()
        scores = evaluator._score({"input_ids": torch.tensor([[101, 102, 0, 0]])}, [])
        assert scores.numel() == 0
        evaluator.model.assert_not_called()


class TestCompute:
    def test_uniform_logits_give_pppl_equal_vocab(self) -> None:
        vocab, seq_len = 50, 8
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {"input_shapes": [[1, seq_len]]}
        model.return_value = {"logits": torch.zeros(1, seq_len, vocab)}

        evaluator = _make_evaluator(model=model, max_length=seq_len)
        evaluator._tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 10, 20, 30, 40, 102, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
        }
        evaluator.data = [
            {"text": "hello"},
            {"text": "world"},
            {"text": ""},  # skipped
        ]
        result = evaluator.compute()
        assert result["nll"] == pytest.approx(math.log(vocab), abs=1e-3)
        assert result["pseudo_perplexity"] == pytest.approx(vocab, rel=1e-3)

    def test_only_real_tokens_scored(self) -> None:
        """Special tokens and padding must not trigger forward passes."""
        vocab, seq_len = 50, 6
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {"input_shapes": [[1, seq_len]]}
        model.return_value = {"logits": torch.zeros(1, seq_len, vocab)}

        evaluator = _make_evaluator(model=model, max_length=seq_len)
        evaluator._tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 10, 20, 102, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
        }
        evaluator.data = [{"text": "hi"}]
        evaluator.compute()
        # real tokens are the two at indices 1, 2 -> 2 forwards
        assert model.call_count == 2

    def test_missing_mask_token_raises(self) -> None:
        evaluator = _make_evaluator()
        evaluator._tokenizer.mask_token_id = None
        with pytest.raises(RuntimeError, match="no mask token"):
            evaluator.compute()

    def test_all_empty_samples_raises(self) -> None:
        evaluator = _make_evaluator()
        evaluator.data = [{"text": ""}, {"text": "  "}]
        with pytest.raises(ValueError, match="No tokens"):
            evaluator.compute()
