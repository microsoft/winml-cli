# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLFillMaskEvaluator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from winml.modelkit.eval import WinMLFillMaskEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tokenizer(vocab_size=100, pad_token_id=0, mask_token_id=103):
    """Create a mock tokenizer with the minimum interface needed."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = pad_token_id
    tokenizer.mask_token_id = mask_token_id
    tokenizer.mask_token = "[MASK]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = None
    tokenizer.__len__ = lambda self: vocab_size

    def _get_special_tokens_mask(ids, already_has_special_tokens=True):
        return [1 if i in (0,) else 0 for i in range(len(ids))]

    tokenizer.get_special_tokens_mask = _get_special_tokens_mask
    return tokenizer


def _make_evaluator(
    model=None,
    max_length=None,
    columns_mapping=None,
):
    """Instantiate WinMLFillMaskEvaluator by patching external dependencies."""
    from winml.modelkit.datasets import DatasetConfig
    from winml.modelkit.eval import WinMLEvaluationConfig

    mapping = columns_mapping or {"input_column": "text"}

    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: 5
    mock_ds.shuffle.return_value = mock_ds
    mock_ds.select.return_value = mock_ds

    if model is None:
        model = MagicMock()
        model.config.label2id = None
        io_config = {}
        if max_length is not None:
            io_config["input_shapes"] = [[1, max_length]]
        model.io_config = io_config

    config = WinMLEvaluationConfig(
        model_id="test/mock-bert",
        task="fill-mask",
        dataset=DatasetConfig(
            path="Salesforce/wikitext",
            name="wikitext-2-raw-v1",
            columns_mapping=mapping,
        ),
    )

    mock_tokenizer = _make_tokenizer()

    with patch("datasets.load_dataset", return_value=mock_ds), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        return WinMLFillMaskEvaluator(config, model)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestFillMaskSchema:
    def test_schema_has_text_column(self) -> None:
        schema = WinMLFillMaskEvaluator.schema_info()
        assert len(schema) == 1
        assert schema[0].name == "text"

    def test_schema_column_type(self) -> None:
        schema = WinMLFillMaskEvaluator.schema_info()
        assert schema[0].type == "Value(string)"


# ---------------------------------------------------------------------------
# prepare_pipeline
# ---------------------------------------------------------------------------

class TestFillMaskPreparePipeline:
    def test_returns_none(self) -> None:
        evaluator = _make_evaluator()
        assert evaluator.pipe is None

    def test_loads_tokenizer(self) -> None:
        evaluator = _make_evaluator()
        assert evaluator._tokenizer is not None
        assert evaluator._tokenizer.mask_token_id == 103


# ---------------------------------------------------------------------------
# _get_max_length
# ---------------------------------------------------------------------------

class TestGetMaxLength:
    def test_fixed_shape_returns_length(self) -> None:
        evaluator = _make_evaluator(max_length=128)
        assert evaluator._get_max_length() == 128

    def test_no_io_config_returns_none(self) -> None:
        evaluator = _make_evaluator()
        assert evaluator._get_max_length() is None

    def test_dynamic_shape_returns_none(self) -> None:
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {"input_shapes": [["batch", "seq"]]}
        evaluator = _make_evaluator(model=model)
        assert evaluator._get_max_length() is None


# ---------------------------------------------------------------------------
# _extract_logits
# ---------------------------------------------------------------------------

class TestExtractLogits:
    def test_from_dict_with_logits_key(self) -> None:
        evaluator = _make_evaluator()
        logits = torch.randn(1, 10, 100)
        result = evaluator._extract_logits({"logits": logits, "hidden_states": None})
        assert torch.equal(result, logits)

    def test_from_dict_without_logits_key(self) -> None:
        evaluator = _make_evaluator()
        logits = torch.randn(1, 10, 100)
        result = evaluator._extract_logits({"output": logits})
        assert torch.equal(result, logits)

    def test_from_dataclass(self) -> None:
        evaluator = _make_evaluator()
        logits = torch.randn(1, 10, 100)
        output = MagicMock()
        output.logits = logits
        result = evaluator._extract_logits(output)
        assert torch.equal(result, logits)


# ---------------------------------------------------------------------------
# _tokenize_and_mask
# ---------------------------------------------------------------------------

class TestTokenizeAndMask:
    def test_empty_text_returns_none(self) -> None:
        evaluator = _make_evaluator()
        result = evaluator._tokenize_and_mask(
            "", evaluator._tokenizer, MagicMock(), None,
        )
        assert result is None

    def test_whitespace_only_returns_none(self) -> None:
        evaluator = _make_evaluator()
        result = evaluator._tokenize_and_mask(
            "   ", evaluator._tokenizer, MagicMock(), None,
        )
        assert result is None

    def test_short_text_returns_none(self) -> None:
        """Text with fewer than 3 non-pad tokens should be skipped."""
        evaluator = _make_evaluator()

        tokenizer = evaluator._tokenizer
        # Returns only 2 non-pad tokens → skip
        encoding = {
            "input_ids": torch.tensor([[101, 102, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        }
        tokenizer.return_value = encoding

        result = evaluator._tokenize_and_mask(
            "Hi", tokenizer, MagicMock(), max_length=4,
        )
        assert result is None

    def test_valid_text_returns_tuple(self) -> None:
        """Valid text should return (model_inputs, labels)."""
        evaluator = _make_evaluator()

        tokenizer = evaluator._tokenizer
        encoding = {
            "input_ids": torch.tensor([[101, 1996, 4937, 2938, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        tokenizer.return_value = encoding

        collator = MagicMock()
        collator.return_value = {
            "input_ids": torch.tensor([[101, 1996, 103, 2938, 102]]),
            "labels": torch.tensor([[-100, -100, 4937, -100, -100]]),
        }

        result = evaluator._tokenize_and_mask(
            "The cat sat", tokenizer, collator, max_length=None,
        )

        assert result is not None
        model_inputs, labels = result
        assert "input_ids" in model_inputs
        assert "attention_mask" in model_inputs
        assert labels.shape == torch.Size([1, 5])

    def test_masked_input_ids_used(self) -> None:
        """model_inputs should use masked input_ids, not the original."""
        evaluator = _make_evaluator()

        tokenizer = evaluator._tokenizer
        encoding = {
            "input_ids": torch.tensor([[101, 1996, 4937, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        tokenizer.return_value = encoding

        masked_ids = torch.tensor([[101, 103, 4937, 102]])
        collator = MagicMock()
        collator.return_value = {
            "input_ids": masked_ids,
            "labels": torch.tensor([[-100, 1996, -100, -100]]),
        }

        result = evaluator._tokenize_and_mask(
            "The cat", tokenizer, collator, max_length=None,
        )
        model_inputs, _ = result
        assert torch.equal(model_inputs["input_ids"], masked_ids)


# ---------------------------------------------------------------------------
# compute (integration with mocked model)
# ---------------------------------------------------------------------------

class TestFillMaskCompute:
    def test_compute_returns_metrics(self) -> None:
        """Verify compute() returns cross_entropy and perplexity."""
        vocab_size = 50
        seq_len = 8

        model = MagicMock()
        model.config.label2id = None
        model.io_config = {"input_shapes": [[1, seq_len]]}

        # Model returns random logits
        model.return_value = {"logits": torch.randn(1, seq_len, vocab_size)}

        evaluator = _make_evaluator(model=model, max_length=seq_len)

        # Mock tokenizer to return valid encoding
        tokenizer = evaluator._tokenizer
        input_ids = torch.tensor([[101, 10, 20, 30, 40, 102, 0, 0]])
        encoding = {
            "input_ids": input_ids,
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
        }
        tokenizer.return_value = encoding

        # Fake dataset with 3 samples
        evaluator.data = [
            {"text": "Hello world foo"},
            {"text": "Another sentence here"},
            {"text": ""},  # empty — should be skipped
        ]

        with patch("transformers.DataCollatorForLanguageModeling") as MockCollator:
            collator_instance = MagicMock()
            collator_instance.return_value = {
                "input_ids": torch.tensor([[101, 103, 20, 103, 40, 102, 0, 0]]),
                "labels": torch.tensor([[-100, 10, -100, 30, -100, -100, -100, -100]]),
            }
            MockCollator.return_value = collator_instance

            result = evaluator.compute()

        assert "cross_entropy" in result
        assert "perplexity" in result
        assert result["cross_entropy"] > 0
        assert result["perplexity"] > 1.0

    def test_compute_no_mask_token_raises(self) -> None:
        """Should raise if tokenizer has no mask token."""
        evaluator = _make_evaluator()
        evaluator._tokenizer.mask_token_id = None

        with pytest.raises(RuntimeError, match="no mask token"):
            evaluator.compute()

    def test_compute_all_empty_raises(self) -> None:
        """Should raise if all samples are empty (no masked tokens)."""
        evaluator = _make_evaluator()
        evaluator.data = [{"text": ""}, {"text": "  "}]

        with patch("transformers.DataCollatorForLanguageModeling"):
            with pytest.raises(ValueError, match="No masked tokens"):
                evaluator.compute()


# ---------------------------------------------------------------------------
# align_labels (no-op)
# ---------------------------------------------------------------------------

class TestFillMaskAlignLabels:
    def test_returns_dataset_unchanged(self) -> None:
        evaluator = _make_evaluator()
        mock_ds = MagicMock()
        mock_config = MagicMock()
        result = evaluator.align_labels(mock_ds, mock_config)
        assert result is mock_ds
