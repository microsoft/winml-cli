# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for WinMLImageToTextEvaluator."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from winml.modelkit.eval import WinMLImageToTextEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_processor(vocab_size: int = 100, bos_id: int = 0) -> MagicMock:
    """Build a mock processor with image_processor + tokenizer.

    Token IDs are clamped to [0, vocab_size-1] so index-into-logits never
    goes out of bounds regardless of the vocab_size argument.
    """
    proc = MagicMock()

    def _process_images(images, return_tensors=None, **kwargs):
        return {"pixel_values": torch.zeros(1, 3, 64, 64)}

    proc.side_effect = _process_images
    proc.__call__ = _process_images

    tokenizer = MagicMock()

    def _tokenize(text, return_tensors=None, truncation=None, max_length=None, **kwargs):
        # Return 4-token sequence with IDs in [0, vocab_size-1]
        t1 = min(1, vocab_size - 1)
        t2 = min(2, vocab_size - 1)
        t3 = min(3, vocab_size - 1)
        return {"input_ids": torch.tensor([[bos_id, t1, t2, t3]])}

    tokenizer.side_effect = _tokenize
    tokenizer.__call__ = _tokenize
    proc.tokenizer = tokenizer

    return proc


def _make_evaluator(
    model: MagicMock | None = None,
    columns_mapping: dict | None = None,
    num_ds_items: int = 5,
) -> WinMLImageToTextEvaluator:
    """Instantiate evaluator by patching external dependencies."""
    from winml.modelkit.datasets import DatasetConfig
    from winml.modelkit.eval import WinMLEvaluationConfig

    mapping = columns_mapping or {}

    mock_ds = MagicMock()
    mock_ds.__len__ = lambda self: num_ds_items
    mock_ds.shuffle.return_value = mock_ds
    mock_ds.select.return_value = mock_ds
    mock_ds.column_names = ["image", "caption"]

    if model is None:
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {}

    config = WinMLEvaluationConfig(
        model_id="test/mock-donut",
        task="image-to-text",
        dataset=DatasetConfig(
            path="nlphuji/flickr30k",
            columns_mapping=mapping,
        ),
    )

    with patch("datasets.load_dataset", return_value=mock_ds):
        return WinMLImageToTextEvaluator(config, model)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_schema_has_image_and_caption(self) -> None:
        schema = WinMLImageToTextEvaluator.schema_info()
        names = [col.name for col in schema]
        assert "image" in names
        assert "caption" in names

    def test_schema_types(self) -> None:
        schema = WinMLImageToTextEvaluator.schema_info()
        by_name = {col.name: col for col in schema}
        assert by_name["image"].type == "Image"
        assert by_name["caption"].type == "Value(string)"


# ---------------------------------------------------------------------------
# Init / lifecycle
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_column_names(self) -> None:
        ev = _make_evaluator()
        assert ev._input_col == "image"
        assert ev._caption_col == "caption"

    def test_custom_column_names(self) -> None:
        ev = _make_evaluator(columns_mapping={"input_column": "img", "caption_column": "text"})
        assert ev._input_col == "img"
        assert ev._caption_col == "text"

    def test_prepare_pipeline_returns_none(self) -> None:
        ev = _make_evaluator()
        assert ev.pipe is None

    def test_align_labels_is_noop(self) -> None:
        ev = _make_evaluator()
        ds = MagicMock()
        assert ev.align_labels(ds, MagicMock()) is ds


# ---------------------------------------------------------------------------
# _logits
# ---------------------------------------------------------------------------


class TestLogits:
    def test_dict_with_logits_key(self) -> None:
        logits = torch.randn(1, 3, 50)
        ev = _make_evaluator()
        result = ev._logits({"logits": logits, "other": None})
        assert torch.equal(result, logits)

    def test_dict_without_logits_raises(self) -> None:
        ev = _make_evaluator()
        with pytest.raises(KeyError, match="no 'logits' key"):
            ev._logits({"output": torch.randn(1, 3, 50)})

    def test_object_attribute(self) -> None:
        logits = torch.randn(1, 3, 50)
        out = MagicMock()
        out.logits = logits
        ev = _make_evaluator()
        assert torch.equal(ev._logits(out), logits)


# ---------------------------------------------------------------------------
# _decoder_seq_len
# ---------------------------------------------------------------------------


class TestDecoderSeqLen:
    def test_reads_second_input_shape(self) -> None:
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {"input_shapes": [[1, 3, 64, 64], [1, 16]]}
        ev = _make_evaluator(model=model)
        assert ev._decoder_seq_len() == 16

    def test_returns_none_when_no_io_config(self) -> None:
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {}
        ev = _make_evaluator(model=model)
        assert ev._decoder_seq_len() is None

    def test_returns_none_when_dynamic_dims(self) -> None:
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {"input_shapes": [[1, 3, 64, 64], ["batch", "seq"]]}
        ev = _make_evaluator(model=model)
        assert ev._decoder_seq_len() is None


# ---------------------------------------------------------------------------
# _score_sample
# ---------------------------------------------------------------------------


class TestScoreSample:
    def test_returns_token_log_probs(self) -> None:
        vocab = 50

        ev = _make_evaluator()
        proc = _make_processor(vocab_size=vocab)

        # Model returns uniform logits → each token gets log(1/vocab)
        ev.model = MagicMock(return_value={"logits": torch.zeros(1, 3, vocab)})

        log_probs = ev._score_sample("fake_image", "hello world", proc, fixed_seq_len=None)

        assert log_probs is not None
        assert log_probs.shape == (3,)
        expected = -math.log(vocab)
        assert all(abs(float(lp) - expected) < 1e-4 for lp in log_probs)

    def test_skips_short_caption(self) -> None:
        ev = _make_evaluator()
        proc = MagicMock()

        # Only one token → can't teacher-force
        proc.side_effect = lambda images, **kw: {"pixel_values": torch.zeros(1, 3, 64, 64)}
        proc.__call__ = proc.side_effect
        proc.tokenizer = MagicMock(
            side_effect=lambda text, **kw: {"input_ids": torch.tensor([[0]])}
        )
        proc.tokenizer.__call__ = proc.tokenizer.side_effect

        result = ev._score_sample("img", "x", proc, fixed_seq_len=None)
        assert result is None

    def test_image_processor_failure_returns_none(self) -> None:
        ev = _make_evaluator()
        proc = MagicMock()
        proc.side_effect = RuntimeError("bad image")
        proc.__call__ = proc.side_effect

        result = ev._score_sample("bad_img", "hello", proc, fixed_seq_len=None)
        assert result is None

    def test_model_forward_failure_returns_none(self) -> None:
        ev = _make_evaluator()
        proc = _make_processor()
        ev.model = MagicMock(side_effect=RuntimeError("ORT error"))

        result = ev._score_sample("img", "hello", proc, fixed_seq_len=None)
        assert result is None

    def test_missing_pixel_values_returns_none(self) -> None:
        ev = _make_evaluator()
        proc = MagicMock()
        proc.side_effect = lambda images, **kw: {}  # No "pixel_values" key
        proc.__call__ = proc.side_effect

        result = ev._score_sample("img", "hello", proc, fixed_seq_len=None)
        assert result is None

    def test_fixed_seq_len_pads_short_caption(self) -> None:
        """decoder_input_ids is padded to fixed_seq_len when caption is shorter."""
        vocab, fixed = 50, 8
        seen_shapes: list[tuple] = []

        def _forward(**kwargs):
            seen_shapes.append(tuple(kwargs["decoder_input_ids"].shape))
            return {"logits": torch.zeros(1, fixed, vocab)}

        ev = _make_evaluator()
        ev.model = MagicMock(side_effect=_forward)
        proc = _make_processor(vocab_size=vocab)
        # Tokenizer returns only 3 tokens ([bos, t1, t2]) — shorter than fixed+1
        proc.tokenizer.side_effect = lambda text, **kw: {"input_ids": torch.tensor([[0, 1, 2]])}

        result = ev._score_sample("img", "short", proc, fixed_seq_len=fixed)

        assert result is not None
        # decoder_input_ids must be [1, fixed]
        assert seen_shapes == [(1, fixed)]

    def test_fixed_seq_len_truncates_long_caption(self) -> None:
        """decoder_input_ids is truncated to fixed_seq_len when caption is longer."""
        vocab, fixed = 50, 4
        seen_shapes: list[tuple] = []

        def _forward(**kwargs):
            seen_shapes.append(tuple(kwargs["decoder_input_ids"].shape))
            return {"logits": torch.zeros(1, fixed, vocab)}

        ev = _make_evaluator()
        ev.model = MagicMock(side_effect=_forward)
        proc = _make_processor(vocab_size=vocab)
        # Tokenizer returns 10 tokens — longer than fixed+1
        proc.tokenizer.side_effect = lambda text, **kw: {
            "input_ids": torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3, 4, 1]])
        }

        result = ev._score_sample("img", "long caption text here", proc, fixed_seq_len=fixed)

        assert result is not None
        assert seen_shapes == [(1, fixed)]


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------


class TestCompute:
    def test_uniform_logits_give_correct_perplexity(self) -> None:
        """With uniform logits, perplexity should equal vocab size."""
        vocab = 50

        model = MagicMock()
        model.config.label2id = None
        model.io_config = {}
        model.return_value = {"logits": torch.zeros(1, 3, vocab)}

        ev = _make_evaluator(model=model)
        ev._processor = _make_processor(vocab_size=vocab)

        ev.data = [
            {"image": "img1", "caption": "hello world"},
            {"image": "img2", "caption": "foo bar"},
        ]

        result = ev.compute()

        assert "nll" in result
        assert "perplexity" in result
        assert result["perplexity"] == pytest.approx(vocab, rel=1e-2)

    def test_skips_none_image_or_caption(self) -> None:
        vocab = 10
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {}
        model.return_value = {"logits": torch.zeros(1, 3, vocab)}

        ev = _make_evaluator(model=model)
        ev._processor = _make_processor(vocab_size=vocab)

        ev.data = [
            {"image": "img1", "caption": "valid"},
            {"image": None, "caption": "valid"},  # skipped
            {"image": "img2", "caption": None},  # skipped
        ]

        result = ev.compute()
        # Only one valid sample, but should still succeed
        assert "perplexity" in result

    def test_list_caption_uses_first_element(self) -> None:
        vocab = 10
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {}
        model.return_value = {"logits": torch.zeros(1, 3, vocab)}

        ev = _make_evaluator(model=model)
        ev._processor = _make_processor(vocab_size=vocab)

        ev.data = [{"image": "img", "caption": ["first caption", "second caption"]}]

        result = ev.compute()
        assert "perplexity" in result

    def test_empty_list_caption_skipped(self) -> None:
        vocab = 10
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {}
        model.return_value = {"logits": torch.zeros(1, 3, vocab)}

        ev = _make_evaluator(model=model)
        ev._processor = _make_processor(vocab_size=vocab)

        ev.data = [
            {"image": "img1", "caption": []},  # skipped: empty list
            {"image": "img2", "caption": "valid one"},
        ]

        result = ev.compute()
        assert "perplexity" in result

    def test_no_valid_samples_raises(self) -> None:
        ev = _make_evaluator()
        ev._processor = _make_processor()
        ev.data = [
            {"image": None, "caption": None},
            {"image": None, "caption": None},
        ]

        with pytest.raises(ValueError, match="No valid token log-probabilities"):
            ev.compute()

    def test_perplexity_is_exp_nll(self) -> None:
        vocab = 20
        model = MagicMock()
        model.config.label2id = None
        model.io_config = {}
        model.return_value = {"logits": torch.zeros(1, 3, vocab)}

        ev = _make_evaluator(model=model)
        ev._processor = _make_processor(vocab_size=vocab)
        ev.data = [{"image": "img", "caption": "some text"}]

        result = ev.compute()
        assert result["perplexity"] == pytest.approx(math.exp(result["nll"]), rel=1e-4)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered_in_evaluator_registry(self) -> None:
        from winml.modelkit.eval.evaluate import _EVALUATOR_REGISTRY

        assert "image-to-text" in _EVALUATOR_REGISTRY
        assert _EVALUATOR_REGISTRY["image-to-text"] is WinMLImageToTextEvaluator

    def test_default_dataset_registered(self) -> None:
        from winml.modelkit.eval.evaluate import _DEFAULT_DATASETS

        assert "image-to-text" in _DEFAULT_DATASETS
        ds = _DEFAULT_DATASETS["image-to-text"]
        assert ds.path == "clip-benchmark/wds_mscoco_captions"
        assert ds.samples == 100
        assert ds.columns_mapping.get("input_column") == "jpg"
        assert ds.columns_mapping.get("caption_column") == "txt"
