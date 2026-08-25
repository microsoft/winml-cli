# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the causal-LM inference classes in ``genai_causal_lm``.

``HFCausalLM`` is the PyTorch-baseline adapter that honours the same
``encode`` / ``forward`` contract as ``WinMLGenaiCausalLM``.  The HF tokenizer
and model are stubbed so no weights are downloaded; the tests verify the
adapter maps onto the contract exactly (``add_special_tokens=False`` encoding,
``forward`` yielding one float32 ``(vocab,)`` vector per position, trimmed to
``N - 1`` positions).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from winml.modelkit.models.winml import HFCausalLM, WinMLGenaiCausalLM


def _make_adapter(*, logits=None, token_ids=None):
    """Build an ``HFCausalLM`` with stubbed tokenizer/model (no download)."""
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": [5, 6, 7] if token_ids is None else token_ids}

    model = MagicMock()
    # from_pretrained(...).to(device).eval() must yield the same stub.
    model.to.return_value = model
    model.eval.return_value = model
    call_out = MagicMock()
    call_out.logits = logits
    model.return_value = call_out

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
    ):
        adapter = HFCausalLM("dummy/model", torch.device("cpu"))
    return adapter, tokenizer, model


class TestEncode:
    def test_returns_token_ids(self) -> None:
        adapter, _, _ = _make_adapter(token_ids=[10, 20, 30])
        assert adapter.encode("hello world") == [10, 20, 30]

    def test_disables_special_tokens(self) -> None:
        """The genai bundle tokenizer adds no specials; the adapter must match."""
        adapter, tokenizer, _ = _make_adapter()
        adapter.encode("some text")
        tokenizer.assert_called_once_with("some text", add_special_tokens=False)


def test_from_model_adapts_without_reloading_or_moving() -> None:
    tokenizer = MagicMock()
    model = MagicMock()
    model.eval.return_value = model

    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as load_tokenizer,
        patch("transformers.AutoModelForCausalLM.from_pretrained") as load_model,
    ):
        adapter = HFCausalLM.from_model(
            "dummy/model",
            model,
            torch.device("cuda:1"),
            trust_remote_code=True,
        )

    load_tokenizer.assert_called_once_with("dummy/model", trust_remote_code=True)
    load_model.assert_not_called()
    model.eval.assert_called_once_with()
    model.to.assert_not_called()
    assert adapter._model is model
    assert adapter._device == torch.device("cuda:1")


class TestForward:
    def test_yields_one_vector_per_position(self) -> None:
        """One (vocab,) vector per position; trailing row dropped -> N-1 rows."""
        vocab = 5
        logits = torch.arange(3 * vocab, dtype=torch.float32).reshape(1, 3, vocab)
        adapter, _, _ = _make_adapter(logits=logits)
        rows = list(adapter.forward([1, 2, 3]))
        assert len(rows) == 2
        assert all(row.shape == (vocab,) for row in rows)

    def test_logits_match_raw_model(self) -> None:
        vocab = 4
        logits = torch.arange(3 * vocab, dtype=torch.float32).reshape(1, 3, vocab)
        adapter, _, _ = _make_adapter(logits=logits)
        rows = list(adapter.forward([7, 8, 9]))
        np.testing.assert_allclose(np.stack(rows), logits[0, :-1, :].numpy())

    def test_casts_to_float32(self) -> None:
        logits = torch.zeros(1, 3, 5, dtype=torch.float16)
        adapter, _, _ = _make_adapter(logits=logits)
        rows = list(adapter.forward([1, 2, 3]))
        assert all(row.dtype == np.float32 for row in rows)

    def test_feeds_input_ids_as_batched_tensor(self) -> None:
        logits = torch.zeros(1, 3, 5)
        adapter, _, model = _make_adapter(logits=logits)
        list(adapter.forward([11, 22, 33]))
        passed = model.call_args.kwargs["input_ids"]
        assert torch.equal(passed, torch.tensor([[11, 22, 33]]))

    def test_call_is_forward(self) -> None:
        assert HFCausalLM.__call__ is HFCausalLM.forward


def _make_genai_adapter(*, logits, context_length=2048):
    """Build a ``WinMLGenaiCausalLM`` over a stubbed :class:`GenaiSession`.

    ``GenaiSession`` is patched so construction does no real work; the returned
    session exposes the ``_ensure_loaded`` / ``context_length`` / ``_model``
    attributes ``forward`` reads. ``logits`` is the ``(vocab,)`` vector the fake
    generator returns at every position.
    """
    logits = np.asarray(logits, dtype=np.float32)

    with patch("winml.modelkit.models.winml.genai_causal_lm.GenaiSession") as session_cls:
        session = session_cls.return_value
        session.context_length = context_length
        session._model = MagicMock(name="model")
        adapter = WinMLGenaiCausalLM("dummy/bundle")

    gen = MagicMock(name="generator")
    # Fake genai returns (batch, seq, vocab); forward reads [0, -1, :].
    gen.get_logits.return_value = logits.reshape(1, 1, -1)

    og = MagicMock(name="onnxruntime_genai")
    params = og.GeneratorParams.return_value
    og.Generator.return_value = gen
    return adapter, og, params, gen


class TestGenaiForwardSearchOptions:
    """``WinMLGenaiCausalLM.forward`` must isolate teacher forcing from the
    bundle's search policy so a forced target is never re-masked."""

    def test_neutralizes_bundle_search_constraints(self) -> None:
        """A non-default bundle search config (e.g. min_length) is overridden:
        forward always pins the neutralizing options on the generator."""
        adapter, og, params, _ = _make_genai_adapter(logits=[0.0, 1.0, 0.0, 0.0])
        with patch.dict("sys.modules", {"onnxruntime_genai": og}):
            list(adapter.forward([1, 2, 3]))

        params.set_search_options.assert_called_once()
        kwargs = params.set_search_options.call_args.kwargs
        assert kwargs["do_sample"] is False
        assert kwargs["min_length"] == 0
        assert kwargs["no_repeat_ngram_size"] == 0
        assert kwargs["repetition_penalty"] == 1.0
        assert kwargs["num_beams"] == 1

    def test_forces_target_when_model_argmax_differs(self) -> None:
        """When the model would pick another token, forward overrides the
        generator logits so the pinned next input is the corpus target."""
        # get_logits argmax is index 0, but the middle target is token 3.
        adapter, og, _, gen = _make_genai_adapter(logits=[9.0, 0.0, 0.0, 0.0, 0.0])
        with patch.dict("sys.modules", {"onnxruntime_genai": og}):
            list(adapter.forward([0, 3, 0]))

        gen.set_logits.assert_called_once()
        forced = np.asarray(gen.set_logits.call_args.args[0])
        assert int(forced.reshape(-1).argmax()) == 3
