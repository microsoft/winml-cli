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

from winml.modelkit.models.winml import HFCausalLM


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
