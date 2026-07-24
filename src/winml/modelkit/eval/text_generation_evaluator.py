# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Perplexity evaluator for causal-LM (text-generation) models.

Does *not* go through HF's ``pipeline`` / ``evaluate`` libraries: perplexity is
scored by teacher-forcing raw corpus tokens through the model's ``forward``, so
the evaluator only needs a model honoring the causal-LM contract
(``encode(text) -> list[int]`` and ``forward(ids).logits``).  The same code
scores a WinML genai bundle or any object exposing that interface.

Protocol: the dataset text column is concatenated and tokenized with the
model's own tokenizer, capped at ``num_tokens``, then cut into contiguous
non-overlapping ``seqlen``-token blocks (no detokenizer, no sliding window).
Every token after the first in its block is scored once, giving
``perplexity = exp(sum(NLL) / scored_positions)``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from transformers.pipelines.base import Pipeline


logger = logging.getLogger(__name__)

__all__ = ["WinMLTextGenerationEvaluator"]


class WinMLTextGenerationEvaluator(WinMLEvaluator):
    """Evaluator computing disjoint fixed-length perplexity for causal LMs.

    Constructor keeps the standard ``(config, model)`` signature so the registry
    dispatch in :mod:`~winml.modelkit.eval.evaluate` works unmodified. ``model``
    is a causal-LM inference object (e.g.
    :class:`~winml.modelkit.models.winml.genai_causal_lm.WinMLGenaiCausalLM`).

    Two scoring parameters are read from ``dataset.columns_mapping`` so they ride
    the existing ``--column key=value`` CLI path (defaults come from the
    text-generation schema in :mod:`~winml.modelkit.utils.eval_utils`):

    * ``num_tokens`` -- total corpus tokens to score.
    * ``seqlen`` -- non-overlapping block length.
    """

    _TASK = "text-generation"

    def prepare_pipeline(self) -> Pipeline | None:  # type: ignore[override]
        """No HF pipeline -- the model's ``forward`` is driven directly."""
        return None

    def prepare_data(self) -> list[list[int]]:
        """Load, tokenize, and block the corpus into fixed-length token blocks.

        Returns a list of token-ID blocks, each at least 2 tokens long (a block
        needs a first token to condition on and at least one token to score).
        """
        from ..utils.eval_utils import get_default

        mapping = self.config.dataset.columns_mapping
        num_tokens = int(mapping.get("num_tokens", get_default(self._TASK, "num_tokens")))
        seqlen = int(mapping.get("seqlen", get_default(self._TASK, "seqlen")))
        if seqlen < 2:
            raise ValueError(f"seqlen must be at least 2; got {seqlen}.")

        ids = self._load_corpus_tokens(num_tokens)
        blocks = [ids[i : i + seqlen] for i in range(0, len(ids), seqlen)]
        blocks = [b for b in blocks if len(b) >= 2]
        if not blocks:
            raise ValueError(
                f"Corpus produced no scorable blocks (got {len(ids)} tokens, "
                f"seqlen={seqlen}). Increase num_tokens or lower seqlen."
            )
        self._seqlen = seqlen
        logger.info(
            "Perplexity corpus: %d tokens -> %d blocks (seqlen=%d)",
            len(ids),
            len(blocks),
            seqlen,
        )
        return blocks

    def compute(self) -> dict[str, Any]:
        """Score every block and return perplexity plus corpus statistics."""
        model = self.model
        total_nll = 0.0
        scored = 0
        for block in self.data:
            logits = model.forward(block).logits[0]
            targets = np.asarray(block[1:], dtype=np.int64)
            total_nll += _block_nll(logits, targets)
            scored += len(targets)

        if scored == 0:
            raise RuntimeError("Perplexity evaluation scored 0 positions.")

        return {
            "perplexity": float(np.exp(total_nll / scored)),
            "num_scored_positions": scored,
            "num_blocks": len(self.data),
            "seqlen": self._seqlen,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_corpus_tokens(self, num_tokens: int) -> list[int]:
        """Concatenate the dataset text column and tokenize with the model.

        Uses the model's own tokenizer (``model.encode``) so the token stream
        matches the model under test exactly.
        """
        from datasets import load_dataset

        from ..utils.eval_utils import get_default

        ds_config = self.config.dataset
        column = ds_config.columns_mapping.get(
            "input_column", get_default(self._TASK, "input_column")
        )
        dataset = load_dataset(
            ds_config.path,
            name=ds_config.name,
            split=ds_config.split,
            revision=ds_config.revision,
        )
        if column not in dataset.column_names:
            raise ValueError(
                f"Dataset '{ds_config.path}' has no column '{column}'; "
                f"available columns: {sorted(dataset.column_names)}. "
                "Set it via --column input_column=<name>."
            )
        text = "\n\n".join(row for row in dataset[column] if row and row.strip())
        return self.model.encode(text)[:num_tokens]


def _block_nll(logits: np.ndarray, targets: np.ndarray) -> float:
    """Sum of ``-log P(target)`` over positions, from raw (unnormalized) logits."""
    x = logits.astype(np.float64)
    logsumexp = x.max(axis=-1) + np.log(np.exp(x - x.max(axis=-1, keepdims=True)).sum(axis=-1))
    return float((logsumexp - x[np.arange(len(targets)), targets]).sum())
