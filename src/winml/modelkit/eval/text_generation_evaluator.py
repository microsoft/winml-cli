# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Perplexity evaluator for causal-LM (text-generation) models.

Does *not* go through HF's ``pipeline`` / ``evaluate`` libraries: perplexity is
scored by teacher-forcing raw corpus tokens through the model's ``forward``, so
the evaluator only needs a model honoring the causal-LM contract
(``encode(text) -> list[int]`` and ``forward(ids)`` yielding one ``(vocab,)``
logit vector per scored position).  The same code scores a WinML genai bundle
or any object exposing that interface.

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
        num_tokens = self._int_param("num_tokens")
        seqlen = self._int_param("seqlen")
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
        from tqdm import tqdm

        model: Any = self.model
        total_nll = 0.0
        scored = 0
        total_positions = sum(len(block) - 1 for block in self.data)
        with tqdm(total=total_positions, desc="Evaluating perplexity", unit="tok") as bar:
            for block in self.data:
                targets = block[1:]
                for step_logits, target in zip(model.forward(block), targets, strict=True):
                    total_nll += _step_nll(step_logits, target)
                    bar.update(1)
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

    def _int_param(self, name: str) -> int:
        """Read an int scoring parameter from ``columns_mapping``.

        Values ride the standard ``--column key=value`` CLI path (so they are
        strings); the default comes from the text-generation schema in
        :mod:`~winml.modelkit.utils.eval_utils`.
        """
        from ..utils.eval_utils import get_default

        raw = self.config.dataset.columns_mapping.get(name)
        if raw is None:
            raw = get_default(self._TASK, name)
        if raw is None:
            raise ValueError(f"No value or default for scoring parameter '{name}'.")
        return int(raw)

    def _load_corpus_tokens(self, num_tokens: int) -> list[int]:
        """Read the dataset text column and tokenize until ``num_tokens``.

        Perplexity needs a coherent corpus, so rows are consumed in dataset
        order (no shuffle) and concatenated with blank lines. Iteration stops as
        soon as enough rows to cover ``num_tokens`` have been collected. A local
        directory produced by ``Dataset.save_to_disk`` is opened with
        ``load_from_disk``; otherwise the split is fetched via ``load_dataset``.
        The ``streaming`` flag only controls how a hub split is fetched (streamed
        vs downloaded once and cached); either way the same in-order prefix is
        read, so the resulting token stream is identical.

        Uses the model's own tokenizer (``model.encode``) so the token stream
        matches the model under test exactly.
        """
        from pathlib import Path

        from datasets import load_dataset, load_from_disk

        from ..utils.eval_utils import get_default

        model: Any = self.model
        ds_config = self.config.dataset
        column = ds_config.columns_mapping.get(
            "input_column", get_default(self._TASK, "input_column")
        )
        ds_path = Path(ds_config.path).expanduser() if ds_config.path else None
        if ds_path and ds_path.is_dir():
            dataset = load_from_disk(str(ds_path))
        else:
            dataset = load_dataset(
                ds_config.path,
                name=ds_config.name,
                split=ds_config.split,
                revision=ds_config.revision,
                streaming=ds_config.streaming,
            )
        if column not in (dataset.column_names or [column]):
            raise ValueError(
                f"Dataset '{ds_config.path}' has no column '{column}'; "
                f"available columns: {sorted(dataset.column_names or [])}. "
                "Set it via --column input_column=<name>."
            )

        collected: list[str] = []
        approx_tokens = 0
        ids: list[int] = []
        for row in dataset:
            text = row.get(column)
            if not (text and text.strip()):
                continue
            collected.append(text)
            # Cheap per-row length estimate (+2 for the "\n\n" separator) avoids
            # re-encoding the whole corpus every row. Isolated per-row encoding
            # slightly overcounts vs the joined stream (boundary tokens merge),
            # so once the estimate is reached we re-encode the join and keep
            # pulling rows until the *authoritative* token count covers
            # num_tokens (a couple of extra rows at most).
            approx_tokens += len(model.encode(text)) + 2
            if approx_tokens >= num_tokens:
                ids = model.encode("\n\n".join(collected))
                if len(ids) >= num_tokens:
                    break
        else:
            ids = model.encode("\n\n".join(collected))

        if len(ids) < num_tokens:
            logger.warning(
                "Corpus exhausted: collected %d tokens < requested num_tokens=%d.",
                len(ids),
                num_tokens,
            )
        return ids[:num_tokens]


def _step_nll(logits: np.ndarray, target: int) -> float:
    """``-log P(target)`` at one position, from raw (unnormalized) logits."""
    x = logits.astype(np.float64)
    m = x.max()
    logsumexp = m + np.log(np.exp(x - m).sum())
    return float(logsumexp - x[target])
