# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Fill-mask evaluator using pseudo-perplexity and top-k accuracy.

Evaluates masked language models (BERT, RoBERTa, etc.) by:
  1. Tokenizing each text sample.
  2. For each eligible (non-special) token position, creating a masked copy.
  3. Running the fill-mask pipeline on the masked text.
  4. Checking whether the original token appears in the top-k predictions
     and recording its probability.
  5. Aggregating into pseudo-perplexity and top-k accuracy.

Pipeline output contract (HF fill-mask):
    pipe("The [MASK] of France") -> [
        {"score": 0.42, "token": 3000, "token_str": "capital", "sequence": "..."},
        ...
    ]
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING, Any

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..datasets.config import DatasetConfig
    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig

logger = logging.getLogger(__name__)

# Maximum number of token positions to mask per sample to keep runtime bounded.
_MAX_MASKS_PER_SAMPLE = 10


class WinMLFillMaskEvaluator(WinMLEvaluator):
    """Evaluator for fill-mask (masked language modeling) tasks.

    Reports pseudo-perplexity, accuracy@1, and accuracy@5.
    """

    @classmethod
    def schema_info(cls) -> list:
        """Return expected dataset schema for fill-mask evaluation."""
        from .config import SchemaColumn

        return [
            SchemaColumn(
                "text",
                "Value(string)",
                "input_column",
                description="text to evaluate (tokens will be masked one-by-one)",
            ),
        ]

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        self._input_col = config.dataset.columns_mapping.get("input_column", "text")
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Create fill-mask pipeline with tokenizer padding for fixed-shape ONNX.

        FillMaskPipeline passes tokenizer args via ``tokenizer_kwargs``
        (unlike text-classification which uses top-level preprocess params).
        """
        pipe = super().prepare_pipeline()

        if pipe.tokenizer is not None:
            io_config = getattr(self.model, "io_config", None) or {}
            shapes = io_config.get("input_shapes", [[]])
            if shapes and len(shapes[0]) > 1 and isinstance(shapes[0][1], int):
                seq_len = shapes[0][1]
                tok_kwargs = pipe._preprocess_params.setdefault("tokenizer_kwargs", {})
                tok_kwargs.setdefault("padding", "max_length")
                tok_kwargs.setdefault("max_length", seq_len)
                tok_kwargs.setdefault("truncation", True)

        return pipe

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """No-op: fill-mask has no class labels to align."""
        return dataset

    def compute(self) -> dict[str, Any]:
        """Run fill-mask evaluation and return pseudo-perplexity + accuracy."""
        from .metrics.pseudo_perplexity import PseudoPerplexityMetric

        tokenizer = self.pipe.tokenizer
        if tokenizer is None:
            raise RuntimeError("Fill-mask evaluation requires a tokenizer.")

        mask_token = tokenizer.mask_token
        mask_token_id = tokenizer.mask_token_id
        if mask_token is None or mask_token_id is None:
            raise RuntimeError(
                f"Tokenizer for {self.config.model_id} has no mask token."
            )

        special_ids = set(tokenizer.all_special_ids)
        top_k = 5

        neg_log_likelihoods: list[float] = []
        top1_hits: list[bool] = []
        top5_hits: list[bool] = []

        rng = random.Random(42)

        for i, sample in enumerate(self.data):
            text = sample[self._input_col]
            if not text or not text.strip():
                continue

            encoding = tokenizer(
                text, truncation=True, return_offsets_mapping=True, add_special_tokens=True,
            )
            input_ids = encoding["input_ids"]
            offsets = encoding.get("offset_mapping", [])

            # Find maskable positions (non-special tokens)
            maskable = [
                idx for idx, tid in enumerate(input_ids)
                if tid not in special_ids and idx < len(offsets) and offsets[idx] != (0, 0)
            ]

            if not maskable:
                continue

            # Sample positions to keep runtime bounded
            if len(maskable) > _MAX_MASKS_PER_SAMPLE:
                maskable = rng.sample(maskable, _MAX_MASKS_PER_SAMPLE)

            for pos in maskable:
                original_token_id = input_ids[pos]
                # Build masked text by replacing the token span with mask_token
                start, end = offsets[pos]
                masked_text = text[:start] + mask_token + text[end:]

                try:
                    predictions = self.pipe(masked_text, top_k=top_k)
                except Exception:
                    logger.debug("Pipeline failed for sample %d pos %d, skipping.", i, pos)
                    continue

                if not predictions:
                    continue

                # Check top-1 and top-k accuracy
                pred_tokens = [p["token"] for p in predictions]
                top1_hits.append(pred_tokens[0] == original_token_id)
                top5_hits.append(original_token_id in pred_tokens)

                # Find score for original token; if not in top-k, use a floor
                score = None
                for p in predictions:
                    if p["token"] == original_token_id:
                        score = p["score"]
                        break

                if score is not None and score > 0:
                    neg_log_likelihoods.append(-math.log(score))
                else:
                    # Token not in top-k: use last prediction's score as upper bound
                    floor_score = predictions[-1]["score"] if predictions else 1e-9
                    neg_log_likelihoods.append(-math.log(max(floor_score, 1e-9)))

            if (i + 1) % 5 == 0:
                total = len(self.data) if hasattr(self.data, "__len__") else "?"
                logger.info(
                    "Processed %d / %s samples (%d mask positions so far)...",
                    i + 1,
                    total,
                    len(neg_log_likelihoods),
                )

        if not neg_log_likelihoods:
            raise ValueError("No valid mask positions found in dataset.")

        logger.info(
            "Fill-mask evaluation complete: %d mask positions across %d samples.",
            len(neg_log_likelihoods),
            len(self.data) if hasattr(self.data, "__len__") else -1,
        )

        return PseudoPerplexityMetric().compute(
            neg_log_likelihoods, top1_hits, top5_hits,
        )
