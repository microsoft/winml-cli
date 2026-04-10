# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Fill-mask evaluator using standard MLM loss (cross-entropy on masked tokens).

Evaluates masked language models (BERT, RoBERTa, etc.) by:
  1. Tokenizing each text sample.
  2. Randomly masking 15% of tokens (standard MLM protocol via
     DataCollatorForLanguageModeling).
  3. Running a single forward pass to get logits.
  4. Computing cross-entropy loss on the masked positions only.
  5. Aggregating into mean cross-entropy and perplexity.

This follows the standard HF Trainer MLM evaluation methodology:
    perplexity = exp(mean_cross_entropy)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..datasets.config import DatasetConfig
    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig

logger = logging.getLogger(__name__)

# Standard MLM masking probability (BERT paper convention).
_MLM_PROBABILITY = 0.15


class WinMLFillMaskEvaluator(WinMLEvaluator):
    """Evaluator for fill-mask (masked language modeling) tasks.

    Uses standard MLM evaluation: 15% random masking with cross-entropy loss.
    Reports cross_entropy (mean NLL) and perplexity = exp(cross_entropy).
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
                description="text to evaluate (tokens will be randomly masked)",
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

        The pipeline is used to access the tokenizer; the compute() method
        calls the model directly for MLM loss evaluation.
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

    def _get_max_length(self) -> int | None:
        """Get fixed sequence length from ONNX model, or None for dynamic."""
        io_config = getattr(self.model, "io_config", None) or {}
        shapes = io_config.get("input_shapes", [[]])
        if shapes and len(shapes[0]) > 1 and isinstance(shapes[0][1], int):
            return shapes[0][1]
        return None

    def _extract_logits(self, outputs: Any) -> torch.Tensor:
        """Extract logits tensor from model output (dict or HF dataclass)."""
        if isinstance(outputs, dict):
            if "logits" in outputs:
                return outputs["logits"]
            return next(iter(outputs.values()))
        return outputs.logits

    def compute(self) -> dict[str, Any]:
        """Run standard MLM evaluation: 15% masking + cross-entropy loss."""
        from transformers import DataCollatorForLanguageModeling

        tokenizer = self.pipe.tokenizer
        if tokenizer is None:
            raise RuntimeError("Fill-mask evaluation requires a tokenizer.")

        if tokenizer.mask_token_id is None:
            raise RuntimeError(
                f"Tokenizer for {self.config.model_id} has no mask token."
            )

        # Ensure pad token is set (needed for DataCollator)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.mask_token

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=_MLM_PROBABILITY,
        )

        max_length = self._get_max_length()
        total_loss = 0.0
        total_masked_tokens = 0

        for i, sample in enumerate(self.data):
            text = sample[self._input_col]
            if not text or not text.strip():
                continue

            # Tokenize with optional fixed-length padding for ONNX models
            tok_kwargs: dict[str, Any] = {
                "truncation": True,
                "return_tensors": "pt",
            }
            if max_length is not None:
                tok_kwargs["padding"] = "max_length"
                tok_kwargs["max_length"] = max_length
            else:
                tok_kwargs["padding"] = False

            encoding = tokenizer(text, **tok_kwargs)
            input_ids = encoding["input_ids"]  # [1, seq_len]

            # Skip samples that are too short (only special tokens)
            non_special = (input_ids[0] != tokenizer.pad_token_id).sum().item()
            if non_special < 3:
                continue

            # Apply standard 15% MLM masking via DataCollator
            # DataCollator expects list of dicts with "input_ids" key
            batch = data_collator([{"input_ids": input_ids.squeeze(0)}])
            masked_input_ids = batch["input_ids"]  # [1, seq_len]
            labels = batch["labels"]  # [1, seq_len], -100 for non-masked

            # Build model inputs: start from tokenizer outputs, replace input_ids
            # with masked version. This preserves token_type_ids, attention_mask, etc.
            model_inputs = {k: v for k, v in encoding.items() if isinstance(v, torch.Tensor)}
            model_inputs["input_ids"] = masked_input_ids
            if "attention_mask" not in model_inputs:
                if max_length is not None:
                    model_inputs["attention_mask"] = (masked_input_ids != tokenizer.pad_token_id).long()
                else:
                    model_inputs["attention_mask"] = torch.ones_like(masked_input_ids)

            # Forward pass — no grad needed for evaluation
            with torch.no_grad():
                outputs = self.model(**model_inputs)

            logits = self._extract_logits(outputs)  # [1, seq_len, vocab_size]

            # Compute CE loss only on masked positions (where labels != -100)
            mask = labels[0] != -100
            n_masked = mask.sum().item()
            if n_masked == 0:
                continue

            masked_logits = logits[0][mask]  # [n_masked, vocab_size]
            masked_labels = labels[0][mask]  # [n_masked]
            loss = F.cross_entropy(masked_logits, masked_labels, reduction="sum")

            total_loss += loss.item()
            total_masked_tokens += n_masked

            if (i + 1) % 50 == 0:
                total = len(self.data) if hasattr(self.data, "__len__") else "?"
                running_ce = total_loss / total_masked_tokens if total_masked_tokens else 0
                logger.info(
                    "Processed %d / %s samples (%d masked tokens, running CE=%.4f)...",
                    i + 1,
                    total,
                    total_masked_tokens,
                    running_ce,
                )

        if total_masked_tokens == 0:
            raise ValueError("No masked tokens found in dataset.")

        mean_ce = total_loss / total_masked_tokens
        perplexity = math.exp(mean_ce)

        logger.info(
            "Fill-mask evaluation complete: %d masked tokens across %d samples. "
            "CE=%.4f, PPL=%.4f",
            total_masked_tokens,
            len(self.data) if hasattr(self.data, "__len__") else -1,
            mean_ce,
            perplexity,
        )

        return {
            "cross_entropy": round(mean_ce, 4),
            "perplexity": round(perplexity, 4),
        }
