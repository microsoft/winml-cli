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
  5. Aggregating into mean cross-entropy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from .base_evaluator import WinMLEvaluator
from .metrics import CrossEntropyMetric


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..datasets.config import DatasetConfig
    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig

logger = logging.getLogger(__name__)

# Standard MLM masking probability (BERT paper convention).
_MLM_PROBABILITY = 0.15

# Fixed seed for reproducible masking across evaluation runs.
_MLM_SEED = 42


class WinMLFillMaskEvaluator(WinMLEvaluator):
    """Evaluator for fill-mask (masked language modeling) tasks.

    Uses standard MLM evaluation: 15% random masking with cross-entropy loss.
    Reports cross_entropy (mean NLL per masked token).
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
        """No-op: fill-mask evaluation calls the model directly.

        Loads only the tokenizer (stored as ``self._tokenizer``).
        Returns ``None`` since the HF pipeline is not used.
        """
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        return None  # type: ignore[return-value]

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

    def _tokenize_and_mask(
        self,
        text: str,
        tokenizer: Any,
        data_collator: Any,
        max_length: int | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor] | None:
        """Tokenize text, apply MLM masking, and build model inputs.

        Returns:
            ``(model_inputs, labels)`` tuple, or ``None`` if the sample
            should be skipped (empty or too short).
        """
        if not text or not text.strip():
            return None

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
            return None

        # Mask 15% of input_ids; labels hold original tokens at masked
        # positions and -100 elsewhere.
        batch = data_collator([{"input_ids": input_ids.squeeze(0)}])
        masked_input_ids = batch["input_ids"]  # [1, seq_len]
        labels = batch["labels"]  # [1, seq_len], -100 for non-masked

        # Prepare model inputs and ensure attention_mask distinguishes
        # real tokens (1) from padding (0).
        model_inputs = {
            k: v for k, v in encoding.items() if isinstance(v, torch.Tensor)
        }
        model_inputs["input_ids"] = masked_input_ids

        return model_inputs, labels

    def compute(self) -> dict[str, Any]:
        """Run standard MLM evaluation: 15% masking + cross-entropy loss."""
        from transformers import DataCollatorForLanguageModeling

        tokenizer = self._tokenizer

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
            seed=_MLM_SEED,
        )

        max_length = self._get_max_length()
        metric = CrossEntropyMetric()

        for i, sample in enumerate(self.data):
            if (i + 1) % 50 == 0:
                logger.info("%d samples, %d masked tokens", i + 1, metric.total_tokens)

            result = self._tokenize_and_mask(
                sample[self._input_col], tokenizer, data_collator, max_length,
            )
            if result is None:
                continue

            model_inputs, labels = result

            with torch.no_grad():
                outputs = self.model(**model_inputs)

            logits = self._extract_logits(outputs)  # [1, seq_len, vocab_size]
            metric.update(logits[0], labels[0])

        return metric.compute()
