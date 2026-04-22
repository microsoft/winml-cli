# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Image-to-text evaluator using image-conditional perplexity.

Evaluates seq2seq vision-encoder-decoder models (e.g. donut, trocr, vit-gpt2)
by computing the negative log-likelihood (NLL) of reference captions given input
images under teacher forcing:

    For each (image, caption) pair:
        decoder_input  = tokenized_caption[:-1]
        target         = tokenized_caption[1:]
        token_log_prob = log P(target_i | image, decoder_input[:i])

    Aggregate:
        nll        = -mean(token_log_probs)        # lower is better
        perplexity = exp(nll)                      # lower is better

Comparing WinML (ONNX) perplexity to the PyTorch baseline perplexity detects
accuracy regressions from export or quantization.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..datasets.config import DatasetConfig
    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig

logger = logging.getLogger(__name__)


class WinMLImageToTextEvaluator(WinMLEvaluator):
    """Evaluator for image-to-text models via image-conditional perplexity."""

    @classmethod
    def schema_info(cls) -> list:
        """Return expected dataset schema for image-to-text evaluation."""
        from .config import SchemaColumn

        return [
            SchemaColumn("image", "Image", "input_column", description="PIL Image"),
            SchemaColumn(
                "caption",
                "Value(string)",
                "caption_column",
                description="Reference caption text (string or list of strings)",
            ),
        ]

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        self._input_col = config.dataset.columns_mapping.get("input_column", "image")
        self._caption_col = config.dataset.columns_mapping.get("caption_column", "caption")
        self._processor: Any = None
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Bypass the HF pipeline; compute() calls the model directly.

        Teacher-forcing perplexity requires access to raw logits at every
        decoder position, which the HF image-to-text pipeline does not expose.
        """
        return None  # type: ignore[return-value]

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """No-op: image-to-text uses no discrete class labels."""
        return dataset

    def _get_processor(self) -> Any:
        """Lazily load the model's processor (image processor + tokenizer)."""
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.config.model_id)
        return self._processor

    def _logits(self, outputs: Any) -> torch.Tensor:
        """Extract logits tensor from model output (dict or object)."""
        if isinstance(outputs, dict):
            if "logits" not in outputs:
                raise KeyError(f"Model output dict has no 'logits' key; got keys {list(outputs)}.")
            return outputs["logits"]
        return outputs.logits

    def _score_sample(
        self,
        image: Any,
        caption: str,
        processor: Any,
    ) -> torch.Tensor | None:
        """Compute per-token log P(caption | image) for one sample.

        Args:
            image: PIL Image or any format accepted by the model's image processor.
            caption: Reference caption string.
            processor: AutoProcessor instance with image_processor and tokenizer.

        Returns:
            1-D tensor of per-token log-probabilities, or None if the sample
            should be skipped (e.g. too short, processor error).
        """
        # Process image → pixel_values
        try:
            processed = processor(images=image, return_tensors="pt")
        except Exception as exc:
            logger.debug("Image processing failed, skipping sample: %s", exc)
            return None

        pixel_values = processed.get("pixel_values")
        if pixel_values is None:
            logger.debug("Processor did not return 'pixel_values', skipping sample.")
            return None

        # Tokenize caption
        tokenizer = getattr(processor, "tokenizer", processor)
        tok_out = tokenizer(caption, return_tensors="pt", truncation=True, max_length=128)
        input_ids = tok_out["input_ids"]  # [1, seq_len]

        if input_ids.shape[1] < 2:
            logger.debug("Caption too short after tokenization, skipping sample.")
            return None

        # Teacher forcing: shift right so decoder_input[i] predicts target[i]
        decoder_input_ids = input_ids[:, :-1]  # [1, seq_len-1]
        target_ids = input_ids[:, 1:]  # [1, seq_len-1]

        with torch.no_grad():
            try:
                outputs = self.model(
                    pixel_values=pixel_values,
                    decoder_input_ids=decoder_input_ids,
                )
            except Exception as exc:
                logger.debug("Model forward pass failed, skipping sample: %s", exc)
                return None

        logits = self._logits(outputs)  # [1, seq_len-1, vocab_size]

        # Per-token log P(target | context, image)
        log_probs = F.log_softmax(logits[0], dim=-1)  # [seq_len-1, vocab_size]
        target_flat = target_ids[0].long()  # [seq_len-1]
        return log_probs[torch.arange(target_flat.shape[0]), target_flat]  # [seq_len-1]

    def compute(self) -> dict[str, Any]:
        """Run image-conditional perplexity evaluation over the dataset.

        Returns:
            Dict with keys:
                nll:        mean negative log-likelihood per token (lower = better)
                perplexity: exp(nll) (lower = better)

        Raises:
            ValueError: If no valid token log-probabilities were accumulated.
        """
        processor = self._get_processor()

        total_neg_log_p = 0.0
        total_tokens = 0

        for sample in tqdm(self.data, desc="Evaluating image-to-text (perplexity)"):
            image = sample.get(self._input_col)
            caption = sample.get(self._caption_col)

            if image is None or caption is None:
                continue

            # Handle list of captions — take the first
            if isinstance(caption, list):
                if not caption:
                    continue
                caption = caption[0]

            if not isinstance(caption, str) or not caption.strip():
                continue

            token_log_probs = self._score_sample(image, caption, processor)
            if token_log_probs is None or token_log_probs.numel() == 0:
                continue

            total_neg_log_p += float(-token_log_probs.sum().item())
            total_tokens += int(token_log_probs.numel())

        if total_tokens == 0:
            raise ValueError(
                "No valid token log-probabilities accumulated during image-to-text "
                "evaluation. Check that the dataset has non-empty 'image' and "
                f"'{self._caption_col}' columns and that the model accepts "
                "pixel_values + decoder_input_ids."
            )

        mean_nll = total_neg_log_p / total_tokens
        return {
            "nll": round(mean_nll, 4),
            "perplexity": round(math.exp(mean_nll), 4),
        }
