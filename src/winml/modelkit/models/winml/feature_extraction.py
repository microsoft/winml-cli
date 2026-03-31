"""WinML Model for Feature Extraction.

Thin wrapper for feature extraction inference (sentence embeddings, etc.).
Pipeline execution (export/optimize/compile) is done by WinMLAutoModel factory.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from transformers.modeling_outputs import BaseModelOutput

from .base import WinMLPreTrainedModel


if TYPE_CHECKING:
    import numpy as np
    import torch

logger = logging.getLogger(__name__)


class WinMLModelForFeatureExtraction(WinMLPreTrainedModel):
    """WinML model for feature extraction.

    Supports:
    - feature-extraction (text, e.g. sentence-transformers)

    Returns BaseModelOutput with last_hidden_state so the HF
    feature-extraction pipeline can consume it via output[0].

    ONNX output handling:
    - ``last_hidden_state`` (shape [B, seq_len, hidden_dim]): used directly.
    - ``sentence_embedding``  (shape [B, hidden_dim]): unsqueezed to
      [B, 1, hidden_dim] so downstream mean-pooling is a no-op.
    - Any other single output: treated as last_hidden_state.
    """

    def forward(
        self,
        input_ids: torch.Tensor | np.ndarray,
        attention_mask: torch.Tensor | np.ndarray | None = None,
        token_type_ids: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> BaseModelOutput:
        """Run feature extraction inference.

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            token_type_ids: Segment IDs for BERT-like models (B, seq_len)
            **kwargs: Additional arguments (ignored, for HF pipeline compatibility)

        Returns:
            BaseModelOutput with last_hidden_state tensor
        """
        inputs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        formatted = self._format_inputs(**inputs)
        outputs = self._run_inference(formatted)

        # Prefer last_hidden_state; fall back to sentence_embedding or first output
        if "last_hidden_state" in outputs:
            last_hidden_state = outputs["last_hidden_state"]
        elif "sentence_embedding" in outputs:
            # Already pooled [B, hidden_dim] -> wrap as [B, 1, hidden_dim]
            last_hidden_state = outputs["sentence_embedding"].unsqueeze(1)
        else:
            last_hidden_state = next(iter(outputs.values()))
            if last_hidden_state.dim() == 2:
                last_hidden_state = last_hidden_state.unsqueeze(1)

        return BaseModelOutput(last_hidden_state=last_hidden_state)
