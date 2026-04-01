"""WinML Model for Feature Extraction.

Thin wrapper for feature extraction inference (sentence embeddings, etc.).
Pipeline execution (export/optimize/compile) is done by WinMLAutoModel factory.
"""

from __future__ import annotations

import logging
from typing import Any

from transformers.modeling_outputs import BaseModelOutput

from .base import WinMLPreTrainedModel


logger = logging.getLogger(__name__)


class WinMLModelForFeatureExtraction(WinMLPreTrainedModel):
    """WinML model for feature extraction.

    Supports:
    - feature-extraction (text, e.g. sentence-transformers)

    Returns BaseModelOutput with last_hidden_state so the HF
    feature-extraction pipeline can consume it via output[0].

    ONNX output handling (shape-based, architecture-agnostic):
    - 3-D [B, seq_len, hidden_dim]: used directly as last_hidden_state.
    - 2-D [B, hidden_dim]: unsqueezed to [B, 1, hidden_dim] so downstream
      mean-pooling is a no-op.
    """

    def forward(self, **kwargs: Any) -> BaseModelOutput:
        """Run feature extraction inference.

        Accepts all tokenizer/processor outputs via **kwargs and passes them
        directly to the ONNX session, keeping the implementation architecture-agnostic.

        Returns:
            BaseModelOutput with last_hidden_state tensor
        """
        formatted = self._format_inputs(**kwargs)
        outputs = self._run_inference(formatted)

        last_hidden_state = next(iter(outputs.values()))
        if last_hidden_state.dim() == 2:
            # Already pooled [B, hidden_dim] -> wrap as [B, 1, hidden_dim]
            last_hidden_state = last_hidden_state.unsqueeze(1)

        return BaseModelOutput(last_hidden_state=last_hidden_state)
