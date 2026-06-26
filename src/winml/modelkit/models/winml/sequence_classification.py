# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Model for Sequence Classification.

Thin wrapper for sequence/text classification inference.
Pipeline execution (export/optimize/compile) is done by WinMLAutoModel factory.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from transformers.modeling_outputs import SequenceClassifierOutput

from .base import WinMLPreTrainedModel


if TYPE_CHECKING:
    import numpy as np
    import torch

logger = logging.getLogger(__name__)


class WinMLModelForSequenceClassification(WinMLPreTrainedModel):
    """WinML model for sequence classification.

    Supports:
    - text-classification
    - sequence-classification
    - next-sentence-prediction

    Thin wrapper - only handles inference I/O.
    Pipeline execution is done by WinMLAutoModel factory.
    """

    def forward(  # type: ignore[override]  # HF-pipeline base uses generic **kwargs; task-specific signature
        self,
        input_ids: torch.Tensor | np.ndarray,
        attention_mask: torch.Tensor | np.ndarray | None = None,
        token_type_ids: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput:
        """Run sequence classification inference.

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            token_type_ids: Segment IDs for BERT-like models (B, seq_len)
            **kwargs: Additional arguments (ignored, for HF pipeline compatibility)

        Returns:
            SequenceClassifierOutput with logits
        """
        # Build inputs dict - only include non-None values
        accepted_inputs = set(self.io_config.get("input_names", []))
        inputs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None and "attention_mask" in accepted_inputs:
            inputs["attention_mask"] = attention_mask
        if token_type_ids is not None and "token_type_ids" in accepted_inputs:
            inputs["token_type_ids"] = token_type_ids

        # Use base class helpers for validation, formatting, and inference
        formatted = self._format_inputs(**inputs)
        outputs = self._run_inference(formatted)

        # Get logits (by name or first output)
        logits = outputs.get("logits", next(iter(outputs.values())))

        # transformers' Output fields are annotated FloatTensor (legacy, over-narrow);
        # the ONNX session returns a real float Tensor.
        return SequenceClassifierOutput(logits=cast("torch.FloatTensor", logits))

    @property
    def num_labels(self) -> int:
        """Number of classification labels."""
        if self.config is not None:
            return getattr(self.config, "num_labels", 2)
        return 2

    @property
    def id2label(self) -> dict[int, str]:
        """Mapping from label ID to label name."""
        if self.config is not None:
            return getattr(self.config, "id2label", {})
        return {}

    @property
    def label2id(self) -> dict[str, int]:
        """Mapping from label name to label ID."""
        if self.config is not None:
            return getattr(self.config, "label2id", {})
        return {}
