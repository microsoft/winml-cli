# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Model for Question Answering.

Thin wrapper for extractive question answering inference.
Filters out inputs not present in the ONNX model (e.g., token_type_ids
for DeBERTa-v3) and returns QuestionAnsweringModelOutput with
start_logits and end_logits.

Pipeline execution (export/optimize/compile) is done by WinMLAutoModel factory.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from transformers.modeling_outputs import QuestionAnsweringModelOutput

from .base import WinMLPreTrainedModel


if TYPE_CHECKING:
    import numpy as np
    import torch

logger = logging.getLogger(__name__)


class WinMLModelForQuestionAnswering(WinMLPreTrainedModel):
    """WinML model for extractive question answering.

    Mirrors HuggingFace AutoModelForQuestionAnswering.
    Handles models that may or may not accept token_type_ids
    (e.g., DeBERTa-v3 does not).

    Thin wrapper - only handles inference I/O.
    Pipeline execution is done by WinMLAutoModel factory.
    """

    def forward(
        self,
        input_ids: torch.Tensor | np.ndarray | None = None,
        attention_mask: torch.Tensor | np.ndarray | None = None,
        token_type_ids: torch.Tensor | np.ndarray | None = None,
        bbox: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> QuestionAnsweringModelOutput:
        """Run question answering inference.

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            token_type_ids: Segment IDs (B, seq_len). Only passed to ONNX
                if the model actually has this input.  Models like
                DeBERTa-v3 omit this input; it is silently dropped when
                not in the ONNX graph's input list.
            bbox: Layout-aware token bounding boxes (B, seq_len, 4). Only
                passed to ONNX when the exported graph declares a ``bbox`` input.
            **kwargs: Ignored. Accepted for HF pipeline compatibility —
                the pipeline may forward extra keys (e.g. ``offset_mapping``,
                ``overflow_to_sample_mapping``) that are not needed for
                ONNX inference.

        Returns:
            QuestionAnsweringModelOutput with start_logits and end_logits
        """
        if input_ids is None:
            raise ValueError("input_ids must be provided for question answering inference.")

        inputs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        input_names = self.io_config.get("input_names", [])
        if token_type_ids is not None and "token_type_ids" in input_names:
            inputs["token_type_ids"] = token_type_ids
        if bbox is not None and "bbox" in input_names:
            inputs["bbox"] = bbox

        formatted = self._format_inputs(**inputs)
        outputs = self._run_inference(formatted)

        # transformers' Output fields are annotated FloatTensor (legacy, over-narrow);
        # the ONNX session returns real float Tensors.
        return QuestionAnsweringModelOutput(
            start_logits=cast("torch.FloatTensor | None", outputs.get("start_logits")),
            end_logits=cast("torch.FloatTensor | None", outputs.get("end_logits")),
        )
