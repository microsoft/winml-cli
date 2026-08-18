# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML runtime wrapper for tensor-based tabular classification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from transformers.modeling_outputs import SequenceClassifierOutput

from .base import WinMLPreTrainedModel


if TYPE_CHECKING:
    import torch


class WinMLModelForTabularClassification(WinMLPreTrainedModel):
    """Run a tabular classifier whose ONNX input is named ``features``."""

    def forward(self, **kwargs: Any) -> SequenceClassifierOutput:
        """Run inference and expose the classifier logits."""
        outputs = self._run_inference(self._format_inputs(**kwargs))
        logits = outputs.get("logits", next(iter(outputs.values())))
        return SequenceClassifierOutput(logits=cast("torch.FloatTensor", logits))
