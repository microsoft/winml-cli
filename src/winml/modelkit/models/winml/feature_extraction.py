# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""WinML Model for Feature Extraction.

Thin wrapper for feature extraction inference (sentence embeddings, etc.).
Pipeline execution (export/optimize/compile) is done by WinMLAutoModel factory.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

from transformers.utils.generic import ModelOutput

from .base import WinMLPreTrainedModel


logger = logging.getLogger(__name__)


class FeatureExtractionModelOutput(ModelOutput):
    """ModelOutput backed directly by the ONNX session output dict.

    Preserves output names and order so HF pipelines (`output[0]`) and
    TensorSimilarityEvaluator (`output["name"]`) both work without a
    per-schema dataclass.
    """

    def __init__(self, data: dict[str, Any]):
        OrderedDict.__init__(self, data)

    def __post_init__(self) -> None:
        # Bypass ModelOutput's dataclass-driven population; OrderedDict
        # is already populated via __init__.
        pass


class WinMLModelForFeatureExtraction(WinMLPreTrainedModel):
    """WinML model for feature extraction.

    Supports text and image feature-extraction plus sentence-similarity.

    Returns a ModelOutput whose entries mirror the ONNX exporter's declared
    output names and order. HF pipelines consume output[0] positionally;
    TensorSimilarityEvaluator consumes by name. Both work without renaming
    or reshaping ONNX outputs.
    """

    def forward(self, **kwargs: Any) -> ModelOutput:
        """Run feature extraction inference.

        Returns a ModelOutput with one entry per ONNX output in declared
        order. Tensors keep their native rank (no unsqueeze); downstream
        pooling handles 1-D and 2-D after raw[0].
        """
        outputs = self._run_inference(self._format_inputs(**kwargs))
        # WinMLEncoderDecoderModel expects its encoder sub-component to expose
        # hidden states as "last_hidden_state". Alias the primary output when an
        # encoder ONNX graph named it otherwise (e.g. "encoder_hidden_states").
        # Appended last to preserve output[0] and the real ONNX output names.
        if outputs and "last_hidden_state" not in outputs:
            outputs["last_hidden_state"] = next(iter(outputs.values()))
        return FeatureExtractionModelOutput(outputs)
