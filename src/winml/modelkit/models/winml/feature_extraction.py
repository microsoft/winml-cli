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

import numpy as np
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
        inputs = self._format_inputs(**kwargs)
        if "token_type_ids" not in inputs and "input_ids" in inputs:
            input_names = self.io_config.get("input_names", [])
            if "token_type_ids" in input_names:
                input_index = input_names.index("token_type_ids")
                input_types = self.io_config.get("input_types", [])
                input_shapes = self.io_config.get("input_shapes", [])
                if input_index < len(input_types) and input_index < len(input_shapes):
                    required_shape = input_shapes[input_index]
                    actual_shape = inputs["input_ids"].shape
                    shape_matches = len(required_shape) == len(actual_shape) and all(
                        not isinstance(dimension, int)
                        or dimension <= 0
                        or dimension == actual_dimension
                        for dimension, actual_dimension in zip(
                            required_shape, actual_shape, strict=True
                        )
                    )
                    if shape_matches:
                        inputs["token_type_ids"] = np.zeros(
                            actual_shape,
                            dtype=np.dtype(input_types[input_index]),
                        )

        outputs = self._run_inference(inputs)
        # WinMLEncoderDecoderModel expects its encoder sub-component to expose
        # hidden states as "last_hidden_state". Alias the primary output when an
        # encoder ONNX graph named it otherwise (e.g. "encoder_hidden_states").
        # Appended last to preserve output[0] and the real ONNX output names.
        if outputs and "last_hidden_state" not in outputs:
            outputs["last_hidden_state"] = next(iter(outputs.values()))
        return FeatureExtractionModelOutput(outputs)
