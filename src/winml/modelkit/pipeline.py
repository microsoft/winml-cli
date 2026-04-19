# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Shared HF Pipeline factory for WinML models.

Used by both ``winml serve`` (InferenceEngine) and ``winml eval`` (WinMLEvaluator)
to create a ``transformers.pipeline`` backed by a WinMLPreTrainedModel.

The pipeline handles all preprocessing and postprocessing; the WinML model
only provides the ONNX Runtime inference session.

ONNX models have fixed input shapes. This module adapts the pipeline's
tokenizer/image_processor to match those shapes so inputs are correctly
padded/resized before hitting the ONNX runtime.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .models.winml.base import WinMLPreTrainedModel

logger = logging.getLogger(__name__)


def create_pipeline(
    task: str,
    model: WinMLPreTrainedModel,
    model_id: str | None = None,
) -> Any:
    """Create an HF pipeline for a WinML model.

    Automatically adapts tokenizer padding and image processor size
    to match the ONNX model's fixed input shapes.

    Args:
        task: HF task name (e.g. "image-classification")
        model: Loaded WinMLPreTrainedModel instance
        model_id: HF model ID for loading processors (tokenizer, image processor).
                  If None, pipeline will attempt auto-detection.

    Returns:
        A configured ``transformers.Pipeline`` ready for inference.
    """
    from transformers import pipeline

    kwargs: dict[str, Any] = {
        "framework": "pt",
        # "device" is for HF pipeline tensor placement, not ORT EP.
        # WinMLSession handles device delegation internally.
        "device": "cpu",
    }
    if model_id:
        kwargs["tokenizer"] = model_id
        kwargs["feature_extractor"] = model_id
        kwargs["image_processor"] = model_id
        kwargs["processor"] = model_id

    pipe = pipeline(task, model=model, **kwargs)

    # Adapt pipeline to fixed ONNX input shapes
    _adapt_tokenizer_padding(pipe, task, model)
    _adapt_image_processor_size(pipe, task, model)

    logger.info("Created HF pipeline: task=%s model=%s", task, model_id)
    return pipe


def _adapt_tokenizer_padding(pipe: Any, task: str, model: Any) -> None:
    """Pad tokenizer output to match ONNX fixed sequence length.

    ONNX models are exported with a fixed sequence_length dimension.
    Without padding, the tokenizer produces variable-length tensors
    that cause INVALID_ARGUMENT errors at inference time.

    Detection is property-driven (not task-name driven):
    the adaptation fires when the pipeline has a tokenizer AND the
    model's first input shape is 2-D with a fixed integer second
    dimension (batch, sequence_length).  4-D shapes (N, C, H, W) are
    image tensors and are explicitly skipped.
    """
    if pipe.tokenizer is None:
        return

    io_config = getattr(model, "io_config", None) or {}
    shapes = io_config.get("input_shapes", [[]])
    # Only 2-D shapes (batch, seq_len) — skip 4-D image tensors (N, C, H, W)
    if not (shapes and len(shapes[0]) == 2 and isinstance(shapes[0][1], int)):
        return

    max_length = shapes[0][1]

    # Some pipelines (e.g. TokenClassificationPipeline) nest tokenizer
    # settings under a `tokenizer_params` kwarg instead of top-level
    # preprocess params.  Detect this from the pipeline class signature.
    preprocess_sig = inspect.signature(type(pipe).preprocess)
    if "tokenizer_params" in preprocess_sig.parameters:
        pipe._preprocess_params.setdefault("tokenizer_params", {})
        tok_params = pipe._preprocess_params["tokenizer_params"]
        tok_params.setdefault("padding", "max_length")
        tok_params.setdefault("max_length", max_length)
        pipe._preprocess_params.setdefault("truncation", True)
        pipe.tokenizer.model_max_length = max_length
    else:
        pipe._preprocess_params.setdefault("padding", "max_length")
        pipe._preprocess_params.setdefault("max_length", max_length)
        pipe._preprocess_params.setdefault("truncation", True)


def _adapt_image_processor_size(pipe: Any, task: str, model: Any) -> None:
    """Match image processor size to ONNX fixed input shape (NCHW).

    Models with 4D input shapes have fixed spatial dimensions.
    The image processor must resize to exactly those dimensions.

    Detection is property-driven (not task-name driven):
    the adaptation fires when the pipeline has an image_processor AND
    the model's first input shape is 4D (N, C, H, W).
    """
    if not hasattr(pipe, "image_processor"):
        return

    io_config = getattr(model, "io_config", None) or {}
    input_shapes = io_config.get("input_shapes", [])
    if not (input_shapes and len(input_shapes[0]) == 4):
        return

    _, _, h, w = input_shapes[0]
    # Preserve the image processor's existing size key format.
    # Some processors use {"shortest_edge": N}, others {"height": H, "width": W}.
    existing = getattr(pipe.image_processor, "size", {})
    if "shortest_edge" in existing:
        pipe.image_processor.size = {"shortest_edge": min(h, w)}
    else:
        pipe.image_processor.size = {"height": h, "width": w}
    if hasattr(pipe.image_processor, "do_pad"):
        pipe.image_processor.do_pad = False
