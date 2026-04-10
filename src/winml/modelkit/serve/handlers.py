# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Task-specific input/output handlers for InferenceEngine.

Adding a new task requires two steps only:
  1. Write a subclass of TaskHandler (preprocess + postprocess)
  2. Register it in _HANDLER_REGISTRY

InferenceEngine calls resolve_handler() after load() and delegates
all pre/post processing to the returned handler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .schema import Prediction


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class TaskHandler(ABC):
    """Protocol for task-specific pre/post processing."""

    def preprocess(
        self,
        *,
        image_bytes: bytes | None = None,
        text: str | None = None,
        tensor_inputs: dict[str, list] | None = None,
    ) -> dict[str, Any]:
        """Convert raw inputs to model-ready tensors.

        tensor_inputs bypasses task-specific preprocessing entirely —
        the caller is responsible for providing correctly shaped arrays.
        """
        if tensor_inputs is not None:
            return {k: np.array(v) for k, v in tensor_inputs.items()}
        return self._preprocess(image_bytes=image_bytes, text=text)

    @abstractmethod
    def _preprocess(
        self,
        *,
        image_bytes: bytes | None,
        text: str | None,
    ) -> dict[str, Any]:
        """Task-specific preprocessing (called when tensor_inputs is None)."""

    @abstractmethod
    def postprocess(self, output: Any, *, top_k: int = 5) -> list[Prediction] | dict[str, Any]:
        """Convert raw model output to structured predictions."""


# ---------------------------------------------------------------------------
# Shared preprocessing mixins
# ---------------------------------------------------------------------------


class _ImagePreprocessMixin:
    """Shared image preprocessing for all image-input tasks."""

    _processor: Any  # AutoImageProcessor or None

    def _preprocess(self, *, image_bytes: bytes | None, text: str | None) -> dict[str, Any]:
        if image_bytes is None:
            raise ValueError("image_bytes required for image tasks")
        from io import BytesIO

        from PIL import Image

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        if self._processor is not None:
            return dict(self._processor(images=image, return_tensors="pt"))

        # Fallback: standard ImageNet preprocessing
        import torchvision.transforms as T

        transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return {"pixel_values": transform(image).unsqueeze(0)}


class _TextPreprocessMixin:
    """Shared text preprocessing for all text-input tasks."""

    _processor: Any  # AutoTokenizer or None
    _max_length: int | None  # from model.io_config["input_shapes"][0][1], or None

    def _preprocess(self, *, image_bytes: bytes | None, text: str | None) -> dict[str, Any]:
        if text is None:
            raise ValueError("text required for text tasks")
        if self._processor is None:
            raise ValueError("No tokenizer loaded — cannot preprocess text input")
        kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "truncation": True,
            "padding": "max_length",
        }
        if self._max_length is not None:
            kwargs["max_length"] = self._max_length
        return dict(self._processor(text, **kwargs))


# ---------------------------------------------------------------------------
# Shared postprocessing helpers
# ---------------------------------------------------------------------------


def _classification_postprocess(
    output: Any,
    id2label: dict[int, str],
    top_k: int,
) -> list[Prediction]:
    import torch
    import torch.nn.functional as F

    logits = getattr(output, "logits", None)
    if logits is None:
        if hasattr(output, "values"):
            logits = next(iter(output.values()))
        else:
            raise ValueError("Cannot extract logits from model output")

    probs = F.softmax(logits.float(), dim=-1)
    k = min(top_k, probs.shape[-1])
    top = torch.topk(probs, k=k, dim=-1)
    return [
        Prediction(label=id2label.get(idx, str(idx)), score=round(score, 6))
        for idx, score in zip(top.indices[0].tolist(), top.values[0].tolist(), strict=False)
    ]


def _raw_tensor_postprocess(output: Any, attrs: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attr in attrs:
        val = getattr(output, attr, None)
        if val is not None:
            try:
                result[attr] = val.detach().cpu().numpy().tolist()
            except Exception:
                result[attr] = str(val)
    return result or {"raw": str(output)}


# ---------------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------------


class ImageClassificationHandler(_ImagePreprocessMixin, TaskHandler):
    """Image classification — softmax over logits, returns top-k label/score pairs."""

    def __init__(self, processor: Any, id2label: dict[int, str]) -> None:
        self._processor = processor
        self._id2label = id2label

    def postprocess(self, output: Any, *, top_k: int = 5) -> list[Prediction]:
        """Return top-k predicted labels with confidence scores."""
        return _classification_postprocess(output, self._id2label, top_k)


class ObjectDetectionHandler(_ImagePreprocessMixin, TaskHandler):
    """Object detection — returns raw pred_boxes, logits, pred_labels tensors."""

    def __init__(self, processor: Any, id2label: dict[int, str]) -> None:
        self._processor = processor
        self._id2label = id2label

    def postprocess(self, output: Any, *, top_k: int = 5) -> dict[str, Any]:
        """Return raw detection tensors as nested lists."""
        return _raw_tensor_postprocess(output, ["pred_boxes", "logits", "pred_labels"])


class ImageSegmentationHandler(_ImagePreprocessMixin, TaskHandler):
    """Image segmentation — returns raw pred_masks and logits tensors."""

    def __init__(self, processor: Any) -> None:
        self._processor = processor

    def postprocess(self, output: Any, *, top_k: int = 5) -> dict[str, Any]:
        """Return raw segmentation masks as nested lists."""
        return _raw_tensor_postprocess(output, ["pred_masks", "logits"])


class TextClassificationHandler(_TextPreprocessMixin, TaskHandler):
    """Text / sequence classification — softmax over logits, returns top-k label/score pairs."""

    def __init__(
        self, processor: Any, id2label: dict[int, str], max_length: int | None = None
    ) -> None:
        self._processor = processor
        self._id2label = id2label
        self._max_length = max_length

    def postprocess(self, output: Any, *, top_k: int = 5) -> list[Prediction]:
        """Return top-k predicted labels with confidence scores."""
        return _classification_postprocess(output, self._id2label, top_k)


class TokenClassificationHandler(_TextPreprocessMixin, TaskHandler):
    """NER / token-level classification — returns per-token label list."""

    def __init__(
        self, processor: Any, id2label: dict[int, str], max_length: int | None = None
    ) -> None:
        self._processor = processor
        self._id2label = id2label
        self._max_length = max_length
        # Mirror eval behaviour: set model_max_length so the tokenizer never silently truncates
        if processor is not None and max_length is not None:
            processor.model_max_length = max_length

    def postprocess(self, output: Any, *, top_k: int = 5) -> dict[str, Any]:
        """Return per-token entity labels as a list of {token_idx, label} dicts."""
        import torch

        logits = getattr(output, "logits", None)
        if logits is None:
            return {"raw": str(output)}
        preds = torch.argmax(logits, dim=-1)[0].tolist()
        return {
            "entities": [
                {"token_idx": i, "label": self._id2label.get(p, str(p))}
                for i, p in enumerate(preds)
            ]
        }


class _FallbackHandler(TaskHandler):
    """Unknown task — accepts tensor_inputs only, returns raw output."""

    def __init__(self, processor: Any) -> None:
        self._processor = processor

    def _preprocess(self, *, image_bytes: bytes | None, text: str | None) -> dict[str, Any]:
        raise ValueError("Unknown task — cannot preprocess. Provide tensor_inputs directly.")

    def postprocess(self, output: Any, *, top_k: int = 5) -> dict[str, Any]:
        return _raw_tensor_postprocess(
            output, ["logits", "pred_masks", "pred_boxes", "pred_labels"]
        )


# ---------------------------------------------------------------------------
# Registry  (task name → handler class)
# To add a new task: write a handler above, add one line here.
# ---------------------------------------------------------------------------

_HANDLER_REGISTRY: dict[str, type[TaskHandler]] = {
    "image-classification": ImageClassificationHandler,
    "object-detection": ObjectDetectionHandler,
    "image-segmentation": ImageSegmentationHandler,
    "text-classification": TextClassificationHandler,
    "sentiment-analysis": TextClassificationHandler,
    "token-classification": TokenClassificationHandler,
}


def resolve_handler(task: str | None, processor: Any, model: Any) -> TaskHandler:
    """Instantiate the right TaskHandler for the given task.

    Falls back to _FallbackHandler for unknown tasks.
    """
    id2label: dict[int, str] = {}
    if model is not None and getattr(model, "config", None) is not None:
        id2label = getattr(model.config, "id2label", {})

    # Mirror eval pattern: read sequence length from model I/O config
    max_length: int | None = None
    io_config = getattr(model, "io_config", None) or {}
    shapes = io_config.get("input_shapes", [])
    if shapes and len(shapes[0]) > 1:
        max_length = shapes[0][1]

    cls = _HANDLER_REGISTRY.get(task or "")
    if cls is None:
        return _FallbackHandler(processor)

    import inspect

    params = list(inspect.signature(cls.__init__).parameters.keys())[1:]  # skip 'self'
    kwargs: dict[str, Any] = {"processor": processor}
    if "id2label" in params:
        kwargs["id2label"] = id2label
    if "max_length" in params:
        kwargs["max_length"] = max_length
    return cls(**kwargs)
