# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Task registry — single source of truth for user inputs and pipeline dispatch.

Provides:
    TASK_REGISTRY  — maps task name → TaskInputSpec (user_inputs + pipeline mapping)
    InputField, PipelineMapping, TaskInputSpec dataclasses
    BINARY_TYPES   — frozenset of types that require binary decoding
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_TYPES = frozenset({"image", "audio", "video", "text", "json", "number", "boolean"})
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

BINARY_TYPES = frozenset({"image", "audio", "video"})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InputField:
    """Schema for a single user input."""

    name: str
    type: str  # One of _VALID_TYPES
    required: bool
    description: str = ""
    default: Any = None  # Only valid when required is False; None = no default

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"Invalid input name: '{self.name}'")
        if self.type not in _VALID_TYPES:
            raise ValueError(f"Invalid type '{self.type}' for input '{self.name}'")
        if self.required and self.default is not None:
            raise ValueError(f"Input '{self.name}': default is not valid when required is True")
        if not self.description:
            self.description = self.name.replace("_", " ").title()


@dataclass
class PipelineMapping:
    """Describes how to convert user inputs into a pipeline call.

    pipe_input semantics:
        "image"                → pipeline(inputs["image"])                  single value
        ["question","context"] → pipeline({"question":..,"context":..})     dict passthrough

    pipe_input_as_list:
        When True and pipe_input is a list, the dict is unpacked to a list
        (e.g., keypoint-matching expects [img0, img1], not a dict).

    pipe_kwargs:
        Input names routed to pipeline(**kwargs) instead of positional arg.
        Names must match the pipeline kwarg names.

    Binary decoding (image → PIL, audio → dict, video → path) is inferred
    automatically from InputField.type — no explicit decode mapping needed.
    """

    pipe_input: str | list[str]
    pipe_kwargs: list[str] = field(default_factory=list)
    pipe_input_as_list: bool = False


@dataclass
class TaskInputSpec:
    """Single source of truth for a task's user inputs and pipeline dispatch.

    Attributes:
        postprocess: Optional callback
            ``(raw, *, pipeline=None, inputs=None) -> predictions``
            that transforms the HF pipeline's raw output into the standard
            ``list[Prediction] | dict`` format.  The engine passes the
            pipeline instance and validated user inputs as keyword arguments
            so that callbacks can access the tokenizer or original texts
            when needed.  When ``None``, the engine's default normalisation
            logic is used.
    """

    user_inputs: list[InputField]
    mapping: PipelineMapping
    postprocess: Callable[..., Any] | None = None


# ---------------------------------------------------------------------------
# Postprocess callbacks
# ---------------------------------------------------------------------------


def _masked_mean_pool(
    token_embeddings: np.ndarray,
    attention_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Mean-pool token embeddings, optionally weighted by attention mask.

    When *attention_mask* is provided, padding tokens are excluded from the
    average.  This is critical for fixed-shape ONNX models where 98%+ of
    tokens can be padding — without masking the sentence embedding is
    dominated by meaningless padding vectors.
    """
    if attention_mask is not None:
        mask = attention_mask.astype(float)
        denom = mask.sum()
        if denom > 0:
            return cast("np.ndarray", (token_embeddings * mask[:, None]).sum(0) / denom)
    if token_embeddings.ndim > 1:
        return cast("np.ndarray", token_embeddings.mean(axis=0))
    return token_embeddings


def _encode_mask_png(mask: Any) -> str:
    """Encode a PIL Image or numpy-array mask as a base64 PNG string."""
    import base64
    from io import BytesIO

    import numpy as np
    from PIL import Image

    if isinstance(mask, np.ndarray):
        img = Image.fromarray(mask.astype(np.uint8))
    elif isinstance(mask, Image.Image):
        img = mask
    else:
        img = Image.fromarray(np.asarray(mask).astype(np.uint8))

    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _postprocess_inpainting(raw: Any, **_kwargs: Any) -> dict[str, Any]:
    """Encode an inpainted PIL image as a JSON-safe PNG payload."""
    return {
        "image": _encode_mask_png(raw),
        "format": "png",
        "width": raw.width,
        "height": raw.height,
    }


def _postprocess_segmentation(raw: Any, **_kwargs: Any) -> Any:
    """Convert segmentation masks to predictions with coverage score and mask.

    Semantic segmentation pipelines return ``score=None`` and a binary mask
    per class.  We compute the fraction of non-zero pixels (area coverage)
    as a meaningful numeric score, encode the mask as base64 PNG, filter out
    empty masks, and sort by coverage descending.
    """
    import numpy as np

    from .types import Prediction

    predictions: list[Prediction] = []
    for item in raw:
        mask = item.get("mask")
        if mask is None:
            continue
        arr = np.asarray(mask)
        coverage = float(np.count_nonzero(arr)) / arr.size if arr.size > 0 else 0.0
        if coverage <= 0:
            continue
        predictions.append(
            Prediction(
                label=str(item.get("label", "unknown")),
                score=round(coverage, 6),
                mask=_encode_mask_png(mask),
            )
        )
    predictions.sort(key=lambda p: p.score or 0, reverse=True)
    return predictions


def _postprocess_sentence_similarity(
    raw: Any,
    *,
    pipeline: Any = None,
    inputs: dict[str, Any] | None = None,
) -> Any:
    """Compute cosine similarity with attention-masked mean pooling.

    Uses the pipeline's tokenizer (when available) to obtain attention masks
    so that padding tokens are excluded from the mean-pooled sentence
    embeddings.  This matches the approach used by ``winml eval`` and is
    critical for fixed-shape ONNX models where most tokens are padding.
    """
    import numpy as np

    from .types import Prediction

    if not isinstance(raw, list) or len(raw) < 2:
        return {"raw": str(raw)}

    arr_1 = np.array(raw[0], dtype=np.float32).squeeze()
    arr_2 = np.array(raw[1], dtype=np.float32).squeeze()

    # Obtain attention masks from the tokenizer when possible.
    mask_1, mask_2 = None, None
    tokenizer = getattr(pipeline, "tokenizer", None) if pipeline else None
    if tokenizer is not None and inputs is not None:
        params = getattr(pipeline, "_preprocess_params", {})
        tok_kwargs: dict[str, Any] = {
            "padding": params.get("padding", False),
            "max_length": params.get("max_length", None),
            "truncation": params.get("truncation", False),
            "return_tensors": "np",
        }
        text_1 = inputs.get("text_1")
        text_2 = inputs.get("text_2")
        if text_1 is not None and text_2 is not None:
            enc_1 = tokenizer(text_1, **tok_kwargs)
            enc_2 = tokenizer(text_2, **tok_kwargs)
            mask_1 = enc_1["attention_mask"][0]
            mask_2 = enc_2["attention_mask"][0]

    emb_1 = _masked_mean_pool(arr_1, mask_1)
    emb_2 = _masked_mean_pool(arr_2, mask_2)

    norm = float(np.linalg.norm(emb_1) * np.linalg.norm(emb_2))
    sim = float(np.dot(emb_1, emb_2) / norm) if norm > 0 else 0.0
    return [Prediction(label="similarity", score=round(sim, 6))]


# ---------------------------------------------------------------------------
# Task Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, TaskInputSpec] = {
    # -- Single image -------------------------------------------------------
    "image-classification": TaskInputSpec(
        user_inputs=[
            InputField(name="image", type="image", required=True, description="Image to classify"),
        ],
        mapping=PipelineMapping(pipe_input="image"),
    ),
    "object-detection": TaskInputSpec(
        user_inputs=[
            InputField(
                name="image", type="image", required=True, description="Image for object detection"
            ),
        ],
        mapping=PipelineMapping(pipe_input="image"),
    ),
    "image-segmentation": TaskInputSpec(
        user_inputs=[
            InputField(name="image", type="image", required=True, description="Image to segment"),
        ],
        mapping=PipelineMapping(pipe_input="image"),
        postprocess=_postprocess_segmentation,
    ),
    "depth-estimation": TaskInputSpec(
        user_inputs=[
            InputField(
                name="image",
                type="image",
                required=True,
                description="Image for depth estimation",
            ),
        ],
        mapping=PipelineMapping(pipe_input="image"),
    ),
    "image-feature-extraction": TaskInputSpec(
        user_inputs=[
            InputField(
                name="image",
                type="image",
                required=True,
                description="Image to extract features from",
            ),
        ],
        mapping=PipelineMapping(pipe_input="image"),
    ),
    "image-to-image": TaskInputSpec(
        user_inputs=[
            InputField(name="image", type="image", required=True, description="Input image"),
        ],
        mapping=PipelineMapping(pipe_input="image"),
    ),
    "inpainting": TaskInputSpec(
        user_inputs=[
            InputField(name="image", type="image", required=True, description="Input image"),
            InputField(
                name="mask",
                type="image",
                required=True,
                description="Binary mask; non-zero pixels are replaced",
            ),
        ],
        mapping=PipelineMapping(pipe_input=["image", "mask"]),
        postprocess=_postprocess_inpainting,
    ),
    # -- Single text --------------------------------------------------------
    "text-classification": TaskInputSpec(
        user_inputs=[
            InputField(name="text", type="text", required=True, description="Text to classify"),
        ],
        mapping=PipelineMapping(pipe_input="text"),
    ),
    "token-classification": TaskInputSpec(
        user_inputs=[
            InputField(
                name="text",
                type="text",
                required=True,
                description="Text for token classification",
            ),
        ],
        mapping=PipelineMapping(pipe_input="text"),
    ),
    "fill-mask": TaskInputSpec(
        user_inputs=[
            InputField(
                name="text", type="text", required=True, description="Text with [MASK] token"
            ),
        ],
        mapping=PipelineMapping(pipe_input="text"),
    ),
    "text-generation": TaskInputSpec(
        user_inputs=[
            InputField(name="text", type="text", required=True, description="Prompt text"),
        ],
        mapping=PipelineMapping(pipe_input="text"),
    ),
    "text2text-generation": TaskInputSpec(
        user_inputs=[
            InputField(name="text", type="text", required=True, description="Input text"),
        ],
        mapping=PipelineMapping(pipe_input="text"),
    ),
    "feature-extraction": TaskInputSpec(
        user_inputs=[
            InputField(
                name="text",
                type="text",
                required=True,
                description="Text to extract features from",
            ),
        ],
        mapping=PipelineMapping(pipe_input="text"),
    ),
    "summarization": TaskInputSpec(
        user_inputs=[
            InputField(name="text", type="text", required=True, description="Text to summarize"),
        ],
        mapping=PipelineMapping(pipe_input="text"),
    ),
    "translation": TaskInputSpec(
        user_inputs=[
            InputField(name="text", type="text", required=True, description="Text to translate"),
        ],
        mapping=PipelineMapping(pipe_input="text"),
    ),
    "text-to-audio": TaskInputSpec(
        user_inputs=[
            InputField(name="text", type="text", required=True, description="Text to synthesize"),
        ],
        mapping=PipelineMapping(pipe_input="text"),
    ),
    # -- Text + text --------------------------------------------------------
    "question-answering": TaskInputSpec(
        user_inputs=[
            InputField(
                name="question", type="text", required=True, description="The question to answer"
            ),
            InputField(
                name="context", type="text", required=True, description="The context paragraph"
            ),
        ],
        mapping=PipelineMapping(pipe_input=["question", "context"]),
    ),
    "zero-shot-classification": TaskInputSpec(
        user_inputs=[
            InputField(name="text", type="text", required=True, description="Text to classify"),
            InputField(
                name="candidate_labels",
                type="json",
                required=True,
                description="List of candidate label strings",
            ),
        ],
        mapping=PipelineMapping(pipe_input="text", pipe_kwargs=["candidate_labels"]),
    ),
    # -- Single audio -------------------------------------------------------
    "audio-classification": TaskInputSpec(
        user_inputs=[
            InputField(name="audio", type="audio", required=True, description="Audio to classify"),
        ],
        mapping=PipelineMapping(pipe_input="audio"),
    ),
    "automatic-speech-recognition": TaskInputSpec(
        user_inputs=[
            InputField(
                name="audio", type="audio", required=True, description="Audio to transcribe"
            ),
        ],
        mapping=PipelineMapping(pipe_input="audio"),
    ),
    # -- Audio + labels -----------------------------------------------------
    "zero-shot-audio-classification": TaskInputSpec(
        user_inputs=[
            InputField(name="audio", type="audio", required=True, description="Audio to classify"),
            InputField(
                name="candidate_labels",
                type="json",
                required=True,
                description="List of candidate label strings",
            ),
        ],
        mapping=PipelineMapping(pipe_input="audio", pipe_kwargs=["candidate_labels"]),
    ),
    # -- Single video -------------------------------------------------------
    "video-classification": TaskInputSpec(
        user_inputs=[
            InputField(name="video", type="video", required=True, description="Video to classify"),
        ],
        mapping=PipelineMapping(pipe_input="video"),
    ),
    # -- Image + text -------------------------------------------------------
    "visual-question-answering": TaskInputSpec(
        user_inputs=[
            InputField(name="image", type="image", required=True, description="Image to ask about"),
            InputField(
                name="question", type="text", required=True, description="Question about the image"
            ),
        ],
        mapping=PipelineMapping(pipe_input=["image", "question"]),
    ),
    "document-question-answering": TaskInputSpec(
        user_inputs=[
            InputField(name="image", type="image", required=True, description="Document image"),
            InputField(
                name="question",
                type="text",
                required=True,
                description="Question about the document",
            ),
        ],
        mapping=PipelineMapping(pipe_input=["image", "question"]),
    ),
    "image-text-to-text": TaskInputSpec(
        user_inputs=[
            InputField(name="image", type="image", required=True, description="Input image"),
            InputField(name="text", type="text", required=True, description="Text prompt"),
        ],
        mapping=PipelineMapping(pipe_input=["image", "text"]),
    ),
    "image-to-text": TaskInputSpec(
        user_inputs=[
            InputField(name="image", type="image", required=True, description="Image to describe"),
            InputField(
                name="prompt", type="text", required=False, description="Optional text prompt"
            ),
        ],
        mapping=PipelineMapping(pipe_input="image", pipe_kwargs=["prompt"]),
    ),
    "zero-shot-image-classification": TaskInputSpec(
        user_inputs=[
            InputField(name="image", type="image", required=True, description="Image to classify"),
            InputField(
                name="candidate_labels",
                type="json",
                required=True,
                description="List of candidate label strings",
            ),
        ],
        mapping=PipelineMapping(pipe_input="image", pipe_kwargs=["candidate_labels"]),
    ),
    "zero-shot-object-detection": TaskInputSpec(
        user_inputs=[
            InputField(name="image", type="image", required=True, description="Image to search"),
            InputField(
                name="candidate_labels",
                type="json",
                required=True,
                description="List of candidate label strings",
            ),
        ],
        mapping=PipelineMapping(pipe_input="image", pipe_kwargs=["candidate_labels"]),
    ),
    # -- Image pair ---------------------------------------------------------
    "keypoint-matching": TaskInputSpec(
        user_inputs=[
            InputField(name="image_0", type="image", required=True, description="First image"),
            InputField(name="image_1", type="image", required=True, description="Second image"),
        ],
        mapping=PipelineMapping(
            pipe_input=["image_0", "image_1"],
            pipe_input_as_list=True,
        ),
    ),
    # -- Image + spatial (SAM) ----------------------------------------------
    "mask-generation": TaskInputSpec(
        user_inputs=[
            InputField(
                name="image",
                type="image",
                required=True,
                description="Image for mask generation",
            ),
            InputField(
                name="input_points",
                type="json",
                required=False,
                description="Point coordinates [[x,y],...]",
            ),
            InputField(
                name="input_labels",
                type="json",
                required=False,
                description="Point labels [0|1,...]",
            ),
            InputField(
                name="input_boxes",
                type="json",
                required=False,
                description="Bounding boxes [[x1,y1,x2,y2],...]",
            ),
        ],
        mapping=PipelineMapping(
            pipe_input="image",
            pipe_kwargs=["input_points", "input_labels", "input_boxes"],
        ),
    ),
    # -- Table --------------------------------------------------------------
    "table-question-answering": TaskInputSpec(
        user_inputs=[
            InputField(
                name="query", type="text", required=True, description="Question about the table"
            ),
            InputField(
                name="table",
                type="json",
                required=True,
                description="Table as {column: [values]} dict",
            ),
        ],
        mapping=PipelineMapping(pipe_input=["query", "table"]),
    ),
}

# ---------------------------------------------------------------------------
# HF aliases — tasks that share the same input schema and pipeline mapping
# ---------------------------------------------------------------------------

TASK_REGISTRY["sentiment-analysis"] = TASK_REGISTRY["text-classification"]
TASK_REGISTRY["ner"] = TASK_REGISTRY["token-classification"]
TASK_REGISTRY["vqa"] = TASK_REGISTRY["visual-question-answering"]
TASK_REGISTRY["text-to-speech"] = TASK_REGISTRY["text-to-audio"]
TASK_REGISTRY["semantic-segmentation"] = TASK_REGISTRY["image-segmentation"]
TASK_REGISTRY["sentence-similarity"] = TaskInputSpec(
    user_inputs=[
        InputField(name="text_1", type="text", required=True, description="First sentence"),
        InputField(name="text_2", type="text", required=True, description="Second sentence"),
    ],
    mapping=PipelineMapping(pipe_input=["text_1", "text_2"], pipe_input_as_list=True),
    postprocess=_postprocess_sentence_similarity,
)
