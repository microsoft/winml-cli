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
from dataclasses import dataclass, field
from typing import Any


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
    """Single source of truth for a task's user inputs and pipeline dispatch."""

    user_inputs: list[InputField]
    mapping: PipelineMapping


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
