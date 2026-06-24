# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Task input-schema metadata for ``winml eval``.

Shared between the CLI (``winml eval --schema``) and the individual
evaluator classes that need default column names. Lives in ``utils`` so
that importing it does not load the heavy ``winml.modelkit.eval`` package
(which would otherwise drag in ``transformers``/``torch``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, get_args


EvalMode: TypeAlias = Literal["onnx", "compare"]

EVAL_MODES: tuple[EvalMode, ...] = get_args(EvalMode)


@dataclass(frozen=True)
class SchemaItem:
    """One dataset column or one configuration parameter."""

    name: str                       # the --column key (e.g. "input_column")
    description: str                # short sentence
    default: str | None = None      # default value; None = no default (optional entry)
    remap_hint: str | None = None   # value placeholder; None = no --column remap


@dataclass(frozen=True)
class TaskSchema:
    """Input schema description for one task."""

    columns: tuple[SchemaItem, ...]
    params: tuple[SchemaItem, ...] = ()
    roles: tuple[str, ...] | None = None


_IMAGE_CLASSIFICATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column", "input image (PIL.Image)",
            default="image", remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "label_column", "integer class label",
            default="label", remap_hint="<your_label_column>",
        ),
    ),
)

_TEXT_CLASSIFICATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column", "input text",
            default="text", remap_hint="<your_text_column>",
        ),
        SchemaItem(
            "label_column", "class label (ClassLabel or integer)",
            default="label", remap_hint="<your_label_column>",
        ),
        SchemaItem(
            "second_input_column", "second text for sentence-pair tasks (optional)",
            remap_hint="<your_pair_column>",
        ),
    ),
)

_TOKEN_CLASSIFICATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column", "tokenized words (list of strings per sample)",
            default="tokens", remap_hint="<your_tokens_column>",
        ),
        SchemaItem(
            "label_column", "NER tag ID per token",
            default="ner_tags", remap_hint="<your_tags_column>",
        ),
    ),
)

_OBJECT_DETECTION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column", "input image (PIL.Image)",
            default="image", remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "annotation_column",
            "annotation dict containing bbox + category fields",
            default="objects", remap_hint="<your_annotation_column>",
        ),
    ),
    params=(
        SchemaItem(
            "bbox_key",
            "name of the bbox field inside the annotation dict",
            default="bbox", remap_hint="<bbox_field>",
        ),
        SchemaItem(
            "category_key",
            "name of the category field inside the annotation dict",
            default="category", remap_hint="<category_field>",
        ),
        SchemaItem(
            "box_format", "bounding box layout",
            default="xywh", remap_hint="<xywh|xyxy>",
        ),
        SchemaItem(
            "box_coords", "bounding box coordinate system",
            default="absolute", remap_hint="<absolute|normalized>",
        ),
    ),
)

_IMAGE_SEGMENTATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column", "input image (PIL.Image)",
            default="image", remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "annotation_column",
            "single-channel mask image; pixel value = class ID",
            default="annotation", remap_hint="<your_mask_column>",
        ),
    ),
)

_QUESTION_ANSWERING_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "question_column", "question text",
            default="question", remap_hint="<your_question_column>",
        ),
        SchemaItem(
            "context_column", "context passage to read",
            default="context", remap_hint="<your_context_column>",
        ),
        SchemaItem(
            "id_column", "unique question-answer ID",
            default="id", remap_hint="<your_id_column>",
        ),
        SchemaItem(
            "label_column",
            "answers dict with text and answer_start lists",
            default="answers", remap_hint="<your_answers_column>",
        ),
    ),
)

_FEATURE_EXTRACTION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column_1", "first sentence of the pair",
            default="sentence1", remap_hint="<your_first_sentence_column>",
        ),
        SchemaItem(
            "input_column_2", "second sentence of the pair",
            default="sentence2", remap_hint="<your_second_sentence_column>",
        ),
        SchemaItem(
            "score_column",
            "ground-truth similarity score (e.g. [0, 5] for STS-B)",
            default="score", remap_hint="<your_score_column>",
        ),
    ),
)

_IMAGE_FEATURE_EXTRACTION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column", "input image (PIL.Image)",
            default="image", remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "label_column", "integer class label (used for kNN accuracy)",
            default="label", remap_hint="<your_label_column>",
        ),
    ),
)

_IMAGE_TO_TEXT_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column", "input image (PIL.Image)",
            default="image", remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "label_column", "reference caption (string or list of strings)",
            default="text", remap_hint="<your_text_column>",
        ),
    ),
    roles=("encoder", "decoder"),
)

_FILL_MASK_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column", "input text scored via pseudo-perplexity",
            default="text", remap_hint="<your_text_column>",
        ),
    ),
)

_ZERO_SHOT_CLASSIFICATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column", "input text",
            default="text", remap_hint="<your_text_column>",
        ),
        SchemaItem(
            "label_column", "gold label (ClassLabel or string)",
            default="label", remap_hint="<your_label_column>",
        ),
    ),
    params=(
        SchemaItem(
            "candidate_labels",
            "candidate label vocabulary; required if label column is not a ClassLabel",
            default="from dataset ClassLabel.names",
            remap_hint="<comma,separated,labels>",
        ),
        SchemaItem(
            "hypothesis_template",
            "NLI prompt template; {} is replaced with each candidate label",
            default='"This example is {}."',
            remap_hint="<template with {} placeholder>",
        ),
    ),
)

_ZERO_SHOT_IMAGE_CLASSIFICATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column", "input image (PIL.Image)",
            default="image", remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "label_column", "integer class label",
            default="label", remap_hint="<your_label_column>",
        ),
    ),
    roles=("image-encoder", "text-encoder"),
)

_DEPTH_ESTIMATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column", "input image (PIL.Image)",
            default="image", remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "depth_column", "single-channel ground-truth depth image",
            default="depth_map", remap_hint="<your_depth_column>",
        ),
    ),
    params=(
        SchemaItem(
            "align",
            "alignment strategy for predictions",
            default="affine",
            remap_hint="<affine|median|none>",
        ),
        SchemaItem(
            "depth_kind",
            "prediction space",
            default="depth",
            remap_hint="<depth|disparity>",
        ),
        SchemaItem(
            "min_depth",
            "minimum valid ground-truth depth",
            default="1e-3",
            remap_hint="<float>",
        ),
        SchemaItem(
            "max_depth",
            "maximum valid ground-truth depth",
            default="10.0",
            remap_hint="<float|none>",
        ),
    ),
)

TASK_SCHEMAS: dict[str, TaskSchema] = {
    "image-classification": _IMAGE_CLASSIFICATION_SCHEMA,
    "text-classification": _TEXT_CLASSIFICATION_SCHEMA,
    "sequence-classification": _TEXT_CLASSIFICATION_SCHEMA,
    "next-sentence-prediction": _TEXT_CLASSIFICATION_SCHEMA,
    "token-classification": _TOKEN_CLASSIFICATION_SCHEMA,
    "object-detection": _OBJECT_DETECTION_SCHEMA,
    "image-segmentation": _IMAGE_SEGMENTATION_SCHEMA,
    "question-answering": _QUESTION_ANSWERING_SCHEMA,
    "feature-extraction": _FEATURE_EXTRACTION_SCHEMA,
    "sentence-similarity": _FEATURE_EXTRACTION_SCHEMA,
    "image-feature-extraction": _IMAGE_FEATURE_EXTRACTION_SCHEMA,
    "image-to-text": _IMAGE_TO_TEXT_SCHEMA,
    "fill-mask": _FILL_MASK_SCHEMA,
    "zero-shot-classification": _ZERO_SHOT_CLASSIFICATION_SCHEMA,
    "zero-shot-image-classification": _ZERO_SHOT_IMAGE_CLASSIFICATION_SCHEMA,
    "depth-estimation": _DEPTH_ESTIMATION_SCHEMA,
}


def get_default(task: str, name: str) -> str | None:
    """Return the default value for *name* in the schema of *task*.

    Looks across both ``columns`` and ``params``. Returns ``None`` if the
    task or name is unknown, or the entry has no default.
    """
    schema = TASK_SCHEMAS.get(task)
    if schema is None:
        return None
    for item in (*schema.columns, *schema.params):
        if item.name == name:
            return item.default
    return None


class DatasetValidationError(Exception):
    """Dataset failed schema validation against a task's expected columns."""


def validate_dataset_columns(
    dataset: object, task: str, columns_mapping: dict[str, str] | None = None,
) -> None:
    """Check required schema columns exist in *dataset*.

    Resolves each required column as ``columns_mapping.get(key, schema_default)``
    and raises :class:`DatasetValidationError` if any is missing. No-op if the
    task is unknown or the dataset does not expose ``column_names``.
    """
    schema = TASK_SCHEMAS.get(task)
    column_names = getattr(dataset, "column_names", None)
    if schema is None or not isinstance(column_names, (list, tuple)):
        return
    mapping = columns_mapping or {}
    actual = set(column_names)
    missing = [
        (item.name, mapping.get(item.name, item.default))
        for item in schema.columns
        if item.default is not None
        and mapping.get(item.name, item.default) not in actual
    ]
    if missing:
        details = ", ".join(f"{k}='{v}'" for k, v in missing)
        raise DatasetValidationError(
            f"missing required column(s) {details}; dataset has {sorted(actual)}",
        )
