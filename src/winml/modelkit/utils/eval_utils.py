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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, get_args


EvalMode: TypeAlias = Literal["onnx", "compare"]

EVAL_MODES: tuple[EvalMode, ...] = get_args(EvalMode)


@dataclass(frozen=True)
class SchemaItem:
    """One dataset column or one configuration parameter."""

    name: str  # the --column key (e.g. "input_column")
    description: str  # short sentence
    default: str | None = None  # default value; None = no default (optional entry)
    remap_hint: str | None = None  # value placeholder; None = no --column remap


@dataclass(frozen=True)
class TaskSchema:
    """Input schema description for one task."""

    columns: tuple[SchemaItem, ...]
    params: tuple[SchemaItem, ...] = ()
    roles: tuple[str, ...] | None = None


_IMAGE_CLASSIFICATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "input image (PIL.Image)",
            default="image",
            remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "label_column",
            "integer class label",
            default="label",
            remap_hint="<your_label_column>",
        ),
    ),
)

_TEXT_CLASSIFICATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "input text",
            default="text",
            remap_hint="<your_text_column>",
        ),
        SchemaItem(
            "label_column",
            "class label (ClassLabel or integer)",
            default="label",
            remap_hint="<your_label_column>",
        ),
        SchemaItem(
            "second_input_column",
            "second text for sentence-pair tasks (optional)",
            remap_hint="<your_pair_column>",
        ),
    ),
)

_RERANKING_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "query_column",
            "query text (grouped rows typically use the authoritative 'input' column)",
            default="input",
            remap_hint="<your_query_column>",
        ),
        SchemaItem(
            "expected_output_column",
            "JSON/list of relevant candidate IDs for grouped rows",
            default="expected_output",
            remap_hint="<your_relevant_ids_column>",
        ),
        SchemaItem(
            "metadata_column",
            "metadata dict/JSON for grouped rows (used for query_id and provenance)",
            default="metadata",
            remap_hint="<your_metadata_column>",
        ),
        SchemaItem(
            "candidates_column",
            "inline candidate list for grouped rows; each item must expose text and ID fields",
            remap_hint="<your_candidates_column>",
        ),
        SchemaItem(
            "positive_column",
            "relevant passage text list for grouped rows",
            remap_hint="<your_positive_passages_column>",
        ),
        SchemaItem(
            "negative_column",
            "non-relevant passage text list for grouped rows",
            remap_hint="<your_negative_passages_column>",
        ),
        SchemaItem(
            "document_column",
            "candidate document text for pre-expanded pairwise rows",
            remap_hint="<your_document_column>",
        ),
        SchemaItem(
            "group_column",
            "group/query identifier for pre-expanded pairwise rows",
            remap_hint="<your_group_column>",
        ),
        SchemaItem(
            "label_column",
            "binary relevance flag for pre-expanded pairwise rows",
            remap_hint="<your_label_column>",
        ),
        SchemaItem(
            "candidate_id_column",
            "candidate identifier for pre-expanded pairwise rows",
            remap_hint="<your_candidate_id_column>",
        ),
    ),
    params=(
        SchemaItem(
            "candidate_text_key",
            "candidate text field inside grouped-row candidates",
            default="text",
            remap_hint="<candidate_text_key>",
        ),
        SchemaItem(
            "candidate_id_key",
            "candidate ID field inside grouped-row candidates",
            default="id",
            remap_hint="<candidate_id_key>",
        ),
        SchemaItem(
            "metadata_group_key",
            "group/query identifier field inside grouped-row metadata",
            default="query_id",
            remap_hint="<metadata_group_key>",
        ),
        SchemaItem(
            "recall_ks",
            "comma-separated K values for Recall@K",
            default="1,10",
            remap_hint="<k1,k2,...>",
        ),
        SchemaItem(
            "max_candidates",
            "maximum candidates materialized from positive/negative passage lists",
            default="10",
            remap_hint="<positive integer>",
        ),
    ),
)

_TOKEN_CLASSIFICATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "tokenized words (list of strings per sample)",
            default="tokens",
            remap_hint="<your_tokens_column>",
        ),
        SchemaItem(
            "label_column",
            "NER tag ID per token",
            default="ner_tags",
            remap_hint="<your_tags_column>",
        ),
    ),
)

_OBJECT_DETECTION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "input image (PIL.Image)",
            default="image",
            remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "annotation_column",
            "annotation dict containing bbox + category fields",
            default="objects",
            remap_hint="<your_annotation_column>",
        ),
    ),
    params=(
        SchemaItem(
            "bbox_key",
            "name of the bbox field inside the annotation dict",
            default="bbox",
            remap_hint="<bbox_field>",
        ),
        SchemaItem(
            "category_key",
            "name of the category field inside the annotation dict",
            default="category",
            remap_hint="<category_field>",
        ),
        SchemaItem(
            "box_format",
            "bounding box layout",
            default="xywh",
            remap_hint="<xywh|xyxy>",
        ),
        SchemaItem(
            "box_coords",
            "bounding box coordinate system",
            default="absolute",
            remap_hint="<absolute|normalized>",
        ),
    ),
)

_IMAGE_SEGMENTATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "input image (PIL.Image)",
            default="image",
            remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "annotation_column",
            "single-channel mask image; pixel value = class ID",
            default="annotation",
            remap_hint="<your_mask_column>",
        ),
    ),
)

_QUESTION_ANSWERING_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "question_column",
            "question text",
            default="question",
            remap_hint="<your_question_column>",
        ),
        SchemaItem(
            "context_column",
            "context passage to read",
            default="context",
            remap_hint="<your_context_column>",
        ),
        SchemaItem(
            "id_column",
            "unique question-answer ID",
            default="id",
            remap_hint="<your_id_column>",
        ),
        SchemaItem(
            "label_column",
            "answers dict with text and answer_start lists",
            default="answers",
            remap_hint="<your_answers_column>",
        ),
    ),
)

_FEATURE_EXTRACTION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column_1",
            "first sentence of the pair",
            default="sentence1",
            remap_hint="<your_first_sentence_column>",
        ),
        SchemaItem(
            "input_column_2",
            "second sentence of the pair",
            default="sentence2",
            remap_hint="<your_second_sentence_column>",
        ),
        SchemaItem(
            "score_column",
            "ground-truth similarity score (e.g. [0, 5] for STS-B)",
            default="score",
            remap_hint="<your_score_column>",
        ),
    ),
)

_IMAGE_FEATURE_EXTRACTION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "input image (PIL.Image)",
            default="image",
            remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "label_column",
            "integer class label (used for kNN accuracy)",
            default="label",
            remap_hint="<your_label_column>",
        ),
    ),
)

_IMAGE_TO_TEXT_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "input image (PIL.Image)",
            default="image",
            remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "label_column",
            "reference caption (string or list of strings)",
            default="text",
            remap_hint="<your_text_column>",
        ),
    ),
    roles=("encoder", "decoder"),
)

_FILL_MASK_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "input text scored via pseudo-perplexity",
            default="text",
            remap_hint="<your_text_column>",
        ),
    ),
)

_ZERO_SHOT_CLASSIFICATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "input text",
            default="text",
            remap_hint="<your_text_column>",
        ),
        SchemaItem(
            "label_column",
            "gold label (ClassLabel or string)",
            default="label",
            remap_hint="<your_label_column>",
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
            "input_column",
            "input image (PIL.Image)",
            default="image",
            remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "label_column",
            "integer class label",
            default="label",
            remap_hint="<your_label_column>",
        ),
    ),
    roles=("image-encoder", "text-encoder"),
)

_DEPTH_ESTIMATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "input image (PIL.Image)",
            default="image",
            remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "depth_column",
            "single-channel ground-truth depth image",
            default="depth_map",
            remap_hint="<your_depth_column>",
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

_KEYPOINT_DETECTION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "input image (PIL.Image)",
            default="image",
            remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "annotation_column",
            "annotation dict containing per-person keypoints + bbox + area",
            default="objects",
            remap_hint="<your_annotation_column>",
        ),
    ),
    params=(
        SchemaItem(
            "keypoints_key",
            "keypoints field inside the annotation dict (flat [x, y, v] triplets per person)",
            default="keypoints",
            remap_hint="<keypoints_field>",
        ),
        SchemaItem(
            "bbox_key",
            "person bbox field inside the annotation dict",
            default="bbox",
            remap_hint="<bbox_field>",
        ),
        SchemaItem(
            "area_key",
            "person area field inside the annotation dict",
            default="area",
            remap_hint="<area_field>",
        ),
        SchemaItem(
            "box_format",
            "person bounding box layout",
            default="xywh",
            remap_hint="<xywh|xyxy>",
        ),
        SchemaItem(
            "sigmas",
            "per-keypoint OKS sigmas as comma-separated floats; "
            "defaults to the COCO 17-keypoint constants",
            default="COCO 17 sigmas",
            remap_hint="<s1,s2,...>",
        ),
        SchemaItem(
            "keypoint_names",
            "keypoint names in index order as comma-separated strings; "
            "defaults to the COCO 17 names",
            default="COCO 17 names",
            remap_hint="<name1,name2,...>",
        ),
    ),
)

_MASK_GENERATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "input image (PIL.Image)",
            default="image",
            remap_hint="<your_image_column>",
        ),
        SchemaItem(
            "mask_column",
            "binary or instance ground-truth mask (PIL.Image)",
            default="mask",
            remap_hint="<your_mask_column>",
        ),
    ),
    # SAM-family composite: ``image-encoder`` runs once, ``prompt-decoder``
    # consumes the embeddings plus point / box prompts to produce masks.
    roles=("image-encoder", "prompt-decoder"),
)

_TEXT_GENERATION_SCHEMA = TaskSchema(
    columns=(
        SchemaItem(
            "input_column",
            "text field the perplexity corpus is concatenated from",
            default="text",
            remap_hint="<your_text_column>",
        ),
    ),
    params=(
        SchemaItem(
            "num_tokens",
            "total corpus tokens to score",
            default="8192",
            remap_hint="<int>",
        ),
        SchemaItem(
            "seqlen",
            "non-overlapping block length (tokens)",
            default="2048",
            remap_hint="<int>",
        ),
    ),
)

TASK_SCHEMAS: dict[str, TaskSchema] = {
    "image-classification": _IMAGE_CLASSIFICATION_SCHEMA,
    "reranking": _RERANKING_SCHEMA,
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
    "keypoint-detection": _KEYPOINT_DETECTION_SCHEMA,
    "mask-generation": _MASK_GENERATION_SCHEMA,
    "text-generation": _TEXT_GENERATION_SCHEMA,
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


RerankingDatasetMode: TypeAlias = Literal[
    "pairwise",
    "grouped-inline",
    "grouped-text",
    "grouped-authoritative",
]


def get_dataset_column_names(dataset: object) -> tuple[str, ...]:
    """Best-effort column-name extraction for datasets and list-backed test fixtures."""
    column_names = getattr(dataset, "column_names", None)
    if isinstance(column_names, (list, tuple)):
        return tuple(str(name) for name in column_names)
    if isinstance(dataset, Sequence) and not isinstance(dataset, (str, bytes, bytearray)):
        names: set[str] = set()
        for row in dataset:
            if isinstance(row, Mapping):
                names.update(str(name) for name in row)
        return tuple(sorted(names))
    return ()


def _resolved_reranking_column(mapping: dict[str, str], key: str) -> str | None:
    return mapping.get(key, get_default("reranking", key))


def detect_reranking_dataset_mode(
    column_names: set[str] | list[str] | tuple[str, ...],
    columns_mapping: dict[str, str] | None = None,
) -> RerankingDatasetMode:
    """Resolve reranking datasets to pairwise, grouped-inline, or grouped-authoritative."""
    mapping = columns_mapping or {}
    actual = set(column_names)

    query_col = _resolved_reranking_column(mapping, "query_column")
    expected_output_col = _resolved_reranking_column(mapping, "expected_output_column")
    metadata_col = _resolved_reranking_column(mapping, "metadata_column")
    document_col = mapping.get("document_column")
    group_col = mapping.get("group_column")
    label_col = mapping.get("label_column")
    candidates_col = mapping.get("candidates_column")
    positive_col = mapping.get("positive_column")
    negative_col = mapping.get("negative_column")

    grouped_required = tuple(
        name for name in (query_col, expected_output_col, metadata_col) if name is not None
    )
    pairwise_required = tuple(
        name for name in (query_col, document_col, group_col, label_col) if name is not None
    )

    has_grouped_core = len(grouped_required) == 3 and all(
        name in actual for name in grouped_required
    )
    has_pairwise = len(pairwise_required) == 4 and all(name in actual for name in pairwise_required)
    has_grouped_text = (
        query_col is not None
        and positive_col is not None
        and negative_col is not None
        and all(name in actual for name in (query_col, positive_col, negative_col))
    )

    if has_grouped_core and candidates_col is not None and candidates_col in actual:
        return "grouped-inline"
    if has_pairwise:
        return "pairwise"
    if has_grouped_text:
        return "grouped-text"
    if has_grouped_core:
        return "grouped-authoritative"

    grouped_missing = sorted(name for name in grouped_required if name not in actual)
    pairwise_missing = sorted(name for name in pairwise_required if name not in actual)
    raise DatasetValidationError(
        "reranking datasets require pairwise columns "
        f"{sorted(pairwise_required)} or grouped authoritative columns {sorted(grouped_required)}; "
        f"missing pairwise={pairwise_missing} grouped={grouped_missing}; "
        f"dataset has {sorted(actual)}"
    )


def validate_dataset_columns(
    dataset: object,
    task: str,
    columns_mapping: dict[str, str] | None = None,
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
    if task == "reranking":
        detect_reranking_dataset_mode(actual, mapping)
        return
    missing = [
        (item.name, mapping.get(item.name, item.default))
        for item in schema.columns
        if item.default is not None and mapping.get(item.name, item.default) not in actual
    ]
    if missing:
        details = ", ".join(f"{k}='{v}'" for k, v in missing)
        raise DatasetValidationError(
            f"missing required column(s) {details}; dataset has {sorted(actual)}",
        )
