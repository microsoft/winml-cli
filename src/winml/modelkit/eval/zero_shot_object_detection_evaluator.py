# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Open-vocabulary object detection evaluation using grounded processor semantics."""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..utils.eval_utils import DatasetValidationError
from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..models.winml.base import WinMLPreTrainedModel
    from .config import DatasetConfig, WinMLEvaluationConfig

logger = logging.getLogger(__name__)


def _require_string(value: str | None, field: str) -> str:
    """Return a required configured string or fail with an actionable error."""
    if value is None:
        raise ValueError(f"zero-shot object detection requires '{field}' configuration")
    return value


@dataclass(frozen=True)
class QueryChunk:
    """One model invocation's ordered query vocabulary."""

    prompts: tuple[str, ...]
    category_ids: tuple[int, ...]


def _validate_prompt_template(template: str) -> str:
    """Require exactly one positional replacement field."""
    fields = [field for _, field, _, _ in string.Formatter().parse(template) if field is not None]
    if len(fields) != 1 or fields[0] not in ("", "0"):
        raise DatasetValidationError(
            "prompt_template must contain exactly one positional replacement field '{}'.",
        )
    try:
        template.format("category")
    except (IndexError, KeyError, ValueError) as error:
        raise DatasetValidationError(f"invalid prompt_template: {error}") from error
    return template


def _unwrap_sequence_feature(feature: Any) -> Any:
    """Unwrap datasets.Sequence/List wrappers without depending on their version."""
    while hasattr(feature, "feature"):
        feature = feature.feature
    return feature


def _extract_category_vocabulary(
    features: Any,
    annotation_column: str,
    category_key: str,
    explicit_mapping: dict[str, int] | None = None,
) -> list[tuple[int, str]]:
    """Extract ordered ``(dataset ID, name)`` pairs from authoritative metadata."""
    if explicit_mapping is not None:
        if not explicit_mapping:
            raise DatasetValidationError("explicit label mapping must not be empty")
        ids = list(explicit_mapping.values())
        if any(not isinstance(value, int) or value < 0 for value in ids):
            raise DatasetValidationError("explicit label mapping IDs must be non-negative integers")
        if len(set(ids)) != len(ids):
            raise DatasetValidationError("explicit label mapping IDs must be unique")
        return sorted(((category_id, name) for name, category_id in explicit_mapping.items()))

    annotation = features.get(annotation_column)
    annotation = _unwrap_sequence_feature(annotation)
    if not isinstance(annotation, dict) or category_key not in annotation:
        raise DatasetValidationError(
            f"'{annotation_column}' has no category feature '{category_key}'",
        )
    category = _unwrap_sequence_feature(annotation[category_key])
    names = getattr(category, "names", None)
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise DatasetValidationError(
            "category IDs require authoritative names from dataset ClassLabel metadata "
            "or an explicit validated label mapping",
        )
    return list(enumerate(names))


def _normalize_max_queries(max_queries: Any) -> int | None:
    """Validate configured query budget (or allow explicit unbounded mode)."""
    if max_queries is None:
        return None
    if isinstance(max_queries, bool) or not isinstance(max_queries, int):
        raise DatasetValidationError("max_queries must be an integer >= 1 or null")
    if max_queries < 1:
        raise DatasetValidationError("max_queries must be >= 1")
    return max_queries


def _apply_query_budget(
    vocabulary: list[tuple[int, str]],
    max_queries: Any,
) -> tuple[list[tuple[int, str]], dict[str, int]]:
    """Apply deterministic query budget to authoritative vocabulary."""
    if not vocabulary:
        raise DatasetValidationError("category vocabulary must not be empty")
    available = len(vocabulary)
    normalized = _normalize_max_queries(max_queries)
    requested = available if normalized is None else normalized
    used = min(requested, available)
    used_vocabulary = vocabulary[:used]
    return (
        used_vocabulary,
        {
            "query_requested": requested,
            "query_available": available,
            "query_used": used,
            "query_truncated": available - used,
        },
    )


def _filter_annotation_to_vocabulary(
    annotation: dict[str, Any],
    category_key: str,
    selected_category_ids: set[int],
) -> tuple[dict[str, Any], list[int]]:
    """Drop per-instance annotation entries outside the selected query vocabulary."""
    raw_categories = [int(value) for value in annotation[category_key]]
    keep_indices = [
        index
        for index, category in enumerate(raw_categories)
        if category in selected_category_ids
    ]
    if len(keep_indices) == len(raw_categories):
        return annotation, raw_categories

    filtered: dict[str, Any] = {}
    for key, value in annotation.items():
        if isinstance(value, list) and len(value) == len(raw_categories):
            filtered[key] = [value[index] for index in keep_indices]
        else:
            filtered[key] = value
    filtered_categories = [raw_categories[index] for index in keep_indices]
    filtered[category_key] = filtered_categories
    return filtered, filtered_categories


def _query_capacity(io_config: dict[str, Any]) -> int | None:
    """Return the static text-query capacity, or ``None`` for a dynamic axis."""
    names = io_config.get("input_names", [])
    shapes = io_config.get("input_shapes", [])
    try:
        shape = shapes[names.index("input_ids")]
    except (ValueError, IndexError):
        raise ValueError("zero-shot object detection model requires an input_ids input") from None
    if not shape:
        return None
    capacity = shape[0]
    if isinstance(capacity, int):
        if capacity < 1:
            raise ValueError("input_ids query capacity must be at least 1")
        return capacity
    return None


def _sequence_length(io_config: dict[str, Any]) -> int | None:
    names = io_config.get("input_names", [])
    shapes = io_config.get("input_shapes", [])
    try:
        shape = shapes[names.index("input_ids")]
    except (ValueError, IndexError):
        return None
    if len(shape) > 1 and isinstance(shape[1], int):
        return shape[1]
    return None


def _make_query_chunks(
    vocabulary: list[tuple[int, str]],
    template: str,
    capacity: int | None,
) -> list[QueryChunk]:
    """Render one independent category per invocation.

    Grounded object-detection postprocessing selects the highest query logit for
    each predicted box. To prevent categories from competing, replicate one real
    query across every slot required by a static graph. Any selected local slot
    then maps to the same authoritative category. Dynamic graphs use one slot.
    """
    if not vocabulary:
        raise DatasetValidationError("category vocabulary must not be empty")
    width = capacity or 1
    return [
        QueryChunk(
            prompts=(template.format(name),) * width,
            category_ids=(category_id,) * width,
        )
        for category_id, name in vocabulary
    ]


def _select_category_covering_rows(
    rows: list[tuple[int, int, list[int]]],
    category_ids: set[int],
    requested_cap: int,
) -> tuple[list[int], set[int]]:
    """Greedily select deterministic category-covering source row indices."""
    if requested_cap < 1:
        raise DatasetValidationError("samples must be at least 1")
    remaining = list(rows)
    uncovered = set(category_ids)
    selected: list[int] = []
    while remaining and len(selected) < requested_cap and uncovered:
        best = min(
            remaining,
            key=lambda row: (
                -len(set(row[2]) & uncovered),
                -len(row[2]),
                row[1],
                row[0],
            ),
        )
        if not (set(best[2]) & uncovered):
            break
        selected.append(best[0])
        uncovered.difference_update(best[2])
        remaining.remove(best)
    return selected, category_ids - uncovered


def _remap_grounded_output(output: dict[str, Any], chunk: QueryChunk) -> dict[str, list]:
    """Map processor query-local labels to authoritative dataset category IDs."""
    boxes: list = []
    scores: list = []
    labels: list[int] = []
    for box, score, local_label in zip(
        output["boxes"], output["scores"], output["labels"], strict=True
    ):
        local_id = int(local_label.item() if hasattr(local_label, "item") else local_label)
        if local_id < 0 or local_id >= len(chunk.category_ids):
            raise ValueError(f"processor returned out-of-range query label {local_id}")
        category_id = chunk.category_ids[local_id]
        boxes.append(box.tolist() if hasattr(box, "tolist") else list(box))
        scores.append(float(score.item() if hasattr(score, "item") else score))
        labels.append(category_id)
    return {"boxes": boxes, "scores": scores, "labels": labels}


class WinMLZeroShotObjectDetectionEvaluator(WinMLEvaluator):
    """Evaluate text-conditioned detectors against a complete dataset vocabulary."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        from ..utils.eval_utils import get_default

        mapping = config.dataset.columns_mapping
        task = "zero-shot-object-detection"
        self._image_col = _require_string(
            mapping.get("input_column", get_default(task, "input_column")), "input_column"
        )
        self._annotation_col = _require_string(
            mapping.get("annotation_column", get_default(task, "annotation_column")),
            "annotation_column",
        )
        self._bbox_key = _require_string(
            mapping.get("bbox_key", get_default(task, "bbox_key")), "bbox_key"
        )
        self._category_key = _require_string(
            mapping.get("category_key", get_default(task, "category_key")), "category_key"
        )
        self._image_id_col = _require_string(
            mapping.get("image_id_column", get_default(task, "image_id_column")),
            "image_id_column",
        )
        self._box_format = _require_string(
            mapping.get("box_format", get_default(task, "box_format")), "box_format"
        )
        self._box_coords = _require_string(
            mapping.get("box_coords", get_default(task, "box_coords")), "box_coords"
        )
        self._prompt_template = _validate_prompt_template(
            mapping.get("prompt_template", get_default(task, "prompt_template")) or ""
        )
        super().__init__(config, model)

    def prepare_pipeline(self) -> Pipeline:
        """Create the task pipeline and retain its matched processor components."""
        from transformers import AutoProcessor

        pipe = super().prepare_pipeline()
        if self.config.model_id is None:
            raise ValueError(
                "zero-shot object detection evaluation requires --model-id "
                "to load processor metadata"
            )
        processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        processor.tokenizer = pipe.tokenizer
        processor.image_processor = pipe.image_processor
        self._processor = processor
        return pipe

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """Keep dataset IDs unchanged and capture their authoritative vocabulary."""
        self._validate_schema(dataset)
        full_vocabulary = _extract_category_vocabulary(
            dataset.features,
            self._annotation_col,
            self._category_key,
            ds_config.label_mapping,
        )
        self._vocabulary, self._query_accounting = _apply_query_budget(
            full_vocabulary,
            ds_config.max_queries,
        )
        return dataset

    def prepare_data(self) -> Dataset:
        """Scan annotations without image decoding and select category-covering rows."""
        from pathlib import Path

        from datasets import Dataset, Image, load_dataset, load_from_disk

        ds = self.config.dataset
        try:
            ds_path = Path(ds.path).expanduser() if ds.path else None
            if ds_path and ds_path.is_dir():
                source = load_from_disk(str(ds_path))
            else:
                source = load_dataset(
                    ds.path,
                    name=ds.name,
                    split=ds.split,
                    streaming=ds.streaming,
                    revision=ds.revision,
                )
        except Exception as error:
            raise DatasetValidationError(
                f"Failed to load dataset '{ds.path}' "
                f"(name={ds.name!r}, split='{ds.split}'): {error}",
            ) from error

        self._validate_schema(source)
        full_vocabulary = _extract_category_vocabulary(
            source.features,
            self._annotation_col,
            self._category_key,
            ds.label_mapping,
        )
        all_category_ids = {category_id for category_id, _ in full_vocabulary}
        self._vocabulary, self._query_accounting = _apply_query_budget(
            full_vocabulary,
            ds.max_queries,
        )
        category_ids = {category_id for category_id, _ in self._vocabulary}
        original_features = source.features
        image_feature = original_features.get(self._image_col)
        if image_feature is not None:
            source = source.cast_column(self._image_col, Image(decode=False))

        rows: list[tuple[int, int, list[int]]] = []
        annotation_source = source.remove_columns(self._image_col)
        for source_index, sample in enumerate(annotation_source):
            annotations = sample[self._annotation_col]
            categories = [int(value) for value in annotations[self._category_key]]
            unknown = set(categories) - all_category_ids
            if unknown:
                raise DatasetValidationError(
                    f"annotation has unknown category IDs {sorted(unknown)}"
                )
            categories = [category for category in categories if category in category_ids]
            image_id = int(sample.get(self._image_id_col, source_index))
            rows.append((source_index, image_id, categories))

        selected_indices, covered = _select_category_covering_rows(rows, category_ids, ds.samples)
        selected_index_set = set(selected_indices)
        selected_by_index = {
            source_index: {
                **sample,
                self._annotation_col: _filter_annotation_to_vocabulary(
                    sample[self._annotation_col],
                    self._category_key,
                    category_ids,
                )[0],
            }
            for source_index, sample in enumerate(source)
            if source_index in selected_index_set
        }
        selected = [selected_by_index[index] for index in selected_indices]
        self._selection_accounting = {
            "requested": ds.samples,
            "requested_cap": ds.samples,
            "source_rows_scanned": len(rows),
            "selected": len(selected),
            "category_count": len(category_ids),
            "categories_covered": len(covered),
            **self._query_accounting,
        }
        return Dataset.from_list(selected, features=original_features)

    def compute(self) -> dict[str, Any]:
        """Run every category query for every image and compute COCO-style mAP."""
        import torch

        from .metrics import MAPMetric

        io_config = getattr(self.model, "io_config", None)
        if not isinstance(io_config, dict):
            raise TypeError("zero-shot object detection model requires I/O configuration")
        capacity = _query_capacity(io_config)
        chunks = _make_query_chunks(self._vocabulary, self._prompt_template, capacity)
        seq_length = _sequence_length(io_config)
        accepted_inputs = set(io_config.get("input_names", []))
        predictions: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []
        decoded = processed = query_passes = 0

        for sample in self.data:
            annotations = sample[self._annotation_col]
            references.append(
                {
                    "boxes": annotations[self._bbox_key],
                    "labels": [int(value) for value in annotations[self._category_key]],
                }
            )
            image = sample[self._image_col]
            if image is None or not hasattr(image, "size"):
                raise DatasetValidationError(
                    f"column '{self._image_col}' must contain decoded PIL images",
                )
            decoded += 1
            image_predictions: dict[str, list[Any]] = {
                "boxes": [],
                "scores": [],
                "labels": [],
            }
            for chunk in chunks:
                text_kwargs: dict[str, Any] = {
                    "return_tensors": "pt",
                    "padding": "max_length",
                    "truncation": True,
                }
                if seq_length is not None:
                    text_kwargs["max_length"] = seq_length
                text_inputs = self._processor.tokenizer(list(chunk.prompts), **text_kwargs)
                image_inputs = self._processor.image_processor(image, return_tensors="pt")
                model_inputs = {
                    **text_inputs,
                    **image_inputs,
                }
                model_inputs = {
                    name: value for name, value in model_inputs.items() if name in accepted_inputs
                }
                outputs = self.model(**model_inputs)
                query_passes += 1
                grounded = self._processor.post_process_grounded_object_detection(
                    outputs,
                    threshold=0.0,
                    target_sizes=torch.tensor([[image.height, image.width]]),
                )[0]
                remapped = _remap_grounded_output(grounded, chunk)
                for key in image_predictions:
                    image_predictions[key].extend(remapped[key])
            predictions.append(image_predictions)
            processed += 1

        if processed == 0:
            raise DatasetValidationError("evaluation processed no images")

        metrics = MAPMetric().compute(
            predictions=predictions,
            references=references,
            box_format=self._box_format,
            box_coords=self._box_coords,
        )
        metrics.update(
            {
                **self._selection_accounting,
                "decoded": decoded,
                "skipped": 0,
                "processed": processed,
                "failed": 0,
                "query_count": len(self._vocabulary),
                "query_passes": query_passes,
                "query_capacity": capacity or "dynamic",
            }
        )
        return metrics

    def _validate_schema(self, dataset: Any) -> None:
        """Validate required columns and nested annotation fields."""
        columns = set(getattr(dataset, "column_names", []))
        for column in (self._image_col, self._annotation_col):
            if column not in columns:
                raise DatasetValidationError(
                    f"missing required column '{column}'; dataset has {sorted(columns)}",
                )
        annotation = _unwrap_sequence_feature(dataset.features.get(self._annotation_col))
        if not isinstance(annotation, dict):
            raise DatasetValidationError(
                f"column '{self._annotation_col}' must be an annotation struct",
            )
        missing = [key for key in (self._bbox_key, self._category_key) if key not in annotation]
        if missing:
            raise DatasetValidationError(
                f"'{self._annotation_col}' is missing required field(s) {missing}",
            )
