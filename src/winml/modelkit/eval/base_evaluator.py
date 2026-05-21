# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Base WinML evaluator class."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..utils.eval_utils import DatasetValidationError, validate_dataset_columns


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.pipelines.base import Pipeline

    from ..datasets.config import DatasetConfig
    from ..models.winml.base import WinMLPreTrainedModel
    from .config import WinMLEvaluationConfig

logger = logging.getLogger(__name__)

# Tasks not supported as HF pipeline tasks, mapped to their pipeline equivalent.
_PIPELINE_TASK_MAP: dict[str, str] = {
    "sentence-similarity": "feature-extraction",
}


class WinMLEvaluator:
    """Base evaluator. Loads dataset, creates pipeline, runs HF evaluator."""

    def __init__(
        self,
        config: WinMLEvaluationConfig,
        model: WinMLPreTrainedModel,
    ) -> None:
        self.model = model
        self.config = config
        self.data = self.prepare_data()
        self.pipe = self.prepare_pipeline()

    def compute(self) -> dict[str, Any]:
        """Run evaluation and return metrics."""
        import inspect

        from evaluate import evaluator

        logger.info("Running evaluation...")
        task_evaluator = evaluator(self.config.task)

        kwargs: dict[str, Any] = {
            "model_or_pipeline": self.pipe,
            "data": self.data,
            "label_mapping": getattr(self.model.config, "label2id", None),
            **self.config.dataset.columns_mapping,
        }

        sig = inspect.signature(task_evaluator.compute)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not has_var_keyword:
            supported = set(sig.parameters)
            for key in set(kwargs) - supported:
                if key in self.config.dataset.columns_mapping:
                    logger.warning(
                        "Column mapping '%s' not supported by %s evaluator; ignoring.",
                        key,
                        self.config.task,
                    )
                kwargs.pop(key)

        return task_evaluator.compute(**kwargs)

    def prepare_data(self) -> Dataset:
        """Load dataset, shuffle, sample, and align labels."""
        from pathlib import Path

        from datasets import Dataset, load_dataset, load_from_disk

        ds = self.config.dataset
        logger.info(
            "Loading dataset: %s (name=%s, split=%s, samples=%s)",
            ds.path,
            ds.name,
            ds.split,
            ds.samples,
        )
        if ds.path and Path(ds.path).is_dir():
            dataset = load_from_disk(ds.path)
        else:
            dataset = load_dataset(
                ds.path,
                name=ds.name,
                split=ds.split,
                streaming=ds.streaming,
            )

        if ds.streaming:
            if ds.shuffle:
                dataset = dataset.shuffle(seed=ds.seed)
            dataset = Dataset.from_list(list(dataset.take(ds.samples)), features=dataset.features)
        else:
            if ds.shuffle:
                dataset = dataset.shuffle(seed=ds.seed)
            actual_samples = min(ds.samples, len(dataset))
            if actual_samples < ds.samples:
                logger.warning(
                    "Requested %d samples but dataset has %d. Using all.",
                    ds.samples,
                    len(dataset),
                )
            dataset = dataset.select(range(actual_samples))

        validate_dataset_columns(
            dataset, self.config.task, self.config.dataset.columns_mapping,
        )
        return self.align_labels(dataset, ds)

    def prepare_pipeline(self) -> Pipeline:
        """Create HF pipeline for inference. Subclasses override to configure."""
        from transformers import pipeline

        pipeline_task = _PIPELINE_TASK_MAP.get(self.config.task, self.config.task)
        return pipeline(
            pipeline_task,
            model=self.model,
            framework="pt",
            tokenizer=self.config.model_id,
            feature_extractor=self.config.model_id,
            image_processor=self.config.model_id,
            processor=self.config.model_id,
            # "device" is for HF pipeline pytorch tensors, not ORT EP.
            # WinMLSession handles device delegation for ORT.
            device="cpu",
        )

    def _fixed_seq_length(self) -> int | None:
        """Return the model's fixed sequence length, or ``None`` if dynamic.

        Reads ``io_config["input_shapes"]`` and treats an integer second
        dimension as a static sequence length. Subclasses use this to decide
        whether tokenized inputs need to be padded/truncated to a fixed size.
        """
        io_config = getattr(self.model, "io_config", None) or {}
        shapes = io_config.get("input_shapes") or [[]]
        if len(shapes[0]) > 1 and isinstance(shapes[0][1], int):
            return shapes[0][1]
        return None

    def _pad_or_truncate(self, encoding: Any, tokenizer: Any) -> Any:
        """Resize tokenized inputs to the model's fixed sequence length.

        No-op for dynamic-shape models. Otherwise truncates over-length
        tensors and delegates padding to the tokenizer.
        """
        seq_len = self._fixed_seq_length()
        if seq_len is None:
            return encoding
        for key, tensor in list(encoding.items()):
            if hasattr(tensor, "shape") and tensor.dim() >= 2 and tensor.shape[1] > seq_len:
                encoding[key] = tensor[:, :seq_len]
        return tokenizer.pad(
            encoding,
            padding="max_length",
            max_length=seq_len,
            return_tensors="pt",
        )

    def align_labels(self, dataset: Dataset, ds_config: DatasetConfig) -> Dataset:
        """Align dataset labels and filter unsupported IDs.

        Label mapping priority: user-provided > known dataset > model.label2id.
        Only applies to ClassLabel columns (not Sequence or dict).
        Derived classes can override for task-specific behavior.
        """
        try:
            label_column = ds_config.columns_mapping.get("label_column", "label")
            if label_column not in dataset.column_names:
                return dataset

            from datasets import ClassLabel

            if not isinstance(dataset.features[label_column], ClassLabel):
                return dataset

            label_map = self._get_label_mapping(ds_config)
            if not label_map:
                return dataset

            dataset = dataset.align_labels_with_mapping(
                label_map,
                label_column,
            )
            logger.info("Dataset labels aligned for %s.", ds_config.path)
            return self._filter_unsupported_labels(dataset, label_column)
        except (ValueError, KeyError) as e:
            raise DatasetValidationError(
                f"label alignment failed for dataset '{ds_config.path}': {e}",
            ) from e

    def _get_label_mapping(self, ds_config: DatasetConfig) -> dict | None:
        """Resolve label mapping: user-provided > known dataset > model.label2id."""
        from ..datasets.label_utils import get_label_mapping, should_align_labels

        if ds_config.label_mapping:
            return ds_config.label_mapping
        if ds_config.path and should_align_labels(ds_config.path):
            return get_label_mapping(ds_config.path)
        return getattr(self.model.config, "label2id", None)

    def _filter_unsupported_labels(self, dataset: Dataset, label_column: str) -> Dataset:
        """Filter rows whose label ID is not in model's id2label."""
        id2label = getattr(self.model.config, "id2label", None)
        if not id2label:
            return dataset

        supported_ids = {int(k) for k in id2label}
        original_count = len(dataset)
        dataset = dataset.filter(lambda row: row[label_column] in supported_ids)

        if len(dataset) == 0:
            raise DatasetValidationError(
                "No samples remain after label filtering. "
                "Dataset and model labels have no overlap.",
            )

        if len(dataset) < original_count:
            logger.warning(
                "Filtered %d → %d rows (unsupported label IDs removed).",
                original_count,
                len(dataset),
            )
        return dataset
