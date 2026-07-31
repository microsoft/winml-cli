# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Zero-shot classification calibration dataset.

Zero-shot text classifiers consume natural-language premise/hypothesis pairs.
This specialization keeps calibration aligned with that input distribution
instead of reusing generic text-classification defaults.
"""

from __future__ import annotations

import logging
from math import ceil
from numbers import Integral
from random import Random
from typing import TYPE_CHECKING, Any, cast

from datasets import load_dataset
from transformers import AutoTokenizer

from .base import BaseTaskDataset


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)

DEFAULT_ZERO_SHOT_CLASSIFICATION_DATASET = "fancyzhx/ag_news"
DEFAULT_ZERO_SHOT_CLASSIFICATION_SPLIT = "test"
DEFAULT_ZERO_SHOT_TEXT_COLUMN = "text"
DEFAULT_ZERO_SHOT_CANDIDATE_LABELS = ("World", "Sports", "Business", "Sci/Tech")
DEFAULT_ZERO_SHOT_HYPOTHESIS_TEMPLATE = "This text is about {}."


class ZeroShotClassificationDataset(BaseTaskDataset):
    """Calibration dataset for zero-shot text classification models."""

    DEFAULT_SEQ_LEN = 128

    def __init__(
        self,
        model_name: str,
        dataset_name: str | None = None,
        max_samples: int | None = None,
        data_split: str | None = None,
        *,
        candidate_labels: str | Sequence[str] | None = None,
        hypothesis_template: str | None = None,
        input_column: str | None = None,
        max_length: int | None = None,
        io_config: dict | None = None,
        io_mapping: dict | None = None,
        **kwargs: Any,
    ) -> None:
        self._candidate_labels = self._normalize_candidate_labels(candidate_labels)
        self._hypothesis_template = hypothesis_template or DEFAULT_ZERO_SHOT_HYPOTHESIS_TEMPLATE
        self._input_column = input_column or DEFAULT_ZERO_SHOT_TEXT_COLUMN
        self._max_length = max_length
        self._io_config = io_config
        self._io_mapping = io_mapping or {}

        super().__init__(
            model_name=model_name,
            dataset_name=dataset_name,
            max_samples=max_samples,
            data_split=data_split,
            io_config=io_config,
            io_mapping=io_mapping,
            **kwargs,
        )

    @staticmethod
    def _normalize_candidate_labels(candidate_labels: str | Sequence[str] | None) -> list[str]:
        """Return non-empty labels from comma-separated or sequence input."""
        if candidate_labels is None:
            raw_labels: Sequence[str] = DEFAULT_ZERO_SHOT_CANDIDATE_LABELS
        elif isinstance(candidate_labels, str):
            raw_labels = candidate_labels.split(",")
        else:
            raw_labels = candidate_labels

        labels = [str(label).strip() for label in raw_labels if str(label).strip()]
        if not labels:
            raise ValueError("candidate_labels must contain at least one label")
        return labels

    def _get_default_dataset(self) -> None:
        """Set the built-in zero-shot dataset defaults."""
        if self._dataset_name is None:
            self._dataset_name = DEFAULT_ZERO_SHOT_CLASSIFICATION_DATASET
        if self._data_split is None:
            self._data_split = DEFAULT_ZERO_SHOT_CLASSIFICATION_SPLIT

    def _resolve_max_length(self) -> None:
        """Resolve tokenizer max length from ONNX io_config when available."""
        if self._max_length is None:
            self._max_length = self.DEFAULT_SEQ_LEN

        if not self._io_config:
            return

        onnx_name = self._io_mapping.get("input_ids", "input_ids")
        input_config = self._io_config.get(onnx_name)
        if not input_config:
            return

        shape = input_config.get("shape", [])
        if len(shape) > 1 and isinstance(shape[1], Integral):
            self._max_length = int(shape[1])
            logger.info("max_length=%d from io_config[%s]", self._max_length, onnx_name)

    def _source_sample_count(self) -> int | None:
        """Number of source rows needed to produce max_samples calibration pairs."""
        if self._max_samples is None:
            return None
        return ceil(self._max_samples / len(self._candidate_labels))

    def _load_and_sample(self) -> Any:
        """Load the source text dataset and cap rows before pair expansion."""
        subset = self._config.get("subset") or self._config.get("dataset_config_name")
        load_args = [self._dataset_name]
        if subset:
            load_args.append(subset)

        logger.info(
            "Loading zero-shot calibration dataset: %s (split=%s)",
            load_args,
            self._data_split,
        )
        dataset = load_dataset(*load_args, split=self._data_split)

        source_count = self._source_sample_count()
        shuffle = self._config.get("shuffle", False)
        seed = self._config.get("seed", 42)

        if source_count is not None:
            n = min(source_count, len(dataset))
            indices = Random(seed).sample(range(len(dataset)), n) if shuffle else list(range(n))
            dataset = dataset.select(indices)
        elif shuffle:
            dataset = dataset.shuffle(seed=seed)

        return dataset

    def _apply_io_mapping(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Rename tokenizer output fields to ONNX input names when requested."""
        if not self._io_mapping:
            return sample
        return {self._io_mapping.get(key, key): value for key, value in sample.items()}

    def _initialize(self) -> None:
        """Load text rows and tokenize them as zero-shot premise/hypothesis pairs."""
        self._get_default_dataset()
        self._resolve_max_length()
        dataset = self._load_and_sample()
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, use_fast=True)

        samples: list[dict[str, Any]] = []
        for example in dataset:
            if self._input_column not in example:
                raise ValueError(
                    f"Column '{self._input_column}' not found in {self._dataset_name}; "
                    f"available columns: {sorted(example)}"
                )
            premise = str(example[self._input_column])
            for label in self._candidate_labels:
                hypothesis = self._hypothesis_template.format(label)
                tokenized = dict(
                    tokenizer(
                        premise,
                        hypothesis,
                        padding="max_length",
                        truncation=True,
                        max_length=self._max_length,
                        return_tensors="pt",
                    )
                )
                samples.append(self._apply_io_mapping(tokenized))
                if self._max_samples is not None and len(samples) >= self._max_samples:
                    self._dataset = samples
                    return

        self._dataset = samples
        logger.info("Initialized zero-shot calibration dataset with %d pairs", len(samples))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a tokenized premise/hypothesis calibration sample."""
        return cast("dict[str, Any]", self._dataset[idx])

    @property
    def label_col(self) -> str:
        """Zero-shot calibration does not consume dataset labels."""
        return ""

    @property
    def max_length(self) -> int:
        """Sequence length."""
        assert self._max_length is not None, "max_length not resolved"
        return self._max_length
