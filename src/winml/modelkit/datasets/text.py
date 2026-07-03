# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Text dataset for calibration pipelines.

Design:
- Universal tokenization (bert_common.py pattern)
- Feature-based column detection (no hardcoded patterns)
- io_config for ONNX shape extraction
- io_mapping with rename_column() for custom ONNX input names
- Explicit attributes over dict access
"""

from __future__ import annotations

import logging
from random import Random
from typing import Any, cast

from datasets import load_dataset
from datasets.features import ClassLabel, Value
from transformers import AutoTokenizer

from .base import BaseTaskDataset


logger = logging.getLogger(__name__)


class TextDataset(BaseTaskDataset):
    """Dataset for text tasks with universal tokenization.

    Supports:
    - Single sentence and sentence pair tasks (classification, fill-mask, etc.)
    - io_config for automatic shape extraction from ONNX
    - io_mapping for custom ONNX input names

    Priority for max_length: io_config > explicit param > DEFAULT_SEQ_LEN
    """

    DEFAULT_SEQ_LEN = 128

    def __init__(
        self,
        model_name: str,
        dataset_name: str | None = None,
        max_samples: int | None = None,
        data_split: str | None = None,
        max_length: int | None = None,
        io_config: dict | None = None,
        io_mapping: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize text classification dataset.

        Args:
            model_name: HuggingFace model identifier
            dataset_name: Dataset name (default: glue)
            max_samples: Maximum samples (None = use all)
            data_split: Dataset split (default: train)
            max_length: Sequence length (default: from io_config or 128)
            io_config: ONNX input config {name: {shape, dtype}}
            io_mapping: Map dataset fields to ONNX names {dataset_field: onnx_name}
            **kwargs: Additional config (subset, shuffle, seed, etc.)
        """
        # Store explicit attributes before super().__init__ calls _initialize()
        self._max_length = max_length  # May be None, resolved in _initialize
        self._io_config = io_config
        self._io_mapping = io_mapping or {}

        # Text/label columns detected in _initialize
        self._text_cols: list[str] = []
        self._label_col: str = ""
        self._label_feature: ClassLabel | None = None

        super().__init__(
            model_name=model_name,
            dataset_name=dataset_name,
            max_samples=max_samples,
            data_split=data_split,
            **kwargs,
        )

    def _get_default_dataset(self) -> None:
        """Set default dataset if none specified."""
        if self._dataset_name is None:
            self._dataset_name = "glue"
            self._config["subset"] = self._config.get("subset", "mrpc")
            self._data_split = self._data_split or "train"

    def _resolve_max_length(self) -> None:
        """Resolve max_length with priority: io_config > explicit > default."""
        # Start with default
        if self._max_length is None:
            self._max_length = self.DEFAULT_SEQ_LEN

        # io_config overrides (highest priority)
        if self._io_config:
            onnx_name = self._io_mapping.get("input_ids", "input_ids")
            if onnx_name in self._io_config:
                shape = self._io_config[onnx_name]["shape"]
                self._max_length = shape[1]
                logger.info("max_length=%d from io_config[%s]", self._max_length, onnx_name)

    def _apply_io_mapping(self) -> None:
        """Rename columns to match ONNX input names."""
        if not self._io_mapping:
            return

        for dataset_field, onnx_name in self._io_mapping.items():
            if dataset_field in self._dataset.column_names:
                self._dataset = self._dataset.rename_column(dataset_field, onnx_name)
                logger.info("Renamed: %s -> %s", dataset_field, onnx_name)

    def _initialize(self) -> None:
        """Initialize dataset with tokenization pipeline."""
        # 1. Set defaults
        self._get_default_dataset()

        # 2. Resolve max_length (io_config > explicit > default)
        self._resolve_max_length()

        # 3. Load dataset
        subset = self._config.get("subset")
        load_args = [self._dataset_name]
        if subset:
            load_args.append(subset)

        logger.warning("Loading: %s (subset=%s, split=%s)",
                    self._dataset_name, subset, self._data_split)

        try:
            dataset = load_dataset(*load_args, split=self._data_split)
        except Exception as e:
            logger.error("Failed to load dataset %s: %s", self._dataset_name, e)
            raise

        # 4. Detect columns by feature type
        self._detect_columns(dataset)

        # 5. Sample if needed
        shuffle = self._config.get("shuffle", False)
        seed = self._config.get("seed", 42)

        if self._max_samples is not None:
            n = min(self._max_samples, len(dataset))
            indices = (
                Random(seed).sample(range(len(dataset)), n)
                if shuffle
                else list(range(n))
            )
            dataset = dataset.select(indices)
        elif shuffle:
            dataset = dataset.shuffle(seed=seed)

        # 6. Tokenize using bert_common.py pattern: tokenizer(*texts, ...)
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, use_fast=True)

        def tokenize(example: dict) -> dict:
            texts = [example[col] for col in self._text_cols]
            return dict(tokenizer(
                *texts,
                padding="max_length",
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ))

        # 7. Apply tokenization, remove text columns
        self._dataset = (
            dataset
            .map(tokenize, remove_columns=self._text_cols)
            .with_format("torch", output_all_columns=True)
        )

        # 8. Rename columns for custom ONNX
        self._apply_io_mapping()

        logger.info("Initialized: %d samples, max_length=%d, text=%s, label=%s",
                    len(self._dataset), self._max_length, self._text_cols, self._label_col)

    def _detect_columns(self, dataset: Any) -> None:
        """Detect text and label columns by feature type."""
        if not hasattr(dataset, "features"):
            raise ValueError(f"Dataset {self._dataset_name} has no features")

        features = dataset.features
        text_cols = []
        label_col = None

        for name, feature in features.items():
            if isinstance(feature, Value) and feature.dtype == "string":
                text_cols.append(name)
            elif isinstance(feature, ClassLabel):
                label_col = name
                self._label_feature = feature

        if not text_cols:
            raise ValueError(f"No text columns in {self._dataset_name}")
        if not label_col:
            raise ValueError(f"No ClassLabel column in {self._dataset_name}")

        self._text_cols = text_cols[:2]
        self._label_col = label_col

        logger.info("Detected: text=%s, label=%s (%d classes)",
                    self._text_cols, label_col,
                    self._label_feature.num_classes if self._label_feature else 0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get tokenized sample."""
        return cast("dict[str, Any]", self._dataset[idx])

    # Readonly properties
    @property
    def max_length(self) -> int:
        """Sequence length."""
        assert self._max_length is not None, "max_length not resolved"
        return self._max_length

    @property
    def label_col(self) -> str:
        """Label column name."""
        return self._label_col

    @property
    def label_names(self) -> list[str]:
        """Class names."""
        return self._label_feature.names if self._label_feature else []

    @property
    def is_sentence_pair(self) -> bool:
        """True if sentence pair classification."""
        return len(self._text_cols) > 1

    @property
    def text_columns(self) -> list[str]:
        """Text column names."""
        return self._text_cols
