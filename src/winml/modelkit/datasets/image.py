# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Image dataset for general-purpose ML pipelines.

Uses model's own preprocessor for proper image handling - no hardcoded sizes.
Supports explicit column configuration and HuggingFace Features API.
"""

from __future__ import annotations

import logging
from random import Random
from typing import Any

from datasets import load_dataset
from datasets.features import ClassLabel, Image
from transformers import AutoImageProcessor

from .base import BaseTaskDataset
from .label_utils import get_imagenet_label_map, should_align_labels


logger = logging.getLogger(__name__)


class ImageDataset(BaseTaskDataset):
    """Dataset for image tasks with explicit configuration.

    This dataset supports:
    - Automatic preprocessing using the model's own processor
    - Explicit column configuration (no guessing)
    - Label alignment for datasets like ImageNet
    - HuggingFace Features API for metadata discovery
    """


    def _get_default_dataset(self) -> None:
        """Set default dataset configuration if none specified.

        Uses timm/mini-imagenet with train split as sensible defaults
        for testing and development when no dataset is explicitly provided.

        TODO: Expand to support multiple dataset presets:
        - Different model types (vision transformers, CNNs, etc.)
        - Task-specific datasets (classification, detection, etc.)
        - Size variants (mini, full, custom sample counts)
        """
        if self._dataset_name is None:
            self._dataset_name = "timm/mini-imagenet"
            self._data_split = "train"
            self._config.setdefault("streaming", True)

    def _load_and_sample(self) -> Any:
        """Load the configured dataset and apply sample/shuffle.

        Shared by ImageDataset and ObjectDetectionDataset. Column detection
        is *not* done here — callers run their own detection on the returned
        dataset because the column schema differs by task.

        Returns:
            A materialized arrow Dataset of up to ``self._max_samples`` rows.
        """
        # Streaming only helps when capped by max_samples; otherwise we'd
        # iterate the full remote stream into memory, which is worse than a
        # bulk download.
        streaming = self._config.get("streaming", False) and self._max_samples is not None
        logger.info(f"Loading dataset: {self._dataset_name} with split: {self._data_split}")
        try:
            dataset = load_dataset(self._dataset_name, split=self._data_split, streaming=streaming)
        except Exception as e:
            logger.error(f"Failed to load dataset {self._dataset_name}: {e}")
            raise

        shuffle = self._config.get("shuffle", False)
        seed = self._config.get("seed", 42)

        if streaming:
            # Streaming datasets aren't indexable: shuffle reservoir-samples
            # within a buffer; take() pulls only the slice we need.
            if shuffle:
                dataset = dataset.shuffle(seed=seed, buffer_size=1000)
            dataset = dataset.take(self._max_samples)
            from datasets import Dataset as ArrowDataset
            dataset = ArrowDataset.from_list(list(dataset), features=dataset.features)
        elif self._max_samples is not None:
            max_samples = min(self._max_samples, len(dataset))
            indices = (
                Random(seed).sample(range(len(dataset)), max_samples)
                if shuffle
                else list(range(max_samples))
            )
            dataset = dataset.select(indices)
        elif shuffle:
            dataset = dataset.shuffle(seed=seed)

        return dataset

    def _initialize(self) -> None:
        """Initialize the image classification dataset.

        Simplified approach:
        1. Set defaults if needed
        2. Load dataset
        3. Detect columns
        4. Apply efficient processing pipeline
        """
        # 1. Set defaults if no dataset specified
        if self._dataset_name is None:
            self._get_default_dataset()

        # 2. Load + sample (shared with subclasses)
        dataset = self._load_and_sample()

        # 3. Detect columns using Features API
        self._detect_columns(dataset)

        # 4. Load processor and apply batch processing
        processor = AutoImageProcessor.from_pretrained(self._model_name, use_fast=True)

        # 5. Conditional label alignment using should_align_labels()
        if should_align_labels(self._dataset_name):
            dataset = dataset.align_labels_with_mapping(get_imagenet_label_map(), self._label_col)

        # 6. Apply image processing with proper batch dimension
        def preprocess_single_sample(example):
            # Process single image and add batch dimension
            return processor(example[self._image_col].convert("RGB"), return_tensors="pt")

        self._dataset = (
            dataset
            .map(preprocess_single_sample, remove_columns=[self._image_col])
            .with_format("torch", output_all_columns=True)
        )

        logger.info(f"Dataset initialized with {len(self._dataset)} samples")

    def _detect_columns(self, dataset) -> None:
        """Detect image and label columns using HuggingFace Features API.

        Uses proper type checking with HuggingFace Features API to reliably
        identify Image and ClassLabel columns without hardcoded assumptions.
        """
        if not hasattr(dataset, "features"):
            raise ValueError(f"Dataset {self._dataset_name} has no features metadata")

        features = dataset.features

        # Detect columns using proper type checking
        self._image_col = None
        self._label_col = None
        self._label_feature = None  # Store ClassLabel feature for mapping

        for col_name, feature in features.items():
            if isinstance(feature, Image):
                self._image_col = col_name
            elif isinstance(feature, ClassLabel):
                self._label_col = col_name
                self._label_feature = feature  # Keep reference for label operations

        # Ensure required columns were found
        if not self._image_col:
            available_cols = list(features.keys())
            available_types = [type(f).__name__ for f in features.values()]
            raise ValueError(
                f"No Image column found in {self._dataset_name}. "
                f"Available: {dict(zip(available_cols, available_types, strict=False))}"
            )

        if not self._label_col:
            available_cols = list(features.keys())
            available_types = [type(f).__name__ for f in features.values()]
            raise ValueError(
                f"No ClassLabel column found in {self._dataset_name}. "
                f"Available: {dict(zip(available_cols, available_types, strict=False))}"
            )

        # Log successful detection with class information
        num_classes = self._label_feature.num_classes if self._label_feature else "unknown"
        logger.info(
            f"Detected columns - image: '{self._image_col}', "
            f"label: '{self._label_col}' ({num_classes} classes)"
        )

    @property
    def label_col(self) -> str:
        """Get the label column name (readonly)."""
        return self._label_col

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get pre-processed sample.

        Since the dataset is pre-processed during initialization with batch processing,
        this method simply returns the already-processed sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing preprocessed tensors
        """
        return self._dataset[idx]

    @property
    def label_names(self) -> list[str]:
        """Get class names from ClassLabel feature."""
        return self._label_feature.names if self._label_feature else []

