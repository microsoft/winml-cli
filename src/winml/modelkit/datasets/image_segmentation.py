"""Image segmentation dataset for general-purpose ML pipelines.

Uses model's own image processor for proper image + mask handling - supports semantic,
instance, and panoptic segmentation tasks. Supports explicit column configuration and
HuggingFace Features API with mask2former model integration.
"""

from __future__ import annotations

import logging
from random import Random
from typing import Any

import torch
from datasets import load_dataset
from datasets.features import Image
from transformers import AutoImageProcessor

from .base import BaseTaskDataset


logger = logging.getLogger(__name__)


class ImageSegmentationDataset(BaseTaskDataset):
    """Dataset for image segmentation tasks with explicit configuration.

    This dataset supports:
    - Automatic preprocessing using the model's own image processor
    - Semantic, instance, and panoptic segmentation
    - Explicit column configuration (no guessing)
    - HuggingFace Features API for metadata discovery
    - Mask2former model integration
    """

    DEFAULT_DATASET = "nielsr/ade20k-demo"
    DEFAULT_SPLIT = "train"

    def __init__(
        self,
        model_name: str,
        dataset_name: str | None = None,
        max_samples: int | None = None,
        data_split: str | None = None,
        do_reduce_labels: bool = True,
        **kwargs,
    ) -> None:
        """Initialize image segmentation dataset.

        Args:
            model_name: HuggingFace model identifier or path
            dataset_name: Dataset name (uses DEFAULT_DATASET if None)
            max_samples: Maximum number of samples (None = use all)
            data_split: Dataset split (uses DEFAULT_SPLIT if None)
            do_reduce_labels: Whether to reduce labels by 1 (background handling)
            **kwargs: Additional dataset-specific parameters
        """
        # Store segmentation-specific config
        self._do_reduce_labels = do_reduce_labels

        # Use default split if not specified
        if data_split is None:
            data_split = self.DEFAULT_SPLIT

        super().__init__(
            model_name=model_name,
            dataset_name=dataset_name,
            max_samples=max_samples,
            data_split=data_split,
            do_reduce_labels=do_reduce_labels,
            **kwargs,
        )

    def _initialize(self) -> None:
        """Initialize the image segmentation dataset.

        Process:
        1. Set defaults if needed
        2. Load dataset
        3. Detect image and mask columns
        4. Apply efficient preprocessing pipeline
        """
        # 1. Set defaults if no dataset specified
        if self._dataset_name is None:
            self._dataset_name = self.DEFAULT_DATASET

        # 2. Load dataset
        logger.info(f"Loading dataset: {self._dataset_name} with split: {self._data_split}")
        try:
            dataset = load_dataset(self._dataset_name, split=self._data_split)
        except Exception as e:
            logger.error(f"Failed to load dataset {self._dataset_name}: {e}")
            raise

        # 3. Detect columns using Features API
        self._detect_columns(dataset)

        # 4. Efficient sampling and processing pipeline
        shuffle = self._config.get("shuffle", False)
        seed = self._config.get("seed", 42)

        if self._max_samples is not None:
            max_samples = min(self._max_samples, len(dataset))
            indices = (
                Random(seed).sample(range(len(dataset)), max_samples)
                if shuffle
                else list(range(max_samples))
            )
            dataset = dataset.select(indices)
        elif shuffle:
            dataset = dataset.shuffle(seed=seed)

        # 5. Load image processor and apply batch processing
        processor = AutoImageProcessor.from_pretrained(self._model_name, use_fast=True)

        # 6. Apply image + mask processing
        def preprocess_single_sample(example):
            """Preprocess a single image + mask sample for segmentation models."""
            # Get image and mask
            image = example[self._image_col].convert("RGB")
            mask = example[self._mask_col]

            # Process with segmentation-aware processor
            inputs = processor(
                images=image,
                segmentation_maps=mask,
                do_reduce_labels=self._do_reduce_labels,
                return_tensors="pt"
            )

            # Squeeze batch dimension for tensors, keep lists as-is
            processed = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    processed[key] = value.squeeze(0)
                else:
                    # Keep lists and other types unchanged (e.g., mask_labels, class_labels)
                    processed[key] = value
            return processed

        # Remove image and mask columns but keep other metadata
        columns_to_remove = [self._image_col, self._mask_col]

        self._dataset = (
            dataset
            .map(preprocess_single_sample, remove_columns=columns_to_remove)
            .with_format("torch", output_all_columns=True)
        )

        logger.info(f"Dataset initialized with {len(self._dataset)} samples")
        logger.info(f"Image column: {self._image_col}")
        logger.info(f"Mask column: {self._mask_col}")

    def _detect_columns(self, dataset) -> None:
        """Detect image and mask columns using HuggingFace Features API.

        Uses proper type checking to identify Image features and applies
        common naming patterns for segmentation datasets.
        """
        if not hasattr(dataset, "features"):
            raise ValueError(f"Dataset {self._dataset_name} has no features metadata")

        features = dataset.features

        # Initialize column detection
        self._image_col = None
        self._mask_col = None

        # Detect columns using proper type checking and naming patterns
        image_candidates = []
        mask_candidates = []

        for col_name, feature in features.items():
            if isinstance(feature, Image):
                # Check for mask/annotation patterns
                if any(keyword in col_name.lower() for keyword in [
                    'annotation', 'mask', 'label', 'segmentation', 'target', 'gt'
                ]):
                    mask_candidates.append(col_name)
                # Check for image patterns
                elif any(keyword in col_name.lower() for keyword in [
                    'image', 'img', 'photo', 'picture'
                ]) or col_name.lower() == 'image':
                    image_candidates.append(col_name)
                else:
                    # Fallback: if no clear pattern, add to both for later decision
                    image_candidates.append(col_name)

        # Assign columns based on patterns and fallbacks
        if not image_candidates:
            # Look for any Image feature as potential image column
            for col_name, feature in features.items():
                if isinstance(feature, Image):
                    image_candidates.append(col_name)

        # Prioritize column assignment
        if image_candidates:
            # Prefer 'image' if available, otherwise use first candidate
            self._image_col = next(
                (col for col in image_candidates if col.lower() == 'image'),
                image_candidates[0]
            )

        if mask_candidates:
            # Prefer common mask names, otherwise use first candidate
            preferred_mask_names = ['label', 'annotation', 'mask', 'segmentation']
            self._mask_col = next(
                (col for col in mask_candidates if col.lower() in preferred_mask_names),
                mask_candidates[0]
            )

        # Handle case where image was classified as mask (if only 2 Image columns)
        image_features = [col for col, feat in features.items() if isinstance(feat, Image)]
        if len(image_features) == 2 and not self._mask_col:
            # Assume first is image, second is mask
            self._image_col = image_features[0]
            self._mask_col = image_features[1]

        # Ensure required columns were found
        if not self._image_col:
            available_cols = list(features.keys())
            available_types = [
                f"{type(f).__name__}({f.dtype if hasattr(f, 'dtype') else 'N/A'})"
                for f in features.values()
            ]
            raise ValueError(
                f"No suitable image column found in {self._dataset_name}. "
                f"Available: {dict(zip(available_cols, available_types, strict=False))}"
            )

        if not self._mask_col:
            available_cols = list(features.keys())
            available_types = [type(f).__name__ for f in features.values()]
            raise ValueError(
                f"No mask/annotation column found in {self._dataset_name}. "
                f"Available: {dict(zip(available_cols, available_types, strict=False))}"
            )

        # Log successful detection
        logger.info(
            f"Detected columns - image: '{self._image_col}', "
            f"mask: '{self._mask_col}'"
        )

    @property
    def label_col(self) -> str:
        """Get the mask column name (readonly)."""
        return self._mask_col

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get pre-processed sample.

        Since the dataset is pre-processed during initialization with batch processing,
        this method simply returns the already-processed sample containing processed
        images, masks, and any additional metadata.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing preprocessed tensors for segmentation models
        """
        return self._dataset[idx]

    @property
    def mask_col(self) -> str:
        """Get the mask column name (segmentation-specific)."""
        return self._mask_col

    @property
    def image_col(self) -> str:
        """Get the image column name."""
        return self._image_col

    @property
    def preprocessing_config(self) -> dict[str, Any]:
        """Get preprocessing configuration used."""
        return {
            "do_reduce_labels": self._do_reduce_labels,
            "image_column": self._image_col,
            "mask_column": self._mask_col,
        }
