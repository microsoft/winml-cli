# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Object detection dataset for DETR-like models in calibration pipelines.

Extends ImageDataset with ONNX-aware configuration overrides
for object detection models that may have different input requirements.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets.features import Image
from transformers import AutoImageProcessor

from .image import ImageDataset


logger = logging.getLogger(__name__)

# Default image size for object detection models (DETR uses larger images)
DEFAULT_OBJECT_DETECTION_SIZE = 640


class ObjectDetectionDataset(ImageDataset):
    """Dataset for object detection tasks with ONNX-aware configuration.

    Extends ImageDataset to handle DETR-like models that may
    require different preprocessing based on ONNX model structure:

    - Disables padding (do_pad=False) when ONNX model has no pixel_mask input
    - Extracts image size from ONNX model's pixel_values shape
    - Uses 640 as default size (larger than classification's 384)

    The io_config from ONNX model is passed via kwargs and used to derive
    appropriate preprocessing overrides.
    """

    def _derive_overrides(self, io_config: dict[str, Any] | None) -> dict[str, Any]:
        """Derive processor configuration overrides from ONNX io_config.

        Analyzes the ONNX model's input configuration to determine:
        1. Whether padding should be disabled (no pixel_mask input)
        2. What image size to use (from pixel_values shape)

        Args:
            io_config: Dictionary mapping input names to their configs,
                       e.g., {"pixel_values": {"shape": [1, 3, 640, 640]}}

        Returns:
            Dictionary of processor configuration overrides
        """
        overrides: dict[str, Any] = {}

        if io_config is None:
            logger.debug("No io_config provided, using default overrides")
            return overrides

        # Check if pixel_mask is present in ONNX inputs
        has_pixel_mask = "pixel_mask" in io_config
        if not has_pixel_mask:
            overrides["do_pad"] = False
            logger.debug("No pixel_mask in io_config, setting do_pad=False")

        # Extract size from pixel_values shape if available
        if "pixel_values" in io_config:
            shape = io_config["pixel_values"].get("shape", [])
            # Shape is typically [batch, channels, height, width]
            if len(shape) >= 4:
                height = shape[2]
                width = shape[3]
                # Only override if dimensions are concrete (not None/dynamic)
                if height is not None and width is not None:
                    overrides["size"] = {"height": height, "width": width}
                    logger.debug(
                        "Extracted size from io_config: height=%d, width=%d",
                        height,
                        width,
                    )

        return overrides

    def _initialize(self) -> None:
        """Initialize the object detection dataset with ONNX-aware configuration.

        Overrides parent to:
        1. Apply ONNX-derived configuration overrides
        2. Use object detection default size (640)
        3. Skip label alignment (object detection has different label structure)
        """
        # Set defaults if no dataset specified
        if self._dataset_name is None:
            self._get_default_dataset()

        # Load + sample (shared with ImageDataset)
        dataset = self._load_and_sample()

        # Detect image column (object detection may not have simple ClassLabel)
        self._detect_image_column(dataset)

        # Derive ONNX-aware overrides
        io_config = self._config.get("io_config")
        overrides = self._derive_overrides(io_config)

        # Set default size if not derived from io_config
        if "size" not in overrides:
            overrides["size"] = {
                "height": DEFAULT_OBJECT_DETECTION_SIZE,
                "width": DEFAULT_OBJECT_DETECTION_SIZE,
            }

        # Create processor with overrides
        processor = AutoImageProcessor.from_pretrained(
            self._model_name,
            use_fast=True,
            **overrides,
        )

        logger.debug("Created processor with overrides: %s", overrides)

        # Apply image processing
        def preprocess_single_sample(example: dict[str, Any]) -> dict[str, Any]:
            return processor(example[self._image_col].convert("RGB"), return_tensors="pt")

        self._dataset = (
            dataset
            .map(preprocess_single_sample, remove_columns=[self._image_col])
            .with_format("torch", output_all_columns=True)
        )

        logger.info("Dataset initialized with %d samples", len(self._dataset))

    def _detect_image_column(self, dataset: Any) -> None:
        """Detect image column for object detection datasets.

        Object detection datasets may have different structures than
        classification datasets, so we focus on finding the Image column.

        Args:
            dataset: HuggingFace dataset to analyze

        Raises:
            ValueError: If no Image column found
        """
        if not hasattr(dataset, "features"):
            raise ValueError(f"Dataset {self._dataset_name} has no features metadata")

        features = dataset.features

        # Find image column
        self._image_col = None
        self._label_col = None  # May not have simple label column
        self._label_feature = None

        for col_name, feature in features.items():
            if isinstance(feature, Image):
                self._image_col = col_name
                break

        if not self._image_col:
            available_cols = list(features.keys())
            available_types = [type(f).__name__ for f in features.values()]
            raise ValueError(
                f"No Image column found in {self._dataset_name}. "
                f"Available: {dict(zip(available_cols, available_types, strict=False))}"
            )

        # Try to find a label-like column (objects, boxes, annotations)
        for col_name in ["objects", "labels", "label", "annotations"]:
            if col_name in features:
                self._label_col = col_name
                break

        if self._label_col is None:
            # Use first non-image column as fallback
            for col_name in features:
                if col_name != self._image_col:
                    self._label_col = col_name
                    break

        logger.info(
            "Detected columns - image: '%s', label/objects: '%s'",
            self._image_col,
            self._label_col,
        )

    @property
    def label_col(self) -> str:
        """Get the label/objects column name (readonly)."""
        return self._label_col or ""

    @property
    def label_names(self) -> list[str]:
        """Get class names if available.

        Object detection datasets typically use COCO-style labels,
        so this returns an empty list unless explicitly set.
        """
        if self._label_feature is not None and hasattr(self._label_feature, "names"):
            return self._label_feature.names
        return []
