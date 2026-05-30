# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Depth-estimation dataset support for calibration.

This dataset keeps image preprocessing aligned with the exported ONNX model.
When a model expects a fixed ``pixel_values`` shape, the processor is forced
to emit that exact size so calibration samples match the model input.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets.features import Image
from transformers import AutoImageProcessor

from .image import ImageDataset


logger = logging.getLogger(__name__)

# Default fallback image size for depth-estimation models.
DEFAULT_DEPTH_ESTIMATION_SIZE = 518

# Default calibration dataset for depth estimation.
# Using the same dataset family in calibration and evaluation keeps behavior
# consistent when users rely on the built-in defaults.
DEFAULT_DEPTH_ESTIMATION_DATASET = "sayakpaul/nyu_depth_v2"
DEFAULT_DEPTH_ESTIMATION_SPLIT = "validation"
# Use the parquet mirror revision so the dataset can be loaded reliably
# through the standard HuggingFace datasets API.
DEFAULT_DEPTH_ESTIMATION_REVISION = "refs/convert/parquet"


class DepthEstimationDataset(ImageDataset):
    """Depth-estimation dataset with fixed-shape preprocessing.

    This specialization ensures calibration samples follow the input shape of
    the exported ONNX model and works with datasets whose target is a depth map
    instead of a class label.
    """

    def _get_default_dataset(self) -> None:
        """Set the built-in depth-estimation dataset defaults.

        The default points to the NYU depth dataset and a stable revision that
        can be loaded directly through ``datasets``.
        """
        if self._dataset_name is None:
            self._dataset_name = DEFAULT_DEPTH_ESTIMATION_DATASET
            self._data_split = DEFAULT_DEPTH_ESTIMATION_SPLIT
            self._revision = DEFAULT_DEPTH_ESTIMATION_REVISION

    def _derive_overrides(self, io_config: dict[str, Any] | None) -> dict[str, Any]:
        """Build processor overrides from the ONNX input configuration.

        When the model exposes a fixed ``pixel_values`` shape, that shape is
        applied to the image processor and variable-size preprocessing is
        disabled.
        """
        overrides: dict[str, Any] = {
            "keep_aspect_ratio": False,
            "do_pad": False,
        }

        if io_config is None:
            logger.debug("No io_config provided, using default overrides")
            return overrides

        if "pixel_values" in io_config:
            shape = io_config["pixel_values"].get("shape", [])
            # Shape is typically [batch, channels, height, width]
            if len(shape) >= 4:
                height = shape[2]
                width = shape[3]
                if height is not None and width is not None:
                    overrides["size"] = {"height": height, "width": width}
                    logger.debug(
                        "Extracted size from io_config: height=%d, width=%d",
                        height,
                        width,
                    )

        return overrides

    def _initialize(self) -> None:
        """Load the dataset and prepare fixed-shape image tensors."""
        # Use the built-in defaults when the caller does not provide a dataset.
        if self._dataset_name is None:
            self._get_default_dataset()

        # Reuse the parent helper so streaming/shuffle/max_samples behavior
        # stays consistent with ImageDataset and ObjectDetectionDataset.
        dataset = self._load_and_sample(revision=getattr(self, "_revision", None))

        # Detect the input image column and the depth target column.
        self._detect_image_column(dataset)

        # Match processor output to the ONNX input shape when available.
        io_config = self._config.get("io_config")
        overrides = self._derive_overrides(io_config)

        # Fall back to the default square size when the ONNX shape is absent.
        if "size" not in overrides:
            overrides["size"] = {
                "height": DEFAULT_DEPTH_ESTIMATION_SIZE,
                "width": DEFAULT_DEPTH_ESTIMATION_SIZE,
            }

        # Create a processor that emits tensors compatible with calibration.
        processor = AutoImageProcessor.from_pretrained(
            self._model_name,
            use_fast=True,
            **overrides,
        )

        logger.debug("Created processor with overrides: %s", overrides)

        # Convert raw images into model-ready tensors.
        def preprocess_single_sample(example: dict[str, Any]) -> dict[str, Any]:
            return processor(example[self._image_col].convert("RGB"), return_tensors="pt")

        self._dataset = dataset.map(
            preprocess_single_sample, remove_columns=[self._image_col]
        ).with_format("torch", output_all_columns=True)

        logger.info("Dataset initialized with %d samples", len(self._dataset))

    def _detect_image_column(self, dataset: Any) -> None:
        """Detect the input image column for calibration.

        PTQ calibration only consumes ``pixel_values``; the depth target is
        not needed and is not read here.
        """
        if not hasattr(dataset, "features"):
            raise ValueError(f"Dataset {self._dataset_name} has no features metadata")

        features = dataset.features

        self._image_col = None
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

        logger.info("Detected image column: '%s'", self._image_col)

    @property
    def label_col(self) -> str:
        """No label column is used for depth-estimation calibration."""
        return ""

    @property
    def label_names(self) -> list[str]:
        """Depth estimation has no class labels."""
        return []
