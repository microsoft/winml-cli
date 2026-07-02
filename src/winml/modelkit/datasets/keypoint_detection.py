# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Keypoint-detection (pose estimation) dataset support for calibration.

Pose image processors (e.g. ``VitPoseImageProcessor``) crop each person out of a
bounding box before resizing to the model's fixed input, so their ``preprocess``
requires a ``boxes`` argument that plain :class:`ImageDataset` does not supply —
without it calibration silently falls back to random noise, which destroys the
quantized model's accuracy. This specialization feeds a single full-image box per
sample, producing real image activations at the model's ``pixel_values`` shape.
Ground-truth person boxes are not needed for representative calibration.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets.features import Image
from transformers import AutoImageProcessor

from .image import ImageDataset


logger = logging.getLogger(__name__)


class KeypointDetectionDataset(ImageDataset):
    """Calibration dataset for keypoint / pose-estimation models.

    Extends :class:`ImageDataset` to satisfy pose processors that require a
    ``boxes`` argument. Each calibration image is passed with one full-image box
    (``[0, 0, width, height]`` in COCO ``xywh`` format), so the processor crops
    and resizes the whole frame to the model input. This is modality-agnostic:
    any pose processor whose ``preprocess`` takes ``boxes`` works unchanged.
    """

    def _initialize(self) -> None:
        """Load the dataset and prepare pose-cropped ``pixel_values`` tensors."""
        # 1. Fall back to the built-in image dataset defaults when unset.
        if self._dataset_name is None:
            self._get_default_dataset()

        # 2. Load + sample (shared streaming/shuffle/max_samples behavior).
        dataset = self._load_and_sample()

        # 3. Detect the input image column (pose datasets carry no ClassLabel).
        self._detect_image_column(dataset)

        # 4. Load the model's own pose processor.
        processor = AutoImageProcessor.from_pretrained(self._model_name, use_fast=True)

        # 5. Preprocess each image with a single full-image box so the pose
        #    processor emits pixel_values at the model's fixed input shape.
        def preprocess_single_sample(example: dict[str, Any]) -> dict[str, Any]:
            image = example[self._image_col].convert("RGB")
            width, height = image.size
            boxes = [[[0.0, 0.0, float(width), float(height)]]]
            return dict(processor(image, boxes=boxes, return_tensors="pt"))

        self._dataset = (
            dataset.map(preprocess_single_sample, remove_columns=[self._image_col])
            .with_format("torch", output_all_columns=True)
        )

        logger.info("Dataset initialized with %d samples", len(self._dataset))

    def _detect_image_column(self, dataset: Any) -> None:
        """Detect the input image column for calibration.

        PTQ calibration only consumes ``pixel_values``; keypoint targets are not
        needed and are not read here.
        """
        if not hasattr(dataset, "features"):
            raise ValueError(f"Dataset {self._dataset_name} has no features metadata")

        features = dataset.features

        self._image_col = ""
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
