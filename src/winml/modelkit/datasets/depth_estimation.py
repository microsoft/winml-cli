# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Depth estimation dataset for calibration pipelines.

Extends ImageDataset with ONNX-aware configuration overrides
for depth estimation models. Depth-Anything (and similar) preprocessors
default to ``keep_aspect_ratio=True`` and ``do_pad=True``, which produces
variable-shape tensors. ONNX models exported with static input shapes
require a fixed ``(H, W)`` matching ``pixel_values`` in the io_config,
so this dataset forces the processor to emit that exact shape.
"""

from __future__ import annotations

import logging
from random import Random
from typing import Any

from datasets import load_dataset
from datasets.features import Image
from transformers import AutoImageProcessor

from .image import ImageDataset


logger = logging.getLogger(__name__)

# Default image size for depth estimation (matches Depth-Anything-V2)
DEFAULT_DEPTH_ESTIMATION_SIZE = 518

# Default dataset for depth estimation calibration.
# Mirrors evaluator default in winml.modelkit.eval.evaluate._DEFAULT_DATASETS
# so calibration uses the same samples as evaluation.
DEFAULT_DEPTH_ESTIMATION_DATASET = "sayakpaul/nyu_depth_v2"
DEFAULT_DEPTH_ESTIMATION_SPLIT = "validation"
# The NYU dataset is script-based; recent `datasets` releases reject
# script-based datasets, so the parquet mirror revision is required.
DEFAULT_DEPTH_ESTIMATION_REVISION = "refs/convert/parquet"


class DepthEstimationDataset(ImageDataset):
    """Dataset for depth estimation tasks with ONNX-aware configuration.

    Extends ImageDataset to handle depth estimation models that:

    - May default to ``keep_aspect_ratio=True`` (Depth-Anything family),
      producing variable output shapes that mismatch static ONNX inputs.
    - May default to ``do_pad=True``, also producing variable shapes.
    - Have no ``ClassLabel`` column — the label is a depth map (Image).

    The io_config from the ONNX model is passed via ``kwargs`` and used to
    derive the exact ``(H, W)`` to use.
    """

    def _get_default_dataset(self) -> None:
        """Set NYU depth v2 (parquet mirror) as the default for calibration.

        Overrides ``ImageDataset._get_default_dataset``, which would otherwise
        pick ``timm/mini-imagenet`` — unsuitable for depth estimation. We use
        the same dataset the evaluator uses so calibration and evaluation
        see consistent samples. The parquet revision avoids the script-based
        loading path that newer ``datasets`` releases reject.
        """
        if self._dataset_name is None:
            self._dataset_name = DEFAULT_DEPTH_ESTIMATION_DATASET
            self._data_split = DEFAULT_DEPTH_ESTIMATION_SPLIT
            self._revision = DEFAULT_DEPTH_ESTIMATION_REVISION

    def _derive_overrides(self, io_config: dict[str, Any] | None) -> dict[str, Any]:
        """Derive processor configuration overrides from ONNX io_config.

        Forces fixed-shape preprocessing by:

        1. Setting ``size`` to ``pixel_values`` shape (if known).
        2. Disabling ``keep_aspect_ratio`` (Depth-Anything-specific).
        3. Disabling ``do_pad``.

        Args:
            io_config: Dictionary mapping input names to their configs,
                       e.g., ``{"pixel_values": {"shape": [1, 3, 518, 518]}}``.

        Returns:
            Dictionary of processor configuration overrides.
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
        """Initialize the depth estimation dataset.

        Overrides parent to:

        1. Apply ONNX-derived processor overrides (size, keep_aspect_ratio, do_pad).
        2. Use depth-estimation default size when io_config is unavailable.
        3. Skip ClassLabel detection (depth maps are images, not labels).
        """
        # Apply task-specific defaults when caller did not specify a dataset
        # (e.g. quantization calibration path goes through universal_calib_dataset).
        if self._dataset_name is None:
            self._get_default_dataset()

        revision = getattr(self, "_revision", None)

        # Load dataset
        logger.info(
            "Loading depth estimation dataset: %s with split: %s (revision=%s)",
            self._dataset_name,
            self._data_split,
            revision,
        )
        try:
            dataset = load_dataset(
                self._dataset_name,
                split=self._data_split,
                revision=revision,
            )
        except Exception as e:
            logger.error("Failed to load dataset %s: %s", self._dataset_name, e)
            raise

        # Detect image column (depth datasets have no ClassLabel)
        self._detect_image_column(dataset)

        # Efficient sampling
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

        # Derive ONNX-aware overrides
        io_config = self._config.get("io_config")
        overrides = self._derive_overrides(io_config)

        # Set default size if not derived from io_config
        if "size" not in overrides:
            overrides["size"] = {
                "height": DEFAULT_DEPTH_ESTIMATION_SIZE,
                "width": DEFAULT_DEPTH_ESTIMATION_SIZE,
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
        """Detect image column for depth estimation datasets.

        Depth estimation datasets have an Image column for the input photo
        and typically a second Image column for the ground-truth depth map.
        No ClassLabel column exists.

        Args:
            dataset: HuggingFace dataset to analyze.

        Raises:
            ValueError: If no Image column found.
        """
        if not hasattr(dataset, "features"):
            raise ValueError(f"Dataset {self._dataset_name} has no features metadata")

        features = dataset.features

        # Find the first Image column (input image)
        self._image_col = None
        self._label_col = None
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

        # Try to find a depth-map column (second Image or commonly-named field)
        for col_name in ["depth_map", "depth", "depths"]:
            if col_name in features and col_name != self._image_col:
                self._label_col = col_name
                break

        if self._label_col is None:
            # Fallback: any other column (Image or otherwise)
            for col_name in features:
                if col_name != self._image_col:
                    self._label_col = col_name
                    break

        logger.info(
            "Detected columns - image: '%s', depth: '%s'",
            self._image_col,
            self._label_col,
        )

    @property
    def label_col(self) -> str:
        """Get the depth-map column name (readonly)."""
        return self._label_col or ""

    @property
    def label_names(self) -> list[str]:
        """Depth estimation has no class labels."""
        return []
