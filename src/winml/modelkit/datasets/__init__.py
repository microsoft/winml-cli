# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Predefined datasets for calibration presets.

This module provides organized, extensible datasets for different model tasks,
making it easy to add new datasets and improve calibration quality over time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from onnxruntime.quantization import CalibrationDataReader

from .base import BaseTaskDataset
from .data_utils import format_data
from .depth_estimation import DEFAULT_DEPTH_ESTIMATION_SIZE, DepthEstimationDataset
from .image import ImageDataset
from .image_segmentation import ImageSegmentationDataset
from .keypoint_detection import KeypointDetectionDataset
from .mask_generation import MaskGenerationDataset
from .object_detection import DEFAULT_OBJECT_DETECTION_SIZE, ObjectDetectionDataset
from .processor_utils import get_image_processor_config
from .random_dataset import RandomDataset
from .text import TextDataset


if TYPE_CHECKING:
    from collections.abc import Set

    import numpy as np

logger = logging.getLogger(__name__)

# Task dataset mapping for olive_runner
# Maps task types to dataset classes
TASK_DATASET_MAPPING = {
    "image-classification": ImageDataset,
    "image-feature-extraction": ImageDataset,
    "object-detection": ObjectDetectionDataset,
    "text-classification": TextDataset,
    "text-feature-extraction": TextDataset,
    "feature-extraction": TextDataset,
    "sentence-similarity": TextDataset,
    "next-sentence-prediction": TextDataset,
    "fill-mask": TextDataset,
    "zero-shot-classification": TextDataset,
    "image-segmentation": ImageSegmentationDataset,
    "mask-generation": MaskGenerationDataset,
    "depth-estimation": DepthEstimationDataset,
    "keypoint-detection": KeypointDetectionDataset,
    "random": RandomDataset,
    # Add more task types as needed
}


def _resolve_dataset_class(task: str) -> tuple[type, str]:
    """Resolve the dataset class for ``task``.

    Every task now maps directly to one dataset class: detection yields modality-aware
    tasks (e.g. ``image-feature-extraction`` rather than the lossy ``feature-extraction``),
    so the previous io_config-based reverse reconstruction is no longer needed.
    """
    return TASK_DATASET_MAPPING[task], task


def _dataset_produces_any_input(dataset: Any, input_names: set[str]) -> bool:
    """Whether the dataset's first sample shares any field with the ONNX input names.

    Used to detect a calibration modality mismatch (e.g. a text dataset's ``input_ids``
    for an audio model that wants ``input_values``). Conservative: an empty model-input
    set, an empty dataset, or any sampling error returns ``True`` (no fallback), so the
    check only triggers a fallback on a clear, total mismatch.
    """
    if not input_names:
        return True
    try:
        sample = dataset[0]
    except Exception:
        return True
    return bool(set(sample.keys()) & input_names)


def universal_calib_dataset(
    model_name: str,
    task: str,
    dataset_name: str | None = None,
    max_samples: int | None = None,
    data_split: str | None = None,
    **kwargs: Any,
) -> Any:
    """Universal dataset function that creates concrete datasets based on task.

    Factory function for creating task-specific datasets.

    Args:
        model_name: HuggingFace model name
        task: Task type (e.g., "image-classification", "text-classification")
        dataset_name: Optional dataset name (uses defaults if None)
        max_samples: Maximum number of samples (uses all if None)
        data_split: Dataset split to use (uses default if None)
        **kwargs: Additional parameters passed to dataset constructor

    Returns:
        Dataset instance ready for use

    Raises:
        ValueError: If task type is not supported or parameters invalid
        RuntimeError: If dataset creation fails
    """
    # Input validation
    if not task:
        raise ValueError("task parameter is required")
    if not model_name:
        raise ValueError("model_name parameter is required")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive if specified")

    model_path = kwargs.get("model_path")
    io_config = kwargs.get("io_config")

    # Fallback to random dataset for unsupported tasks
    if task not in TASK_DATASET_MAPPING:
        supported = list(TASK_DATASET_MAPPING.keys())
        logger.warning(
            "Task '%s' not in supported tasks %s, falling back to RandomDataset",
            task,
            supported,
        )
        task = "random"

    dataset_class, task = _resolve_dataset_class(task)

    # Craft kwargs - only add optional parameters if provided
    dataset_kwargs = {
        "model_name": model_name,
        **kwargs,
    }
    if dataset_name is not None:
        dataset_kwargs["dataset_name"] = dataset_name
    if max_samples is not None:
        dataset_kwargs["max_samples"] = max_samples
    if data_split is not None:
        dataset_kwargs["data_split"] = data_split

    def _random_fallback(reason: str) -> Any:
        # Only reached under the `model_path is not None` guard at the call site below.
        assert model_path is not None
        logger.warning("Falling back to RandomDataset for calibration: %s", reason)
        return RandomDataset(model_path=model_path, max_samples=max_samples or 100)

    # Create dataset instance (loading happens in constructor with lazy pattern).
    logger.info("Creating %s dataset with %s", task, dataset_class.__name__)
    try:
        dataset = dataset_class(**dataset_kwargs)
    except Exception as e:
        # A task-specific dataset that cannot be built for this model (e.g. an audio
        # backbone that stays feature-extraction -> TextDataset, which needs a tokenizer
        # the model lacks) degrades to RandomDataset when an ONNX model_path is known.
        if dataset_class is not RandomDataset and model_path is not None:
            return _random_fallback(f"{dataset_class.__name__} construction failed: {e}")
        raise RuntimeError(f"Failed to create {task} dataset: {e}") from e

    # Modality mismatch: the dataset produces none of the ONNX model's input tensors
    # (e.g. a text dataset's input_ids for an audio model that wants input_values). Fall
    # back so calibration feeds inputs the model accepts instead of failing in the ORT
    # session. Modality-agnostic — also covers image/video and any future modality.
    if (
        dataset_class is not RandomDataset
        and model_path is not None
        and io_config
        and not _dataset_produces_any_input(dataset, set(io_config))
    ):
        return _random_fallback(
            f"{dataset_class.__name__} produces none of the model inputs {sorted(io_config)}"
        )

    logger.info("Created dataset with %d samples for calibration", len(dataset))
    return dataset


class DatasetCalibrationReader(CalibrationDataReader):  # type: ignore[misc]
    """Calibration data reader that wraps universal_calib_dataset.

    Bridges HuggingFace-style datasets to ORT's calibration API by:
    - Creating dataset via universal_calib_dataset
    - Converting PyTorch tensors to numpy arrays
    - Filtering out non-input keys (labels, metadata)
    - Implementing get_next() iterator protocol

    Example:
        >>> from winml.modelkit.datasets import DatasetCalibrationReader
        >>> from winml.modelkit.quant import quantize_onnx, QuantizeConfig
        >>>
        >>> reader = DatasetCalibrationReader(
        ...     model_name="facebook/convnext-tiny-224",
        ...     task="image-classification",
        ...     max_samples=100,
        ... )
        >>> result = quantize_onnx("model.onnx", QuantizeConfig(calibration_data=reader))
    """

    DEFAULT_EXCLUDE_KEYS: frozenset[str] = frozenset({"label", "labels", "idx", "sample_id"})

    def __init__(
        self,
        model_name: str,
        task: str,
        max_samples: int = 100,
        dataset_name: str | None = None,
        data_split: str | None = None,
        exclude_keys: Set[str] | None = None,
        model_path: Any = None,
        **dataset_kwargs: Any,
    ) -> None:
        """Initialize calibration data reader.

        Args:
            model_name: HuggingFace model name for preprocessing
            task: Task type (image-classification, text-classification, etc.)
            max_samples: Maximum calibration samples
            dataset_name: Specific dataset (None = task default)
            data_split: Dataset split (None = task default)
            exclude_keys: Keys to exclude from output (default: {"label", "labels"})
            model_path: Path to ONNX model (for extracting io_config)
            **dataset_kwargs: Additional args passed to dataset constructor
        """
        self.model_name = model_name
        self.task = task
        self.max_samples = max_samples
        self.exclude_keys = frozenset(exclude_keys) if exclude_keys else self.DEFAULT_EXCLUDE_KEYS
        self._index = 0

        # Extract io_config from ONNX model for sequence length, dtype casting, and input filtering
        self._io_config: dict | None = None
        self._valid_input_names: set[str] | None = None
        if model_path is not None:
            self._io_config = self._extract_io_config(model_path)
            io_config = self._io_config
            if io_config:
                self._valid_input_names = set(io_config.keys())
                dataset_kwargs["io_config"] = io_config
            # Pass model_path for RandomDataset fallback on unsupported tasks
            dataset_kwargs["model_path"] = model_path

        # Create dataset via factory
        self._dataset = universal_calib_dataset(
            model_name=model_name,
            task=task,
            dataset_name=dataset_name,
            max_samples=max_samples,
            data_split=data_split,
            **dataset_kwargs,
        )

        logger.info(
            "Created DatasetCalibrationReader: task=%s, samples=%d",
            task,
            len(self._dataset),
        )

    def _extract_io_config(self, model_path: Any) -> dict | None:
        """Extract io_config from ONNX model for shape and dtype.

        Uses get_io_config() from winml.modelkit.onnx and transforms output
        to the format expected by format_data():
        ``{name: {"shape": [...], "dtype": np.dtype}}``.

        Args:
            model_path: Path to ONNX model

        Returns:
            io_config dict with input shapes and dtypes, or None on failure
        """
        try:
            from ..onnx import get_io_config

            raw = get_io_config(str(model_path))
            names = raw.get("input_names", [])
            shapes = raw.get("input_shapes", [])
            types = raw.get("input_types", [])

            io_config = {}
            for name, shape, dtype in zip(names, shapes, types, strict=False):
                io_config[name] = {"shape": shape, "dtype": dtype}

            return io_config if io_config else None

        except Exception as e:
            logger.debug("Failed to extract io_config: %s", e)
            return None

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return next sample as numpy dict, or None when exhausted.

        Uses format_data() to filter to valid ONNX inputs and cast to
        expected dtypes (e.g., int64 -> int32 for QNN compatibility).
        """
        if self._index >= len(self._dataset):
            return None

        sample = self._dataset[self._index]
        self._index += 1

        return format_data(
            sample,
            self._io_config,
            exclude_keys=self.exclude_keys,
        )

    def rewind(self) -> None:
        """Reset to beginning."""
        self._index = 0

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._dataset)


__all__ = [  # noqa: RUF022
    # Factory function
    "universal_calib_dataset",
    # Calibration reader
    "DatasetCalibrationReader",
    # Config
    "DEFAULT_OBJECT_DETECTION_SIZE",
    "DEFAULT_DEPTH_ESTIMATION_SIZE",
    # Dataset classes
    "BaseTaskDataset",
    "DepthEstimationDataset",
    "ImageDataset",
    "ImageSegmentationDataset",
    "KeypointDetectionDataset",
    "ObjectDetectionDataset",
    "RandomDataset",
    "TextDataset",
    # Utilities
    "format_data",
    "get_image_processor_config",
    # Mapping
    "TASK_DATASET_MAPPING",
]
