# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Base class for task-specific calibration datasets.

General-purpose base class with readonly properties and explicit configuration.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any


logger = logging.getLogger(__name__)


class BaseTaskDataset(ABC):
    """Base class for task-specific datasets with readonly properties.

    This class provides a general-purpose interface for datasets that can be
    used with any ML framework, not just Olive. Once instantiated, all
    properties are readonly to ensure dataset immutability and thread safety.

    Attributes:
        model_name: HuggingFace model identifier or local model path
        dataset_name: Dataset identifier (HF dataset or local path)
        data_split: Dataset split to use (e.g., 'train', 'validation', 'test')
    """

    # Override in subclasses if a default dataset is preferred
    DEFAULT_DATASET: str | None = None

    def __init__(
        self,
        model_name: str,
        dataset_name: str | None = None,
        max_samples: int | None = None,
        data_split: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize dataset with readonly properties.

        Args:
            model_name: HuggingFace model identifier or path
            dataset_name: Dataset name (uses DEFAULT_DATASET if None)
            max_samples: Maximum number of samples (None = use all)
            data_split: Dataset split (None = let subclass decide)
            **kwargs: Additional dataset-specific parameters
        """
        # Store as private attributes to enforce readonly access
        self._model_name = model_name
        self._dataset_name = dataset_name or self.DEFAULT_DATASET
        self._max_samples = max_samples
        self._data_split = data_split

        # Store additional kwargs for subclass use
        self._config = kwargs

        # Subclasses should populate these during initialization.
        # Typed as Any because each subclass uses a different dataset library
        # (HF datasets.Dataset, torch DataLoader, plain list[dict], ...).
        self._dataset: Any = None
        self._metadata: dict[str, Any] = {}  # Dataset metadata

        # Initialize subclass-specific data
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize dataset-specific data.

        Subclasses must implement this to load their data and set up
        any necessary preprocessing pipelines.
        """

    # Readonly properties
    @property
    def model_name(self) -> str:
        """Get the model name (readonly)."""
        return self._model_name

    @property
    def dataset_name(self) -> str | None:
        """Get the dataset name (readonly)."""
        return self._dataset_name

    @property
    def data_split(self) -> str | None:
        """Get the dataset split (readonly)."""
        return self._data_split


    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._dataset is None:
            return 0
        return len(self._dataset)

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single preprocessed sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing preprocessed sample data
        """

    @property
    @abstractmethod
    def label_col(self) -> str:
        """Get the label column name (readonly).

        Returns:
            Name of the label column used by this dataset
        """
