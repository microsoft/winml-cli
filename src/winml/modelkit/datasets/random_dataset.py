"""Random Dataset for Universal ONNX Model Testing.

Generates synthetic data from ONNX model I/O specs for quick QDQ workflow
testing. Auto-reads winml.io.inputs metadata for correct value ranges.
"""

from __future__ import annotations

import logging
import random
from typing import Any, ClassVar

import numpy as np
import torch
from datasets import Dataset


logger = logging.getLogger(__name__)


class RandomDataset:
    """Universal random dataset for ONNX model testing.

    Generates synthetic data by reading ONNX model input specs (shapes, dtypes)
    and winml.io.inputs metadata (value ranges) via get_io_config().
    Model-agnostic and requires no real data or downloads.

    Args:
        model_path: Path to ONNX model file
        max_samples: Maximum number of samples to generate (default: 100)
        seed: Random seed for reproducible data generation (default: 42)
        **kwargs: Additional keyword arguments (ignored)
    """

    TASK_TYPE = "random"
    DEFAULT_DATASETS: ClassVar[list[str]] = ["random"]

    def __init__(
        self,
        model_path: str,
        max_samples: int = 100,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.model_path = model_path
        self.max_samples = max_samples
        self.seed = seed

        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Cache io_config (loads ONNX once)
        from ..onnx import get_io_config

        self._io_config = get_io_config(model_path)

        # Build InputTensorSpec list for reuse across samples
        from ..onnx import InputTensorSpec

        value_ranges = self._io_config.get("value_ranges", {})
        self._input_specs: list[InputTensorSpec] = []
        for name, shape, dtype in zip(
            self._io_config["input_names"],
            self._io_config["input_shapes"],
            self._io_config["input_types"],
            strict=False,
        ):
            resolved_shape = tuple(d if d is not None and d > 0 else 1 for d in shape)
            dtype_str = str(dtype).replace("numpy.", "")
            self._input_specs.append(
                InputTensorSpec(
                    name=name,
                    dtype=dtype_str,
                    shape=resolved_shape,
                    value_range=value_ranges.get(name),
                )
            )

        # Generate synthetic dataset
        self.dataset = self._load_dataset()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single preprocessed sample."""
        return self.dataset[idx]

    @property
    def label_col(self) -> str:
        """Label column name (readonly). RandomDataset has no labels."""
        return "sample_id"

    def _generate_random_sample(self) -> dict[str, Any]:
        """Generate a single random sample as torch tensors.

        Uses cached InputTensorSpec list built from ONNX model I/O config.
        Each spec's to_tensor() handles value_range, dtype, and shape correctly.
        """
        return {spec.name: spec.to_tensor() for spec in self._input_specs}

    def _load_dataset(self) -> Dataset:
        """Generate synthetic dataset with random samples as tensors."""
        samples = []
        for i in range(self.max_samples):
            sample = self._generate_random_sample()
            sample[self.label_col] = torch.tensor(i, dtype=torch.long)
            samples.append(sample)

        dataset = Dataset.from_list(samples)
        dataset.set_format("torch")
        return dataset

    def get_data_config(self) -> dict[str, Any]:
        """Get Olive data configuration for random dataset."""
        return {
            "name": "random_data",
            "type": "RandomDataset",
            "params": {
                "model_path": self.model_path,
                "max_samples": self.max_samples,
                "seed": self.seed,
            },
            "load_dataset_config": {
                "data_name": "random",
                "split": "train",
                "subset": None,
            },
        }
