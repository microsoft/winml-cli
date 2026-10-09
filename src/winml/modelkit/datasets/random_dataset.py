# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Random Dataset for Universal ONNX Model Testing.

Generates synthetic data from ONNX model I/O specs for quick QDQ workflow
testing. Auto-reads winml.io.inputs metadata for correct value ranges.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, cast

import torch
from datasets import Dataset


logger = logging.getLogger(__name__)


class RandomDataset:
    """Universal random dataset for ONNX model testing.

    Generates synthetic data by reading ONNX model input specs (shapes, dtypes)
    and winml.io.inputs metadata (value ranges) via get_io_config().
    Samples are generated on demand and replayed deterministically by index.
    Model-agnostic and requires no real data or downloads.

    Args:
        model_path: Path to ONNX model file
        max_samples: Maximum number of samples to generate (default: 100)
        seed: Random seed for reproducible data generation (default: 42)
        **kwargs: Additional keyword arguments
    """

    TASK_TYPE = "random"
    DEFAULT_DATASETS: ClassVar[list[str]] = ["random"]

    def __init__(
        self,
        model_path: str | None,
        max_samples: int = 100,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.max_samples = max_samples
        self.seed = seed

        # Cache io_config (loads ONNX once)
        from ..onnx import get_io_config

        self._io_config = (
            get_io_config(model_path) if model_path is not None else kwargs["io_config"]
        )

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
        return cast("dict[str, Any]", self.dataset[idx])

    @property
    def label_col(self) -> str:
        """Label column name (readonly). RandomDataset has no labels."""
        return "sample_id"

    def _generate_random_sample(self, index: int) -> dict[str, Any]:
        """Generate a single random sample as torch tensors.

        Uses cached InputTensorSpec list built from ONNX model I/O config.
        Each spec's to_tensor() handles value_range, dtype, and shape correctly.
        """
        generator = torch.Generator().manual_seed(self.seed + index)
        sample = {
            spec.name: spec.to_tensor(generator=generator)
            for spec in self._input_specs
            if spec.name
        }
        sample[self.label_col] = torch.tensor(index, dtype=torch.long)
        return sample

    def _load_dataset(self) -> Dataset:
        """Store sample indices and generate tensors only for requested rows."""
        columns = [spec.name for spec in self._input_specs if spec.name] + [self.label_col]

        def generate_batch(batch: dict[str, list[int]]) -> dict[str, list[Any]]:
            samples = [self._generate_random_sample(index) for index in batch[self.label_col]]
            return {name: [sample[name] for sample in samples] for name in columns}

        dataset = Dataset.from_dict({self.label_col: range(self.max_samples)})
        return dataset.with_transform(generate_batch)

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
