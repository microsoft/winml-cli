# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Random dataset for calibration when no specific dataset is provided."""

from __future__ import annotations

import numpy as np
import onnx

from ..onnx import load_onnx
from .registry import DataRegistry


@DataRegistry.register_dataset()
class RandomDataset:
    """Random dataset that generates synthetic data for calibration.

    Uses model input shape and type information to generate appropriate
    random data for quantization calibration when specific datasets
    aren't available or specified.
    """

    def __init__(self, config):
        """Initialize random dataset.

        Args:
            config: DataConfig object containing model_input and load_dataset_config with:
                - size (int, optional): Number of random samples to generate (default: 10)
        """
        if config is None:
            raise ValueError("DataConfig is required for RandomDataset")

        load_config = config.load_dataset_config
        model_path = config.model_input
        self.size = load_config.get("size", 10)

        if not model_path:
            raise ValueError("model_input must be specified in DataConfig")

        self.model_path = model_path
        self.samples = []
        self._load_model_and_generate()

    def _load_model_and_generate(self):
        """Load ONNX model and generate random data based on input specifications."""
        try:
            model = load_onnx(self.model_path, load_weights=False, validate=False)
            inputs = model.graph.input

            for _ in range(self.size):
                sample = {}
                for input_info in inputs:
                    name = input_info.name
                    shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
                    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[
                        input_info.type.tensor_type.elem_type
                    ]

                    # Handle dynamic dimensions (replace 0 with 1)
                    shape = [1 if dim == 0 else dim for dim in shape]

                    sample[name] = self._generate_data(shape, dtype)

                self.samples.append(sample)

        except Exception as e:
            raise RuntimeError(f"Failed to load model or generate random data: {e}")

    def _generate_data(self, shape, dtype):
        """Generate data for a given shape and dtype.

        Args:
            shape: Shape of the tensor to generate
            dtype: NumPy dtype of the tensor

        Returns:
            NumPy array with generated data
        """
        if np.issubdtype(dtype, np.floating):
            return np.random.rand(*shape).astype(dtype)
        if np.issubdtype(dtype, np.integer):
            return np.random.randint(0, 100, size=shape, dtype=dtype)
        if dtype == np.bool_:
            return np.random.choice([False, True], size=shape)
        return np.random.rand(*shape).astype(np.float32)

    def __len__(self):
        """Return dataset length."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get item by index.

        Args:
            idx: Index of the sample

        Returns:
            Dict containing input tensors
        """
        return self.samples[idx]
