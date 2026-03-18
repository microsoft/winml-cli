"""Data formatting utilities for calibration pipelines.

Provides format_data() for converting dataset outputs to numpy arrays
matching ONNX model's expected input names and dtypes. Filters to valid
inputs and casts dtypes (e.g., int64 -> int32 for QNN compatibility).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


def format_data(
    data: dict[str, Any],
    io_config: dict[str, dict] | None = None,
    *,
    exclude_keys: frozenset[str] | None = None,
) -> dict[str, np.ndarray]:
    """Format dataset sample to match ONNX model's expected inputs.

    Converts torch tensors to numpy arrays, filters to valid input names,
    and casts to expected dtypes (e.g., int64 -> int32 for QNN).

    Args:
        data: Sample dict from a dataset. Values can be torch tensors,
            numpy arrays, or Python scalars.
        io_config: ONNX input config from model parsing.
            Format: {"input_ids": {"shape": [1, 512], "dtype": np.dtype("int32")}}
            When provided, filters to known inputs and casts to expected dtypes.
        exclude_keys: Keys to exclude (e.g., {"label", "labels"}).

    Returns:
        Dict mapping input names to numpy arrays with correct dtypes.

    Example:
        >>> io_config = {
        ...     "input_ids": {"shape": [1, 512], "dtype": np.dtype("int32")},
        ...     "attention_mask": {"shape": [1, 512], "dtype": np.dtype("int32")},
        ... }
        >>> sample = {"input_ids": torch.ones(1, 512, dtype=torch.int64), "label": 0}
        >>> result = format_data(sample, io_config, exclude_keys=frozenset({"label"}))
        >>> result["input_ids"].dtype  # int32 (cast from int64)
    """
    exclude = exclude_keys or frozenset()
    valid_names = set(io_config.keys()) if io_config else None

    result = {}
    for key, value in data.items():
        if key in exclude:
            continue
        if valid_names is not None and key not in valid_names:
            continue

        # Convert to numpy
        if hasattr(value, "cpu"):
            arr = value.cpu().numpy()
        elif hasattr(value, "numpy"):
            arr = value.numpy()
        else:
            arr = np.asarray(value)

        # Cast to ONNX-expected dtype
        if io_config and key in io_config:
            expected_dtype = io_config[key].get("dtype")
            if expected_dtype is not None and arr.dtype != expected_dtype:
                arr = np.ascontiguousarray(arr, dtype=expected_dtype)

        result[key] = arr

    return result
