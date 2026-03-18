#!/usr/bin/env python3
"""Manual Model Input Generator - Pure PyTorch.

This module provides manual input tensor generation from specifications.
No external dependencies on Optimum or transformers.

For Optimum-based automatic input generation, use modelkit.export.io:
    - resolve_io_specs(model_type, task, hf_config)
    - generate_dummy_inputs(model, task)

Example:
    >>> from winml.modelkit.core.model_input_generator import generate_dummy_inputs_from_specs
    >>>
    >>> specs = {
    ...     "input_ids": {"dtype": "int", "shape": [1, 128], "range": [0, 30000]},
    ...     "attention_mask": {"dtype": "int", "shape": [1, 128], "range": [0, 1]},
    ... }
    >>> inputs = generate_dummy_inputs_from_specs(specs)
    >>> inputs["input_ids"].shape
    torch.Size([1, 128])
"""

from __future__ import annotations

import logging
from typing import Any

import torch


logger = logging.getLogger(__name__)


def generate_dummy_inputs_from_specs(
    input_specs: dict[str, dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Generate dummy inputs from manual specifications.

    This function creates PyTorch tensors based on explicit specifications,
    without requiring model loading or Optimum/transformers dependencies.

    Args:
        input_specs: Input specifications with format:
            {
                "input_name": {
                    "dtype": "int" | "float",  # Required
                    "shape": [1, 128],         # Required
                    "range": [0, 1000]         # Optional: [min, max]
                }
            }

    Returns:
        Dictionary mapping input names to generated tensors

    Raises:
        ValueError: If required fields are missing or invalid
        TypeError: If shape is not a list

    Example:
        >>> specs = {
        ...     "pixel_values": {
        ...         "dtype": "float",
        ...         "shape": [1, 3, 224, 224],
        ...         "range": [0.0, 1.0]
        ...     }
        ... }
        >>> inputs = generate_dummy_inputs_from_specs(specs)
        >>> inputs["pixel_values"].shape
        torch.Size([1, 3, 224, 224])
    """
    inputs = {}

    for name, spec in input_specs.items():
        # Validate required fields
        if "dtype" not in spec:
            raise ValueError(f"Missing 'dtype' in input spec for '{name}'")
        if "shape" not in spec:
            raise ValueError(f"Missing 'shape' in input spec for '{name}'")

        # Parse dtype
        dtype_str = spec["dtype"].lower()
        if dtype_str in ["int", "long", "int64"]:
            dtype = torch.long
        elif dtype_str in ["float", "float32"]:
            dtype = torch.float32
        else:
            raise ValueError(
                f"Unsupported dtype '{spec['dtype']}' for '{name}'. Use 'int' or 'float'"
            )

        # Parse shape
        shape = spec["shape"]
        if not isinstance(shape, list):
            raise TypeError(f"Shape must be a list for '{name}', got {type(shape)}")

        # Generate values
        if "range" in spec:
            if len(spec["range"]) != 2:
                raise ValueError(f"Range must have exactly 2 values [min, max] for '{name}'")
            min_val, max_val = spec["range"]

            if dtype == torch.long:
                inputs[name] = torch.randint(min_val, max_val + 1, shape, dtype=dtype)
            else:
                inputs[name] = torch.rand(shape, dtype=dtype) * (max_val - min_val) + min_val
        else:
            # Default ranges
            if dtype == torch.long:
                inputs[name] = torch.randint(0, 2, shape, dtype=dtype)  # Default: 0 or 1
            else:
                inputs[name] = torch.rand(shape, dtype=dtype)  # Default: [0, 1)

        logger.info(
            "Generated '%s': shape=%s, dtype=%s", name, list(inputs[name].shape), inputs[name].dtype
        )

    return inputs
