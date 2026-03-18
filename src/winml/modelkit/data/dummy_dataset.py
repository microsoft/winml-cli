"""Dummy dataset for testing with all-ones data."""

from __future__ import annotations

import numpy as np

from .random_dataset import RandomDataset
from .registry import DataRegistry


@DataRegistry.register_dataset()
class DummyDataset(RandomDataset):
    """Dummy dataset that generates all-ones data for testing.

    Inherits from RandomDataset but overrides data generation to use
    all ones instead of random values. Useful for deterministic testing
    and debugging.
    """

    def _generate_data(self, shape, dtype):
        """Generate all-ones data for a given shape and dtype.

        Args:
            shape: Shape of the tensor to generate
            dtype: NumPy dtype of the tensor

        Returns:
            NumPy array filled with ones (or True for boolean)
        """
        if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.integer):
            return np.ones(shape, dtype=dtype)
        if dtype == np.bool_:
            return np.ones(shape, dtype=np.bool_)
        return np.ones(shape).astype(np.float32)
