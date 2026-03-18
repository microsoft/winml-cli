# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Central registry for dataset class management."""

from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


class DataRegistry:
    """Central registry for dataset class management.

    Enables dynamic dataset instantiation from configuration using a
    decorator-based registration pattern.
    """

    _datasets: dict[str, type] = {}

    @classmethod
    def register_dataset(cls, name: str | None = None) -> Callable[[T], T]:
        """Decorator to register a dataset class.

        Args:
            name: Optional name to register the dataset under. If not provided,
                uses the class's __name__ attribute.

        Returns:
            Decorator function that registers the class and returns it unchanged

        Example:
            @DataRegistry.register_dataset()
            class ImageClassificationDataset(Dataset):
                pass

            @DataRegistry.register_dataset("custom_name")
            class MyDataset(Dataset):
                pass
        """

        def decorator(dataset_class: T) -> T:
            dataset_name = name or dataset_class.__name__
            cls._datasets[dataset_name] = dataset_class
            return dataset_class

        return decorator

    @classmethod
    def get_component(cls, name: str) -> type:
        """Retrieve registered class or function by name.

        Args:
            name: Name of the registered dataset class

        Returns:
            The registered dataset class

        Raises:
            ValueError: If the requested dataset name is not registered

        Example:
            dataset_class = DataRegistry.get_component("ImageClassificationDataset")
            dataset = dataset_class(config)
        """
        if name not in cls._datasets:
            available = ", ".join(cls._datasets.keys())
            raise ValueError(
                f"Unknown dataset: '{name}'. "
                f"Available datasets: {available if available else 'none'}"
            )
        return cls._datasets[name]
