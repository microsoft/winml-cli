"""Label utilities for dataset-specific label mappings.

Handles label alignment for datasets like ImageNet that need synset ID to class index mapping.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_imagenet_label_map() -> dict[str, int]:
    """Get ImageNet label mapping from synset IDs to class indices.

    Returns mapping like: {"n01440764": 0, "n01443537": 1, ...}
    This ensures correct label alignment for ImageNet models.

    Uses in-memory caching only - no local file persistence.
    """
    # Download from PyTorch vision repo and cache in memory
    logger.info("Fetching ImageNet class index...")
    import requests

    url = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/imagenet_class_index.json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content = response.json()
        logger.debug("Successfully fetched ImageNet class index")
    except Exception as e:
        logger.error(f"Failed to fetch ImageNet labels: {e}")
        raise

    # Convert {0: ["n01440764", "tench"], ...} to {synset: index}
    return {v[0]: int(k) for k, v in content.items()}


def get_label_mapping(dataset_name: str) -> dict[str, Any] | None:
    """Get label mapping for a specific dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Label mapping dict or None if no mapping needed
    """
    # ImageNet variants need label alignment
    imagenet_variants = [
        "imagenet-1k",
        "imagenet",
        "imagenet2012",
        "timm/imagenet-1k",
        "ILSVRC/imagenet-1k",
    ]

    if any(variant in dataset_name.lower() for variant in imagenet_variants):
        return get_imagenet_label_map()

    # CIFAR and other datasets typically don't need alignment
    return None


def should_align_labels(dataset_name: str) -> bool:
    """Check if a dataset needs label alignment.

    Args:
        dataset_name: Name of the dataset

    Returns:
        True if labels need alignment
    """
    return get_label_mapping(dataset_name) is not None
