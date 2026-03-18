# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Utility functions for loading and configuring image processors.

Provides tools for loading HuggingFace preprocessor configurations
without instantiation, enabling ONNX-aware configuration overrides.
"""

from __future__ import annotations

import logging
from typing import Any

from transformers.image_processing_utils import ImageProcessingMixin


logger = logging.getLogger(__name__)


def get_image_processor_config(
    pretrained_model_name_or_path: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load image processor configuration from HuggingFace and merge with overrides.

    Uses ImageProcessingMixin.get_image_processor_dict() to load the raw
    preprocessor_config.json without instantiating a processor. This enables
    inspection and modification of configuration before processor creation.

    Args:
        pretrained_model_name_or_path: HuggingFace model name or local path
        **kwargs: Override values to merge into the loaded config

    Returns:
        Merged configuration dictionary with kwargs taking precedence

    Example:
        >>> config = get_image_processor_config(
        ...     "facebook/detr-resnet-50",
        ...     do_pad=False,
        ...     size={"height": 640, "width": 640},
        ... )
        >>> config["do_pad"]
        False
    """
    try:
        # get_image_processor_dict returns (config_dict, unused_kwargs)
        config_dict, _ = ImageProcessingMixin.get_image_processor_dict(
            pretrained_model_name_or_path,
        )
        logger.debug(
            "Loaded image processor config from %s: %s",
            pretrained_model_name_or_path,
            list(config_dict.keys()),
        )
    except Exception as e:
        logger.warning(
            "Failed to load image processor config from %s: %s. Using empty config.",
            pretrained_model_name_or_path,
            e,
        )
        config_dict = {}

    # Merge with kwargs (kwargs take precedence)
    merged_config = {**config_dict, **kwargs}

    if kwargs:
        logger.debug("Applied config overrides: %s", list(kwargs.keys()))

    return merged_config
