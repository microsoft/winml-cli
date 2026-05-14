# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Data loading and preprocessing components for WinML CLI."""

from . import (
    dummy_dataset,
    image_classification_dataset,
    random_dataset,
)
from .data_config import DataConfig
from .registry import DataRegistry


__all__ = ["DataConfig", "DataRegistry"]
