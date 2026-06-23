# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Utility modules for ModelExport."""

from .config_utils import merge_config
from .constants import normalize_ep_name


__all__ = [
    "merge_config",
    "normalize_ep_name",
]
