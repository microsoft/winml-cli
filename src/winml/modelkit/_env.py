# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Environment variable helpers used during early package initialization."""

from __future__ import annotations

import os


_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def env_flag_enabled(name: str) -> bool:
    """Return whether an environment variable is set to a truthy flag value."""
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES
