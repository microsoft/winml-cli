# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML ModelKit serving package.

Public API:
    InferenceEngine  — core inference component
    create_app       — FastAPI application factory
"""

from .app import create_app
from .engine import InferenceEngine


__all__ = ["InferenceEngine", "create_app"]
