# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML CLI serving package.

Public API:
    InferenceEngine  — core inference component (re-exported from inference/)
    create_app       — FastAPI application factory
"""

from ..inference import InferenceEngine
from .app import create_app


__all__ = ["InferenceEngine", "create_app"]
