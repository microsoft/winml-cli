# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Custom ONNX graph fusions.

These fusions extend ORT's transformer optimizer framework with
additional pattern-matching fusions. They inherit from ORT's Fusion
base class and use the same graph traversal API (match_parent_path,
get_constant_value, etc.).

Each fusion is applied via FusionPipe after ORT's built-in fusions.
"""

from __future__ import annotations

from .fusion_rmsnorm import FusionRMSNorm


__all__ = ["FusionRMSNorm"]
