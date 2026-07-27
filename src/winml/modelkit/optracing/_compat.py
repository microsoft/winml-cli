# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Deprecated compatibility helpers for ``winml.modelkit.optracing``."""

from __future__ import annotations

import warnings


def warn_deprecated(symbol: str, replacement: str, *, stacklevel: int = 2) -> None:
    """Emit a caller-attributed deprecation warning for an old optracing symbol."""
    warnings.warn(
        f"winml.modelkit.optracing.{symbol} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )
