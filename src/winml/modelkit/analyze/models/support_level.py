# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Support level classification enum."""

from enum import StrEnum


class SupportLevel(StrEnum):
    """Support level classification."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
