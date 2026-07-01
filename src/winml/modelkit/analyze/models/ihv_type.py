# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""IHV type enum."""

from enum import Enum


class IHVType(str, Enum):
    """IHV (Independent Hardware Vendor) type."""

    QC = "QC"
    INTEL = "Intel"
    AMD = "AMD"
    NVIDIA = "NVIDIA"
    MICROSOFT = "Microsoft"
    UNKNOWN = "Unknown"
