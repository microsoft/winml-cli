# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""IHV type enum."""

from enum import StrEnum


class IHVType(StrEnum):
    """IHV (Independent Hardware Vendor) type."""

    QC = "QC"
    INTEL = "Intel"
    AMD = "AMD"
