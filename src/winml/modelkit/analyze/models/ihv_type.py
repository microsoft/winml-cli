"""IHV type enum."""

from enum import Enum


class IHVType(str, Enum):
    """IHV (Independent Hardware Vendor) type."""

    QC = "QC"
    INTEL = "Intel"
    AMD = "AMD"
