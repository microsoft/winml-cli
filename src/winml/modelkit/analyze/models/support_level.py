"""Support level classification enum."""

from enum import Enum


class SupportLevel(str, Enum):
    """Support level classification."""

    WHITE = "white"
    GRAY = "gray"
    BLACK = "black"
    UNKNOWN = "unknown"
