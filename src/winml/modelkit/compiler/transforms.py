# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""EP-specific graph transform registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    import onnx

    from ..utils.constants import EPAlias


class TransformError(Exception):
    """Raised when a graph transform fails."""


@runtime_checkable
class GraphTransform(Protocol):
    """Protocol for EP-specific graph transforms.

    Error handling contract:
    - transform() raises TransformError on failure (no partial transforms)
    - Must return valid ONNX ModelProto
    - Should be idempotent (applying twice = applying once)
    """

    def applies_to(self, ep: EPAlias) -> bool:
        """Check if this transform applies to the given EP."""
        ...

    def transform(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Transform model. Raises TransformError on failure."""
        ...


_TRANSFORMS: list[GraphTransform] = []


def register_transform(transform: GraphTransform) -> None:
    """Register an EP-specific graph transform."""
    _TRANSFORMS.append(transform)


def get_transforms_for_ep(ep: EPAlias) -> list[GraphTransform]:
    """Return all registered transforms that apply to the given EP."""
    return [t for t in _TRANSFORMS if t.applies_to(ep)]


def clear_transforms() -> None:
    """Clear registry (for testing)."""
    _TRANSFORMS.clear()
