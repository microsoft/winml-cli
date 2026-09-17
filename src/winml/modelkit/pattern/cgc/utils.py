# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Static-value guards shared by explicit compatibility patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...onnx import ONNXDomain


if TYPE_CHECKING:
    import numpy as np

    from .. import PatternMatcher


def _static_tensor(name: str, matcher: PatternMatcher) -> np.ndarray | None:
    """Read only direct initializers or standard Constants, never input defaults."""
    producer = matcher.producer_lookup.get(name)
    if producer is None or producer[2] not in {"Initializer", "Constant"}:
        return None
    if producer[2] == "Constant":
        node = matcher.node_lookup[producer[0]]
        if node.domain not in {"", ONNXDomain.AI_ONNX.value}:
            return None
    return matcher.tensor_values.get(name)


def _depends_on_overridable_initializer(name: str, matcher: PatternMatcher) -> bool:
    """Reject static-shape proofs that might depend on overridable defaults."""
    overridable = {value.name for value in matcher.graph.input} & {
        value.name for value in matcher.graph.initializer
    }
    if not overridable:
        return False
    pending = [name]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if current in overridable:
            return True
        producer = matcher.producer_lookup.get(current)
        node = matcher.node_lookup.get(producer[0]) if producer else None
        if node is not None:
            pending.extend(node.input)
    return False
