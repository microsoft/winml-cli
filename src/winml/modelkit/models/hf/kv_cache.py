# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared KV cache utilities for ONNX export wrappers.

Provides ``CapturingStaticCache`` — a ``StaticCache`` subclass that captures
each layer's new-token KV from ``update()``, eliminating the scatter→gather
round-trip in the exported ONNX graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from transformers import StaticCache


if TYPE_CHECKING:
    import torch


class CapturingStaticCache(StaticCache):
    """StaticCache that captures each layer's new-token KV from ``update()``.

    Standard ``StaticCache.update()`` does ``index_copy_`` (ScatterElements in
    ONNX) to write the new KV into the full buffer, then returns the full
    buffer for attention.  The old approach then used ``gather``
    (GatherElements) to extract the same KV back — a pointless round-trip.

    This subclass intercepts ``update()`` to save the *incoming*
    ``key_states`` / ``value_states`` before they enter the buffer, so the
    wrapper can return them directly as ONNX outputs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Capture new-token KV, then delegate to parent ``index_copy_``."""
        self.captured[layer_idx] = (key_states, value_states)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)
