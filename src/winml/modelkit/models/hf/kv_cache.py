# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared KV cache utilities for ONNX export wrappers.

Provides ``CapturingStaticCache`` — a ``StaticCache`` subclass that captures
each layer's new-token KV from ``update()``, eliminating the scatter→gather
round-trip in the exported ONNX graph.

Also provides ``PastKeyValueInputGenerator`` — a reusable ``DummyInputGenerator``
for static KV cache inputs (``past_{i}_key``, ``past_{i}_value``), shared by
T5, Qwen, and future models with static KV cache export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optimum.utils.input_generators import DummyInputGenerator
from transformers import StaticCache


if TYPE_CHECKING:
    import torch
    from optimum.utils import NormalizedConfig


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


class PastKeyValueInputGenerator(DummyInputGenerator):
    """Generates ``past_{i}_key`` / ``past_{i}_value`` tensors for static KV cache.

    Reads ``num_layers``, ``num_attention_heads``, ``head_dim``, and
    ``max_cache_len`` from the ``NormalizedConfig``.  Each model's
    ``NORMALIZED_CONFIG_CLASS`` maps these to the appropriate HF config fields
    (e.g. T5: ``head_dim="d_kv"``, ``max_cache_len="n_positions"``).
    """

    SUPPORTED_INPUT_NAMES = ()  # dynamic — built in __init__

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        self.batch_size = batch_size
        self.num_layers: int = normalized_config.num_layers
        self.num_heads: int = normalized_config.num_attention_heads
        self.head_dim: int = normalized_config.head_dim
        self.max_cache_len: int = normalized_config.max_cache_len
        self.SUPPORTED_INPUT_NAMES = tuple(
            name for i in range(self.num_layers) for name in (f"past_{i}_key", f"past_{i}_value")
        )

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        """Return a random float tensor of shape ``[batch, heads, max_cache_len, head_dim]``."""
        return self.random_float_tensor(
            (self.batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            framework=framework,
            dtype=float_dtype,
        )
