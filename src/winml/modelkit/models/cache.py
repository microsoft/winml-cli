# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Static append-only KV cache for encoder-decoder ONNX export and inference.

Provides:
- StaticWriteLayer: Single layer with fixed-size buffer, scatter-write at
  cache_position. No shifting — KV_index always equals sequence_position.
- StaticWriteCache: Multi-layer cache using StaticWriteLayer.
- StaticWriteEncoderDecoderCache: Wraps StaticWriteCache (self-attn) + empty
  DynamicCache (cross-attn). Forces is_updated=False so encoder_hidden_states
  is never constant-folded.

Design:
    The buffer is [batch, heads, max_decode_length, d_kv], zero-initialized.
    Each step writes new KV at the position given by cache_position (via
    torch.scatter, traceable). KV_index = sequence_position always holds,
    so T5's relative position bias (which does memory_position = arange(key_length))
    computes correct distances.

    cache_position and decoder_attention_mask are explicit ONNX inputs,
    overriding T5Stack's auto-generated values.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from transformers.cache_utils import (
    Cache,
    CacheLayerMixin,
    DynamicCache,
    EncoderDecoderCache,
)


class StaticWriteLayer(CacheLayerMixin):
    """Single layer static KV buffer with scatter-write.

    update() writes new KV at cache_position via torch.scatter.
    Returns the full buffer for attention. No in-place mutation on the
    input tensor (scatter returns a new tensor).
    """

    def __init__(self, keys: Tensor, values: Tensor) -> None:
        super().__init__()
        self.keys = keys  # [batch, heads, max_decode_len, d_kv]
        self.values = values
        self.is_initialized = True
        self.max_decode_len = keys.shape[2]

    @classmethod
    def from_zeros(
        cls,
        batch: int,
        num_heads: int,
        max_decode_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> StaticWriteLayer:
        """Create zero-initialized layer."""
        keys = torch.zeros(batch, num_heads, max_decode_len, head_dim, dtype=dtype, device=device)
        values = torch.zeros(batch, num_heads, max_decode_len, head_dim, dtype=dtype, device=device)
        return cls(keys, values)

    def lazy_initialization(self, key_states: Tensor) -> None:
        """Not used — StaticWriteLayer is always eagerly initialized."""

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Write new KV at cache_position via scatter. Returns full buffer.

        Args:
            key_states: [batch, heads, 1, d_kv]
            value_states: [batch, heads, 1, d_kv]
            cache_kwargs: Must contain "cache_position" tensor [1] with
                the write index. Passed by T5Attention automatically.

        Returns:
            (keys, values) each [batch, heads, max_decode_len, d_kv].
        """
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
        if cache_position is not None:
            # scatter: write key_states at cache_position along dim=2
            idx = cache_position.view(1, 1, -1, 1).expand_as(key_states)
            self.keys = self.keys.scatter(2, idx, key_states)
            self.values = self.values.scatter(2, idx, value_states)
        return self.keys, self.values

    def get_seq_length(self) -> int:
        """Return max_decode_len (physical buffer size)."""
        return self.max_decode_len

    def get_max_cache_shape(self) -> int:
        """Return max_decode_len."""
        return self.max_decode_len

    def get_mask_sizes(self, cache_position: Tensor) -> tuple[int, int]:
        """Return (kv_length, kv_offset) for mask generation."""
        return self.max_decode_len, 0


class StaticWriteCache(Cache):
    """Fixed-size append-only KV cache for self-attention.

    Each layer is a StaticWriteLayer with shape [batch, heads, max_decode_len, d_kv].
    """

    def __init__(self, layers: list[StaticWriteLayer]) -> None:
        super().__init__(layers=layers)

    @classmethod
    def from_kv_pairs(cls, kv_pairs: list[tuple[Tensor, Tensor]]) -> StaticWriteCache:
        """Create from existing KV tensor pairs (for export wrapper)."""
        return cls([StaticWriteLayer(k, v) for k, v in kv_pairs])

    @classmethod
    def from_zeros(
        cls,
        num_layers: int,
        batch: int,
        num_heads: int,
        max_decode_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> StaticWriteCache:
        """Create zero-initialized cache for inference start."""
        return cls(
            [
                StaticWriteLayer.from_zeros(
                    batch, num_heads, max_decode_len, head_dim, dtype, device
                )
                for _ in range(num_layers)
            ]
        )


class StaticWriteEncoderDecoderCache(EncoderDecoderCache):
    """EncoderDecoderCache with StaticWriteCache for self-attention.

    Cross-attention cache is left empty (DynamicCache) so that T5 always
    recomputes cross-attention K/V from encoder_hidden_states (prevents
    constant-folding during ONNX export).
    """

    def __init__(
        self,
        self_attention_cache: StaticWriteCache,
        cross_attention_cache: DynamicCache,
        fill_count: int = 0,
    ) -> None:
        super().__init__(self_attention_cache, cross_attention_cache)
        # Force cross-attention recomputation every step.
        for layer_idx in self.is_updated:
            self.is_updated[layer_idx] = False
        self.fill_count = fill_count

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return fill_count (logical sequence length, not buffer size).

        This controls cache_position auto-generation in T5Stack. When we
        pass cache_position explicitly, this value is unused. But returning
        fill_count is correct for any code that queries logical length.
        """
        return self.fill_count

    @classmethod
    def from_zeros(
        cls,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_decode_len: int,
        batch: int = 1,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> StaticWriteEncoderDecoderCache:
        """Create zero-initialized cache for pipeline inference start."""
        self_attn = StaticWriteCache.from_zeros(
            num_layers,
            batch,
            num_heads,
            max_decode_len,
            head_dim,
            dtype,
            device,
        )
        cross_attn = DynamicCache()
        return cls(self_attn, cross_attn, fill_count=0)
