# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML KV cache classes for ONNX export and inference.

Hierarchy::

    StaticCache (HF transformers)
      └─ WinMLCache                        — common interface
           ├─ WinMLStaticCache             — ScatterElements (index_copy_), T5/Qwen
           └─ WinMLSlidingWindowCache      — Slice+Concat (FIFO), Mu2

Cache type compatibility:

- **WinMLStaticCache**: ``index_copy_`` at ``cache_position`` keeps
  ``buffer_position == sequence_position``.  Cannot evict — ``max_cache_len``
  must be ≥ total generated tokens.

- **WinMLSlidingWindowCache**: Slice+Concat eviction; works for RoPE models
  (Mu2, Qwen, Llama) where position is baked into K when K is computed, and
  for learned relative position bias (T5) as long as the wrapper feeds
  ``cache_position`` as the query's *buffer index* (see
  ``get_query_cache_position``).  The invariant ``cache_position = buffer_idx
  of query`` makes ``j - cache_position`` the correct relative distance for
  both cache types, so no per-model compute_bias patch is required.

Common interface (called by ``WinMLEncoderDecoderModel.forward``):

- ``position_input_name``: ONNX input name (``"cache_position"`` or ``"position_id"``)
- ``build_decoder_mask(max_len)``: 2D attention mask for current step
- ``get_query_cache_position(max_len)``: buffer indices of query tokens
  (used by HF's ``create_causal_mask`` and by T5's ``compute_bias``)
- ``update_all_layers(outputs)``: write present KV from ONNX output, advance step
- ``reset()``: zero out for new generation
- ``create(config, kv_shape, dtype)``: factory from ONNX metadata

Also provides ``PastKeyValueInputGenerator`` — a reusable ``DummyInputGenerator``
for static KV cache inputs (``past_{i}_key``, ``past_{i}_value``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from optimum.utils.input_generators import DummyInputGenerator
from transformers import StaticCache


if TYPE_CHECKING:
    import torch
    from optimum.utils import NormalizedConfig
    from transformers import PretrainedConfig


# =============================================================================
# WinMLCache — common interface
# =============================================================================


class WinMLCache(StaticCache, ABC):
    """Abstract base for WinML KV caches (export + inference).

    Subclasses set ``position_input_name``, implement ``build_decoder_mask``,
    and override ``update()`` for cache-specific write logic.

    ``step`` tracks the absolute generation position
    (used for RoPE and mask construction).
    ``num_layers`` is set from ``config.num_hidden_layers``.
    """

    #: ONNX input name for the position tensor (subclasses override).
    #: Empty string is a sentinel — concrete subclasses must set a real value.
    position_input_name: ClassVar[str] = ""

    def __init__(self, config: PretrainedConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.step: int = 0
        # StaticCache.__init__ already built ``self.layers`` using
        # ``config.get_text_config(decoder=True).num_hidden_layers``, which is
        # the decoder's layer count (e.g., 6 for distilbart-cnn-12-6).
        # Reading the outer ``config.num_hidden_layers`` would instead give the
        # encoder's count (12 for distilbart), so we take len(self.layers) --
        # correct for both symmetric and asymmetric encoder-decoder models.
        self.num_layers: int = len(self.layers)
        #: New-token KV captured during ``update()``, keyed by layer index.
        #: Export wrappers read ``captured[i]`` to build ONNX present outputs.
        self.captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    # ----- Interface for WinMLEncoderDecoderModel.forward -----

    @abstractmethod
    def build_decoder_mask(self, max_len: int, num_new_tokens: int = 1) -> torch.Tensor:
        """Build the decoder attention mask for the current step.

        Args:
            max_len: Total cache buffer length.
            num_new_tokens: Number of new tokens being added (1 for gen,
                chunk_len for prefill).
        """

    @abstractmethod
    def get_query_cache_position(self, max_len: int, num_new_tokens: int = 1) -> torch.Tensor:
        """Buffer indices of the query tokens for HF's ``cache_position`` input.

        HF's ``create_causal_mask`` uses ``cache_position`` as the query's
        *buffer index* (``kv_idx <= q_idx``).  For static cache the buffer index
        equals the sequence position (``step``); for sliding window it is the
        rightmost slot(s) because new tokens are written at the right end.

        Returns:
            ``[num_new_tokens]`` int64 tensor of buffer positions for the new
            tokens being processed this step.
        """

    @abstractmethod
    def prepare_prefill_chunk(
        self,
        chunk_ids: torch.Tensor,
        start: int,
        prefill_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Pad tokens and build position IDs for one prefill chunk.

        Args:
            chunk_ids: ``[1, chunk_len]`` — real tokens for this chunk.
            start: Absolute position of the first real token.
            prefill_seq_len: ONNX model's fixed prefill input length.

        Returns:
            padded_ids: ``[1, prefill_seq_len]`` — padded input token IDs.
            position_ids: ``[1, prefill_seq_len]`` — position encoding input.
            pad_len: Number of leading padding positions (0 for right-pad).
        """

    def update_all_layers(self, outputs: dict[str, Any]) -> None:
        """Write present KV for all layers via ``update()`` and advance step.

        Step advances by N where N is the seq_len of the present KV tensors
        (1 for gen, chunk_len for prefill).
        """
        import torch

        n = 0
        for i in range(self.num_layers):
            k = outputs[f"present_{i}_key"]
            v = outputs[f"present_{i}_value"]
            k = k if isinstance(k, torch.Tensor) else torch.tensor(k)
            v = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            n = k.size(2)
            ck = {"cache_position": torch.arange(self.step, self.step + n, dtype=torch.int64)}
            self.update(k, v, i, cache_kwargs=ck)
        self.step += n

    def reset(self) -> None:
        """Zero out all layers and reset step (start of new generation)."""
        self.step = 0
        self.captured.clear()
        for i in range(self.num_layers):
            self.layers[i].keys.zero_()
            self.layers[i].values.zero_()

    @classmethod
    def create(
        cls, config: PretrainedConfig, kv_shape: list[int], dtype: torch.dtype
    ) -> WinMLCache:
        """Create and initialize a cache from ONNX KV shape metadata.

        Args:
            config: HF model config (must have ``num_hidden_layers``).
            kv_shape: ``[batch, heads, max_cache_len, head_dim]`` from ONNX.
            dtype: KV dtype (fp32 or fp16).
        """
        import torch

        cache = cls(config, max_cache_len=kv_shape[2])
        cache.early_initialization(
            batch_size=1,
            num_heads=kv_shape[1],
            head_dim=kv_shape[3],
            dtype=dtype,
            device=torch.device("cpu"),
        )
        return cache


# =============================================================================
# WinMLStaticCache — ScatterElements (index_copy_)
# =============================================================================


class WinMLStaticCache(WinMLCache):
    """Cache using ``index_copy_`` at ``cache_position`` (ScatterElements).

    **Export**: intercepts ``update()`` to capture incoming KV for ONNX output.
    **Inference**: ``update_all_layers`` writes new-token KV at the current step.
    Mask is left-aligned: ``[1, 1, ..., 1, 0, 0, ..., 0]``.
    """

    position_input_name: str = "cache_position"

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Capture new-token KV, then write via multi-dim ``index_put_`` (ScatterND).

        Using full (b, h, pos) coord tuples forces the ONNX exporter to emit
        ScatterND rather than ScatterElements (which ``index_copy_`` along a
        single axis would produce).
        """
        import torch

        self.captured[layer_idx] = (key_states, value_states)
        cache_position = cache_kwargs["cache_position"]

        k_out = self.layers[layer_idx].keys
        v_out = self.layers[layer_idx].values

        bsz, n_heads, n_new, _ = key_states.shape
        bi = torch.arange(bsz, device=k_out.device).view(bsz, 1, 1).expand(bsz, n_heads, n_new)
        hi = (
            torch.arange(n_heads, device=k_out.device)
            .view(1, n_heads, 1)
            .expand(bsz, n_heads, n_new)
        )
        pi = cache_position.view(1, 1, n_new).expand(bsz, n_heads, n_new)
        k_out.index_put_((bi, hi, pi), key_states)
        v_out.index_put_((bi, hi, pi), value_states)

        return k_out, v_out

    def build_decoder_mask(self, max_len: int, num_new_tokens: int = 1) -> torch.Tensor:
        """Left-aligned: first ``step + num_new_tokens`` positions are 1."""
        import torch

        mask = torch.zeros(1, max_len, dtype=torch.int64)
        mask[0, : self.step + num_new_tokens] = 1
        return mask

    def get_query_cache_position(self, max_len: int, num_new_tokens: int = 1) -> torch.Tensor:
        """Buffer index == sequence position for static cache: ``[step..step+N)``."""
        import torch

        return torch.arange(self.step, self.step + num_new_tokens, dtype=torch.int64)

    def prepare_prefill_chunk(
        self,
        chunk_ids: torch.Tensor,
        start: int,
        prefill_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Right-pad: real tokens at START, padding at end."""
        import torch

        chunk_len = chunk_ids.shape[1]
        padded_ids = torch.zeros(1, prefill_seq_len, dtype=chunk_ids.dtype)
        padded_ids[0, :chunk_len] = chunk_ids[0]

        position_ids = torch.arange(start, start + prefill_seq_len, dtype=torch.int64).unsqueeze(0)

        return padded_ids, position_ids, 0


# =============================================================================
# WinMLSlidingWindowCache — Slice + Concat (FIFO)
# =============================================================================


class WinMLSlidingWindowCache(WinMLCache):
    """FIFO cache: evict oldest, append new at end (Slice+Concat).

    **Export**: ``update()`` does Slice+Concat on the buffer and captures
    the new-token KV (same as ``WinMLStaticCache.captured``).  Present KV
    output is the new token only ``[batch, heads, 1, head_dim]``.
    **Inference**: ``update_all_layers`` does Slice+Concat from present KV.
    Mask is right-aligned: ``[0, 0, ..., 0, 1, 1, ..., 1]``.
    """

    position_input_name: str = "position_id"

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drop N oldest, append N new KV at end (N = key_states.size(2)).

        Works for both single-token gen (N=1) and multi-token prefill (N>1).
        """
        import torch

        self.captured[layer_idx] = (key_states, value_states)

        n = key_states.size(2)
        old_k = self.layers[layer_idx].keys[:, :, n:, :]
        new_k = torch.cat([old_k, key_states], dim=2)
        self.layers[layer_idx].keys = new_k

        old_v = self.layers[layer_idx].values[:, :, n:, :]
        new_v = torch.cat([old_v, value_states], dim=2)
        self.layers[layer_idx].values = new_v

        return new_k, new_v

    def build_decoder_mask(self, max_len: int, num_new_tokens: int = 1) -> torch.Tensor:
        """Right-aligned: rightmost ``step + num_new_tokens`` positions are 1."""
        import torch

        filled = min(self.step + num_new_tokens, max_len)
        mask = torch.zeros(1, max_len, dtype=torch.int64)
        mask[0, max(0, max_len - filled) :] = 1
        return mask

    def get_query_cache_position(self, max_len: int, num_new_tokens: int = 1) -> torch.Tensor:
        """Query tokens sit at the rightmost ``num_new_tokens`` buffer slots.

        Because new tokens are always written at the right end of the buffer
        (Slice+Concat), the query's buffer index is ``[max_len-N..max_len)`` —
        independent of the absolute sequence position.  HF's causal mask
        then allows attention to every prior buffer slot (``j <= max_len-1``),
        and the 2D ``build_decoder_mask`` selects the filled region within that.
        """
        import torch

        return torch.arange(max_len - num_new_tokens, max_len, dtype=torch.int64)

    def prepare_prefill_chunk(
        self,
        chunk_ids: torch.Tensor,
        start: int,
        prefill_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Left-pad: padding at start, real tokens at END."""
        import torch

        chunk_len = chunk_ids.shape[1]
        pad_len = prefill_seq_len - chunk_len

        padded_ids = torch.zeros(1, prefill_seq_len, dtype=chunk_ids.dtype)
        padded_ids[0, pad_len:] = chunk_ids[0]

        # Padding positions get 0 — RoPE computes embeddings for position 0 on these,
        # but the attention mask at build_decoder_mask masks them out before softmax,
        # so the RoPE artifacts don't influence outputs.
        position_ids = torch.zeros(1, prefill_seq_len, dtype=torch.int64)
        position_ids[0, pad_len:] = torch.arange(start, start + chunk_len, dtype=torch.int64)

        return padded_ids, position_ids, pad_len

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Filled positions: ``min(step, max_cache_len)``."""
        max_len = self.layers[layer_idx].keys.shape[2]
        return min(self.step, max_len)


# =============================================================================
# PastKeyValueInputGenerator
# =============================================================================


class PastKeyValueInputGenerator(DummyInputGenerator):
    """Generates ``past_{i}_key`` / ``past_{i}_value`` tensors for static KV cache.

    Reads ``num_layers``, ``num_attention_heads``, ``head_dim``, and
    ``max_cache_len`` from the ``NormalizedConfig``.
    """

    SUPPORTED_INPUT_NAMES = ()  # dynamic — built in __init__

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = 1,
        max_cache_len: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.batch_size = batch_size
        self.num_layers: int = normalized_config.num_layers
        self.num_heads: int = normalized_config.num_attention_heads
        # head_dim: prefer explicit config attr (T5's d_kv, Qwen's head_dim),
        # otherwise derive from hidden_size / num_attention_heads (Marian, BART).
        try:
            self.head_dim: int = normalized_config.head_dim
        except AttributeError:
            self.head_dim = normalized_config.hidden_size // normalized_config.num_attention_heads
        self.max_cache_len: int = max_cache_len or normalized_config.max_cache_len
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
