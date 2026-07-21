# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Base class for decoder export wrappers with a traceable KV cache.

HuggingFace decoders represent the KV cache as a Python ``Cache`` object,
which has no shape at the ONNX graph boundary — a caller cannot supply past
KV through it, and HF cannot return new-token KV out of it as named tensors.
``WinMLDecoderWrapper`` adapts an HF decoder so its KV cache *is* exportable:
past KV become flat per-layer ONNX inputs, and new-token KV become flat
per-layer ONNX outputs.  Transformer math inside HF runs unmodified.

What ``forward`` does during ``torch.onnx.export``:

1. Build a ``WinMLCache`` whose per-layer storage is **aliased** to the ONNX
   past-KV input tensors.  HF then reads/writes the cache as usual, and every
   such op surfaces as a graph edge on the named graph input.
2. Hand the cache to the unmodified HF decoder via its ``past_key_values=``
   argument.
3. Pack ``logits`` plus the per-layer new-token K/V (captured by
   ``WinMLStaticCache.update``) into the ONNX output tuple.

The ONNX input name ordering and decoder layer count are read from the
family's ``OnnxConfig`` at construction time, so per-architecture shape
knowledge lives in one place.  A per-family subclass only has to:

* set ``_HF_MODEL_CLS`` and ``_IO_CONFIG_CLS`` class constants,
* override ``_invoke_hf`` with the HF decoder call,
* (optional) override ``_make_cache`` — call ``super()`` first, then attach
  any trace-time side channel to the returned cache (e.g.,
  ``cache.set_trace_position(...)``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from transformers import PreTrainedModel

from ..winml.kv_cache import WinMLCache, WinMLStaticCache


if TYPE_CHECKING:
    from transformers.cache_utils import CacheLayerMixin


class WinMLDecoderWrapper(nn.Module, ABC):
    """Abstract base class for static-KV-cache decoder export wrappers.

    Concrete subclasses must:
    - Set ``_HF_MODEL_CLS`` and ``_IO_CONFIG_CLS`` class attributes.
    - Implement ``_invoke_hf`` (the family-specific HF decoder call).

    Instance attributes (set by ``from_pretrained``):
        model         — the HF model, called by ``_invoke_hf``
        config        — full HF ``PretrainedConfig``
        onnx_config   — instance of ``_IO_CONFIG_CLS`` — source of truth for
                        ONNX input name ordering and decoder layer count
        num_layers    — derived from ``onnx_config._normalized_config.num_layers``
    """

    _HF_MODEL_CLS: ClassVar[type[PreTrainedModel]]  # set per-subclass to a concrete HF model class
    _IO_CONFIG_CLS: ClassVar[type]
    _TASK: ClassVar[str] = "text2text-generation"
    _CACHE_CLS: ClassVar[type[WinMLCache]] = WinMLStaticCache

    # ---- Instance attrs ----
    model: nn.Module
    config: Any
    onnx_config: OnnxConfig
    num_layers: int

    # =====================================================================
    # Factory
    # =====================================================================

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> WinMLDecoderWrapper:
        """Load the HF model and wrap it for export."""
        full = cls._HF_MODEL_CLS.from_pretrained(model_name_or_path, **kwargs)
        self = cls()
        self.model = full
        self.config = full.config
        self.onnx_config = cls._IO_CONFIG_CLS(full.config, task=cls._TASK)
        self.num_layers = self.onnx_config._normalized_config.num_layers
        self.eval()
        return self

    # =====================================================================
    # torch.onnx.export entry point
    # =====================================================================

    def get_export_args(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Order dict inputs positionally to match the IOConfig's input order."""
        return tuple(inputs.values())

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Execute the three-step adapter.  Subclasses override ``_invoke_hf``."""
        inputs = dict(zip(self.onnx_config.inputs.keys(), args, strict=True))

        # 1. Create cache aliased to ONNX past-KV inputs.
        cache = self._make_cache(inputs)
        # 2. Invoke HF.
        logits = self._invoke_hf(cache, inputs)
        # 3. Pack captured new-token KV as ONNX outputs.
        result: list[torch.Tensor] = [logits]
        for i in range(self.num_layers):
            k, v = cache.captured[i]
            result.extend([k, v])
        return tuple(result)

    # =====================================================================
    # Shared helpers
    # =====================================================================

    def _make_cache(self, inputs: dict[str, torch.Tensor]) -> WinMLCache:
        """Alias the flat past-KV inputs onto a fresh ``WinMLCache``.

        Reads ONNX input names via ``self.onnx_config``'s semantic
        accessors (``past_key_input_names``, ``past_value_input_names``)
        so families that use non-default names only need to override
        those accessors — no string formats here.  All cache shape info
        (batch, heads, max_cache_len, head_dim) is probed from the first
        past-key tensor.
        """
        cfg = self.onnx_config
        key_names = cfg.past_key_input_names
        value_names = cfg.past_value_input_names
        sample_k = inputs[key_names[0]]
        decoder_config = self.config.get_text_config(decoder=True)
        cache = self._CACHE_CLS(decoder_config, max_cache_len=sample_k.size(2))
        cache.early_initialization(
            batch_size=sample_k.size(0),
            num_heads=sample_k.size(1),
            head_dim=sample_k.size(3),
            dtype=sample_k.dtype,
            device=sample_k.device,
        )
        # Alias each cache slot to the corresponding ONNX input tensor.  After
        # this loop, any read/write HF performs on ``cache.layers[i].keys`` is
        # an op on the named graph input — that's how the cache becomes
        # "visible" at the ONNX boundary.
        for i, (key_name, value_name) in enumerate(zip(key_names, value_names, strict=True)):
            # ``Cache.layers`` is typed as a union including
            # ``LinearAttentionCacheLayerMixin`` (no keys/values); a WinML static
            # cache always holds ``CacheLayerMixin`` layers, so narrow the type.
            layer = cast("CacheLayerMixin", cache.layers[i])
            layer.keys = inputs[key_name]
            layer.values = inputs[value_name]
        return cache

    @abstractmethod
    def _invoke_hf(
        self, cache: WinMLCache, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Call the HF decoder with ``past_key_values=<cache>``.  Returns logits."""


class WinMLStaticCacheDecoderIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum/transformers base is untyped
    """Semantic-name contract used by ``WinMLDecoderWrapper._make_cache``.

    Subclasses declare their own ``inputs`` / ``outputs`` bodies (each
    family is free to pick its own ONNX input names).  Override these
    accessors only if the family uses non-default per-layer KV naming.

    Defaults match HuggingFace's encoder-decoder convention.
    """

    @property
    def past_key_input_names(self) -> list[str]:
        """ONNX input names for past KV keys, ordered by layer index."""
        return [f"past_{i}_key" for i in range(self._normalized_config.num_layers)]

    @property
    def past_value_input_names(self) -> list[str]:
        """ONNX input names for past KV values, ordered by layer index."""
        return [f"past_{i}_value" for i in range(self._normalized_config.num_layers)]


__all__ = ["WinMLDecoderWrapper", "WinMLStaticCacheDecoderIOConfig"]
