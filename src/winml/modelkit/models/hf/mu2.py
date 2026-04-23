# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Mu2 encoder-decoder model with KV cache.

Export wrappers, OnnxConfig registrations, and ``WinMLMu2Model`` inference
class for Mu2 (custom ``trust_remote_code`` model).

Export Strategy:
- Mu2EncoderWrapper (``feature-extraction``): encoder-only ONNX.
- Mu2DecoderWrapper (``text2text-generation``): decoder with
  ``WinMLSlidingWindowCache`` (Slice+Concat, no ScatterElements).
  Present KV output is the new-token KV only.

Custom model integration (``auto_map``):
    The Mu2 model uses ``trust_remote_code=True`` with ``auto_map`` in
    ``config.json`` pointing to ``modeling_mu.py`` / ``configuration_mu.py``
    alongside the weights.  KV cache support was added to the model source
    (``MuAttentionSDPA`` accepts ``past_key_value`` + ``cache_position``).

Key decisions:
- Uses ``WinMLSlidingWindowCache`` (not Static) because Mu2 uses RoPE,
  not learned relative position bias.  RoPE is baked into K tensors,
  so buffer positions don't affect attention — sliding window is safe.
- The decoder ONNX input is ``position_id`` (absolute seq position for
  RoPE), not ``cache_position`` (which implies buffer-position indexing).
- Mu2's ``generate_sin_cos_pos_emb`` was patched for transformers < 5.x
  compatibility (computes inv_freq directly instead of using
  ``LlamaRotaryEmbedding.compute_default_rope_parameters``).
- Mu2's ``Mu2Config`` must pass ``pad_token_id`` / ``bos_token_id`` /
  ``eos_token_id`` to ``super().__init__()`` or PretrainedConfig
  overrides them to None.

Cache type:

The default configuration uses ``WinMLSlidingWindowCache`` (FIFO
Slice+Concat).  ``WinMLEncoderDecoderModel`` is cache-agnostic — mask
construction and cache updates are delegated to the cache class via
``build_decoder_mask``, ``position_input_name``, and
``update_all_layers``.  To switch to ``WinMLStaticCache`` (index_copy_):

1. **Export wrapper**: change ``Mu2DecoderWrapper.forward()`` to use
   ``WinMLStaticCache`` and rename the position arg from ``position_id``
   to ``cache_position``.
2. **OnnxConfig inputs**: change ``"position_id"`` to
   ``"cache_position"`` in ``Mu2DecoderIOConfig.inputs``.
3. **Inference**: override ``get_cache_class()`` to return
   ``WinMLStaticCache``.  ``WinMLEncoderDecoderModel`` uses
   ``cache.position_input_name`` to select the correct ONNX input name
   automatically.

Usage::

    wmk config -m path/to/mu2 --task translation --trust-remote-code -o mu2.json
    wmk build -c mu2_encoder.json -m path/to/mu2 --trust-remote-code -o output/encoder
    wmk build -c mu2_decoder.json -m path/to/mu2 --trust-remote-code -o output/decoder
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyTextInputGenerator

from ...config import WinMLBuildConfig
from ...export import register_onnx_overwrite
from ...optim import WinMLOptimizationConfig
from ..winml.composite_model import register_composite_model
from ..winml.encoder_decoder import EncoderDecoderInputGenerator, WinMLEncoderDecoderModel
from ..winml.kv_cache import PastKeyValueInputGenerator, WinMLSlidingWindowCache


# =============================================================================
# Wrapper nn.Modules
# =============================================================================


class Mu2EncoderWrapper(nn.Module):
    """Wraps Mu2 encoder for standalone ONNX export."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.config = model.config

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> Mu2EncoderWrapper:
        """Load full Mu2, extract encoder."""
        from transformers import AutoModelForSeq2SeqLM

        full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **kwargs)
        wrapper = cls(full_model)
        wrapper.eval()
        return wrapper

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return encoder last hidden state."""
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask.bool()
        ).last_hidden_state


class Mu2DecoderWrapper(nn.Module):
    """Wraps Mu2 decoder for ONNX export.

    Delegates to the model's own decoder (which now accepts ``past_key_values``
    and ``cache_position``).  This wrapper just builds the cache from flat
    KV inputs, calls the decoder, and collects captured KV outputs.

    Same pattern as ``T5DecoderWrapper``.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.config = model.config
        self.num_layers = model.config.n_decoder_layer

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> Mu2DecoderWrapper:
        """Load full Mu2, wrap for cached decoder export."""
        from transformers import AutoModelForSeq2SeqLM

        full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **kwargs)
        wrapper = cls(full_model)
        wrapper.eval()
        return wrapper

    def get_export_args(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Convert dict inputs to positional args for torch.onnx.export."""
        return tuple(inputs.values())

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run decoder with FIFO KV cache (Slice+Concat).

        Positional args (order matches OnnxConfig.inputs):
            decoder_input_ids, encoder_hidden_states, attention_mask (encoder),
            decoder_attention_mask, position_id,
            past_0_key, past_0_value, past_1_key, past_1_value, ...

        Returns:
            (logits, present_0_key, present_0_value, ...) where each
            present KV is the new-token slice only [batch, n_kv_head, seq_len, head_dim]
            (raw key_states/value_states captured before Slice+Concat in WinMLSlidingWindowCache).
        """
        decoder_input_ids = args[0]
        encoder_hidden_states = args[1]
        encoder_attention_mask = args[2]  # "attention_mask" in OnnxConfig
        decoder_attention_mask = args[3]
        position_id = args[4]  # absolute sequence position for RoPE
        kv_start = 5

        # Build WinMLSlidingWindowCache (FIFO: Slice+Concat instead of ScatterElements)
        cache = WinMLSlidingWindowCache(self.config, max_cache_len=args[kv_start].size(2))
        cache.early_initialization(
            batch_size=decoder_input_ids.size(0),
            num_heads=self.config.n_kv_head,
            head_dim=self.config.head_dim,
            dtype=args[kv_start].dtype,
            device=decoder_input_ids.device,
        )
        for i in range(self.num_layers):
            cache.layers[i].keys = args[kv_start + i * 2]
            cache.layers[i].values = args[kv_start + i * 2 + 1]

        # Delegate to model's decoder — position_id is passed as cache_position
        # for RoPE computation (WinMLSlidingWindowCache.update ignores it for indexing)
        hidden_states = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=cache,
            cache_position=position_id,
        )
        logits = self.model.lm_head(hidden_states)

        # Output new-token KV only (same as T5 — captured during update)
        result: list[torch.Tensor] = [logits]
        for i in range(self.num_layers):
            k, v = cache.captured[i]
            result.extend([k, v])
        return tuple(result)


# =============================================================================
# OnnxConfig Registrations
# =============================================================================


@register_onnx_overwrite("mu2", "feature-extraction", library_name="transformers")
class Mu2EncoderIOConfig(OnnxConfig):
    """ONNX config for Mu2 encoder (feature-extraction task)."""

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        vocab_size="vocab_size",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return {
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        }


@register_onnx_overwrite("mu2", "text2text-generation", library_name="transformers")
class Mu2DecoderIOConfig(OnnxConfig):
    """ONNX config for Mu2 decoder with static KV cache."""

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        hidden_size="n_embd",
        num_layers="n_decoder_layer",
        num_attention_heads="n_kv_head",
        head_dim="head_dim",
        max_cache_len="block_size",
        vocab_size="vocab_size",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (
        EncoderDecoderInputGenerator,
        PastKeyValueInputGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        result: dict[str, dict[int, str]] = {
            "decoder_input_ids": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "decoder_attention_mask": {0: "batch_size"},
            "position_id": {},
        }
        num_layers = self._normalized_config.num_layers
        for i in range(num_layers):
            result[f"past_{i}_key"] = {0: "batch_size"}
            result[f"past_{i}_value"] = {0: "batch_size"}
        return result

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        result: dict[str, dict[int, str]] = {"logits": {0: "batch_size"}}
        num_layers = self._normalized_config.num_layers
        for i in range(num_layers):
            result[f"present_{i}_key"] = {0: "batch_size"}
            result[f"present_{i}_value"] = {0: "batch_size"}
        return result


# =============================================================================
# Model Class Mapping + WinML Inference Model
# =============================================================================

MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("mu2", "feature-extraction"): Mu2EncoderWrapper,
    ("mu2", "text2text-generation"): Mu2DecoderWrapper,
}

MU2_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        gelu_fusion=True,
        fuse_rmsnorm=True,
        matmul_add_fusion=True,
        clamp_constant_values=True,
        remove_isnan_in_attention_mask=True,
    ),
)


@register_composite_model("mu2", "translation")
class WinMLMu2Model(WinMLEncoderDecoderModel):
    """Mu2 encoder-decoder model with sliding-window KV cache.

    Only differs from T5 in ``get_cache_class`` and ``_SUB_MODEL_CONFIG``.
    All forward/cache logic lives in ``WinMLEncoderDecoderModel``.
    """

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }

    @classmethod
    def get_cache_class(cls) -> type:  # noqa: D102
        return WinMLSlidingWindowCache

    @property
    def generation_config(self):  # noqa: D102
        if not hasattr(self, "_generation_config"):
            from transformers import GenerationConfig

            gc_kw: dict[str, Any] = {}
            if self.config is not None:
                for attr in (
                    "decoder_start_token_id",
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                ):
                    val = getattr(self.config, attr, None)
                    if val is not None:
                        gc_kw[attr] = val
            gc_kw.setdefault("max_new_tokens", self._max_dec - 1)
            gc_kw.setdefault("num_beams", 1)
            gc_kw.setdefault("do_sample", False)
            self._generation_config = GenerationConfig(**gc_kw)
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value: Any) -> None:
        self._generation_config = value


__all__ = [
    "MODEL_CLASS_MAPPING",
    "MU2_CONFIG",
    "Mu2DecoderIOConfig",
    "Mu2DecoderWrapper",
    "Mu2EncoderIOConfig",
    "Mu2EncoderWrapper",
    "WinMLMu2Model",
]
