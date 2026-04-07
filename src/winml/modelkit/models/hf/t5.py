# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""T5 HuggingFace Model Configuration.

Provides encoder/decoder export wrappers and OnnxConfig registrations for
T5 encoder-decoder models with static KV cache.

Export Strategy (split by task):
- T5EncoderWrapper + T5EncoderIOConfig: ``feature-extraction`` task
  → encoder-only ONNX (input_ids, attention_mask → encoder_hidden_states)
- T5DecoderWrapper + T5DecoderIOConfig: ``text2text-generation`` task
  → decoder ONNX with static buffer input + single-token KV output.
    Uses HF StaticCache (index_copy_ at cache_position) for attention.
    Output is only the new token's KV [batch, heads, 1, d_kv].

Model: google-t5/t5-small, google-t5/t5-base, etc.

Usage:
    wmk config -m google-t5/t5-small --task feature-extraction       → encoder
    wmk config -m google-t5/t5-small --task text2text-generation      → decoder
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import (
    DummyInputGenerator,
    DummyTextInputGenerator,
)
from transformers import StaticCache, T5ForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from ...export import register_onnx_overwrite


# =============================================================================
# Capturing StaticCache — eliminates gather from ONNX output
# =============================================================================


class _CapturingStaticCache(StaticCache):
    """StaticCache that captures each layer's new-token KV from ``update()``.

    Standard ``StaticCache.update()`` does ``index_copy_`` (ScatterElements in
    ONNX) to write the new KV into the full buffer, then returns the full
    buffer for attention. The old approach then used ``gather``
    (GatherElements) to extract the same KV back from the buffer — a
    pointless round-trip.

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


# =============================================================================
# Wrapper nn.Modules (with from_pretrained, like SAM2 wrappers)
# =============================================================================


class T5EncoderWrapper(nn.Module):
    """Wraps T5 encoder for standalone ONNX export.

    Loads the full T5ForConditionalGeneration and extracts the encoder.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> T5EncoderWrapper:
        """Load full T5, extract encoder."""
        full_model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        wrapper = cls(full_model.encoder)
        wrapper.eval()
        return wrapper

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return encoder last hidden state."""
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state


class T5DecoderWrapper(nn.Module):
    """Wraps T5ForConditionalGeneration with static KV cache I/O.

    Input: full static buffer ``[batch, heads, max_decode, d_kv]`` per layer.
    Output: only the new token's KV ``[batch, heads, 1, d_kv]`` per layer.

    Uses HF ``StaticCache`` (``index_copy_`` at ``cache_position``) wrapped
    in ``EncoderDecoderCache`` (cross-attn empty → always recomputed from
    ``encoder_hidden_states``). ``KV_index = sequence_position`` holds, so
    T5's relative position bias computes correct distances.

    The inference wrapper (WinMLModelForSeq2SeqLM) uses the same
    ``StaticCache`` class — it writes the single-token output KV back
    into the buffer via ``cache.update()`` before the next step.
    """

    def __init__(self, model: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        # Expose config for OnnxConfig / NormalizedConfig access
        self.config = model.config

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> T5DecoderWrapper:
        """Load full T5, wrap with static cache."""
        full_model = T5ForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        num_layers = full_model.config.num_layers
        wrapper = cls(full_model, num_layers)
        wrapper.eval()
        return wrapper

    def get_export_args(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Convert dict inputs to positional args for torch.onnx.export."""
        return tuple(inputs.values())

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run decoder with static KV cache.

        Positional args (order matches OnnxConfig.inputs):
            decoder_input_ids, encoder_hidden_states, attention_mask,
            decoder_attention_mask, cache_position,
            past_0_key, past_0_value, past_1_key, past_1_value, ...

        Returns:
            (logits, present_0_key, present_0_value, ...) where each
            present KV is [batch, heads, 1, d_kv] — the new token only.
        """
        decoder_input_ids = args[0]
        encoder_hidden_states = args[1]
        attention_mask = args[2]
        decoder_attention_mask = args[3]
        cache_position = args[4]
        kv_start = 5

        # Build CapturingStaticCache from input KV tensors.
        # update() uses index_copy_ at cache_position for correct attention,
        # and captures the incoming key/value states for direct output
        # (eliminating the old scatter→gather round-trip in the ONNX graph).
        self_attn_cache = _CapturingStaticCache(self.config, max_cache_len=args[kv_start].size(2))
        self_attn_cache.early_initialization(
            batch_size=decoder_input_ids.size(0),
            num_heads=args[kv_start].size(1),
            head_dim=args[kv_start].size(3),
            dtype=args[kv_start].dtype,
            device=decoder_input_ids.device,
        )
        for i in range(self.num_layers):
            self_attn_cache.layers[i].keys = args[kv_start + i * 2]
            self_attn_cache.layers[i].values = args[kv_start + i * 2 + 1]

        # EncoderDecoderCache is structurally required: T5Attention routes
        # self-attention → self_attention_cache, cross-attention → cross_attention_cache.
        # Without the wrapper, both would share the same cache + layer indices.
        # DynamicCache for cross-attn is a no-op during export (each layer
        # computes fresh from encoder_hidden_states, never reuses).
        cross_attn_cache = DynamicCache()
        cache = EncoderDecoderCache(self_attn_cache, cross_attn_cache)

        out = self.model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )

        # Return new-token KV directly from the capturing cache.
        # The old approach did gather(ScatterElements output) — a round-trip.
        # _CapturingStaticCache already saved the incoming key/value states.
        result: list[torch.Tensor] = [out.logits]
        for i in range(self.num_layers):
            k, v = self_attn_cache.captured[i]
            result.extend([k, v])
        return tuple(result)


# =============================================================================
# Custom DummyInputGenerators
# =============================================================================


class T5DecoderBaseInputGenerator(DummyInputGenerator):
    """Generates decoder base inputs: decoder_input_ids, encoder_hidden_states,
    attention_mask, decoder_attention_mask, cache_position.
    """  # noqa: D205

    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "encoder_hidden_states",
        "attention_mask",
        "decoder_attention_mask",
        "cache_position",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        self.batch_size = batch_size
        self.d_model = normalized_config.hidden_size
        self.enc_seq = getattr(normalized_config, "sequence_length", 16)
        self.max_decode = getattr(normalized_config, "max_decode_length", 32)
        self.vocab_size = normalized_config.vocab_size

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        if input_name == "decoder_input_ids":
            return self.random_int_tensor(
                (self.batch_size, 1),
                max_value=self.vocab_size,
                framework=framework,
                dtype=int_dtype,
            )
        if input_name == "encoder_hidden_states":
            return self.random_float_tensor(
                (self.batch_size, self.enc_seq, self.d_model),
                framework=framework,
                dtype=float_dtype,
            )
        if input_name == "attention_mask":
            return torch.ones(self.batch_size, self.enc_seq, dtype=torch.int64)
        if input_name == "decoder_attention_mask":
            return torch.ones(self.batch_size, self.max_decode, dtype=torch.int64)
        if input_name == "cache_position":
            return torch.tensor([5], dtype=torch.int64)  # arbitrary position for tracing
        raise ValueError(f"Unknown input: {input_name}")


class T5KVCacheInputGenerator(DummyInputGenerator):
    """Generates KV cache tensors: past_{i}_key, past_{i}_value."""

    SUPPORTED_INPUT_NAMES = ()  # dynamic — handled via supports()

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        self.batch_size = batch_size
        self.num_layers = normalized_config.num_layers
        self.num_heads = normalized_config.num_attention_heads
        self.d_kv = getattr(normalized_config, "key_value_dim", 64)
        self.max_decode = getattr(normalized_config, "max_decode_length", 32)
        # Build supported names dynamically
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
        return self.random_float_tensor(
            (self.batch_size, self.num_heads, self.max_decode, self.d_kv),
            framework=framework,
            dtype=float_dtype,
        )


# =============================================================================
# OnnxConfig Registrations
# =============================================================================


@register_onnx_overwrite("t5", "feature-extraction", library_name="transformers")
class T5EncoderIOConfig(OnnxConfig):
    """ONNX config for T5 encoder (feature-extraction task).

    Inputs:  input_ids, attention_mask
    Outputs: encoder_hidden_states
    """

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


@register_onnx_overwrite("t5", "text2text-generation", library_name="transformers")
class T5DecoderIOConfig(OnnxConfig):
    """ONNX config for T5 decoder with static KV cache.

    Inputs:  decoder_input_ids, encoder_hidden_states, attention_mask,
             decoder_attention_mask, cache_position, past_{i}_key/value
    Outputs: logits, present_{i}_key/value

    Input past KV: full static buffer [batch, heads, max_decode, d_kv].
    Output present KV: new token only [batch, heads, 1, d_kv].
    """

    # T5Config: d_model, num_layers, num_heads, d_kv, vocab_size.
    # sequence_length uses Optimum default (16) — NOT n_positions (512, too large).
    # max_decode_length is not in T5Config — defaults to 32 in the generator.
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        hidden_size="d_model",
        num_layers="num_layers",
        num_attention_heads="num_heads",
        key_value_dim="d_kv",
        vocab_size="vocab_size",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (
        T5DecoderBaseInputGenerator,
        T5KVCacheInputGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        result: dict[str, dict[int, str]] = {
            "decoder_input_ids": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "decoder_attention_mask": {0: "batch_size"},
            "cache_position": {},
        }
        num_layers = self._normalized_config.num_layers
        for i in range(num_layers):
            result[f"past_{i}_key"] = {0: "batch_size"}
            result[f"past_{i}_value"] = {0: "batch_size"}
        return result

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        result: dict[str, dict[int, str]] = {
            "logits": {0: "batch_size"},
        }
        num_layers = self._normalized_config.num_layers
        for i in range(num_layers):
            result[f"present_{i}_key"] = {0: "batch_size"}
            result[f"present_{i}_value"] = {0: "batch_size"}
        return result


# =============================================================================
# Model Class Mapping (same pattern as SAM2 and CLIP)
# =============================================================================

MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("t5", "feature-extraction"): T5EncoderWrapper,
    ("t5", "text2text-generation"): T5DecoderWrapper,
}

__all__ = [
    "MODEL_CLASS_MAPPING",
    "T5DecoderIOConfig",
    "T5DecoderWrapper",
    "T5EncoderIOConfig",
    "T5EncoderWrapper",
]
