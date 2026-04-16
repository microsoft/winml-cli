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

from typing import Any, ClassVar

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyTextInputGenerator
from transformers import T5ForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from ...export import register_onnx_overwrite
from ..winml.composite_model import register_composite_model
from .encoder_decoder import EncoderDecoderInputGenerator, WinMLEncoderDecoderModel
from .kv_cache import PastKeyValueInputGenerator, WinMLStaticCache


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

    The inference wrapper (WinMLT5Model) uses the same
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

        # Build WinMLStaticCache from input KV tensors.
        # update() uses index_copy_ at cache_position for correct attention,
        # and captures the incoming key/value states for direct output
        # (eliminating the old scatter→gather round-trip in the ONNX graph).
        self_attn_cache = WinMLStaticCache(self.config, max_cache_len=args[kv_start].size(2))
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
        # WinMLStaticCache already saved the incoming key/value states.
        result: list[torch.Tensor] = [out.logits]
        for i in range(self.num_layers):
            k, v = self_attn_cache.captured[i]
            result.extend([k, v])
        return tuple(result)


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

    # T5Config: d_model, num_layers, num_heads, d_kv, vocab_size, n_positions.
    # sequence_length uses Optimum default (16) — NOT n_positions (512, too large).
    # head_dim maps to d_kv for PastKeyValueInputGenerator.
    # max_cache_len maps to n_positions (decoder static buffer size).
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        hidden_size="d_model",
        num_layers="num_layers",
        num_attention_heads="num_heads",
        head_dim="d_kv",
        max_cache_len="n_positions",
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


# =============================================================================
# WinMLT5Model — inference wrapper (registered as pipeline model)
# =============================================================================


@register_composite_model("t5", "translation")
@register_composite_model("t5", "summarization")
class WinMLT5Model(WinMLEncoderDecoderModel):
    """T5 encoder-decoder model for seq2seq tasks (translation, summarization).

    Declares T5 sub-component tasks and generation config defaults.
    All encoder-decoder forward/cache logic lives in ``WinMLEncoderDecoderModel``.
    """

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }

    @classmethod
    def get_cache_class(cls) -> type:
        """T5 requires WinMLStaticCache (cannot use sliding window).

        T5's relative position bias (``T5Attention.compute_bias``) computes
        ``memory_position = arange(key_length)`` — it assumes buffer
        position == sequence position.  With sliding window, KV entries
        shift left each step, so buffer positions no longer correspond to
        sequence positions, producing wrong relative distances.
        """
        return WinMLStaticCache

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
            # Static batch=1 ONNX models don't support beam search
            gc_kw.setdefault("num_beams", 1)
            gc_kw.setdefault("do_sample", False)
            self._generation_config = GenerationConfig(**gc_kw)
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value: Any) -> None:
        self._generation_config = value


__all__ = [
    "MODEL_CLASS_MAPPING",
    "T5DecoderIOConfig",
    "T5DecoderWrapper",
    "T5EncoderIOConfig",
    "T5EncoderWrapper",
    "WinMLT5Model",
]
