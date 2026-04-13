# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Mu2 HuggingFace Model Configuration.

Provides encoder/decoder export wrappers and OnnxConfig registrations for
Mu2 encoder-decoder models with static KV cache.

Export Strategy (split by task):
- Mu2EncoderWrapper + Mu2EncoderIOConfig: ``feature-extraction`` task
  → encoder-only ONNX (input_ids, attention_mask → encoder_hidden_states)
- Mu2DecoderWrapper + Mu2DecoderIOConfig: ``text2text-generation`` task
  → decoder ONNX with static KV buffer input + single-token KV output.
    Input past KV: full static buffer [batch, n_kv_head, max_decode, head_dim].
    Output present KV: new token only [batch, n_kv_head, 1, head_dim].

The Mu2 model's native attention (MuAttentionSDPA) does NOT support HF's
cache mechanism.  The decoder wrapper reimplements the decoder forward pass
using the original layer weights, adding CapturingStaticCache for
self-attention KV.  Cross-attention KV is always recomputed from
encoder_hidden_states (no cache needed).

Model: local Mu2ForCausalLM with trust_remote_code=True.

Usage:
    wmk config -m path/to/mu2 --task feature-extraction   → encoder
    wmk config -m path/to/mu2 --task text2text-generation  → decoder
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyTextInputGenerator

from ...export import register_onnx_overwrite
from ..winml.pipeline_model import register_pipeline_model
from .encoder_decoder import EncoderDecoderInputGenerator, WinMLEncoderDecoderModel
from .kv_cache import CapturingStaticCache as _CapturingStaticCache
from .kv_cache import PastKeyValueInputGenerator


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
    """Wraps Mu2 decoder with CapturingStaticCache for ONNX export.

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
        """Run decoder with static KV cache.

        Positional args (order matches OnnxConfig.inputs):
            decoder_input_ids, encoder_hidden_states, attention_mask (encoder),
            decoder_attention_mask, cache_position,
            past_0_key, past_0_value, past_1_key, past_1_value, ...

        Returns:
            (logits, present_0_key, present_0_value, ...) where each
            present KV is [batch, n_kv_head, 1, head_dim].
        """
        decoder_input_ids = args[0]
        encoder_hidden_states = args[1]
        encoder_attention_mask = args[2]  # "attention_mask" in OnnxConfig
        decoder_attention_mask = args[3]
        cache_position = args[4]
        kv_start = 5

        # Build CapturingStaticCache from input KV tensors
        self_attn_cache = _CapturingStaticCache(self.config, max_cache_len=args[kv_start].size(2))
        self_attn_cache.early_initialization(
            batch_size=decoder_input_ids.size(0),
            num_heads=self.config.n_kv_head,
            head_dim=self.config.head_dim,
            dtype=args[kv_start].dtype,
            device=decoder_input_ids.device,
        )
        for i in range(self.num_layers):
            self_attn_cache.layers[i].keys = args[kv_start + i * 2]
            self_attn_cache.layers[i].values = args[kv_start + i * 2 + 1]

        # Delegate to model's decoder (now supports past_key_values + cache_position)
        hidden_states = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=self_attn_cache,
            cache_position=cache_position,
        )
        logits = self.model.lm_head(hidden_states)

        # Collect captured KV
        result: list[torch.Tensor] = [logits]
        for i in range(self.num_layers):
            k, v = self_attn_cache.captured[i]
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
            "cache_position": {},
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


@register_pipeline_model("mu2", "translation")
class WinMLMu2Model(WinMLEncoderDecoderModel):
    """Mu2 encoder-decoder model for translation.

    Declares Mu2 sub-component tasks and generation config defaults.
    All encoder-decoder forward/cache logic lives in ``WinMLEncoderDecoderModel``.
    """

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }

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
    "Mu2DecoderIOConfig",
    "Mu2DecoderWrapper",
    "Mu2EncoderIOConfig",
    "Mu2EncoderWrapper",
    "WinMLMu2Model",
]
