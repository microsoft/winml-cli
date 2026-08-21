# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Marian (Helsinki-NLP/opus-mt) HuggingFace Model Configuration.

Provides encoder/decoder export wrappers and OnnxConfig registrations for
Marian translation models with a static KV cache.

Export Strategy (split by task):
- MarianEncoderWrapper + MarianEncoderIOConfig: ``feature-extraction`` task
  → encoder-only ONNX (input_ids, attention_mask → encoder_hidden_states)
- MarianDecoderWrapper + MarianDecoderIOConfig: ``text2text-generation`` task
  → decoder ONNX with a static cache buffer input + single-token KV output.

Transformers cache compatibility:

Transformers 4.57 passes ``cache_position`` through Marian's decoder and
attention layers. Transformers 5 removes that explicit parameter and calls
``Cache.update`` without cache kwargs; it also derives position IDs and the
causal mask from ``Cache.get_seq_length``. The wrapper handles both APIs:

1. ``WinMLStaticCache.set_trace_position`` supplies the explicit ONNX input
    when Transformers omits ``cache_kwargs`` from ``Cache.update``.
2. On the Transformers 5 path, the positional-embedding patch reads that same
    input directly, avoiding the unsupported integer ``arange + length`` graph.
3. A precomputed 4D additive mask bypasses Transformers 5's internal dynamic
    causal-mask offsets. Transformers 4 keeps its native mask path.

The static cache writes new KV at ``cache_position`` through ScatterND and
preserves ``buffer_position == sequence_position``.

Models: Helsinki-NLP/opus-mt-fr-en, opus-mt-en-ru, opus-mt-es-en, etc.

Usage:
    winml config -m Helsinki-NLP/opus-mt-fr-en --task feature-extraction       → encoder
    winml config -m Helsinki-NLP/opus-mt-fr-en --task text2text-generation     → decoder
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from optimum.exporters.onnx import OnnxConfig
from optimum.exporters.onnx.model_patcher import PatchingSpec
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyTextInputGenerator
from transformers import MarianMTModel
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from ...config import WinMLBuildConfig
from ...export import register_onnx_overwrite
from ...optim import WinMLOptimizationConfig
from ..winml.composite_model import register_composite_model
from ..winml.encoder_decoder import EncoderDecoderInputGenerator, WinMLEncoderDecoderModel
from ..winml.kv_cache import PastKeyValueInputGenerator, WinMLStaticCache


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig
    from transformers.models.marian.modeling_marian import MarianSinusoidalPositionalEmbedding

logger = logging.getLogger(__name__)


# =============================================================================
# Patch for Transformers 5 positional lookup
# =============================================================================
# Transformers 5 no longer forwards ``cache_position`` through MarianDecoder.
# The export wrapper stores the tensor on the embedding module, and this patch
# reads it directly. Without the attribute, it preserves stock behavior for
# Transformers 4 and other Marian export paths.


def _patched_marian_sinusoidal_forward(
    self: MarianSinusoidalPositionalEmbedding,  # monkey-patched onto this HF module
    input_ids_shape: torch.Size,
    past_key_values_length: int = 0,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Patched ``MarianSinusoidalPositionalEmbedding.forward``.

    When the export wrapper stores ``cache_position`` on ``position_id``, use
    it directly as the sin/cos lookup index. Without that attribute, preserve
    the stock implementation for Transformers 4 and other export paths.
    """
    abs_pos = getattr(self, "position_id", None)
    if abs_pos is not None:
        return F.embedding(abs_pos, self.weight)
    # Fallback: unchanged HF behavior
    if position_ids is None:
        _, seq_len = input_ids_shape[:2]
        position_ids = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
    return F.embedding(position_ids, self.weight)


def _build_marian_patching_specs() -> list[PatchingSpec]:
    """Return PatchingSpec list for Marian.

    Returns [] if MarianSinusoidalPositionalEmbedding is unavailable.
    """
    try:
        from transformers.models.marian.modeling_marian import MarianSinusoidalPositionalEmbedding
    except ImportError:
        logger.debug("MarianSinusoidalPositionalEmbedding not found; sin/cos patch skipped.")
        return []
    return [
        PatchingSpec(
            o=MarianSinusoidalPositionalEmbedding,
            name="forward",
            custom_op=_patched_marian_sinusoidal_forward,
        ),
    ]


# =============================================================================
# Wrapper nn.Modules (with from_pretrained, matching T5/Mu2 pattern)
# =============================================================================


class MarianEncoderWrapper(nn.Module):
    """Wraps Marian encoder for standalone ONNX export.

    Loads the full MarianMTModel and extracts the encoder.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> MarianEncoderWrapper:
        """Load full MarianMTModel, extract encoder."""
        full_model = MarianMTModel.from_pretrained(model_name_or_path, **kwargs)
        wrapper = cls(full_model.get_encoder())
        wrapper.eval()
        return wrapper

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return encoder last hidden state."""
        # self.encoder is a torch submodule (untyped __call__ -> Any).
        return cast(
            "torch.Tensor",
            self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state,
        )


class MarianDecoderWrapper(nn.Module):
    """Wraps ``MarianMTModel`` with static KV cache I/O.

    Input: full buffer ``[batch, heads, max_decode, d_kv]`` per layer.
    Output: only the new token's KV ``[batch, heads, 1, d_kv]`` per layer.

    ``cache_position`` is an explicit ONNX input used for both the static
    cache write and Marian's absolute sinusoidal position.
    """

    def __init__(self, model: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        # Expose config for OnnxConfig / NormalizedConfig access
        # model is typed nn.Module, so torch's __getattr__ types .config as
        # Tensor | Module; it is really the model's PretrainedConfig.
        self.config: PretrainedConfig = cast("PretrainedConfig", model.config)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> MarianDecoderWrapper:
        """Load full MarianMTModel and wrap its decoder with a static cache."""
        full_model = MarianMTModel.from_pretrained(model_name_or_path, **kwargs)
        num_layers = full_model.config.decoder_layers
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
            present KV is ``[batch, heads, 1, d_kv]`` — the new token only.
        """
        decoder_input_ids = args[0]
        encoder_hidden_states = args[1]
        attention_mask = args[2]
        decoder_attention_mask = args[3]
        cache_position = args[4]
        kv_start = 5

        max_cache_len = args[kv_start].size(2)
        self_attn_cache = WinMLStaticCache(self.config, max_cache_len=max_cache_len)
        self_attn_cache.early_initialization(
            batch_size=decoder_input_ids.size(0),
            num_heads=args[kv_start].size(1),
            head_dim=args[kv_start].size(3),
            dtype=args[kv_start].dtype,
            device=decoder_input_ids.device,
        )
        for i in range(self.num_layers):
            self_attn_cache._layer(i).keys = args[kv_start + i * 2]
            self_attn_cache._layer(i).values = args[kv_start + i * 2 + 1]

        self_attn_cache.set_trace_position(cache_position)
        model = cast("MarianMTModel", self.model)
        decoder = model.get_decoder()
        if "cache_position" not in inspect.signature(decoder.forward).parameters:
            decoder.embed_positions.position_id = cache_position
            expanded_mask = decoder_attention_mask[:, None, None, :].to(
                dtype=encoder_hidden_states.dtype
            )
            decoder_attention_mask = (1.0 - expanded_mask) * torch.finfo(
                encoder_hidden_states.dtype
            ).min

        # EncoderDecoderCache routes self-attention vs cross-attention to
        # separate caches.  DynamicCache for cross-attn is a no-op during
        # export (each layer computes fresh from encoder_hidden_states).
        cross_attn_cache = DynamicCache()
        cache = EncoderDecoderCache(self_attn_cache, cross_attn_cache)

        out = model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )

        # Return new-token KV directly from the capturing cache.
        result: list[torch.Tensor] = [out.logits]
        for i in range(self.num_layers):
            k, v = self_attn_cache.captured[i]
            result.extend([k, v])
        return tuple(result)


# =============================================================================
# OnnxConfig Registrations
# =============================================================================


@register_onnx_overwrite("marian", "feature-extraction", library_name="transformers")
class MarianEncoderIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for Marian encoder (feature-extraction task).

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


class _MarianDecoderNormalizedConfig(NormalizedConfig):  # type: ignore[misc]  # optimum base is untyped
    """NormalizedConfig for Marian decoder-side export.

    Maps NormalizedConfig attributes to MarianConfig's decoder-side attrs.
    ``head_dim`` is derived — MarianConfig has no such attr natively.
    """

    VOCAB_SIZE = "vocab_size"
    HIDDEN_SIZE = "d_model"
    NUM_LAYERS = "decoder_layers"
    NUM_ATTENTION_HEADS = "decoder_attention_heads"
    MAX_CACHE_LEN = "max_position_embeddings"

    @property
    def head_dim(self) -> int:
        # hidden_size / num_attention_heads come from the untyped NormalizedConfig base.
        return cast("int", self.hidden_size // self.num_attention_heads)


@register_onnx_overwrite("marian", "text2text-generation", library_name="transformers")
class MarianDecoderIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for Marian decoder with static KV cache.

    Inputs:  decoder_input_ids, encoder_hidden_states, attention_mask,
             decoder_attention_mask, cache_position, past_{i}_key/value
    Outputs: logits, present_{i}_key/value

    ``cache_position`` is both the static buffer index and absolute sequence
    position. Transformers 5 receives it through the export compatibility
    path described by ``MarianDecoderWrapper``.

    Input past KV: full buffer ``[batch, heads, max_decode, d_kv]``.
    Output present KV: new token only ``[batch, heads, 1, d_kv]``.
    """

    NORMALIZED_CONFIG_CLASS = _MarianDecoderNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        EncoderDecoderInputGenerator,
        PastKeyValueInputGenerator,
    )
    PATCHING_SPECS = _build_marian_patching_specs()

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
# Model Class Mapping + Build Config
# =============================================================================

MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("marian", "feature-extraction"): MarianEncoderWrapper,
    ("marian", "text2text-generation"): MarianDecoderWrapper,
}

MARIAN_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        gelu_fusion=True,
        matmul_add_fusion=True,
        clamp_constant_values=True,
        remove_isnan_in_attention_mask=True,
    ),
)


# =============================================================================
# WinMLMarianModel — inference wrapper (registered as composite model)
# =============================================================================


@register_composite_model("marian", "translation")
class WinMLMarianModel(WinMLEncoderDecoderModel):
    """Marian encoder-decoder model for translation.

    Declares Marian sub-component tasks and generation-config defaults.
    All encoder-decoder forward/cache logic lives in
    ``WinMLEncoderDecoderModel``. Uses ``WinMLStaticCache`` so inference cache
    positions match the decoder's exported ``cache_position`` input.
    """

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }

    @classmethod
    def get_cache_class(cls) -> type:
        """Use the same static cache semantics as the exported decoder."""
        return WinMLStaticCache

    @property
    def generation_config(self) -> GenerationConfig:  # noqa: D102
        if not hasattr(self, "_generation_config"):
            from transformers import GenerationConfig

            gc_kw: dict[str, Any] = {}
            if self.config is not None:
                for attr in (
                    "decoder_start_token_id",
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                    "forced_eos_token_id",
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
    "MARIAN_CONFIG",
    "MODEL_CLASS_MAPPING",
    "MarianDecoderIOConfig",
    "MarianDecoderWrapper",
    "MarianEncoderIOConfig",
    "MarianEncoderWrapper",
    "WinMLMarianModel",
]
