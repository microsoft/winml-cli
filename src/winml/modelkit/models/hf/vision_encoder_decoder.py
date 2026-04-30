# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Vision-encoder-decoder export — generic encoder + decoder.

Works for any HF ``VisionEncoderDecoderModel`` where:

- The vision encoder produces ``encoder_hidden_states`` of shape
  ``[batch, encoder_seq, encoder.hidden_size]`` (ViT/DeiT/Beit geometry).
- The inner decoder is a CausalLM with cross-attention (TrOCR / MBart /
  Marian inner causal-LM family).

Dispatch happens at the HF model level: ``VisionDecoderWrapper`` loads the
full ``VisionEncoderDecoderModel`` and delegates to ``model.decoder``
polymorphically — no inner-arch registry, no per-family wrapper classes.

Per-architecture field-name differences (``decoder.d_model`` vs
``decoder.n_embd``, etc.) are handled by ``_VedDecoderNormalizedConfig``,
which delegates to Optimum's ``NormalizedConfigManager`` for each subconfig.

PATCHING_SPECS
--------------
Bundles trace-time fixes for inner-decoder positional-embedding ops that
don't trace cleanly under static-cache export.  Each ``PatchingSpec`` is
class-targeted, so it's a no-op on graphs that don't contain that class.
Adding fixes for new families is purely additive.

TrOCR positional-embedding patch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``TrOCRDecoder.forward`` (transformers ≥4.57) drives the learned positional
embedding via::

    past_kv_len = past_key_values.get_seq_length()
    embed_pos = self.embed_positions(input, past_key_values_length=past_kv_len)

and ``TrOCRLearnedPositionalEmbedding.forward`` derives positions with
``torch.arange(past_kv_len, past_kv_len + seq_len)`` — which traces as a
``Range`` op driven by a symbolic scalar; NPU compilers reject that.

We side-channel the absolute seq pos onto the embedding module as a tensor
attribute named ``position_id`` and ``PATCHING_SPECS``-replace
``TrOCRLearnedPositionalEmbedding.forward`` with a variant that reads that
attribute and does a plain ``Embedding`` lookup (preserving TrOCR's
``+offset``).  The patched forward falls back to stock HF behavior when
``position_id`` is not set, so any TrOCR embedding used outside this
exporter is unaffected.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.exporters.onnx.model_patcher import PatchingSpec
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyVisionInputGenerator
from optimum.utils.normalized_config import NormalizedConfigManager
from transformers import VisionEncoderDecoderModel
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from ...config import WinMLBuildConfig
from ...export import register_onnx_overwrite
from ...optim import WinMLOptimizationConfig
from ..winml.encoder_decoder import EncoderDecoderInputGenerator
from ..winml.kv_cache import PastKeyValueInputGenerator
from .decoder_wrapper import WinMLDecoderWrapper, WinMLStaticCacheDecoderIOConfig


logger = logging.getLogger(__name__)


# =============================================================================
# WinML Build Config
# =============================================================================

VISION_ENCODER_DECODER_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        gelu_fusion=True,
        layer_norm_fusion=True,
        matmul_add_fusion=True,
        clip_constant_values=True,
        reshape_mergedreshape=True,
    ),
)


# =============================================================================
# Encoder
# =============================================================================


class VisionEncoderWrapper(nn.Module):
    """Extracts the vision backbone of a ``VisionEncoderDecoderModel`` for export."""

    def __init__(self, encoder: nn.Module, config: Any) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> VisionEncoderWrapper:
        """Load full ``VisionEncoderDecoderModel`` and wrap its encoder."""
        full = VisionEncoderDecoderModel.from_pretrained(model_name_or_path, **kwargs)
        wrapper = cls(full.encoder, full.config)
        wrapper.eval()
        return wrapper

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Trace ``pixel_values → encoder_hidden_states``."""
        return self.encoder(pixel_values=pixel_values).last_hidden_state


@register_onnx_overwrite(
    "vision-encoder-decoder", "feature-extraction", library_name="transformers"
)
class VisionEncoderIOConfig(OnnxConfig):
    """ONNX config for the vision encoder."""

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="encoder.num_channels",
        image_size="encoder.image_size",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return {
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return {
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        }


# =============================================================================
# TrOCR positional-embedding patch (class-targeted; no-op on other archs)
# =============================================================================


def _patched_trocr_learned_positional_embedding_forward(
    self,
    input_ids: torch.Tensor,
    past_key_values_length: Any = 0,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Patched ``TrOCRLearnedPositionalEmbedding.forward``.

    If a ``position_id`` tensor attribute has been set on this module by
    the export wrapper, use it as the lookup index (with TrOCR's
    ``+self.offset`` preserved) and ignore the kwargs that HF would
    otherwise derive via ``torch.arange``.  Without ``position_id`` set,
    behavior is bit-identical to the original HF implementation.
    """
    abs_pos = getattr(self, "position_id", None)
    if abs_pos is not None:
        if abs_pos.dim() == 1:
            abs_pos = abs_pos.unsqueeze(0)
        return nn.Embedding.forward(self, abs_pos + self.offset)
    if position_ids is None:
        bsz, seq_len = input_ids.shape[:2]
        position_ids = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        ).expand(bsz, -1)
    else:
        position_ids = position_ids.unsqueeze(0)
    return nn.Embedding.forward(self, position_ids + self.offset)


def _patched_trocr_sinusoidal_positional_embedding_forward(
    self,
    input_ids: torch.Tensor,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    """Patched ``TrOCRSinusoidalPositionalEmbedding.forward``.

    Stock HF derives ``position_ids`` via ``cumsum(input_ids != pad)``
    plus a tensor add of ``past_key_values_length`` — both of which trace
    as dynamic Cast/Add ops the analyzer can't lower.  We instead read
    the absolute seq pos from the ``position_id`` attribute (same
    side-channel the learned-embedding patch uses), shift by HF's
    standard ``padding_idx + 1`` offset, and look up the frozen sin/cos
    table directly.

    Falls back to stock HF behavior when ``position_id`` is unset.
    """
    abs_pos = getattr(self, "position_id", None)
    if abs_pos is not None:
        if abs_pos.dim() == 1:
            abs_pos = abs_pos.unsqueeze(0)
        position_ids = (abs_pos + self.padding_idx + 1).long()
        weights = self.weights.to(self._float_tensor)
        return weights.index_select(0, position_ids.view(-1)).view(
            position_ids.size(0), position_ids.size(1), -1
        ).detach()
    # Fallback: bit-identical to stock HF behavior.
    bsz, seq_len = input_ids.size()
    position_ids = self.create_position_ids_from_input_ids(
        input_ids, self.padding_idx, past_key_values_length
    ).to(input_ids.device)
    max_pos = self.padding_idx + 1 + seq_len
    if self.weights is None or max_pos > self.weights.size(0):
        self.weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
    self.weights = self.weights.to(self._float_tensor)
    return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()


def _build_ved_patching_specs() -> list[PatchingSpec]:
    """Aggregate class-targeted patches for VED inner decoders.

    Each spec is a no-op on graphs that don't contain the targeted class,
    so bundling patches for multiple inner-decoder families is safe.
    """
    specs: list[PatchingSpec] = []
    try:
        from transformers.models.trocr.modeling_trocr import (
            TrOCRLearnedPositionalEmbedding,
            TrOCRSinusoidalPositionalEmbedding,
        )
    except ImportError:
        logger.debug("TrOCR positional-embedding classes not found; patches skipped.")
    else:
        specs.append(
            PatchingSpec(
                o=TrOCRLearnedPositionalEmbedding,
                name="forward",
                custom_op=_patched_trocr_learned_positional_embedding_forward,
            )
        )
        specs.append(
            PatchingSpec(
                o=TrOCRSinusoidalPositionalEmbedding,
                name="forward",
                custom_op=_patched_trocr_sinusoidal_positional_embedding_forward,
            )
        )
    return specs


# =============================================================================
# Decoder NormalizedConfig + dummy generator
# =============================================================================


class _VedDecoderNormalizedConfig(NormalizedConfig):
    """VED decoder NormalizedConfig.

    Per-architecture field paths (``decoder.d_model`` vs ``decoder.n_embd``
    etc.) are resolved by Optimum's ``NormalizedConfigManager`` against each
    subconfig.  Only VED-level concerns (encoder geometry, cross-attention
    hidden size, max cache len) are expressed locally.
    """

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        dec_cls = NormalizedConfigManager.get_normalized_config_class(
            config.decoder.model_type
        )
        enc_cls = NormalizedConfigManager.get_normalized_config_class(
            config.encoder.model_type
        )
        self._dec = dec_cls(config.decoder)
        self._enc = enc_cls(config.encoder)

    @property
    def vocab_size(self) -> int:
        return self._dec.vocab_size

    @property
    def hidden_size(self) -> int:
        return self._dec.hidden_size

    @property
    def num_layers(self) -> int:
        # Not every model family defines ``decoder_layers`` on
        # ``config.decoder``; fall back to Optimum's NormalizedConfig.
        decoder_layers = getattr(self.config.decoder, "decoder_layers", None)
        return decoder_layers if decoder_layers is not None else self._dec.num_layers

    @property
    def num_attention_heads(self) -> int:
        return self._dec.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def max_cache_len(self) -> int:
        # Optimum's normalized configs don't uniformly expose this; read
        # the raw decoder config field that BART/TrOCR-family use.
        return self.config.decoder.max_position_embeddings

    @property
    def encoder_hidden_size(self) -> int:
        # Decoder-side property: the cross-attn K/V projection input dim.
        # Falls back to encoder.hidden_size when no explicit projection is
        # configured (HF convention when enc.hidden_size matches).
        cah = getattr(self.config.decoder, "cross_attention_hidden_size", None)
        return cah if cah is not None else self._enc.hidden_size

    @property
    def image_size(self) -> int | list[int]:
        # Some model types ship a scalar (square input);
        # others ship a ``[H, W]`` list.
        return self.config.encoder.image_size

    @property
    def patch_size(self) -> int:
        return self.config.encoder.patch_size

    @property
    def encoder_seq_length(self) -> int:
        """Output sequence length of the vision encoder.

        Handles both ``image_size`` shapes:

        - Scalar (square ViT-style input): ``(image_size / patch_size)**2``
          patch tokens plus one CLS-style token.
        - ``[H, W]`` list (hierarchical Swin-style input): the patch grid
          ``(H / patch_size) * (W / patch_size)`` divided by
          ``2**(2*(N-1))`` where ``N = len(depths)`` is the number of
          stages — each of the ``N-1`` stage transitions halves
          spatial resolution.  No CLS token.
        """
        enc = self.config.encoder
        patch_size = enc.patch_size
        image_size = enc.image_size

        # Scalar image_size: square ViT-style encoder.
        if not isinstance(image_size, (list, tuple)):
            return (image_size // patch_size) ** 2 + 1

        # [H, W] image_size: hierarchical Swin-style encoder.
        h, w = image_size[0], image_size[1]
        shrink = 2 ** (len(enc.depths) - 1)
        return (h // patch_size // shrink) * (w // patch_size // shrink)


class VedDecoderInputGenerator(EncoderDecoderInputGenerator):
    """Dummy input generator for VED decoder export."""

    def __init__(self, task: str, normalized_config: Any, **kwargs: Any) -> None:
        super().__init__(task, normalized_config, **kwargs)
        self.enc_seq = normalized_config.encoder_seq_length
        # ``encoder_hidden_states`` last dim is the cross-attn K/V input dim.
        self.d_model = normalized_config.encoder_hidden_size


# =============================================================================
# Decoder IOConfig
# =============================================================================


@register_onnx_overwrite(
    "vision-encoder-decoder", "text2text-generation", library_name="transformers"
)
class VisionDecoderIOConfig(WinMLStaticCacheDecoderIOConfig):
    """ONNX config for the VED decoder with static KV cache.

    Inputs:  decoder_input_ids, encoder_hidden_states, decoder_attention_mask,
             cache_position, past_{i}_key / past_{i}_value
    Outputs: logits, present_{i}_key / present_{i}_value
    """

    NORMALIZED_CONFIG_CLASS = _VedDecoderNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        VedDecoderInputGenerator,
        PastKeyValueInputGenerator,
    )
    PATCHING_SPECS = _build_ved_patching_specs()

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        result: dict[str, dict[int, str]] = {
            "decoder_input_ids": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size"},
            "decoder_attention_mask": {0: "batch_size"},
            "cache_position": {},
        }
        for i in range(self._normalized_config.num_layers):
            result[f"past_{i}_key"] = {0: "batch_size"}
            result[f"past_{i}_value"] = {0: "batch_size"}
        return result

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        result: dict[str, dict[int, str]] = {"logits": {0: "batch_size"}}
        for i in range(self._normalized_config.num_layers):
            result[f"present_{i}_key"] = {0: "batch_size"}
            result[f"present_{i}_value"] = {0: "batch_size"}
        return result


# =============================================================================
# Decoder Wrapper — generic; uses ``model.decoder`` polymorphically
# =============================================================================


class VisionDecoderWrapper(WinMLDecoderWrapper):
    """Static-KV-cache decoder export for any VED model.

    Loads the full ``VisionEncoderDecoderModel`` and stores ``model.decoder``
    (the inner causal LM, e.g. ``TrOCRForCausalLM``) as ``self.model``.  The
    trace bypasses ``VisionEncoderDecoderModel.enc_to_dec_proj`` — the
    encoder ONNX must produce hidden states already at the cross-attention
    input dim (which it does, by the IOConfig's ``encoder_hidden_size``).
    """

    _HF_MODEL_CLS = VisionEncoderDecoderModel
    _IO_CONFIG_CLS = VisionDecoderIOConfig

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> VisionDecoderWrapper:
        full = VisionEncoderDecoderModel.from_pretrained(model_name_or_path, **kwargs)
        self = cls()
        self.model = full.decoder           # inner causal LM, called directly
        self.config = full.config           # full VED config drives the IOConfig
        self.onnx_config = cls._IO_CONFIG_CLS(full.config, task=cls._TASK)
        self.num_layers = self.onnx_config._normalized_config.num_layers
        self.eval()
        return self

    def _make_cache(self, inputs: dict[str, torch.Tensor]) -> Any:
        cache = super()._make_cache(inputs)
        # HF decoders that use BART-style learned positional embeddings read
        # ``past_key_values.get_seq_length()`` to drive the position offset.
        # Override so the mask shape reflects the actual generation step.
        position = inputs["cache_position"].squeeze()
        cache.get_seq_length = lambda layer_idx=0: position
        return cache

    def _invoke_hf(self, cache: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        # TrOCR side-channel: write the absolute seq pos onto the inner
        # decoder's ``embed_positions`` so the patched forward (see module
        # docstring) does an Embedding lookup instead of ``torch.arange``.
        # Defensive walk — silently skipped on inner decoders that don't
        # have this module path.  The patched forward also no-ops without
        # ``position_id`` set, so this is benign for non-TrOCR archs.
        decoder_module = getattr(self.model, "model", None)
        inner_decoder = getattr(decoder_module, "decoder", None) if decoder_module is not None else None
        embed_positions = getattr(inner_decoder, "embed_positions", None) if inner_decoder is not None else None
        if embed_positions is not None:
            embed_positions.position_id = inputs["cache_position"]

        # Pass position_ids when the inner decoder accepts it.  Without this,
        # decoders that derive position_ids from ``past_kv_len`` internally
        # (BERT, GPT-2) emit Unsqueeze/Add chains driven by the dynamic
        # ``get_seq_length()`` value — those ops survive into the optimized
        # graph and the analyzer can't lower them.  TrOCRForCausalLM has no
        # ``position_ids`` parameter and stays on the patched-embedding path.
        forward_params = inspect.signature(self.model.forward).parameters
        extra: dict[str, Any] = {}
        if "position_ids" in forward_params:
            extra["position_ids"] = inputs["cache_position"]

        outputs = self.model(
            input_ids=inputs["decoder_input_ids"],
            attention_mask=inputs["decoder_attention_mask"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            encoder_attention_mask=None,  # vision encoders have no padding
            past_key_values=EncoderDecoderCache(cache, DynamicCache()),
            use_cache=True,
            cache_position=inputs["cache_position"],
            return_dict=True,
            **extra,
        )
        return outputs.logits


# =============================================================================
# Model Class Mapping
# =============================================================================

MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("vision-encoder-decoder", "feature-extraction"): VisionEncoderWrapper,
    ("vision-encoder-decoder", "text2text-generation"): VisionDecoderWrapper,
}


__all__ = [
    "MODEL_CLASS_MAPPING",
    "VISION_ENCODER_DECODER_CONFIG",
    "VisionDecoderIOConfig",
    "VisionDecoderWrapper",
    "VisionEncoderIOConfig",
    "VisionEncoderWrapper",
]
