# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""TrOCR decoder export — vision-encoder-decoder with TrOCR inner causal-LM.

Registered for ``("vision-encoder-decoder", "text2text-generation")``;
the ``VisionDecoderWrapper`` dispatcher routes to ``TocrDecoderWrapper``
when ``config.decoder.model_type == "trocr"``.

Models: microsoft/trocr-base-printed, microsoft/trocr-large-*, etc.

Why a positional-embedding patch is needed
------------------------------------------
``TrOCRDecoder.forward`` (transformers 4.57.6) drives the learned positional
embedding via::

    past_key_values_length = past_key_values.get_seq_length()
    embed_pos = self.embed_positions(input, past_key_values_length=past_key_values_length)

and ``TrOCRLearnedPositionalEmbedding.forward`` derives positions with
``torch.arange(past_key_values_length, past_key_values_length + seq_len)`` —
which traces as a ``Range`` op driven by a symbolic scalar.  NPU compilers
reject that.

To avoid the ``Range``, we side-channel the absolute seq pos onto the
embedding module as a tensor attribute named ``position_id`` and
``PATCHING_SPECS``-replace ``TrOCRLearnedPositionalEmbedding.forward`` with
a variant that reads that attribute and does a plain ``Embedding`` lookup
(adding TrOCR's ``+offset``).  The pattern mirrors the bart/marian
sliding-window patches in ``hf/bart.py`` / ``hf/marian.py``.

The patched forward falls back to stock HF behavior when ``position_id``
is not set, so any non-TrOCR ``nn.Embedding`` instance (and a TrOCR
embedding used outside this exporter) is unaffected.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from optimum.exporters.onnx.model_patcher import PatchingSpec
from optimum.utils import NormalizedConfig
from transformers import VisionEncoderDecoderModel
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from ..winml.encoder_decoder import EncoderDecoderInputGenerator
from ..winml.kv_cache import PastKeyValueInputGenerator
from .decoder_wrapper import WinMLDecoderWrapper, WinMLStaticCacheDecoderIOConfig


logger = logging.getLogger(__name__)


# =============================================================================
# Positional-embedding patch (side-channel)
# =============================================================================


def _patched_tocr_learned_positional_embedding_forward(
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
    # Fallback: bit-identical to stock HF behavior.
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


def _build_tocr_patching_specs() -> list[PatchingSpec]:
    """Return PatchingSpec list for TrOCR, or [] if the target class is unavailable."""
    try:
        from transformers.models.trocr.modeling_trocr import TrOCRLearnedPositionalEmbedding
    except ImportError:
        logger.debug("TrOCRLearnedPositionalEmbedding not found; learned-embedding patch skipped.")
        return []
    return [
        PatchingSpec(
            o=TrOCRLearnedPositionalEmbedding,
            name="forward",
            custom_op=_patched_tocr_learned_positional_embedding_forward,
        ),
    ]


# =============================================================================
# IOConfig + dummy generators
# =============================================================================


class TocrDecoderInputGenerator(EncoderDecoderInputGenerator):
    """Dummy input generator for TrOCR decoder export."""

    def __init__(self, task: str, normalized_config: Any, **kwargs: Any) -> None:
        super().__init__(task, normalized_config, **kwargs)
        self.enc_seq = (normalized_config.image_size // normalized_config.patch_size) ** 2 + 1
        self.d_model = normalized_config.encoder_hidden_size


class _TocrDecoderNormalizedConfig(NormalizedConfig):
    """NormalizedConfig for TrOCR — nested ``decoder.*`` and ``encoder.*`` paths."""

    VOCAB_SIZE = "decoder.vocab_size"
    HIDDEN_SIZE = "decoder.d_model"
    NUM_LAYERS = "decoder.decoder_layers"
    NUM_ATTENTION_HEADS = "decoder.decoder_attention_heads"
    MAX_CACHE_LEN = "decoder.max_position_embeddings"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def encoder_hidden_size(self) -> int:
        return self.config.encoder.hidden_size

    @property
    def image_size(self) -> int:
        return self.config.encoder.image_size

    @property
    def patch_size(self) -> int:
        return self.config.encoder.patch_size


class TocrDecoderIOConfig(WinMLStaticCacheDecoderIOConfig):
    """ONNX config for TrOCR decoder with static KV cache.

    Inputs:  decoder_input_ids, encoder_hidden_states, decoder_attention_mask,
             cache_position, past_{i}_key / past_{i}_value
    Outputs: logits, present_{i}_key / present_{i}_value
    """

    NORMALIZED_CONFIG_CLASS = _TocrDecoderNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        TocrDecoderInputGenerator,
        PastKeyValueInputGenerator,
    )
    PATCHING_SPECS = _build_tocr_patching_specs()

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
# Wrapper
# =============================================================================


class TocrDecoderWrapper(WinMLDecoderWrapper):
    """Static-KV-cache decoder export for TrOCR (VED with TrOCR inner causal-LM)."""

    _HF_MODEL_CLS = VisionEncoderDecoderModel
    _IO_CONFIG_CLS = TocrDecoderIOConfig

    def _make_cache(self, inputs: dict[str, torch.Tensor]) -> Any:
        cache = super()._make_cache(inputs)
        # ``TrOCRDecoder.forward`` reads ``past_key_values.get_seq_length()`` and
        # threads it as ``past_key_values_length`` into the causal-mask prep.
        # Override so the mask shape reflects the actual generation step.
        position = inputs["cache_position"].squeeze()
        cache.get_seq_length = lambda layer_idx=0: position
        return cache

    def _invoke_hf(self, cache: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Side-channel the absolute seq pos to the (patched) learned embedding.
        # Stock HF derives positions via ``torch.arange(past_kv_len, ...)`` in
        # ``TrOCRLearnedPositionalEmbedding.forward``; the patch reads this
        # attribute and does a plain Embedding lookup instead — avoiding the
        # ``Range`` op.  See module docstring.
        decoder = self.model.decoder.model.decoder  # VED -> TrOCRForCausalLM -> TrOCRDecoder
        decoder.embed_positions.position_id = inputs["cache_position"]

        outputs = self.model.decoder(
            input_ids=inputs["decoder_input_ids"],
            attention_mask=inputs["decoder_attention_mask"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            encoder_attention_mask=None,  # vision encoders have no padding
            past_key_values=EncoderDecoderCache(cache, DynamicCache()),
            use_cache=True,
            cache_position=inputs["cache_position"],
            return_dict=True,
        )
        return outputs.logits


__all__ = [
    "TocrDecoderIOConfig",
    "TocrDecoderInputGenerator",
    "TocrDecoderWrapper",
]
