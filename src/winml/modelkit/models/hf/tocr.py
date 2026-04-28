# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""TrOCR decoder export — vision-encoder-decoder with TrOCR inner causal-LM.

Registered for ``("vision-encoder-decoder", "text2text-generation")``;
the ``VisionDecoderWrapper`` dispatcher routes to ``TocrDecoderWrapper``
when ``config.decoder.model_type == "trocr"``.

Models: microsoft/trocr-base-printed, microsoft/trocr-large-*, etc.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn as nn
from optimum.exporters.onnx.model_patcher import ModelPatcher
from optimum.utils import NormalizedConfig
from transformers import VisionEncoderDecoderModel
from transformers.cache_utils import DynamicCache, EncoderDecoderCache


if TYPE_CHECKING:
    from optimum.exporters.onnx import OnnxConfig

from ..winml.encoder_decoder import EncoderDecoderInputGenerator
from ..winml.kv_cache import PastKeyValueInputGenerator
from .decoder_wrapper import WinMLDecoderWrapper, WinMLStaticCacheDecoderIOConfig


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
        # Without this, embed_positions reads cache.get_seq_length() (= buffer
        # length) and indexes the position table out of range.
        position = inputs["cache_position"].squeeze()
        cache.get_seq_length = lambda layer_idx=0: position
        return cache

    def _invoke_hf(self, cache: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
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


# =============================================================================
# Internal: TrOCR static-shape patches for ONNX export
# =============================================================================
# TrOCRAttention reshapes Q/K/V to ``(bsz * num_heads, ...)`` and
# TrOCRLearnedPositionalEmbedding builds positions via ``torch.arange`` —
# both produce symbolic shapes that NPU compilers reject.  The patches below
# replace both with static-shape equivalents only during ``torch.onnx.export``.


def _patched_tocr_attention_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    attention_mask: torch.Tensor | None = None,
    layer_head_mask: torch.Tensor | None = None,
    output_attentions: bool = False,
    cache_position: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """4-D matmul replacement for TrOCR attention; assumes ``bsz == 1``."""
    assert hidden_states.size(0) == 1, (
        "TrOCR static-shape patch assumes batch_size=1 (WinML split-export contract)."
    )
    is_cross_attention = key_value_states is not None
    bsz = 1
    tgt_len = hidden_states.size(1)

    q = (
        (self.q_proj(hidden_states) * self.scaling)
        .view(bsz, tgt_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    is_updated = False
    curr_past_key_value = None
    if past_key_values is not None:
        if isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            curr_past_key_value = (
                past_key_values.cross_attention_cache
                if is_cross_attention
                else past_key_values.self_attention_cache
            )
        else:
            curr_past_key_value = past_key_values

    if is_cross_attention and curr_past_key_value is not None and is_updated:
        k = curr_past_key_value.layers[self.layer_idx].keys
        v = curr_past_key_value.layers[self.layer_idx].values
    else:
        current_states = key_value_states if is_cross_attention else hidden_states
        k = (
            self.k_proj(current_states)
            .view(bsz, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(current_states)
            .view(bsz, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        if curr_past_key_value is not None:
            cache_pos = cache_position if not is_cross_attention else None
            k, v = curr_past_key_value.update(
                k, v, self.layer_idx, {"cache_position": cache_pos}
            )
            if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                past_key_values.is_updated[self.layer_idx] = True

    attn_weights = torch.matmul(q, k.transpose(-2, -1))
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    if layer_head_mask is not None:
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights
    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = (
        torch.matmul(attn_probs, v)
        .transpose(1, 2)
        .reshape(bsz, tgt_len, self.embed_dim)
    )
    attn_output = self.out_proj(attn_output)
    return attn_output, (attn_weights if output_attentions else None)


def _patched_tocr_learned_positional_embedding_forward(
    self,
    input_ids: torch.Tensor,
    past_key_values_length: Any = 0,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Static-shape replacement for ``TrOCRLearnedPositionalEmbedding.forward``."""
    if position_ids is None:
        if isinstance(past_key_values_length, torch.Tensor):
            position_ids = past_key_values_length.reshape(1, 1)
        else:
            position_ids = torch.full(
                (1, 1),
                int(past_key_values_length),
                dtype=torch.long,
                device=self.weight.device,
            )
    else:
        position_ids = position_ids.unsqueeze(0)
    return nn.Embedding.forward(self, position_ids + self.offset)


class _TocrStaticShapePatcher(ModelPatcher):
    """Applies TrOCR static-shape patches during ``torch.onnx.export``."""

    _ATTN_REQUIRED: ClassVar[tuple[str, ...]] = (
        "q_proj", "k_proj", "v_proj", "out_proj",
        "num_heads", "head_dim", "embed_dim",
    )
    _POS_REQUIRED: ClassVar[tuple[str, ...]] = (
        "offset", "weight", "num_embeddings", "embedding_dim",
    )

    def __init__(
        self,
        config: OnnxConfig,
        model: nn.Module,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config, model, model_kwargs=model_kwargs)
        self._patched: list[tuple[nn.Module, Any]] = []

    def __enter__(self):
        super().__enter__()
        for _name, module in self._model.named_modules():
            patch_fn: Any = None
            if all(hasattr(module, a) for a in self._ATTN_REQUIRED):
                patch_fn = _patched_tocr_attention_forward
            elif isinstance(module, nn.Embedding) and all(
                hasattr(module, a) for a in self._POS_REQUIRED
            ):
                patch_fn = _patched_tocr_learned_positional_embedding_forward
            if patch_fn is not None:
                self._patched.append((module, module.forward))
                module.forward = types.MethodType(patch_fn, module)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module, original in self._patched:
            module.forward = original
        self._patched.clear()
        super().__exit__(exc_type, exc_val, exc_tb)


TocrDecoderIOConfig._MODEL_PATCHER = _TocrStaticShapePatcher


__all__ = [
    "TocrDecoderIOConfig",
    "TocrDecoderInputGenerator",
    "TocrDecoderWrapper",
]
