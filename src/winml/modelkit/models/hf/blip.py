# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""BLIP HuggingFace Model Configuration — split image-to-text export.

Export strategy (one ONNX per sub-component):

- ``BlipVisionEncoderWrapper`` + ``BlipVisionEncoderIOConfig``
  (task ``feature-extraction``) — encoder-only ONNX
  (``pixel_values → encoder_hidden_states``).

- ``BlipDecoderWrapper`` + ``BlipDecoderIOConfig``
  (task ``text2text-generation``) — decoder ONNX with a static KV cache:
  full per-layer KV buffers as inputs, new-token K/V as outputs.

The decoder wrapper follows the three-step adapter documented in
``DECODER_KV_CACHE_EXPORT_SPEC.md``.  All transformer math stays inside
HF's ``BlipTextLMHeadModel``; the wrapper only plumbs KV tensors.

BLIP-specific trace-time adjustments:

- **3-D ``decoder_attention_mask``** — ``BlipTextModel.get_extended_attention_mask``
  has a ``dim == 3`` branch that broadcasts our mask without reconstructing a
  causal triangle.  Passing a ``[1, 1, max_cache_len]`` mask routes through
  that branch.
- **Explicit ``position_ids``** — ``BlipTextEmbeddings`` would otherwise
  derive positions from ``past_key_values_length`` (which traces as 0 for a
  static cache), baking the wrong position into the embedding lookup.
  Supplying ``position_ids = cache_position.unsqueeze(0)`` fixes that.

Model: Salesforce/blip-image-captioning-base, *-large, etc.
Pipeline task: ``image-to-text``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyVisionInputGenerator
from transformers import BlipForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from ...config import WinMLBuildConfig
from ...export import MaxLengthTextInputGenerator, register_onnx_overwrite
from ...optim import WinMLOptimizationConfig
from ..winml.composite_model import register_composite_model
from ..winml.encoder_decoder import EncoderDecoderInputGenerator, WinMLEncoderDecoderModel
from ..winml.kv_cache import PastKeyValueInputGenerator, WinMLStaticCache
from .decoder_wrapper import WinMLDecoderWrapper, WinMLStaticCacheDecoderIOConfig


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig


# =============================================================================
# WinML Build Config
# =============================================================================

BLIP_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        gelu_fusion=True,
        layer_norm_fusion=True,
        matmul_add_fusion=True,
    ),
)


# =============================================================================
# Monolithic ONNX Export Config (single-file image-to-text export)
# =============================================================================
#
# TODO: remove once ``winml export`` supports composite models. This monolithic
# config is a temporary fallback so ``winml export`` can produce a single ONNX
# for BLIP at ``image-to-text`` / ``image-text-to-text``. The split encoder +
# decoder configs above are the production path used by the composite pipeline
# (``winml config`` / ``winml build``); when ``winml export`` learns to emit
# multiple ONNX files for composite models, delete this class. See issue #636.


@register_onnx_overwrite("blip", "image-to-text", library_name="transformers")
@register_onnx_overwrite("blip", "image-text-to-text", library_name="transformers")
class BlipCaptioningIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """Monolithic ONNX config for BLIP captioning — single-graph fallback.

    Traces ``BlipForConditionalGeneration.forward`` with pixel_values +
    decoder input_ids/attention_mask -> logits. Only used by ``winml export``
    (single-file path); the composite encoder + decoder split below is the
    production export route used by ``winml config`` / ``winml build``.
    """

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="vision_config.num_channels",
        image_size="vision_config.image_size",
        vocab_size="text_config.vocab_size",
        sequence_length="text_config.max_position_embeddings",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyVisionInputGenerator,
        MaxLengthTextInputGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return {
            "pixel_values": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {
            "logits": {0: "batch_size", 1: "sequence_length"},
        }


# =============================================================================
# Encoder
# =============================================================================


class BlipVisionEncoderWrapper(nn.Module):
    """Wraps BLIP's ``vision_model`` for encoder-only ONNX export."""

    def __init__(self, vision_model: nn.Module, config: Any) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> BlipVisionEncoderWrapper:
        """Load full ``BlipForConditionalGeneration`` and wrap its vision tower."""
        full = BlipForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        wrapper = cls(full.vision_model, full.config)
        wrapper.eval()
        return wrapper

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Trace ``pixel_values → encoder_hidden_states``."""
        # self.vision_model is a torch submodule (untyped __call__ -> Any).
        return cast("torch.Tensor", self.vision_model(pixel_values=pixel_values).last_hidden_state)


@register_onnx_overwrite("blip", "feature-extraction", library_name="transformers")
class BlipVisionEncoderIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for the BLIP vision encoder.

    ``image-feature-extraction`` is a synonym that Optimum's TasksManager
    maps to ``feature-extraction`` at lookup time, so we register under the
    canonical name.  The composite's sub-task uses the ``image-…`` form so
    quantisation picks ``ImageDataset`` for calibration.
    """

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="vision_config.num_channels",
        image_size="vision_config.image_size",
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
# Decoder
# =============================================================================


class BlipDecoderInputGenerator(EncoderDecoderInputGenerator):
    """Dummy input generator for BLIP decoder export.

    Reads the vision sequence length from the image/patch grid (vs the default
    text length 16), and the cross-attn K/V projection width from the vision
    encoder's ``hidden_size``.
    """

    def __init__(self, task: str, normalized_config: Any, **kwargs: Any) -> None:
        super().__init__(task, normalized_config, **kwargs)
        image_size = normalized_config.image_size
        patch_size = normalized_config.patch_size
        self.enc_seq = (image_size // patch_size) ** 2 + 1
        self.d_model = normalized_config.vision_hidden_size


@register_onnx_overwrite("blip", "text2text-generation", library_name="transformers")
class BlipDecoderIOConfig(WinMLStaticCacheDecoderIOConfig):
    """ONNX config for the BLIP text decoder with static KV cache.

    Inputs:  decoder_input_ids, encoder_hidden_states, decoder_attention_mask,
             cache_position, past_{i}_key / past_{i}_value
    Outputs: logits, present_{i}_key / present_{i}_value
    """

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        hidden_size="text_config.hidden_size",
        num_layers="text_config.num_hidden_layers",
        num_attention_heads="text_config.num_attention_heads",
        max_cache_len="text_config.max_position_embeddings",
        vocab_size="text_config.vocab_size",
        vision_hidden_size="vision_config.hidden_size",
        image_size="vision_config.image_size",
        patch_size="vision_config.patch_size",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (
        BlipDecoderInputGenerator,
        PastKeyValueInputGenerator,
    )

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        nc = self._normalized_config
        nc.head_dim = nc.hidden_size // nc.num_attention_heads

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


class BlipDecoderWrapper(WinMLDecoderWrapper):
    """BLIP text decoder export — see module docstring for trace-time notes."""

    _HF_MODEL_CLS = BlipForConditionalGeneration
    _IO_CONFIG_CLS = BlipDecoderIOConfig

    def _invoke_hf(self, cache: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        encoder_hidden_states = inputs["encoder_hidden_states"]
        # Vision tokens have no padding — all-ones mask traces as a Constant.
        enc_mask = torch.ones(
            encoder_hidden_states.size()[:-1],
            dtype=torch.long,
            device=encoder_hidden_states.device,
        )
        decoder_mask = (1 - inputs["decoder_attention_mask"]).to(dtype=encoder_hidden_states.dtype)
        decoder_mask = decoder_mask.unsqueeze(1).unsqueeze(1)
        decoder_mask = decoder_mask * torch.finfo(encoder_hidden_states.dtype).min
        # self.model is nn.Module; torch's __getattr__ types text_decoder as
        # Tensor | Module, so narrow to a callable Module.
        outputs = cast("nn.Module", self.model.text_decoder)(
            input_ids=inputs["decoder_input_ids"],
            # HF's causal-mask reconstruction traces as ops the NPU analyzer
            # doesn't support; pass an additive 4-D mask to bypass reconstruction.
            attention_mask=decoder_mask,
            # Without explicit position_ids, BlipTextModel would derive them
            # from past_kv_len=0 (a frozen constant in the trace), giving every
            # step position 0 instead of the actual step index.
            position_ids=inputs["cache_position"].unsqueeze(0),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=enc_mask,
            past_key_values=EncoderDecoderCache(cache, DynamicCache()),
            use_cache=True,
            cache_position=inputs["cache_position"],
            return_dict=True,
        )
        return cast("torch.Tensor", outputs.logits)


# =============================================================================
# Inference composite model
# =============================================================================


@register_composite_model("blip", "image-to-text")
class WinMLBlipImageToText(WinMLEncoderDecoderModel):
    """BLIP image-to-text inference model."""

    main_input_name = "pixel_values"

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "encoder": "image-feature-extraction",
        "decoder": "text2text-generation",
    }

    def __init__(
        self,
        sub_models: dict[str, Any],
        config: PretrainedConfig,
        device: str = "cpu",
    ) -> None:
        super().__init__(sub_models, config, device)
        # BLIP defaults ``is_encoder_decoder`` to False because it ships a
        # custom ``generate()``.  We always go through HF's standard
        # encoder-decoder path, so flip the flag on.
        self.config.is_encoder_decoder = True
        # WinMLCache reads config.num_hidden_layers; BLIP nests it under text_config.
        self.config.num_hidden_layers = config.text_config.num_hidden_layers

    @classmethod
    def get_cache_class(cls) -> type:  # noqa: D102
        # BLIP's text decoder uses absolute position embeddings;
        # ``WinMLStaticCache`` preserves ``buffer_idx == seq_pos``.
        return WinMLStaticCache

    @property
    def generation_config(self) -> GenerationConfig:  # noqa: D102
        if not hasattr(self, "_generation_config"):
            from transformers import GenerationConfig

            tc = self.config.text_config
            bos = tc.bos_token_id
            kw: dict[str, Any] = {
                # BLIP doesn't declare decoder_start_token_id — fall back to bos.
                "decoder_start_token_id": bos,
                "bos_token_id": bos,
                # BLIP's real terminator is sep_token_id; the declared eos_token_id
                # points to a BERT [unused] slot the model never emits.
                "eos_token_id": tc.sep_token_id,
                "pad_token_id": tc.pad_token_id,
            }
            kw.setdefault("max_new_tokens", self._max_dec - 1)
            kw.setdefault("num_beams", 1)  # static batch=1 ONNX → no beams
            kw.setdefault("do_sample", False)  # deterministic greedy
            self._generation_config = GenerationConfig(**kw)
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value: Any) -> None:
        self._generation_config = value


# =============================================================================
# Model Class Mapping
# =============================================================================

# ``image-feature-extraction`` is normalized to ``feature-extraction`` by
# Optimum's TasksManager before this lookup, so the encoder key uses the
# normalized task name.
MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("blip", "feature-extraction"): BlipVisionEncoderWrapper,
    ("blip", "text2text-generation"): BlipDecoderWrapper,
}


__all__ = [
    "BLIP_CONFIG",
    "MODEL_CLASS_MAPPING",
    "BlipDecoderIOConfig",
    "BlipDecoderInputGenerator",
    "BlipDecoderWrapper",
    "BlipVisionEncoderIOConfig",
    "BlipVisionEncoderWrapper",
    "WinMLBlipImageToText",
]
