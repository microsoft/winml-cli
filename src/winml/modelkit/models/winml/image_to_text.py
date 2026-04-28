# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML inference classes for image-to-text composite models.

One concrete class per family — each reads its own family's config conventions
directly, with no walker or per-family fallback.

Hierarchy::

    WinMLEncoderDecoderModel (GenerationMixin) — encoder-decoder generation loop
      └─ _ImageToTextBase                       — universal image-to-text scaffolding
           ├─ WinMLBlipImageToText             — BLIP
           └─ WinMLVEDImageToText              — vision-encoder-decoder (TrOCR / Donut / ...)

The base class holds only the truly universal pieces (input name, sub-task
mapping, static-cache class, static-batch generation flags).  Subclasses
contribute the two family-specific parts: where ``num_hidden_layers`` lives in
the config tree, and which token IDs go into the ``GenerationConfig``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from .composite_model import register_composite_model
from .encoder_decoder import WinMLEncoderDecoderModel
from .kv_cache import WinMLStaticCache


if TYPE_CHECKING:
    from transformers import PretrainedConfig


class _ImageToTextBase(WinMLEncoderDecoderModel):
    """Universal scaffolding shared by every image-to-text family."""

    # GenerationMixin routes ``main_input_name`` into the encoder.
    main_input_name = "pixel_values"

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        # ``image-feature-extraction`` is a TasksManager synonym of the
        # canonical ``feature-extraction`` IOConfig task.  The pre-normalisation
        # name flows into ``quant.task`` so calibration picks ``ImageDataset``.
        "encoder": "image-feature-extraction",
        "decoder": "text2text-generation",
    }

    def __init__(self, sub_models: dict[str, Any], config: PretrainedConfig) -> None:
        super().__init__(sub_models, config)
        # Multimodal configs (BLIP, Pix2Struct) default ``is_encoder_decoder``
        # to False because they ship a custom ``generate()``.  We always go
        # through HF's standard encoder-decoder path, so flip the flag on.
        self.config.is_encoder_decoder = True

    @classmethod
    def get_cache_class(cls) -> type:
        # BLIP and BART-family decoders use absolute position embeddings;
        # ``WinMLStaticCache`` preserves ``buffer_idx == seq_pos``.
        return WinMLStaticCache

    @property
    def generation_config(self):
        if not hasattr(self, "_generation_config"):
            from transformers import GenerationConfig

            kw = self._build_gen_kwargs()
            kw.setdefault("max_new_tokens", self._max_dec - 1)
            kw.setdefault("num_beams", 1)        # static batch=1 ONNX → no beams
            kw.setdefault("do_sample", False)    # deterministic greedy
            self._generation_config = GenerationConfig(**kw)
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value: Any) -> None:
        self._generation_config = value

    def _build_gen_kwargs(self) -> dict[str, Any]:
        """Family-specific token IDs for ``GenerationConfig``."""
        raise NotImplementedError


@register_composite_model("blip", "image-to-text")
class WinMLBlipImageToText(_ImageToTextBase):
    """BLIP image-to-text inference model."""

    def __init__(self, sub_models: dict[str, Any], config: PretrainedConfig) -> None:
        super().__init__(sub_models, config)
        # WinMLCache reads config.num_hidden_layers; BLIP nests it under text_config.
        self.config.num_hidden_layers = config.text_config.num_hidden_layers

    def _build_gen_kwargs(self) -> dict[str, Any]:
        tc = self.config.text_config
        bos = tc.bos_token_id
        return {
            # BLIP doesn't declare decoder_start_token_id — fall back to bos.
            "decoder_start_token_id": bos,
            "bos_token_id": bos,
            # BLIP's real terminator is sep_token_id; the declared eos_token_id
            # points to a BERT [unused] slot the model never emits.
            "eos_token_id": tc.sep_token_id,
            "pad_token_id": tc.pad_token_id,
        }


@register_composite_model("vision-encoder-decoder", "image-to-text")
class WinMLVEDImageToText(_ImageToTextBase):
    """Vision-encoder-decoder image-to-text inference model (TrOCR, Donut, ...)."""

    @classmethod
    def get_sub_model_config(
        cls, hf_config: PretrainedConfig | None = None
    ) -> dict[str, str] | None:
        """Split-build only when the inner decoder family is supported.

        Non-supported inner decoders (e.g. ``bert`` for manga-ocr, ``gpt2``
        for vit-gpt2-image-captioning) fall through to monolithic build.
        """
        from ..hf.vision_encoder_decoder import _INNER_DECODER_REGISTRY

        decoder = getattr(hf_config, "decoder", None)
        if decoder is not None and decoder.model_type in _INNER_DECODER_REGISTRY:
            return super().get_sub_model_config(hf_config)
        return None

    def __init__(self, sub_models: dict[str, Any], config: PretrainedConfig) -> None:
        super().__init__(sub_models, config)
        # WinMLCache reads config.num_hidden_layers; VED nests it under decoder.
        self.config.num_hidden_layers = config.decoder.num_hidden_layers

    def _build_gen_kwargs(self) -> dict[str, Any]:
        dc = self.config.decoder
        return {
            "decoder_start_token_id": dc.decoder_start_token_id,
            "bos_token_id": dc.bos_token_id,
            "eos_token_id": dc.eos_token_id,
            "pad_token_id": dc.pad_token_id,
        }


__all__ = [
    "WinMLBlipImageToText",
    "WinMLVEDImageToText",
]
