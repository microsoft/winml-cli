# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Vision-encoder-decoder export — generic encoder + decoder dispatcher.

- ``VisionEncoderWrapper`` / ``VisionEncoderIOConfig`` — encoder side, generic
  for any VED inner architecture (ViT/DeiT/Swin/...).
- ``VisionDecoderWrapper`` — dispatcher; routes by ``config.decoder.model_type``
  to a family-specific wrapper in ``_INNER_DECODER_REGISTRY``.

Pipeline task: ``image-to-text``.  Family wrappers (e.g., ``TocrDecoderWrapper``
in ``tocr.py``) own the actual KV-cache export logic.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyVisionInputGenerator
from transformers import VisionEncoderDecoderModel

from ...config import WinMLBuildConfig
from ...export import register_onnx_overwrite
from ...optim import WinMLOptimizationConfig
from .tocr import TocrDecoderIOConfig, TocrDecoderWrapper


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
# Decoder dispatcher — wrapper + IOConfig
# =============================================================================


# Inner decoder ``model_type`` → (family wrapper, family IOConfig).
_INNER_DECODER_REGISTRY: dict[str, tuple[type, type]] = {
    "trocr": (TocrDecoderWrapper, TocrDecoderIOConfig),
}


class VisionDecoderWrapper:
    """Decoder wrapper for VED models.

    The concrete wrapper is determined by ``config.decoder.model_type``
    (e.g. ``trocr``, ``bert``, ``gpt2``); see ``_INNER_DECODER_REGISTRY``
    for currently supported inner decoders.
    """

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> Any:
        """Read inner decoder model_type from config; defer to the family wrapper."""
        from transformers import AutoConfig

        inner_type = AutoConfig.from_pretrained(model_name_or_path).decoder.model_type
        entry = _INNER_DECODER_REGISTRY.get(inner_type)
        if entry is None:
            raise NotImplementedError(
                f"decoder model_type={inner_type!r} is not supported"
            )
        wrapper_cls, _ = entry
        return wrapper_cls.from_pretrained(model_name_or_path, **kwargs)


@register_onnx_overwrite(
    "vision-encoder-decoder", "text2text-generation", library_name="transformers"
)
class VisionDecoderIOConfig(OnnxConfig):
    """Decoder IOConfig for VED models.

    The concrete IOConfig is determined by ``config.decoder.model_type``
    (e.g. ``trocr``, ``bert``, ``gpt2``); see ``_INNER_DECODER_REGISTRY``
    for currently supported inner decoders.
    """

    def __new__(cls, config: Any, *args: Any, **kwargs: Any) -> OnnxConfig:
        if cls is VisionDecoderIOConfig:
            inner_type = config.decoder.model_type
            entry = _INNER_DECODER_REGISTRY.get(inner_type)
            if entry is None:
                raise NotImplementedError(
                    f"decoder model_type={inner_type!r} is not supported"
                )
            _, io_config_cls = entry
            return io_config_cls(config, *args, **kwargs)
        return super().__new__(cls)


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
    "VisionDecoderWrapper",
    "VisionEncoderIOConfig",
    "VisionEncoderWrapper",
]
