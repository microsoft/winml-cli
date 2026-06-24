# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""WinML Model for Zero-Shot Image Classification.

Split-encoder composite wrapper for dual-encoder families (CLIP, SigLIP, …).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
from transformers.utils.generic import ModelOutput

from .composite_model import WinMLCompositeModel, register_composite_model


logger = logging.getLogger(__name__)


@dataclass
class ZeroShotImageClassifierOutput(ModelOutput):
    """Output container for dual-encoder zero-shot image classification."""

    logits_per_image: torch.Tensor | None = None
    logits_per_text: torch.Tensor | None = None
    text_embeds: torch.Tensor | None = None
    image_embeds: torch.Tensor | None = None


# Sub-encoder ONNX output names we accept, in priority order. The first name
# present in the session outputs wins. CLIP's WithProjection classes expose
# ``image_embeds`` / ``text_embeds`` natively; SigLIP (no WithProjection
# variants) exposes the projected embedding under ``pooler_output``.
_EMBED_OUTPUT_KEYS: tuple[str, ...] = ("image_embeds", "text_embeds", "pooler_output")


@register_composite_model("clip", "zero-shot-image-classification")
@register_composite_model("siglip", "zero-shot-image-classification")
class WinMLModelForZeroShotImageClassification(WinMLCompositeModel):
    """WinML model for zero-shot image classification.

    Supports dual-encoder families: CLIP, SigLIP.

    Thin composite wrapper - orchestrates two sub-encoders (``image-encoder``
    and ``text-encoder``) and combines their projected embeddings into
    ``logits_per_image``.
    """

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "image-encoder": "image-feature-extraction",
        "text-encoder": "feature-extraction",
    }

    def __init__(
        self,
        sub_models: dict[str, Any],
        config: Any = None,
    ) -> None:
        super().__init__(sub_models, config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ZeroShotImageClassifierOutput:
        """Run split-encoder zero-shot image classification.

        Returns:
            ZeroShotImageClassifierOutput with:

            - ``logits_per_image`` — cosine similarity ``[B, N]`` between each of ``B``
              images and ``N`` candidate text classes. ``logit_scale`` / ``logit_bias``
              are not applied.
            - ``logits_per_text`` — transpose of ``logits_per_image``, shape ``[N, B]``.
            - ``image_embeds`` — L2-normalized projected image embeddings ``[B, D]``.
            - ``text_embeds`` — L2-normalized projected text embeddings ``[N, D]``.
        """
        image_embeds = self._run_vision(self._preprocess_vision(pixel_values))
        text_embeds = self._run_text(self._preprocess_text(input_ids, attention_mask))

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-9)

        logits_per_image = image_embeds @ text_embeds.T

        return ZeroShotImageClassifierOutput(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_image.T,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
        )

    def _preprocess_vision(self, pixel_values: torch.Tensor | None) -> dict[str, np.ndarray]:
        """Torch→numpy via the sub-model's formatter."""
        return self.sub_models["image-encoder"]._format_inputs(pixel_values=pixel_values)

    def _run_vision(self, inputs: dict[str, np.ndarray]) -> torch.Tensor:
        """Run vision encoder over ``M`` images, batching per the ONNX's fixed batch dim."""
        return self._run_batched(self.sub_models["image-encoder"], inputs, "pixel_values")

    def _preprocess_text(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> dict[str, np.ndarray]:
        """Torch→numpy + pad/truncate text inputs to the ONNX's fixed seq_len."""
        text = self.sub_models["text-encoder"]
        raw: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            raw["attention_mask"] = attention_mask
        inputs = text._format_inputs(**raw)
        expected = text.io_config["input_shapes"][0][-1]
        return {k: self._pad_or_truncate(v, expected).astype(np.int64) for k, v in inputs.items()}

    def _run_text(self, inputs: dict[str, np.ndarray]) -> torch.Tensor:
        """Run text encoder over ``N`` texts, batching per the ONNX's fixed batch dim."""
        return self._run_batched(self.sub_models["text-encoder"], inputs, "input_ids")

    def _run_batched(
        self,
        sub_model: Any,
        inputs: dict[str, np.ndarray],
        leading_key: str,
    ) -> torch.Tensor:
        """Run ``sub_model``'s ONNX session over ``inputs`` in chunks of its fixed batch size.

        The ONNX model typically has a fixed batch size ``B``. We iterate in
        chunks of ``B``; when the leading dim ``N`` isn't a multiple of ``B``,
        the final chunk is zero-padded to ``B`` and the padding rows are
        stripped from the output. Dynamic batch dims (``B`` non-positive /
        non-int) fall back to a single call with the full input.

        ``leading_key`` names the input whose leading dim defines ``N``.
        """
        batch_size = sub_model.io_config["input_shapes"][0][0]
        n = inputs[leading_key].shape[0]

        if not isinstance(batch_size, int) or batch_size <= 0:
            return torch.from_numpy(self._pick_embeds(sub_model._session.run(inputs)))

        all_embeds = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = {k: v[start:end] for k, v in inputs.items()}
            pad = batch_size - (end - start)
            if pad:
                chunk = {
                    k: np.pad(v, ((0, pad), *([(0, 0)] * (v.ndim - 1))), constant_values=0)
                    for k, v in chunk.items()
                }
            embeds = self._pick_embeds(sub_model._session.run(chunk))
            if pad:
                embeds = embeds[: batch_size - pad]
            all_embeds.append(embeds)
        return torch.from_numpy(np.concatenate(all_embeds, axis=0))

    @staticmethod
    def _pad_or_truncate(arr: np.ndarray, target_len: int | None) -> np.ndarray:
        # No padding/truncation when the ONNX model has a dynamic sequence-length dim.
        if not isinstance(target_len, int) or target_len <= 0:
            return arr
        if arr.shape[1] < target_len:
            pad_width = target_len - arr.shape[1]
            return np.pad(arr, ((0, 0), (0, pad_width)), constant_values=0)
        if arr.shape[1] > target_len:
            logger.warning(
                "Truncating text input from %d to %d tokens to match the ONNX model's "
                "fixed sequence length; trailing prompt tokens will not be seen by the model.",
                arr.shape[1],
                target_len,
            )
            return arr[:, :target_len]
        return arr

    @staticmethod
    def _pick_embeds(outputs: dict[str, np.ndarray]) -> np.ndarray:
        """Pick projected embedding from an ONNX output dict by priority list."""
        for key in _EMBED_OUTPUT_KEYS:
            if key in outputs:
                return outputs[key]
        raise KeyError(
            f"None of {_EMBED_OUTPUT_KEYS} found in ONNX outputs. "
            f"Available: {list(outputs)}",
        )
