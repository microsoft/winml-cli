# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Specialized image-to-text inference support."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from .base import WinMLPreTrainedModel


def _convert_images_to_rgb(images: Any) -> Any:
    """Normalize supported image inputs to RGB before processor conversion."""
    from transformers.image_transforms import to_pil_image
    from transformers.image_utils import load_image

    if isinstance(images, list | tuple):
        return [_convert_images_to_rgb(image) for image in images]
    if isinstance(images, np.ndarray | torch.Tensor) and images.ndim == 4:
        return [_convert_images_to_rgb(image) for image in images]
    if isinstance(images, str):
        try:
            return load_image(images).convert("RGB")
        except ValueError:
            return images
    if isinstance(images, Image.Image):
        return images.convert("RGB")
    if isinstance(images, np.ndarray):
        if images.ndim == 2:
            return Image.fromarray(images).convert("RGB")
        return to_pil_image(images).convert("RGB")
    if isinstance(images, torch.Tensor):
        if images.ndim == 2:
            images = images.unsqueeze(0)
        return to_pil_image(images).convert("RGB")
    return images


class WinMLModelForMgpstrSceneTextRecognition(WinMLPreTrainedModel):
    """Expose MGP-STR's ordered ONNX heads in its Transformers output contract."""

    main_input_name = "pixel_values"

    def forward(self, **kwargs: Any) -> Any:
        """Run the graph and return character, BPE, then WordPiece logits."""
        from transformers.models.mgp_str.modeling_mgp_str import MgpstrModelOutput

        try:
            pixel_values = kwargs.pop("pixel_values")
        except KeyError as exc:
            raise ValueError("MGP-STR inference requires 'pixel_values'.") from exc

        outputs = self._run_inference(self._format_inputs(pixel_values=pixel_values))
        return MgpstrModelOutput(
            logits=cast(
                "Any",
                (
                    outputs["char_logits"],
                    outputs["bpe_logits"],
                    outputs["wp_logits"],
                ),
            )
        )


class MgpstrImageToTextPipeline:
    """Preprocess and decode MGP-STR's character, BPE, and WordPiece heads."""

    def __init__(
        self,
        model: WinMLPreTrainedModel,
        model_id: str,
        *,
        device: str = "cpu",
        trust_remote_code: bool = False,
    ) -> None:
        from transformers import AutoProcessor

        processor_kwargs = {"trust_remote_code": True} if trust_remote_code else {}
        self.model = model
        self.processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
        self.image_processor = self.processor.image_processor
        self.tokenizer = getattr(self.processor, "char_tokenizer", None)
        self.device = device
        self._preprocess_params: dict[str, Any] = {}

    def _sanitize_parameters(self, **_kwargs: Any) -> tuple[dict, dict, dict]:
        """Expose the pipeline parameter-introspection contract."""
        return {}, {}, {}

    def __call__(self, images: Any, *, prompt: str | None = None, **_kwargs: Any) -> Any:
        """Recognize text in one image or a batch of images."""
        if prompt is not None:
            raise ValueError("MGP-STR scene text recognition does not accept a text prompt.")

        model_inputs = self.processor(
            images=_convert_images_to_rgb(images),
            return_tensors="pt",
        )
        outputs = self.model(**model_inputs)
        if isinstance(outputs, Mapping) and all(
            name in outputs for name in ("char_logits", "bpe_logits", "wp_logits")
        ):
            logits = tuple(outputs[name] for name in ("char_logits", "bpe_logits", "wp_logits"))
        elif isinstance(outputs, Mapping):
            logits = outputs["logits"]
        else:
            logits = outputs.logits
        decoded = self.processor.batch_decode(logits)

        records = []
        for index, text in enumerate(decoded["generated_text"]):
            record: dict[str, Any] = {"generated_text": text}
            if "scores" in decoded:
                score = decoded["scores"][index]
                record["score"] = float(score.item() if hasattr(score, "item") else score)
            records.append(record)
        return records
