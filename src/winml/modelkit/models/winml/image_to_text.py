# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Image-to-text inference wrappers and pipelines."""

from __future__ import annotations

from typing import Any

from .base import WinMLPreTrainedModel


class WinMLModelForMgpstrSceneTextRecognition(WinMLPreTrainedModel):
    """Expose MGP-STR's three ONNX heads in the Transformers output contract."""

    main_input_name = "pixel_values"

    def forward(self, pixel_values: Any, **_kwargs: Any) -> Any:
        """Run the ONNX graph and return its ordered three-head logits tuple."""
        from transformers.models.mgp_str.modeling_mgp_str import MgpstrModelOutput

        outputs = self._run_inference(self._format_inputs(pixel_values=pixel_values))
        return MgpstrModelOutput(
            logits=(
                outputs["char_logits"],
                outputs["bpe_logits"],
                outputs["wp_logits"],
            )
        )


class MgpstrImageToTextPipeline:
    """Preprocess images and decode MGP-STR's character/BPE/WordPiece heads."""

    def __init__(self, model: WinMLPreTrainedModel, model_id: str) -> None:
        from transformers import AutoProcessor

        self.model = model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.image_processor = self.processor.image_processor
        self.tokenizer = getattr(self.processor, "char_tokenizer", None)
        self._preprocess_params: dict[str, Any] = {}

    def _sanitize_parameters(self, **_kwargs: Any) -> tuple[dict, dict, dict]:
        """Match the pipeline introspection contract used by inference clients."""
        return {}, {}, {}

    def __call__(self, images: Any, *, prompt: str | None = None, **_kwargs: Any) -> Any:
        """Recognize text in one image or a batch of images."""
        if prompt is not None:
            raise ValueError("MGP-STR scene text recognition does not accept a text prompt.")

        model_inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model(**model_inputs)
        decoded = self.processor.batch_decode(outputs.logits)

        records = []
        for index, text in enumerate(decoded["generated_text"]):
            record: dict[str, Any] = {"generated_text": text}
            if "scores" in decoded:
                score = decoded["scores"][index]
                record["score"] = float(score.item() if hasattr(score, "item") else score)
            records.append(record)
        return records
