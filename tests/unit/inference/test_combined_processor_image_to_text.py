# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for capability-selected combined image/text preprocessing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature

from winml.modelkit.inference.pipeline import (
    CombinedProcessorImageToTextPipeline,
    PipelineCapability,
    create_pipeline,
)


class RecordingCombinedProcessor:
    """Small processor double with the same batch surface as HF processors."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.tokenizer = RecordingTokenizer()

    def __call__(
        self, *, images: Image.Image, text: str, return_tensors: str
    ) -> BatchFeature:
        self.calls.append(
            {"images": images, "text": text, "return_tensors": return_tensors}
        )
        return BatchFeature(
            {
                "input_ids": torch.tensor([[7, 8]], dtype=torch.long),
                "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }
        )


class RecordingTokenizer:
    """Tokenizer double used by the combined processor."""

    model_max_length = 1

    def decode(self, output_ids: Any, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return f"decoded-{output_ids.tolist()}"


class FakePipeline(SimpleNamespace):
    """Pipeline double that supports tokenizer adaptation."""

    def preprocess(self, inputs: Any, **kwargs: Any) -> Any:
        return inputs


def test_combined_processor_receives_image_and_prompt() -> None:
    processor = RecordingCombinedProcessor()
    pipe = object.__new__(CombinedProcessorImageToTextPipeline)
    pipe.processor = processor
    pipe.framework = "pt"
    pipe.model = SimpleNamespace(dtype=torch.float32)
    image = Image.new("RGB", (2, 2))

    batch = pipe.preprocess(image, prompt="<CAPTION>")

    assert processor.calls == [
        {"images": image, "text": "<CAPTION>", "return_tensors": "pt"}
    ]
    assert set(batch) == {"input_ids", "pixel_values", "attention_mask"}
    assert batch["input_ids"].dtype == torch.long
    assert batch["pixel_values"].dtype == torch.float32


class CapabilityModel:
    """Minimal model surface required by the shared pipeline factory."""

    pipeline_capabilities = frozenset(
        {PipelineCapability.COMBINED_IMAGE_TEXT_PROCESSOR}
    )
    io_config: ClassVar[dict[str, list[list[int]]]] = {
        "input_shapes": [[1, 2], [1, 3, 2, 2]]
    }
    processor = RecordingCombinedProcessor()

    @classmethod
    def create_combined_processor(cls, model_id: str) -> object:
        assert model_id == "local-model"
        return cls.processor


def test_factory_uses_capability_selected_pipeline(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_pipeline(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["tokenizer"] is CapabilityModel.processor.tokenizer
        assert not isinstance(kwargs["tokenizer"], str)
        assert "feature_extractor" not in kwargs
        assert "image_processor" not in kwargs
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakePipeline(tokenizer=kwargs["tokenizer"], _preprocess_params={})

    monkeypatch.setattr("transformers.pipeline", fake_pipeline)

    pipe = create_pipeline("image-to-text", CapabilityModel(), "local-model")

    assert captured["kwargs"]["pipeline_class"] is CombinedProcessorImageToTextPipeline
    assert captured["kwargs"]["processor"] is CapabilityModel.processor
    assert CombinedProcessorImageToTextPipeline.postprocess(
        pipe,
        [torch.tensor([1, 2])],
    ) == [{"generated_text": "decoded-[1, 2]"}]


def test_factory_keeps_default_pipeline_without_capability(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_pipeline(*args: Any, **kwargs: Any) -> Any:
        captured["kwargs"] = kwargs
        return SimpleNamespace(tokenizer=None)

    monkeypatch.setattr("transformers.pipeline", fake_pipeline)

    create_pipeline(
        "image-to-text",
        SimpleNamespace(io_config={"input_shapes": [[1, 3, 2, 2]]}),
        "local-model",
    )

    assert "pipeline_class" not in captured["kwargs"]
