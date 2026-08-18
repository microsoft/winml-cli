# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for specialized MGP-STR image-to-text inference."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from winml.modelkit.models.winml import get_winml_class
from winml.modelkit.models.winml.image_to_text import (
    MgpstrImageToTextPipeline,
    WinMLModelForMgpstrSceneTextRecognition,
    _convert_images_to_rgb,
)


def test_specialized_wrapper_is_registered() -> None:
    assert (
        get_winml_class("mgp_str", "image-to-text")
        is WinMLModelForMgpstrSceneTextRecognition
    )


def test_wrapper_forward_signature_and_missing_input() -> None:
    signature = inspect.signature(WinMLModelForMgpstrSceneTextRecognition.forward)
    assert list(signature.parameters) == ["self", "kwargs"]
    assert signature.parameters["kwargs"].kind == inspect.Parameter.VAR_KEYWORD

    model = object.__new__(WinMLModelForMgpstrSceneTextRecognition)
    with pytest.raises(ValueError, match="requires 'pixel_values'"):
        model.forward()


def test_wrapper_preserves_three_head_order() -> None:
    model = object.__new__(WinMLModelForMgpstrSceneTextRecognition)
    model._format_inputs = MagicMock(return_value={"pixel_values": np.zeros((1, 3, 32, 128))})
    expected = {
        "char_logits": torch.tensor([1]),
        "bpe_logits": torch.tensor([2]),
        "wp_logits": torch.tensor([3]),
    }
    model._run_inference = MagicMock(return_value=expected)

    output = model.forward(pixel_values=torch.zeros(1, 3, 32, 128))

    assert output.logits == (
        expected["char_logits"],
        expected["bpe_logits"],
        expected["wp_logits"],
    )


@pytest.mark.parametrize(
    "image",
    [
        Image.new("L", (128, 32)),
        np.zeros((32, 128), dtype=np.uint8),
        torch.zeros(1, 32, 128),
    ],
    ids=["grayscale-pil", "2d-numpy", "single-channel-torch"],
)
def test_supported_grayscale_inputs_normalize_to_rgb(image) -> None:
    converted = _convert_images_to_rgb(image)

    assert isinstance(converted, Image.Image)
    assert converted.mode == "RGB"
    assert converted.size == (128, 32)


@pytest.mark.parametrize(
    "outputs",
    [
        {
            "char_logits": torch.tensor([1]),
            "bpe_logits": torch.tensor([2]),
            "wp_logits": torch.tensor([3]),
        },
        {"logits": (torch.tensor([1]), torch.tensor([2]), torch.tensor([3]))},
    ],
    ids=["raw-named-heads", "mapping-backed-logits"],
)
def test_pipeline_decodes_heads_in_order(outputs) -> None:
    model = MagicMock(return_value=outputs)
    processor = MagicMock()
    processor.image_processor = SimpleNamespace(size={"height": 32, "width": 128})
    processor.char_tokenizer = MagicMock()
    processor.return_value = {"pixel_values": torch.zeros(1, 3, 32, 128)}
    processor.batch_decode.return_value = {"generated_text": ["word"]}

    with patch("transformers.AutoProcessor.from_pretrained", return_value=processor):
        pipeline = MgpstrImageToTextPipeline(model, "org/model")

    result = pipeline(Image.new("L", (128, 32)))

    assert result == [{"generated_text": "word"}]
    assert processor.call_args.kwargs["images"].mode == "RGB"
    assert processor.return_value["pixel_values"].shape == (1, 3, 32, 128)
    decoded = processor.batch_decode.call_args.args[0]
    assert tuple(int(item.item()) for item in decoded) == (1, 2, 3)
