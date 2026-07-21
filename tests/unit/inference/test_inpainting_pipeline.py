# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the generic direct-ONNX inpainting pipeline."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import torch
from PIL import Image

from winml.modelkit.inference.inpainting import WinMLInpaintingPipeline


class _EchoInpaintingModel:
    io_config: ClassVar[dict] = {
        "input_names": ["source_pixels", "edit_region"],
        "input_shapes": [[1, 3, 2, 2], [1, 1, 2, 2]],
        "output_names": ["completed_image"],
        "output_shapes": [[1, 3, 2, 2]],
    }

    def __init__(self) -> None:
        self.inputs: dict[str, np.ndarray] = {}

    def __call__(self, **kwargs):
        self.inputs = kwargs
        return {"completed_image": torch.from_numpy(kwargs["source_pixels"] * 255.0)}


class TestWinMLInpaintingPipeline:
    def test_prepares_bgr_image_and_binary_mask(self) -> None:
        model = _EchoInpaintingModel()
        pipeline = WinMLInpaintingPipeline(model)  # type: ignore[arg-type]
        pixels = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
        mask = np.array([[0, 1], [127, 255]], dtype=np.uint8)

        output = pipeline(
            {
                "image": Image.fromarray(pixels, mode="RGB"),
                "mask": Image.fromarray(mask, mode="L"),
            }
        )

        assert model.inputs["source_pixels"].shape == (1, 3, 2, 2)
        np.testing.assert_allclose(
            model.inputs["source_pixels"][0, :, 0, 0],
            np.array([30, 20, 10], dtype=np.float32) / 255.0,
        )
        np.testing.assert_array_equal(
            model.inputs["edit_region"],
            np.array([[[[0, 1], [1, 1]]]], dtype=np.float32),
        )
        np.testing.assert_array_equal(np.asarray(output), pixels)

    def test_requires_image_and_mask_tensor_roles(self) -> None:
        model = _EchoInpaintingModel()
        model.io_config = {
            "input_names": ["pixels"],
            "input_shapes": [[1, 3, 2, 2]],
            "output_names": ["output"],
            "output_shapes": [[1, 3, 2, 2]],
        }
        with pytest.raises(ValueError, match=r"image .* and mask"):
            WinMLInpaintingPipeline(model)  # type: ignore[arg-type]

    def test_requires_matching_image_output(self) -> None:
        model = _EchoInpaintingModel()
        model.io_config = {
            **model.io_config,
            "output_names": ["embedding"],
            "output_shapes": [[1, 128]],
        }
        with pytest.raises(ValueError, match="3-channel ONNX image output"):
            WinMLInpaintingPipeline(model)  # type: ignore[arg-type]

    def test_selects_named_image_output(self) -> None:
        model = _EchoInpaintingModel()
        model.io_config = {
            **model.io_config,
            "output_names": ["scores", "completed_image"],
            "output_shapes": [[1, 10], [1, 3, 2, 2]],
        }
        pipeline = WinMLInpaintingPipeline(model)  # type: ignore[arg-type]
        assert pipeline._output_name == "completed_image"
