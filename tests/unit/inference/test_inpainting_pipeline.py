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


_LAMA_RUNTIME = {
    "pipeline": "inpainting",
    "options": {
        "image_input_name": "source_pixels",
        "mask_input_name": "edit_region",
        "output_name": "completed_image",
        "image_color_order": "bgr",
        "image_value_range": [0, 1],
        "mask_semantics": "nonzero-is-hole",
        "output_color_order": "bgr",
        "output_value_range": [0, 255],
    },
}


def _make_pipeline(model: _EchoInpaintingModel) -> WinMLInpaintingPipeline:
    return WinMLInpaintingPipeline(model, runtime_config=_LAMA_RUNTIME)  # type: ignore[arg-type]


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
        pipeline = _make_pipeline(model)
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
        with pytest.raises(ValueError, match="absent or not distinct"):
            _make_pipeline(model)

    def test_requires_matching_image_output(self) -> None:
        model = _EchoInpaintingModel()
        model.io_config = {
            **model.io_config,
            "output_names": ["embedding"],
            "output_shapes": [[1, 128]],
        }
        with pytest.raises(ValueError, match="3-channel ONNX output"):
            _make_pipeline(model)

    def test_selects_named_image_output(self) -> None:
        model = _EchoInpaintingModel()
        model.io_config = {
            **model.io_config,
            "output_names": ["scores", "completed_image"],
            "output_shapes": [[1, 10], [1, 3, 2, 2]],
        }
        pipeline = _make_pipeline(model)
        assert pipeline._output_name == "completed_image"

    def test_requires_explicit_runtime_contract(self) -> None:
        with pytest.raises(ValueError, match=r"explicit runtime\.options"):
            WinMLInpaintingPipeline(  # type: ignore[arg-type]
                _EchoInpaintingModel(), runtime_config=None
            )

    def test_rejects_unsupported_contract_instead_of_guessing(self) -> None:
        runtime = {
            **_LAMA_RUNTIME,
            "options": {**_LAMA_RUNTIME["options"], "image_value_range": [0, 2]},
        }
        with pytest.raises(ValueError, match="Unsupported inpainting image_value_range"):
            WinMLInpaintingPipeline(  # type: ignore[arg-type]
                _EchoInpaintingModel(), runtime_config=runtime
            )

    def test_applies_declared_non_lama_rgb_range_and_mask_contract(self) -> None:
        class _AlternateContractModel(_EchoInpaintingModel):
            def __call__(self, **kwargs):
                self.inputs = kwargs
                return {"completed_image": torch.from_numpy(kwargs["source_pixels"])}

        runtime = {
            "pipeline": "inpainting",
            "options": {
                **_LAMA_RUNTIME["options"],
                "image_color_order": "rgb",
                "image_value_range": [-1, 1],
                "mask_semantics": "zero-is-hole",
                "output_color_order": "rgb",
                "output_value_range": [-1, 1],
            },
        }
        model = _AlternateContractModel()
        pipeline = WinMLInpaintingPipeline(model, runtime_config=runtime)  # type: ignore[arg-type]
        pixels = np.full((2, 2, 3), [10, 20, 30], dtype=np.uint8)
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)

        output = pipeline(
            {
                "image": Image.fromarray(pixels, mode="RGB"),
                "mask": Image.fromarray(mask, mode="L"),
            }
        )

        np.testing.assert_allclose(
            model.inputs["source_pixels"][0, :, 0, 0],
            np.array([10, 20, 30], dtype=np.float32) / 255.0 * 2.0 - 1.0,
        )
        np.testing.assert_array_equal(
            model.inputs["edit_region"],
            np.array([[[[1, 0], [0, 1]]]], dtype=np.float32),
        )
        np.testing.assert_allclose(np.asarray(output), pixels, atol=1)
