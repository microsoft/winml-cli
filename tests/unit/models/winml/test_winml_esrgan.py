# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``WinMLESRGANForImageToImage`` and its patch helpers.

The class adds a Real-ESRGAN-style ``predict(lr_image) -> PIL.Image``
method on top of the generic image-to-image runtime model. We test:

* The task-class resolver picks the specialised subclass for
  ``("esrgan", "image-to-image")``.
* The helper functions copied from upstream's ``utils.py`` round-trip on
  shapes that are not patch-aligned.
* ``predict`` calls the underlying ``forward`` for every patch, batches
  per ``batch_size``, threads ``self.config.scale`` correctly, and
  returns a PIL image of size ``(W * scale, H * scale)``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def patches_size() -> int:
    return 192


@pytest.fixture
def padding() -> int:
    return 24


@pytest.fixture
def pad_size() -> int:
    return 15


# =============================================================================
# Resolver / inheritance
# =============================================================================


class TestResolver:
    def test_specialised_class_registered(self):
        from winml.modelkit.models.winml import (
            WINML_MODEL_CLASS_MAPPING,
            WinMLESRGANForImageToImage,
            get_winml_class,
        )

        assert WINML_MODEL_CLASS_MAPPING[("esrgan", "image-to-image")] == (
            "WinMLESRGANForImageToImage"
        )
        assert get_winml_class("esrgan", "image-to-image") is WinMLESRGANForImageToImage
        # Resolver normalises mixed case
        assert get_winml_class("ESRGAN", "image-to-image") is WinMLESRGANForImageToImage

    def test_subclasses_generic_image_to_image(self):
        from winml.modelkit.models.winml import (
            WinMLESRGANForImageToImage,
            WinMLModelForImageToImage,
        )

        assert issubclass(WinMLESRGANForImageToImage, WinMLModelForImageToImage)


# =============================================================================
# Patch helpers — these are copied verbatim from upstream, so we test the
# shape contracts we depend on rather than the implementation details.
# =============================================================================


class TestPatchHelpers:
    def test_pad_reflect_roundtrip(self, pad_size):
        from winml.modelkit.models.winml.esrgan import pad_reflect, unpad_image

        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=(40, 50, 3), dtype=np.uint8)
        padded = pad_reflect(img, pad_size)
        assert padded.shape == (40 + 2 * pad_size, 50 + 2 * pad_size, 3)
        # Centre region is the original image, untouched
        np.testing.assert_array_equal(padded[pad_size:-pad_size, pad_size:-pad_size, :], img)
        # unpad_image inverts pad_reflect exactly
        np.testing.assert_array_equal(unpad_image(padded, pad_size), img)

    def test_split_and_stitch_roundtrip_with_extension(self, patches_size, padding):
        """Image whose H/W are not multiples of patch_size is edge-extended,
        split, then perfectly stitched back to its original H/W."""
        from winml.modelkit.models.winml.esrgan import (
            split_image_into_overlapping_patches,
            stich_together,
        )

        rng = np.random.default_rng(1)
        # 369 = 192 + 177, 250 = 192 + 58 — both require x_extend/y_extend > 0
        img = rng.integers(0, 255, size=(369, 250, 3), dtype=np.uint8)
        patches, p_shape = split_image_into_overlapping_patches(
            img, patch_size=patches_size, padding_size=padding
        )
        # Each patch is patches_size + 2 * padding on a side
        assert patches.shape[1] == patches_size + 2 * padding
        assert patches.shape[2] == patches_size + 2 * padding
        # scale=1 stitch: same shapes, no upscale → must recover the original
        reconstructed = stich_together(
            patches.astype(np.float64) / 255.0,
            padded_image_shape=p_shape,
            target_shape=img.shape,
            padding_size=padding,
        )
        # Convert back to uint8 for exact comparison
        recovered = (reconstructed * 255).round().astype(np.uint8)
        np.testing.assert_array_equal(recovered, img)


# =============================================================================
# predict() — mock the inherited forward() so we don't need an ONNX session.
# =============================================================================


def _make_model_with_fake_forward(scale: int, batch: int | None = 1):
    """Construct a WinMLESRGANForImageToImage that skips real init.

    Forward returns zeros sized for the scale; lets us assert shape /
    invocation count without a real ORT session. ``batch`` controls the
    ONNX input batch dim that ``predict`` reads from ``self.io_config``
    (use ``None`` to simulate a dynamic-batch export).
    """
    from winml.modelkit.models.winml import WinMLESRGANForImageToImage
    from winml.modelkit.models.winml.image_to_image import ImageReconstructionOutput

    instance = WinMLESRGANForImageToImage.__new__(WinMLESRGANForImageToImage)
    instance.config = SimpleNamespace(scale=scale)
    # ``io_config`` is a read-only property on the base class that delegates
    # to ``_session.io_config``; stub the session so the property returns
    # the shape we control here.
    instance._session = SimpleNamespace(io_config={"input_shapes": [[batch, 3, 240, 240]]})

    def fake_forward(pixel_values, **_kw):
        n, c, h, w = pixel_values.shape
        return ImageReconstructionOutput(
            reconstruction=torch.zeros(n, c, h * scale, w * scale, dtype=torch.float32)
        )

    instance.forward = MagicMock(side_effect=fake_forward)
    return instance


class TestPredict:
    @pytest.mark.parametrize("scale", [2, 4, 8])
    def test_predict_returns_pil_image_with_upscaled_size(self, scale: int):
        model = _make_model_with_fake_forward(scale=scale)
        lr = Image.new("RGB", (50, 40), color=(127, 127, 127))  # PIL size = (W, H)

        sr = model.predict(lr)

        assert isinstance(sr, Image.Image)
        assert sr.size == (50 * scale, 40 * scale)
        assert sr.mode == "RGB"

    def test_predict_reads_scale_from_config(self):
        # If predict didn't read self.config.scale, it would default to
        # the model's class-level scale (no such default exists) and the
        # output shape would not match.
        model = _make_model_with_fake_forward(scale=4)
        lr = Image.new("RGB", (32, 32))
        assert model.predict(lr).size == (128, 128)

    def test_predict_invokes_forward_at_least_once_with_correct_patch_shape(
        self,
        patches_size,
        padding,
    ):
        model = _make_model_with_fake_forward(scale=2)
        lr = Image.new("RGB", (32, 32))
        model.predict(lr)

        # First call's pixel_values is the first batch of patches; each patch
        # is shape-specialised to patches_size + 2 * padding on each spatial side.
        assert model.forward.call_count >= 1
        first_call_pv = model.forward.call_args_list[0].kwargs["pixel_values"]
        assert isinstance(first_call_pv, torch.Tensor)
        assert first_call_pv.shape[1] == 3
        assert first_call_pv.shape[2] == patches_size + 2 * padding
        assert first_call_pv.shape[3] == patches_size + 2 * padding

    def test_predict_infers_batch_size_from_session_input_shape(self):
        """batch_size is no longer a kwarg — it's read from io_config[input_shapes]."""
        # Static batch=4 export
        model = _make_model_with_fake_forward(scale=2, batch=4)
        # 600x400 LR with patches_size=192, padding=24, pad_size=15:
        # post-reflect-pad → 630x430, after extending to multiples of 192:
        # 768x576 → 4 * 3 = 12 patches. With inferred batch_size=4 → 3 forward calls.
        lr = Image.new("RGB", (600, 400))
        model.predict(lr)
        assert model.forward.call_count == 3

    def test_predict_defaults_to_batch_1_when_dynamic(self):
        """When the ONNX export left the batch dim dynamic, fall back to 1."""
        model = _make_model_with_fake_forward(scale=2, batch=None)
        lr = Image.new("RGB", (600, 400))
        model.predict(lr)
        # 12 patches, batch=1 -> 12 forward calls
        assert model.forward.call_count == 12

    def test_predict_supports_arbitrary_input_sizes(self):
        """Non-square, non-patch-multiple sizes must work end-to-end."""
        model = _make_model_with_fake_forward(scale=2)
        for w, h in [(100, 80), (333, 217), (1, 1)]:
            lr = Image.new("RGB", (w, h))
            sr = model.predict(lr)
            assert sr.size == (w * 2, h * 2)
