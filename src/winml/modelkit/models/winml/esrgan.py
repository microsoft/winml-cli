# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Real-ESRGAN-specialised WinML inference wrapper.

Adds a torch-based ``predict(lr_image) -> PIL.Image`` method on top of the
generic :class:`WinMLModelForImageToImage`. ``predict`` is the official
``RealESRGAN.model.predict`` flow from sberbank-ai/Real-ESRGAN ported as
directly as possible — same kwargs, same patch geometry, same torch ops —
with the underlying ``self.model(...)`` call replaced by ``self.forward()``
on the ONNX-backed runtime model. The helpers (``pad_reflect``,
``split_image_into_overlapping_patches``, ``stich_together``,
``unpad_image``, plus the private ``pad_patch`` / ``unpad_patches``) are
copied verbatim from ``RealESRGAN/utils.py``.

Wired into the task-class resolver via
``WINML_MODEL_CLASS_MAPPING[("esrgan", "image-to-image")]``.

Architecture / algorithm reference: sberbank-ai/Real-ESRGAN
(BSD-3-Clause license).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from .image_to_image import WinMLModelForImageToImage


if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers — copied verbatim from sberbank-ai/Real-ESRGAN/RealESRGAN/utils.py
# =============================================================================


def pad_reflect(image, pad_size):
    """Reflect-pad ``image`` (H, W, C) by ``pad_size`` on each side (verbatim upstream)."""
    imsize = image.shape
    height, width = imsize[:2]
    new_img = np.zeros([height + pad_size * 2, width + pad_size * 2, imsize[2]]).astype(np.uint8)
    new_img[pad_size:-pad_size, pad_size:-pad_size, :] = image

    new_img[0:pad_size, pad_size:-pad_size, :] = np.flip(image[0:pad_size, :, :], axis=0)  # top
    new_img[-pad_size:, pad_size:-pad_size, :] = np.flip(image[-pad_size:, :, :], axis=0)  # bottom
    new_img[:, 0:pad_size, :] = np.flip(new_img[:, pad_size : pad_size * 2, :], axis=1)  # left
    new_img[:, -pad_size:, :] = np.flip(new_img[:, -pad_size * 2 : -pad_size, :], axis=1)  # right

    return new_img


def unpad_image(image, pad_size):
    """Inverse of :func:`pad_reflect` (verbatim upstream)."""
    return image[pad_size:-pad_size, pad_size:-pad_size, :]


def pad_patch(image_patch, padding_size, channel_last=True):
    """Pads image_patch with with padding_size edge values."""
    if channel_last:
        return np.pad(
            image_patch,
            ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
            "edge",
        )
    return np.pad(
        image_patch,
        ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
        "edge",
    )


def unpad_patches(image_patches, padding_size):
    """Strip the spatial border added by :func:`pad_patch` (verbatim upstream)."""
    return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]


def split_image_into_overlapping_patches(image_array, patch_size, padding_size=2):
    """Splits the image into partially overlapping patches.

    The patches overlap by padding_size pixels.
    Pads the image twice:
        - first to have a size multiple of the patch size,
        - then to have equal padding at the borders.

    Args:
        image_array: numpy array of the input image.
        patch_size: size of the patches from the original image (without padding).
        padding_size: size of the overlapping area.
    """
    xmax, ymax, _ = image_array.shape
    x_remainder = xmax % patch_size
    y_remainder = ymax % patch_size

    # modulo here is to avoid extending of patch_size instead of 0
    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size

    # make sure the image is divisible into regular patches
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), "edge")

    # add padding around the image to simplify computations
    padded_image = pad_patch(extended_image, padding_size, channel_last=True)

    xmax, ymax, _ = padded_image.shape
    patches = []

    x_lefts = range(padding_size, xmax - padding_size, patch_size)
    y_tops = range(padding_size, ymax - padding_size, patch_size)

    for x in x_lefts:
        for y in y_tops:
            x_left = x - padding_size
            y_top = y - padding_size
            x_right = x + patch_size + padding_size
            y_bottom = y + patch_size + padding_size
            patch = padded_image[x_left:x_right, y_top:y_bottom, :]
            patches.append(patch)

    return np.array(patches), padded_image.shape


def stich_together(patches, padded_image_shape, target_shape, padding_size=4):
    """Reconstruct the image from overlapping patches.

    After scaling, shapes and padding should be scaled too.

    Args:
        patches: patches obtained with split_image_into_overlapping_patches
        padded_image_shape: shape of the padded image contructed in
            split_image_into_overlapping_patches
        target_shape: shape of the final image
        padding_size: size of the overlapping area.
    """
    xmax, ymax, _ = padded_image_shape
    patches = unpad_patches(patches, padding_size)
    patch_size = patches.shape[1]
    n_patches_per_row = ymax // patch_size

    complete_image = np.zeros((xmax, ymax, 3))

    row = -1
    col = 0
    for i in range(len(patches)):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
        complete_image[
            row * patch_size : (row + 1) * patch_size,
            col * patch_size : (col + 1) * patch_size,
            :,
        ] = patches[i]
        col += 1
    return complete_image[0 : target_shape[0], 0 : target_shape[1], :]


# =============================================================================
# Specialised WinML class
# =============================================================================


class WinMLESRGANForImageToImage(WinMLModelForImageToImage):
    """ESRGAN-specialised ``WinMLModelForImageToImage`` with patch-based SR.

    The exported ONNX session is shape-specialised to a fixed patch tensor
    (``patches_size + 2 * padding`` on each spatial side) and an upscale
    factor encoded on :attr:`config.scale`. :meth:`predict` accepts any-size
    PIL image, runs the official Real-ESRGAN patch flow, and returns the
    upscaled PIL image.
    """

    def predict(
        self,
        lr_image: PILImage,
        patches_size: int = 192,
        padding: int = 24,
        pad_size: int = 15,
    ) -> PILImage:
        """Port of ``RealESRGAN.model.predict``.

        Equivalent to the original line-for-line, with ``self.model(...)``
        replaced by the WinML ``self.forward(pixel_values=...).reconstruction``
        call. ``self.scale`` is read from ``self.config.scale``; ``batch_size``
        is no longer a kwarg — it is inferred from the ONNX input's batch
        dim (``1`` when the export left the batch dim dynamic). No torch
        device placement: the ONNX session handles its own device via the
        configured EP, so tensors stay on CPU.
        """
        scale = int(self.config.scale)
        # Inferred from the session's input shape; dynamic batch -> 1.
        batch_size = self.io_config["input_shapes"][0][0] or 1
        lr_image = np.array(lr_image)
        lr_image = pad_reflect(lr_image, pad_size)

        patches, p_shape = split_image_into_overlapping_patches(
            lr_image, patch_size=patches_size, padding_size=padding
        )
        img = torch.FloatTensor(patches / 255).permute((0, 3, 1, 2)).detach()

        with torch.no_grad():
            res = self.forward(pixel_values=img[0:batch_size]).reconstruction
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat(
                    (res, self.forward(pixel_values=img[i : i + batch_size]).reconstruction),
                    0,
                )

        sr_image = res.permute((0, 2, 3, 1)).clamp_(0, 1).cpu()
        np_sr_image = sr_image.numpy()

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)  # noqa: RUF005 — verbatim upstream
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)  # noqa: RUF005 — verbatim upstream
        np_sr_image = stich_together(
            np_sr_image,
            padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape,
            padding_size=padding * scale,
        )
        sr_img = (np_sr_image * 255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size * scale)
        return Image.fromarray(sr_img)
