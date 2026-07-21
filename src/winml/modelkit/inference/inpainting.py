# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pipeline adapter for direct-ONNX image-inpainting models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image


if TYPE_CHECKING:
    from ..models.winml.base import WinMLPreTrainedModel


class WinMLInpaintingPipeline:
    """Prepare an image/mask pair and postprocess a direct ONNX result.

    The adapter derives spatial sizes and tensor roles from ONNX metadata. It
    implements the OpenCV inpainting interchange contract: float32 NCHW BGR
    image values in ``[0, 1]`` and a float32 NCHW binary mask. The first
    three-channel output is converted back to an RGB PIL image.
    """

    def __init__(self, model: WinMLPreTrainedModel) -> None:
        self.model = model
        self._inputs, self._output_name = self._resolve_contract(model.io_config)

    def _sanitize_parameters(self, **_kwargs: Any) -> tuple[dict, dict, dict]:
        """Expose the standard pipeline parameter-discovery contract."""
        return {}, {}, {}

    @staticmethod
    def _resolve_contract(
        io_config: dict[str, Any],
    ) -> tuple[dict[str, tuple[str, list[Any]]], str]:
        names = list(io_config.get("input_names", []))
        shapes = list(io_config.get("input_shapes", []))
        output_names = list(io_config.get("output_names", []))
        output_shapes = list(io_config.get("output_shapes", []))
        try:
            entries = list(zip(names, shapes, strict=True))
            outputs = list(zip(output_names, output_shapes, strict=True))
        except ValueError as exc:
            raise ValueError("Inpainting ONNX I/O metadata is incomplete.") from exc

        def _pick(preferred_name: str, channels: int) -> tuple[str, list[Any]] | None:
            by_name = next((entry for entry in entries if entry[0].lower() == preferred_name), None)
            if by_name is not None and len(by_name[1]) == 4 and by_name[1][1] == channels:
                return by_name
            return next(
                (entry for entry in entries if len(entry[1]) == 4 and entry[1][1] == channels),
                None,
            )

        image = _pick("image", 3)
        mask = _pick("mask", 1)
        if image is None or mask is None or image[0] == mask[0]:
            raise ValueError(
                "Inpainting requires distinct 4-D image (3-channel) and mask "
                "(1-channel) ONNX inputs."
            )
        image_size = WinMLInpaintingPipeline._spatial_size(image[1], "image")
        mask_size = WinMLInpaintingPipeline._spatial_size(mask[1], "mask")
        if image_size != mask_size:
            raise ValueError("Inpainting image and mask inputs must have the same spatial size.")

        output = next(
            (
                entry
                for entry in outputs
                if entry[0].lower() in {"output", "image", "images"}
                and len(entry[1]) == 4
                and entry[1][1] == 3
            ),
            None,
        )
        if output is None:
            output = next(
                (entry for entry in outputs if len(entry[1]) == 4 and entry[1][1] == 3),
                None,
            )
        if output is None:
            raise ValueError("Inpainting requires a 4-D 3-channel ONNX image output.")
        output_size = WinMLInpaintingPipeline._spatial_size(output[1], "output")
        if output_size != image_size:
            raise ValueError("Inpainting output must match the image input spatial size.")

        return {"image": image, "mask": mask}, output[0]

    @staticmethod
    def _spatial_size(shape: list[Any], role: str) -> tuple[int, int]:
        if len(shape) != 4 or not isinstance(shape[2], int) or not isinstance(shape[3], int):
            raise ValueError(f"Inpainting {role} input requires fixed NCHW height and width.")
        return shape[3], shape[2]

    @staticmethod
    def _prepare_image(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR))
        bgr = rgb[..., ::-1].astype(np.float32) / 255.0
        return np.ascontiguousarray(bgr.transpose(2, 0, 1)[None, ...])

    @staticmethod
    def _prepare_mask(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
        grayscale = np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST))
        binary = (grayscale > 0).astype(np.float32)
        return binary[None, None, ...]

    @staticmethod
    def _to_image(output: Any) -> Image.Image:
        if isinstance(output, torch.Tensor):
            array = output.detach().cpu().numpy()
        else:
            array = np.asarray(output)
        if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 3:
            raise ValueError(
                f"Inpainting output must have shape [1, 3, height, width], got {list(array.shape)}."
            )
        image = array[0].transpose(1, 2, 0)
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        rgb = np.clip(image[..., ::-1], 0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")

    def __call__(self, inputs: dict[str, Image.Image], **_kwargs: Any) -> Image.Image:
        """Run inpainting for a decoded ``{"image": ..., "mask": ...}`` input."""
        image_name, image_shape = self._inputs["image"]
        mask_name, mask_shape = self._inputs["mask"]
        model_inputs = {
            image_name: self._prepare_image(
                inputs["image"], self._spatial_size(image_shape, "image")
            ),
            mask_name: self._prepare_mask(inputs["mask"], self._spatial_size(mask_shape, "mask")),
        }
        outputs = self.model(**model_inputs)
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError("Inpainting model returned no named ONNX outputs.")
        if self._output_name not in outputs:
            raise ValueError(
                f"Inpainting model did not return expected ONNX output '{self._output_name}'."
            )
        return self._to_image(outputs[self._output_name])
