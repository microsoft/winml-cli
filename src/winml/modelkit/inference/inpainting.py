# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Data-driven pipeline adapter for direct-ONNX image-inpainting models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image


if TYPE_CHECKING:
    from ..models.winml.base import WinMLPreTrainedModel


@dataclass(frozen=True)
class _InpaintingContract:
    image_input_name: str
    mask_input_name: str
    output_name: str
    image_color_order: str
    image_value_range: tuple[float, float]
    mask_semantics: str
    output_color_order: str
    output_value_range: tuple[float, float]

    @classmethod
    def from_runtime_config(cls, runtime_config: dict[str, Any] | None) -> _InpaintingContract:
        """Validate and materialize the explicit runtime contract."""
        if not isinstance(runtime_config, dict) or runtime_config.get("pipeline") != "inpainting":
            raise ValueError(
                "Inpainting requires runtime.pipeline='inpainting' and explicit runtime.options."
            )
        options = runtime_config.get("options")
        if not isinstance(options, dict):
            raise TypeError("Inpainting runtime.options must be an object.")

        required = {
            "image_input_name",
            "mask_input_name",
            "output_name",
            "image_color_order",
            "image_value_range",
            "mask_semantics",
            "output_color_order",
            "output_value_range",
        }
        missing = sorted(required - options.keys())
        unknown = sorted(options.keys() - required)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown: {', '.join(unknown)}")
            raise ValueError(f"Invalid inpainting runtime options ({'; '.join(details)}).")

        def _color_order(name: str) -> str:
            value = options[name]
            if value not in {"rgb", "bgr"}:
                raise ValueError(f"Inpainting {name} must be 'rgb' or 'bgr'.")
            return str(value)

        def _value_range(name: str) -> tuple[float, float]:
            value = options[name]
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"Inpainting {name} must be a two-number array.")
            result = (float(value[0]), float(value[1]))
            if result not in {(0.0, 1.0), (-1.0, 1.0), (0.0, 255.0)}:
                raise ValueError(
                    f"Unsupported inpainting {name} {value}; supported ranges are "
                    "[0, 1], [-1, 1], and [0, 255]."
                )
            return result

        mask_semantics = options["mask_semantics"]
        if mask_semantics not in {"nonzero-is-hole", "zero-is-hole"}:
            raise ValueError(
                "Inpainting mask_semantics must be 'nonzero-is-hole' or 'zero-is-hole'."
            )
        for name in ("image_input_name", "mask_input_name", "output_name"):
            if not isinstance(options[name], str) or not options[name].strip():
                raise ValueError(f"Inpainting {name} must be a non-empty string.")

        return cls(
            image_input_name=options["image_input_name"],
            mask_input_name=options["mask_input_name"],
            output_name=options["output_name"],
            image_color_order=_color_order("image_color_order"),
            image_value_range=_value_range("image_value_range"),
            mask_semantics=mask_semantics,
            output_color_order=_color_order("output_color_order"),
            output_value_range=_value_range("output_value_range"),
        )


class WinMLInpaintingPipeline:
    """Prepare an image/mask pair and postprocess a direct ONNX result.

    Tensor roles, color order, value ranges, mask polarity, and output semantics
    come from a checked build/runtime contract. ONNX shape metadata is used only
    to validate that explicit contract and determine spatial dimensions.
    """

    def __init__(
        self,
        model: WinMLPreTrainedModel,
        *,
        runtime_config: dict[str, Any] | None,
    ) -> None:
        self.model = model
        self._contract = _InpaintingContract.from_runtime_config(runtime_config)
        self._inputs, self._output_name = self._resolve_io(model.io_config, self._contract)

    def _sanitize_parameters(self, **_kwargs: Any) -> tuple[dict, dict, dict]:
        """Expose the standard pipeline parameter-discovery contract."""
        return {}, {}, {}

    @staticmethod
    def _resolve_io(
        io_config: dict[str, Any],
        contract: _InpaintingContract,
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

        image = next((entry for entry in entries if entry[0] == contract.image_input_name), None)
        mask = next((entry for entry in entries if entry[0] == contract.mask_input_name), None)
        if image is None or mask is None or image[0] == mask[0]:
            raise ValueError(
                "The inpainting runtime contract names inputs that are absent or not distinct."
            )
        if len(image[1]) != 4 or image[1][1] != 3 or len(mask[1]) != 4 or mask[1][1] != 1:
            raise ValueError(
                "Inpainting contract inputs must identify a 4-D 3-channel image and "
                "a 4-D 1-channel mask."
            )
        image_size = WinMLInpaintingPipeline._spatial_size(image[1], "image")
        mask_size = WinMLInpaintingPipeline._spatial_size(mask[1], "mask")
        if image_size != mask_size:
            raise ValueError("Inpainting image and mask inputs must have the same spatial size.")

        output = next((entry for entry in outputs if entry[0] == contract.output_name), None)
        if output is None or len(output[1]) != 4 or output[1][1] != 3:
            raise ValueError(
                "The inpainting runtime contract must identify a 4-D 3-channel ONNX output."
            )
        output_size = WinMLInpaintingPipeline._spatial_size(output[1], "output")
        if output_size != image_size:
            raise ValueError("Inpainting output must match the image input spatial size.")

        return {"image": image, "mask": mask}, output[0]

    @staticmethod
    def _spatial_size(shape: list[Any], role: str) -> tuple[int, int]:
        if len(shape) != 4 or not isinstance(shape[2], int) or not isinstance(shape[3], int):
            raise ValueError(f"Inpainting {role} input requires fixed NCHW height and width.")
        return shape[3], shape[2]

    def _prepare_image(self, image: Image.Image, size: tuple[int, int]) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR))
        pixels = rgb[..., ::-1] if self._contract.image_color_order == "bgr" else rgb
        low, high = self._contract.image_value_range
        pixels = pixels.astype(np.float32) / 255.0 * (high - low) + low
        return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None, ...])

    def _prepare_mask(self, mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
        grayscale = np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST))
        binary = (grayscale > 0).astype(np.float32)
        if self._contract.mask_semantics == "zero-is-hole":
            binary = 1.0 - binary
        return binary[None, None, ...]

    def _to_image(self, output: Any) -> Image.Image:
        if isinstance(output, torch.Tensor):
            array = output.detach().cpu().numpy()
        else:
            array = np.asarray(output)
        if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 3:
            raise ValueError(
                f"Inpainting output must have shape [1, 3, height, width], got {list(array.shape)}."
            )
        image = array[0].transpose(1, 2, 0)
        low, high = self._contract.output_value_range
        image = (image - low) / (high - low) * 255.0
        if self._contract.output_color_order == "bgr":
            image = image[..., ::-1]
        rgb = np.clip(image, 0, 255).astype(np.uint8)
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
