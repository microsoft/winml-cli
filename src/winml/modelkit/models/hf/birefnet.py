# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""BiRefNet Hugging Face model configuration."""

from __future__ import annotations

import sys
from typing import Any, cast

import torch
import torch.nn.functional as F
from optimum.exporters.onnx import OnnxConfig
from optimum.exporters.onnx.model_patcher import ModelPatcher, PatchingSpec
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyVisionInputGenerator

from ...export import register_onnx_overwrite


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize a scalar or two-dimensional convolution parameter."""
    return value if isinstance(value, tuple) else (value, value)


def _exportable_deform_conv2d(
    input: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch equivalent of ``torchvision.ops.deform_conv2d``.

    The legacy ONNX exporter cannot lower ``torchvision::deform_conv2d``.
    This implementation expresses the same sampling operation with
    ``grid_sample`` followed by grouped linear projection, both of which have
    standard ONNX representations at the configured opset.
    """
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dilation_h, dilation_w = _pair(dilation)
    batch_size = input.shape[0]
    in_channels = int(input.shape[1])
    out_channels = int(weight.shape[0])
    channels_per_group = int(weight.shape[1])
    kernel_h = int(weight.shape[2])
    kernel_w = int(weight.shape[3])
    output_h, output_w = offset.shape[-2:]
    kernel_points = kernel_h * kernel_w
    offset_groups = int(offset.shape[1]) // (2 * kernel_points)
    if in_channels % offset_groups != 0:
        raise ValueError("Input channels must be divisible by deformable offset groups")
    conv_groups = in_channels // channels_per_group
    if out_channels % conv_groups != 0:
        raise ValueError("Output channels must be divisible by convolution groups")

    padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h))
    padded_h, padded_w = padded.shape[-2:]
    offsets = offset.reshape(
        batch_size,
        offset_groups,
        kernel_points,
        2,
        output_h,
        output_w,
    )
    modulation = (
        mask.reshape(batch_size, offset_groups, kernel_points, output_h, output_w)
        if mask is not None
        else None
    )
    channels_per_offset_group = in_channels // offset_groups
    base_y = torch.arange(output_h, device=input.device, dtype=input.dtype) * stride_h
    base_x = torch.arange(output_w, device=input.device, dtype=input.dtype) * stride_w
    grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")

    sampled_groups: list[torch.Tensor] = []
    for offset_group in range(offset_groups):
        channel_start = offset_group * channels_per_offset_group
        channel_end = channel_start + channels_per_offset_group
        point_samples: list[torch.Tensor] = []
        for point in range(kernel_points):
            kernel_y, kernel_x = divmod(point, kernel_w)
            sample_y = grid_y + kernel_y * dilation_h + offsets[:, offset_group, point, 0]
            sample_x = grid_x + kernel_x * dilation_w + offsets[:, offset_group, point, 1]
            normalized_y = sample_y * (2.0 / (padded_h - 1)) - 1.0
            normalized_x = sample_x * (2.0 / (padded_w - 1)) - 1.0
            grid = torch.stack((normalized_x, normalized_y), dim=-1)
            sampled = F.grid_sample(
                padded[:, channel_start:channel_end],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            if modulation is not None:
                sampled = sampled * modulation[:, offset_group, point].unsqueeze(1)
            point_samples.append(sampled)
        sampled_groups.append(torch.stack(point_samples, dim=2))
    sampled_input = torch.cat(sampled_groups, dim=1)

    outputs: list[torch.Tensor] = []
    output_channels_per_group = out_channels // conv_groups
    flattened_weight = weight.flatten(2)
    for conv_group in range(conv_groups):
        input_start = conv_group * channels_per_group
        input_end = input_start + channels_per_group
        output_start = conv_group * output_channels_per_group
        output_end = output_start + output_channels_per_group
        outputs.append(
            torch.einsum(
                "nckhw,ock->nohw",
                sampled_input[:, input_start:input_end],
                flattened_weight[output_start:output_end],
            )
        )
    result = torch.cat(outputs, dim=1)
    if bias is not None:
        result = result + bias.reshape(1, -1, 1, 1)
    return result


class _BiRefNetModelPatcher(ModelPatcher):  # type: ignore[misc]
    """Patch the remote module's imported deformable-convolution function."""

    def __init__(self, config: OnnxConfig, model: torch.nn.Module, **kwargs: Any) -> None:
        remote_module = sys.modules[model.__class__.__module__]
        original = getattr(remote_module, "deform_conv2d", None)
        if original is None:
            raise ValueError(
                "BiRefNet remote module does not expose the expected deform_conv2d function"
            )
        config.PATCHING_SPECS = [
            PatchingSpec(
                o=remote_module,
                name="deform_conv2d",
                custom_op=_exportable_deform_conv2d,
                orig_op=original,
            )
        ]
        super().__init__(config, model, **kwargs)


class _BiRefNetVisionInputGenerator(DummyVisionInputGenerator):  # type: ignore[misc]
    """Generate the ``x`` image tensor consumed by remote BiRefNet classes."""

    SUPPORTED_INPUT_NAMES = ("x",)
    _NORMALIZED_MIN = -2.1179039301310043
    _NORMALIZED_MAX = 2.64

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = 1,
        num_channels: int = 3,
        width: int = 1024,
        height: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task,
            normalized_config,
            batch_size=batch_size,
            num_channels=num_channels,
            width=width,
            height=height,
            **kwargs,
        )

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        """Generate an ImageNet-normalized RGB tensor for ``forward(x)``."""
        del input_name, int_dtype
        return cast(
            "torch.Tensor",
            self.random_float_tensor(
                shape=[self.batch_size, self.num_channels, self.height, self.width],
                min_value=self._NORMALIZED_MIN,
                max_value=self._NORMALIZED_MAX,
                framework=framework,
                dtype=float_dtype,
            ),
        )


@register_onnx_overwrite(
    "SegformerForSemanticSegmentation",
    "image-segmentation",
    library_name="transformers",
)
class BiRefNetIOConfig(OnnxConfig):  # type: ignore[misc]
    """ONNX contract for custom-code BiRefNet image segmentation models.

    BiRefNet's remote ``forward(x)`` accepts a normalized RGB image and, in
    evaluation mode, returns a one-element list containing full-resolution,
    single-channel foreground logits. The checkpoint's own inference example
    uses 1024 by 1024 images.
    """

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(allow_new=True)
    DUMMY_INPUT_GENERATOR_CLASSES = (_BiRefNetVisionInputGenerator,)
    _MODEL_PATCHER = _BiRefNetModelPatcher

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return the remote model's exact forward input contract."""
        return {"x": {0: "batch_size", 2: "height", 3: "width"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return the flattened foreground-logit output contract."""
        return {"logits": {0: "batch_size", 2: "height", 3: "width"}}
