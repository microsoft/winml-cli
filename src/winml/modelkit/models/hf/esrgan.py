# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Real-ESRGAN (RRDBNet) as a HuggingFace PreTrainedModel.

Implements the RRDBNet architecture from Real-ESRGAN as a PreTrainedModel
so it can be saved/loaded via save_pretrained/from_pretrained and exported
to ONNX via the standard HuggingFace/Optimum pipeline.

Architecture reference: sberbank-ai/Real-ESRGAN (BSD-3-Clause license).

Classes:
    ESRGANConfig: PretrainedConfig with RRDBNet hyperparameters.
    ESRGANPreTrainedModel: Base PreTrainedModel (config_class, init_weights).
    ResidualDenseBlock: 5-conv dense block with residual scaling.
    RRDB: Residual-in-Residual Dense Block (3x ResidualDenseBlock).
    ESRGANForImageSuperResolution: Full RRDBNet for image super-resolution.

Note:
    ``ESRGANForImageSuperResolution.from_pretrained`` is overridden so that
    a ``.pth``-only HF repo (e.g. ``ai-forever/Real-ESRGAN``) is supported
    end-to-end without an offline ``save_pretrained`` conversion step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedVisionConfig
from optimum.utils.input_generators import DummyVisionInputGenerator
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ImageSuperResolutionOutput

from ...export import register_onnx_overwrite


logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================


class ESRGANConfig(PretrainedConfig):
    """Configuration for Real-ESRGAN RRDBNet architecture.

    The ``model_type`` is intentionally uppercase ``"ESRGAN"`` so that
    HuggingFace's name-based fallback in
    :meth:`AutoConfig.from_pretrained` (which does a case-sensitive
    substring match against the repo name) succeeds for community repos
    like ``ai-forever/Real-ESRGAN`` that ship raw ``.pth`` weights with
    no ``config.json``.

    Attributes:
        num_in_ch: Number of input channels.
        num_out_ch: Number of output channels.
        num_feat: Number of intermediate feature channels.
        num_block: Number of RRDB blocks in the body.
        num_grow_ch: Growth channel count inside ResidualDenseBlock.
        scale: Upscaling factor (1, 2, 4, or 8).
    """

    model_type = "ESRGAN"

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_block = num_block
        self.num_grow_ch = num_grow_ch
        self.scale = scale


# =============================================================================
# Weight initialisation helper
# =============================================================================


def default_init_weights(module_list: list[nn.Module] | nn.Module, scale: float = 1.0) -> None:
    """Kaiming normal init for Conv2d layers with optional scale multiplier.

    Mirrors the sberbank-ai/Real-ESRGAN initialisation used in ResidualDenseBlock.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()


# =============================================================================
# Building blocks
# =============================================================================


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block with 5 convolutions.

    Each conv receives the concatenation of all preceding feature maps.
    A 0.2 residual scaling is applied before adding back to the input.
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Initialise weights (conv5 uses scale=0.1 for stability)
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
            scale=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dense forward: each conv sees all prior feature maps."""
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Residual scaling
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block (3x ResidualDenseBlock)."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 3 RDB blocks with 0.2 residual scaling."""
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


# =============================================================================
# PreTrainedModel base
# =============================================================================


class ESRGANPreTrainedModel(PreTrainedModel):
    """Base PreTrainedModel for Real-ESRGAN variants."""

    config_class = ESRGANConfig
    base_model_prefix = "esrgan"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        """No-op: RRDBNet blocks self-initialise via default_init_weights."""


# =============================================================================
# Full model
# =============================================================================


class ESRGANForImageSuperResolution(ESRGANPreTrainedModel):
    """RRDBNet for image super-resolution.

    Architecture:
        - Optional pixel_unshuffle for scale 1 or 2
        - conv_first -> body (N x RRDB) -> conv_body (skip connection)
        - Upsampling via nearest-neighbour interpolation + conv
        - conv_hr -> conv_last for final output

    Attribute names match sberbank-ai/Real-ESRGAN for weight compatibility.
    """

    def __init__(self, config: ESRGANConfig) -> None:
        super().__init__(config)

        scale = config.scale
        num_feat = config.num_feat
        num_grow_ch = config.num_grow_ch

        # For scale <= 2, pixel_unshuffle compresses spatial dims
        # and increases channel count before the network body
        if scale == 2:
            num_in_ch = config.num_in_ch * 4
        elif scale == 1:
            num_in_ch = config.num_in_ch * 16
        else:
            num_in_ch = config.num_in_ch

        self.scale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(config.num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling convolutions
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, config.num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Initialize weights and apply final processing (PreTrainedModel)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool | None = None,
    ) -> ImageSuperResolutionOutput | tuple[torch.Tensor]:
        """Run super-resolution on input images.

        Args:
            pixel_values: Input tensor of shape (B, C, H, W).
            return_dict: Whether to return ImageSuperResolutionOutput or tuple.

        Returns:
            ImageSuperResolutionOutput with reconstruction, or tuple if return_dict=False.
        """
        if return_dict is None:
            return_dict = (
                self.config.use_return_dict
                if hasattr(self.config, "use_return_dict")
                else True
            )

        feat = pixel_values

        # Pixel unshuffle for scale <= 2
        if self.scale == 2:
            feat = F.pixel_unshuffle(feat, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(feat, downscale_factor=4)

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.scale == 8:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode="nearest")))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        if not return_dict:
            return (out,)

        return ImageSuperResolutionOutput(reconstruction=out)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        **kwargs: Any,
    ) -> ESRGANForImageSuperResolution:
        """Load Real-ESRGAN from a ``.pth``-only HF repo.

        Real-ESRGAN distributions on the Hub (e.g. ``ai-forever/Real-ESRGAN``)
        ship raw ``.pth`` checkpoints with no ``config.json`` /
        ``pytorch_model.bin``, so the standard
        :meth:`PreTrainedModel.from_pretrained` flow can't load them. Because
        :class:`ESRGANConfig` is registered with :class:`AutoConfig`,
        :meth:`ESRGANConfig.from_pretrained` already returns a default-valued
        config when ``config.json`` is missing on the Hub — so we just:

        1. Build the config via that path (or accept one passed through ``config=``).
        2. Override ``scale`` from a ``scale=`` kwarg if the caller specifies one.
        3. Download ``RealESRGAN_x{scale}.pth`` to a temp dir, ``load_state_dict``
           it, then discard the temp dir.

        Local directories that already contain ``config.json``/weights are
        delegated to :meth:`PreTrainedModel.from_pretrained` unchanged.
        """
        import tempfile

        from huggingface_hub import hf_hub_download

        scale_hint = kwargs.pop("scale", None)

        # Local directory with config.json -> use base implementation as-is.
        local_path = Path(pretrained_model_name_or_path)
        if local_path.exists() and (local_path / "config.json").exists():
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Build an ESRGANConfig either from a caller-provided one or by
        # invoking ESRGANConfig.from_pretrained (which returns defaults when
        # the Hub repo has no config.json).
        config = kwargs.pop("config", None)
        if not isinstance(config, ESRGANConfig):
            config = ESRGANConfig.from_pretrained(str(pretrained_model_name_or_path))

        if scale_hint is not None:
            config.scale = int(scale_hint)

        weight_file = f"RealESRGAN_x{config.scale}.pth"
        with tempfile.TemporaryDirectory(prefix="esrgan-weights-") as tmpdir:
            logger.info(
                "Downloading %s from %s (scale=%d) to %s",
                weight_file,
                pretrained_model_name_or_path,
                config.scale,
                tmpdir,
            )
            pth_path = hf_hub_download(
                repo_id=str(pretrained_model_name_or_path),
                filename=weight_file,
                local_dir=tmpdir,
            )
            state = torch.load(pth_path, map_location="cpu", weights_only=True)

        if isinstance(state, dict):
            if "params_ema" in state:
                state = state["params_ema"]
            elif "params" in state:
                state = state["params"]

        model = cls(config)
        model.load_state_dict(state, strict=True)
        return model


# =============================================================================
# Class mapping for the loader
# =============================================================================
# ``model_type`` keys are lowercased before MODEL_CLASS_MAPPING lookup
# (see ``loader.task._get_custom_model_class``), so we use ``"esrgan"`` here
# even though the canonical model_type on the config is uppercase ``ESRGAN``.
#
# The ``("esrgan", None)`` sentinel encodes the per-model-type default task
# for auto-detection — see the ``[3a]`` block in ``loader/task.py``. Because
# the sentinel is consulted ahead of architecture-based detection, the
# resolver does not need ``ESRGANForImageSuperResolution`` to be importable
# from the ``transformers`` package.
MODEL_CLASS_MAPPING: dict[tuple[str, str | None], type] = {
    ("esrgan", None): ESRGANForImageSuperResolution,
    ("esrgan", "image-to-image"): ESRGANForImageSuperResolution,
}


# =============================================================================
# ONNX export config
# =============================================================================


@register_onnx_overwrite("ESRGAN", "image-to-image", library_name="transformers")
class ESRGANIOConfig(OnnxConfig):
    """ONNX export config for Real-ESRGAN.

    Inputs:
        - pixel_values: {0: "batch_size", 2: "height", 3: "width"}

    Outputs:
        - reconstruction: {0: "batch_size", 2: "height", 3: "width"}
    """

    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return input tensor names and their dynamic axes."""
        return {"pixel_values": {0: "batch_size", 2: "height", 3: "width"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return output tensor names and their dynamic axes."""
        return {"reconstruction": {0: "batch_size", 2: "height", 3: "width"}}


