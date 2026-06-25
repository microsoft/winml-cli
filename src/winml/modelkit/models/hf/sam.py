# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""SAM/SAM2 HuggingFace model patches and ONNX export configs.

Provides QNN-compatible patches and ONNX export configs for both:
- SAM v1 (facebook/sam-vit-*)
- SAM2 / SAM2-video (facebook/sam2-hiera-*)

Key features:
- SAM2 QNN-compatible patches: 5D window partition, arithmetic masking
- Split and task-specific exports: encoder, full model, and decoder wrappers

Patch targets (applied via Sam2ModelPatcher during export):
- Sam2MultiScaleBlock: 5D window partition (6D->5D for QNN)
- Sam2PromptEncoder: Arithmetic masking (torch.where->arithmetic for ONNX)

Export coverage:
- SAM2: image encoder, full model, mask-generation decoder wrapper
- SAM v1: mask-generation decoder wrapper

Exports:
    Sam2NormalizedVisionConfig: NormalizedVisionConfig with 1024 image_size
    Sam2ImageEncoderIOConfig: ONNX config for SAM2 image encoder
    Sam2IOConfig: ONNX config for SAM2 full model
    Sam2MaskGenerationIOConfig: ONNX config for SAM2 mask-generation wrapper
    SamMaskGenerationIOConfig: ONNX config for SAM v1 mask-generation wrapper
    Sam2ModelPatcher: Custom ModelPatcher for SAM2 export patches
    _patched_sam2_multiscale_block_forward: Patched forward (internal)
    _patched_sam2_prompt_encoder_forward: Patched forward (internal)
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from optimum.exporters.onnx import OnnxConfig
from optimum.exporters.onnx.model_patcher import ModelPatcher
from optimum.utils import NormalizedVisionConfig
from optimum.utils.input_generators import (
    DummyInputGenerator,
    DummyVisionInputGenerator,
)
from transformers import Sam2Model, SamModel

from ...export import register_onnx_overwrite


if TYPE_CHECKING:
    from collections.abc import Callable

    from optimum.utils import NormalizedConfig


# =============================================================================
# Custom Model Class: Sam2VisionEncoder
# =============================================================================
# Sam2VisionModel cannot load weights from a Sam2VideoModel checkpoint because
# checkpoint keys are prefixed with "vision_encoder." (e.g., "vision_encoder.backbone.*")
# but Sam2VisionModel expects unprefixed keys (e.g., "backbone.*").
# This wrapper loads the full Sam2VideoModel and extracts the vision_encoder submodule,
# flattening the FPN tuple outputs into individual tensor outputs for ONNX compatibility.


class Sam2VisionEncoder(torch.nn.Module):
    """Wrapper that loads Sam2VideoModel, extracts vision_encoder.

    Flattens FPN tuple outputs for ONNX export.

    Sam2VisionModel.forward() returns Sam2VisionEncoderOutput where
    fpn_hidden_states is a tuple of 3 tensors (one per FPN level).
    Optimum's ModelPatcher output filter matches by dict key name against
    the ONNX config's output names, so tuple-of-tensor fields are invisible
    to the filter, producing an empty ONNX graph.

    This wrapper flattens the FPN outputs into individual tensor entries
    with names matching Sam2ImageEncoderIOConfig.outputs:
        fpn_hidden_states[2] -> image_embeddings   [B, 256, 64, 64]
        fpn_hidden_states[0] -> high_res_features1  [B, 256, 256, 256]
        fpn_hidden_states[1] -> high_res_features2  [B, 256, 128, 128]
    """

    def __init__(self, vision_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.config = vision_encoder.config

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> Sam2VisionEncoder:
        full_model = Sam2Model.from_pretrained(model_name_or_path, **kwargs)
        return cls(full_model.vision_encoder)

    def forward(
        self, pixel_values: torch.Tensor | None = None, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        out = self.vision_encoder(pixel_values=pixel_values, **kwargs)
        fpn = out.fpn_hidden_states
        return {
            "image_embeddings": fpn[2],
            "high_res_features1": fpn[0],
            "high_res_features2": fpn[1],
        }


class SAM2MaskGeneration(torch.nn.Module):
    """Export wrapper for SAM2 mask generation (decoder portion).

    Composes prompt_encoder + mask_decoder + positional embeddings
    into a single module with explicit I/O signature.

    Mirrors Sam2Model.forward flow:
        1. Add no_memory_embedding to image_embeddings (fpn level 2)
        2. Encode prompts (points + optional mask)
        3. Compute positional embeddings
        4. Apply conv_s0/conv_s1 to raw FPN high-res features
        5. Run mask decoder
        6. Upsample to full resolution

    Inputs:
        input_points:       [B, 1, N, 2]       - Point coordinates in pixels
        input_labels:       [B, 1, N]           - Point labels (0=neg, 1=pos, -1=pad)
        image_embeddings:   [B, 256, 64, 64]    - FPN level 2 from encoder
        high_res_features0: [B, 256, 256, 256]  - FPN level 0 from encoder (raw)
        high_res_features1: [B, 256, 128, 128]  - FPN level 1 from encoder (raw)
        mask_input:         [B, 1, 256, 256]    - Previous mask (for refinement)
        use_mask_input:     [B]                 - Flag: 0.0=ignore mask, 1.0=use mask

    Outputs:
        masks:          [B, 3, 1024, 1024] - Full resolution masks
        iou_scores:     [B, 3]             - IoU predictions per mask
        low_res_masks:  [B, 3, 256, 256]   - Low-res masks (for next iteration)
    """

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> SAM2MaskGeneration:
        """Load from a HuggingFace Sam2Model checkpoint."""
        sam2_model = Sam2Model.from_pretrained(model_name_or_path, **kwargs)
        return cls(sam2_model)

    def __init__(self, sam2_model: Sam2Model) -> None:
        super().__init__()

        self.prompt_encoder = sam2_model.prompt_encoder
        self.mask_decoder = sam2_model.mask_decoder
        self.shared_image_embedding = sam2_model.shared_image_embedding
        self.image_embedding_size = self.prompt_encoder.image_embedding_size

        # no_memory_embedding: added to fpn level 2 (matches Sam2Model.forward)
        self.no_memory_embedding = sam2_model.no_memory_embedding

        # High-res projections (originally applied in get_image_features)
        self.conv_s0 = sam2_model.mask_decoder.conv_s0  # 256 -> 32
        self.conv_s1 = sam2_model.mask_decoder.conv_s1  # 256 -> 64

    def _get_image_positional_embeddings(self, batch_size: int = 1) -> torch.Tensor:
        """Replicates Sam2Model.get_image_wide_positional_embeddings()."""
        size = self.image_embedding_size
        # positional_embedding is a registered Parameter reached via torch's
        # __getattr__ (typed Tensor | Module); narrow to Tensor for device/dtype.
        pos_emb = cast("torch.Tensor", self.shared_image_embedding.positional_embedding)
        target_device = pos_emb.device
        target_dtype = pos_emb.dtype

        grid = torch.ones(size, device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size[0]
        x_embed = x_embed / size[1]

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        positional_embedding = positional_embedding.permute(2, 0, 1).unsqueeze(0)
        # shared_image_embedding is a torch submodule (untyped __call__ -> Any).
        return cast("torch.Tensor", positional_embedding.repeat(batch_size, 1, 1, 1))

    def forward(
        self,
        input_points: torch.Tensor,
        input_labels: torch.Tensor,
        image_embeddings: torch.Tensor,
        high_res_features0: torch.Tensor,
        high_res_features1: torch.Tensor,
        mask_input: torch.Tensor,
        use_mask_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run mask generation from pre-computed encoder features."""
        batch_size = image_embeddings.shape[0]

        # 1. Add no_memory_embedding to image_embeddings
        no_mem = self.no_memory_embedding.permute(0, 2, 1).unsqueeze(-1)
        image_embeddings = image_embeddings + no_mem

        # 2. Prompt embeddings
        # Get sparse embeddings (without mask — mask blending handled below)
        sparse_embeddings, _ = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=None,
            input_masks=None,
        )

        # Arithmetic mask blending via use_mask_input flag
        # (avoids torch.where for ONNX/QNN compatibility)
        mask_dense = self.prompt_encoder.mask_embed(mask_input)
        no_mask_dense = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch_size,
            -1,
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        )
        flag = use_mask_input.reshape(-1, 1, 1, 1).to(mask_dense.dtype)
        dense_embeddings = (1.0 - flag) * no_mask_dense + flag * mask_dense

        # 3. Positional embeddings
        image_positional_embeddings = self._get_image_positional_embeddings(batch_size)

        # 4. Apply high-res projections (conv_s0, conv_s1)
        high_res_proj0 = self.conv_s0(high_res_features0)  # [B, 32, 256, 256]
        high_res_proj1 = self.conv_s1(high_res_features1)  # [B, 64, 128, 128]

        # 5. Mask decoder
        low_res_masks, iou_pred, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            high_resolution_features=[high_res_proj0, high_res_proj1],
        )

        # Squeeze point_batch_size dimension
        low_res_masks = low_res_masks.squeeze(1)  # [B, 3, 256, 256]
        iou_scores = iou_pred.squeeze(1)  # [B, 3]

        # 6. Upsample to full resolution
        masks = torch.nn.functional.interpolate(
            low_res_masks,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )

        return masks, iou_scores, low_res_masks


class SAMMaskGeneration(torch.nn.Module):
    """Export wrapper for SAM v1 mask generation (decoder portion).

    Composes prompt_encoder + mask_decoder + positional embeddings
    into a single module with explicit I/O signature.

    Mirrors SamModel.forward flow:
        1. Encode prompts (points + optional mask)
        2. Compute positional embeddings
        3. Run mask decoder

    Inputs:
        input_points:     [B, 1, N, 2]     - Point coordinates in pixels
        input_labels:     [B, 1, N]         - Point labels (0=neg, 1=pos, -1=pad)
        image_embeddings: [B, 256, 64, 64]  - From vision encoder
        mask_input:       [B, 1, 256, 256]  - Previous mask (for refinement)
        use_mask_input:   [B]               - Flag: 0.0=ignore mask, 1.0=use mask

    Outputs:
        masks:          [B, 3, 1024, 1024] - Full resolution masks
        iou_scores:     [B, 3]             - IoU predictions per mask
        low_res_masks:  [B, 3, 256, 256]   - Low-res masks (for next iteration)
    """

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> SAMMaskGeneration:
        """Load from a HuggingFace SamModel checkpoint."""
        sam_model = SamModel.from_pretrained(model_name_or_path, **kwargs)
        return cls(sam_model)

    def __init__(self, sam_model: SamModel) -> None:
        super().__init__()

        self.prompt_encoder = sam_model.prompt_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.shared_image_embedding = sam_model.shared_image_embedding
        self.image_embedding_size = self.prompt_encoder.image_embedding_size
        self.config = sam_model.config

    def _get_image_positional_embeddings(self, batch_size: int = 1) -> torch.Tensor:
        """Replicates SamModel.get_image_wide_positional_embeddings()."""
        size = self.config.prompt_encoder_config.image_embedding_size
        # positional_embedding is a registered Parameter reached via torch's
        # __getattr__ (typed Tensor | Module); narrow to Tensor for device/dtype.
        pos_emb = cast("torch.Tensor", self.shared_image_embedding.positional_embedding)
        target_device = pos_emb.device
        target_dtype = pos_emb.dtype

        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        positional_embedding = positional_embedding.permute(2, 0, 1).unsqueeze(0)
        # shared_image_embedding is a torch submodule (untyped __call__ -> Any).
        return cast("torch.Tensor", positional_embedding.repeat(batch_size, 1, 1, 1))

    def forward(
        self,
        input_points: torch.Tensor,
        input_labels: torch.Tensor,
        image_embeddings: torch.Tensor,
        mask_input: torch.Tensor,
        use_mask_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run mask generation from pre-computed encoder features."""
        batch_size = image_embeddings.shape[0]

        # 1. Prompt embeddings (sparse - points only, mask handled separately)
        sparse_embeddings, _ = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=None,
            input_masks=None,
        )

        # Arithmetic mask blending via use_mask_input flag
        # (avoids torch.where for ONNX/QNN compatibility)
        mask_dense = self.prompt_encoder.mask_embed(mask_input)
        no_mask_dense = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch_size,
            -1,
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        )
        flag = use_mask_input.reshape(-1, 1, 1, 1).to(mask_dense.dtype)
        dense_embeddings = (1.0 - flag) * no_mask_dense + flag * mask_dense

        # 2. Positional embeddings
        image_positional_embeddings = self._get_image_positional_embeddings(batch_size)

        # 3. Mask decoder
        low_res_masks, iou_pred = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        # Squeeze point_batch_size dimension
        low_res_masks = low_res_masks.squeeze(1)  # [B, 3, 256, 256]
        iou_scores = iou_pred.squeeze(1)  # [B, 3]

        # 4. Upsample to full resolution
        masks = torch.nn.functional.interpolate(
            low_res_masks,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
        )

        return masks, iou_scores, low_res_masks


# =============================================================================
# HuggingFace Model Class Mapping
# =============================================================================

# (model_type, task) -> HuggingFace model class
#
# A sentinel entry with task=None encodes the per-model-type default task
# applied during auto-detection (when the user does not pass --task). Its
# value is the default *class*; the resolver reverse-looks-up the task name
# from the matching (model_type, default_task) -> same_class entry. Encoding
# the default inside MODEL_CLASS_MAPPING — instead of a parallel
# MODEL_TASK_DEFAULTS table — keeps the data in one place and structurally
# enforces that a matching class entry must exist (else reverse lookup fails).
#
# Why SAM/SAM2 need this:
# TasksManager.infer_task_from_model() returns "feature-extraction" for
# SamModel / Sam2Model, but the canonical export target for these
# architectures is the mask-generation decoder wrapper. Encoder-only entries
# remain so users can opt in via --task feature-extraction /
# image-feature-extraction (the latter routes perf pipeline to ImageDataset).
# Users wanting the full encoder+decoder monolith use --task image-segmentation.

MODEL_CLASS_MAPPING: dict[tuple[str, str | None], type] = {
    ("sam", None): SAMMaskGeneration,
    ("sam", "mask-generation"): SAMMaskGeneration,
    ("sam2", None): SAM2MaskGeneration,
    ("sam2", "image-segmentation"): Sam2Model,
    ("sam2", "feature-extraction"): Sam2VisionEncoder,
    ("sam2", "image-feature-extraction"): Sam2VisionEncoder,
    ("sam2", "mask-generation"): SAM2MaskGeneration,
    ("sam2-video", None): SAM2MaskGeneration,
    ("sam2-video", "image-segmentation"): Sam2Model,
    ("sam2-video", "feature-extraction"): Sam2VisionEncoder,
    ("sam2-video", "image-feature-extraction"): Sam2VisionEncoder,
    ("sam2-video", "mask-generation"): SAM2MaskGeneration,
}


# Note: No model-specific build config needed. The analyzer autoconf loop
# discovers optimization flags automatically. See issue #232.


def _window_partition(
    hidden_state: torch.Tensor, window_size: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    """QNN-compatible window partition (5D instead of 6D).

    Original HF creates 6D view: [B, H//ws, ws, W//ws, ws, C]
    This version uses 5D: [B*H//ws, ws, W//ws, ws, C]
    """
    B, H, W, C = hidden_state.shape  # noqa: N806 (standard tensor dimension naming)

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        hidden_state = F.pad(hidden_state, (0, 0, 0, pad_w, 0, pad_h))

    pH, pW = H + pad_h, W + pad_w  # noqa: N806

    # 5D reshape (not 6D)
    hidden_state = hidden_state.reshape(
        B * pH // window_size, window_size, pW // window_size, window_size, C
    )
    windows = hidden_state.permute(0, 2, 1, 3, 4).contiguous()
    windows = windows.view(-1, window_size, window_size, C)

    return windows, (pH, pW)


def _window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> torch.Tensor:
    """QNN-compatible window unpartition (5D instead of 6D).

    Original HF creates 6D view: [B, pH//ws, pW//ws, ws, ws, C]
    This version uses 5D: [B*pH//ws, pW//ws, ws, ws, C]
    """
    pH, pW = pad_hw  # noqa: N806 (standard tensor dimension naming)
    H, W = hw  # noqa: N806
    B = windows.shape[0] // (pH * pW // window_size // window_size)  # noqa: N806

    # 5D reshape (not 6D): merge B with pH//ws
    hidden_state = windows.reshape(
        B * pH // window_size, pW // window_size, window_size, window_size, -1
    )
    # permute to interleave: [B*pH//ws, ws, pW//ws, ws, C]
    hidden_state = hidden_state.permute(0, 2, 1, 3, 4).contiguous()
    hidden_state = hidden_state.view(B, pH, pW, -1)

    if pH > H or pW > W:
        hidden_state = hidden_state[:, :H, :W, :].contiguous()

    return hidden_state


def _patched_sam2_multiscale_block_forward(
    self: Any, hidden_states: torch.Tensor, **kwargs: Any
) -> torch.Tensor:
    """Patched Sam2MultiScaleBlock.forward with 5D window functions.

    Target: Sam2MultiScaleBlock
    Changes: Uses _window_partition/_window_unpartition (5D) instead of
             original window_partition/window_unpartition (6D)

    Applied via Sam2ModelPatcher during export.
    """
    # No windowing needed, use original
    if self.window_size <= 0:
        return cast("torch.Tensor", self._original_forward(hidden_states, **kwargs))

    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)

    if self.dim != self.dim_out:
        # Inline import: do_pool is a private API in transformers internals.
        try:
            from transformers.models.sam2.modeling_sam2 import do_pool
        except ImportError as e:
            raise ImportError(
                "do_pool not found; upgrade transformers to a version that includes SAM2 support"
            ) from e
        residual = do_pool(self.proj(hidden_states), self.query_stride)

    window_size = self.window_size
    H, W = None, None  # noqa: N806 (standard tensor dimension naming)
    pad_hw = None

    if self.window_size > 0:
        H, W = hidden_states.shape[1], hidden_states.shape[2]  # noqa: N806
        hidden_states, pad_hw = _window_partition(hidden_states, window_size)

    attn_output = self.attn(hidden_states=hidden_states, **kwargs)
    hidden_states = attn_output

    if self.query_stride:
        window_size = self.window_size // self.query_stride[0]
        H, W = residual.shape[1:3]  # noqa: N806
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        pad_hw = (H + pad_h, W + pad_w)

    if self.window_size > 0:
        # Set together with pad_hw under the same window_size > 0 guard above;
        # mypy can't correlate the two blocks, so assert the shared invariant.
        assert H is not None
        assert W is not None
        assert pad_hw is not None
        hidden_states = _window_unpartition(hidden_states, window_size, pad_hw, (H, W))

    hidden_states = residual + hidden_states
    layernorm_output = self.layer_norm2(hidden_states)
    return cast("torch.Tensor", hidden_states + self.mlp(layernorm_output))


def _patched_sam2_embed_points(
    self: Any, points: torch.Tensor, labels: torch.Tensor, pad: bool
) -> torch.Tensor:
    """Patched _embed_points with arithmetic masking instead of torch.where.

    Internal helper used by _patched_sam2_prompt_encoder_forward.
    """
    points = points + 0.5
    if pad:
        points = F.pad(points, (0, 0, 0, 1), mode="constant", value=0)
        labels = F.pad(labels, (0, 1), mode="constant", value=-1)

    input_shape = (self.input_image_size, self.input_image_size)
    point_embedding = self.shared_embedding(points, input_shape)

    # Replace torch.where(labels == -1) with arithmetic masking
    mask_neg1 = (labels == -1).unsqueeze(-1).to(point_embedding.dtype)
    not_a_point = self.not_a_point_embed.weight.expand_as(point_embedding)
    point_embedding = mask_neg1 * not_a_point + (1 - mask_neg1) * point_embedding

    # Replace torch.where(labels != -10) with arithmetic masking
    mask_neg10 = (labels == -10).unsqueeze(-1).to(point_embedding.dtype)
    point_embedding = (1 - mask_neg10) * point_embedding

    # Add point type embedding
    mask_ge0 = (labels >= 0).unsqueeze(-1).to(point_embedding.dtype)
    point_embed_lookup = self.point_embed(labels.clamp(min=0))
    return cast("torch.Tensor", point_embedding + point_embed_lookup * mask_ge0)


def _patched_sam2_prompt_encoder_forward(
    self: Any,
    input_points: torch.Tensor | None,
    input_labels: torch.Tensor | None,
    input_boxes: torch.Tensor | None,
    input_masks: torch.Tensor | None,
    use_mask_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patched forward with arithmetic mask blending via use_mask_input flag.

    Target: Sam2PromptEncoder
    Changes:
        1. Uses arithmetic masking in _embed_points (replaces torch.where)
        2. Supports use_mask_input flag for arithmetic mask/no-mask blending

    Applied via Sam2ModelPatcher during export.
    """
    # Patch _embed_points to use arithmetic masking
    self._embed_points = types.MethodType(_patched_sam2_embed_points, self)

    # If use_mask_input not provided, use original behavior
    if use_mask_input is None:
        return cast(
            "tuple[torch.Tensor, torch.Tensor]",
            self._original_forward(input_points, input_labels, input_boxes, input_masks),
        )

    # Get batch size
    batch_size = 1
    if input_points is not None:
        batch_size = input_points.shape[0]
    elif input_boxes is not None:
        batch_size = input_boxes.shape[0]

    # Get sparse embeddings (with patched _embed_points)
    sparse_embeddings, _ = self._original_forward(input_points, input_labels, input_boxes, None)

    # Arithmetic mask blending
    mask_dense = self.mask_embed(input_masks)
    no_mask_dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
        batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
    )

    flag = use_mask_input.reshape(-1, 1, 1, 1).to(mask_dense.dtype)
    dense_embeddings = (1.0 - flag) * no_mask_dense + flag * mask_dense

    return sparse_embeddings, dense_embeddings


# =============================================================================
# Custom Model Patcher for SAM2
# =============================================================================

# Target class names for instance-level patching.
# These are matched by type(module).__name__ to stay architecture-agnostic
# (no class import needed; the classes come from transformers internals).
_SAM2_PATCH_TARGETS: dict[str, Callable[..., Any]] = {
    "Sam2MultiScaleBlock": _patched_sam2_multiscale_block_forward,
    "Sam2PromptEncoder": _patched_sam2_prompt_encoder_forward,
}


class Sam2ModelPatcher(ModelPatcher):  # type: ignore[misc]  # optimum base is untyped
    """Custom ModelPatcher that applies SAM2 QNN-compatible patches during export.

    Patches Sam2MultiScaleBlock and Sam2PromptEncoder forward methods on all
    matching instances found via model.named_modules(). Each patched forward
    expects ``self._original_forward`` to be set, which this patcher handles.

    Used as ``_MODEL_PATCHER`` on all SAM2 OnnxConfig classes so patches are
    applied only during the ``patch_model_for_export()`` context.
    """

    def __init__(
        self,
        config: OnnxConfig,
        model: torch.nn.Module,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config, model, model_kwargs=model_kwargs)
        self._sam2_originals: list[tuple[torch.nn.Module, Any]] = []

    def __enter__(self) -> Sam2ModelPatcher:
        super().__enter__()
        for _name, module in self._model.named_modules():
            class_name = type(module).__name__
            patch_fn = _SAM2_PATCH_TARGETS.get(class_name)
            if patch_fn is not None:
                self._sam2_originals.append((module, module.forward))
                module._original_forward = module.forward
                module.forward = types.MethodType(patch_fn, module)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for module, original_forward in self._sam2_originals:
            module.forward = original_forward
            if hasattr(module, "_original_forward"):
                del module._original_forward
        self._sam2_originals.clear()
        super().__exit__(exc_type, exc_val, exc_tb)


# =============================================================================
# Custom Dummy Input Generators for SAM2
# =============================================================================
class Sam2PointsInputGenerator(DummyInputGenerator):  # type: ignore[misc]  # optimum base is untyped
    """Points input generator for SAM2 decoder.

    Generates:
        - input_points: [B, 1, N, 2] point coordinates (0-1024)
        - input_labels: [B, 1, N] point labels (int64)
    """

    SUPPORTED_INPUT_NAMES = ("input_points", "input_labels")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = 1,
        point_batch_size: int = 1,
        nb_points_per_image: int = 5,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.batch_size = batch_size
        self.point_batch_size = point_batch_size
        self.nb_points_per_image = nb_points_per_image

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        if input_name == "input_points":
            shape = [
                self.batch_size,
                self.point_batch_size,
                self.nb_points_per_image,
                2,
            ]
            # Scale to 0-1024 pixel coordinates. optimum's DummyInputGenerator is
            # untyped, so random_*_tensor returns Any.
            return cast(
                "torch.Tensor",
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype) * 1024,
            )
        # input_labels
        shape = [self.batch_size, self.point_batch_size, self.nb_points_per_image]
        # Labels: 1=positive for all test points
        return cast(
            "torch.Tensor",
            self.random_int_tensor(
                shape, max_value=2, min_value=1, framework=framework, dtype=int_dtype
            ),
        )


class Sam2EmbeddingsInputGenerator(DummyInputGenerator):  # type: ignore[misc]  # optimum base is untyped
    """Embeddings input generator for SAM2 mask generation decoder.

    Generates raw (pre-projection) encoder outputs:
        - image_embeddings:   [B, 256, 64, 64]   - FPN level 2
        - high_res_features0: [B, 256, 256, 256]  - FPN level 0 (raw, before conv_s0)
        - high_res_features1: [B, 256, 128, 128]  - FPN level 1 (raw, before conv_s1)
    """

    SUPPORTED_INPUT_NAMES = (
        "image_embeddings",
        "high_res_features0",
        "high_res_features1",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.batch_size = batch_size

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        if input_name == "image_embeddings":
            shape = [self.batch_size, 256, 64, 64]
        elif input_name == "high_res_features0":
            shape = [self.batch_size, 256, 256, 256]
        elif input_name == "high_res_features1":
            shape = [self.batch_size, 256, 128, 128]
        else:
            raise ValueError(f"Unknown input: {input_name}")

        # optimum's DummyInputGenerator is untyped, so random_float_tensor returns Any.
        return cast(
            "torch.Tensor", self.random_float_tensor(shape, framework=framework, dtype=float_dtype)
        )


class Sam2MaskInputGenerator(DummyInputGenerator):  # type: ignore[misc]  # optimum base is untyped
    """Mask input generator for SAM2 decoder refinement.

    Generates:
        - mask_input: [B, 1, 256, 256] previous mask for refinement
        - use_mask_input: [B] flag (0.0=first iteration, 1.0=refinement)
    """

    SUPPORTED_INPUT_NAMES = ("mask_input", "use_mask_input")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.batch_size = batch_size

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        if input_name == "mask_input":
            shape = [self.batch_size, 1, 256, 256]
            return torch.zeros(shape, dtype=torch.float32)
        # use_mask_input
        shape = [self.batch_size]
        # Default to first iteration (0.0 = don't use mask)
        return torch.zeros(shape, dtype=torch.float32)


# =============================================================================
# Normalized Config with Default Image Size
# =============================================================================
class Sam2NormalizedVisionConfig(NormalizedVisionConfig):  # type: ignore[misc]  # optimum base is untyped
    """NormalizedVisionConfig with default image_size for SAM2.

    SAM2 uses 1024x1024 input images by default.
    """

    DEFAULT_IMAGE_SIZE = 1024

    def __getattr__(self, attr_name: str) -> Any:
        """Return default image_size when not found in model config."""
        try:
            return super().__getattr__(attr_name)
        except AttributeError:
            if attr_name == "image_size":
                return self.DEFAULT_IMAGE_SIZE
            raise


# =============================================================================
# Optimum ONNX Export Config Registrations
# =============================================================================


# -----------------------------------------------------------------------------
# Encoder-only export (image-feature-extraction task)
# TasksManager requires canonical name "feature-extraction"; our MODEL_CLASS_MAPPING
# and TASK_DATASET_MAPPING use "image-feature-extraction" for correct routing.
# -----------------------------------------------------------------------------
@register_onnx_overwrite("sam2", "feature-extraction", library_name="transformers")
@register_onnx_overwrite("sam2_video", "feature-extraction", library_name="transformers")
@register_onnx_overwrite("sam2_vision_model", "feature-extraction", library_name="transformers")
class Sam2ImageEncoderIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for SAM2 image encoder (vision_encoder component).

    Task: image-feature-extraction (encoder-only export)
    Model types: sam2, sam2_video, sam2_vision_model

    Inputs:
        - pixel_values: [B, 3, 1024, 1024]

    Outputs:
        - image_embeddings: [B, 256, 64, 64]
        - high_res_features1: [B, 32, 256, 256]
        - high_res_features2: [B, 64, 128, 128]
    """

    NORMALIZED_CONFIG_CLASS = Sam2NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)
    _MODEL_PATCHER = Sam2ModelPatcher

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return input tensors for SAM2 encoder."""
        return {
            "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return output tensors for SAM2 encoder."""
        return {
            "image_embeddings": {0: "batch_size"},
            "high_res_features1": {0: "batch_size"},
            "high_res_features2": {0: "batch_size"},
        }


# -----------------------------------------------------------------------------
# Full model export (image-segmentation task) - encoder + decoder monolith
# -----------------------------------------------------------------------------
@register_onnx_overwrite("sam2", "image-segmentation", library_name="transformers")
@register_onnx_overwrite("sam2_video", "image-segmentation", library_name="transformers")
class Sam2IOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for SAM2 full model (encoder + decoder monolith).

    Task: image-segmentation (full model export)
    Model types: sam2, sam2_video

    Inputs:
        - pixel_values: [B, 3, 1024, 1024] - input image
        - input_points: [B, 1, N, 2] - point prompts
        - input_labels: [B, 1, N] - point labels (0=neg, 1=pos)

    Outputs:
        - masks: [B, 3, 1024, 1024] - predicted masks
        - iou_scores: [B, 3] - mask quality scores
    """

    NORMALIZED_CONFIG_CLASS = Sam2NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyVisionInputGenerator,
        Sam2PointsInputGenerator,
    )
    _MODEL_PATCHER = Sam2ModelPatcher

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return input tensors for full SAM2 model."""
        return {
            "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
            "input_points": {0: "batch_size", 2: "num_points"},
            "input_labels": {0: "batch_size", 2: "num_points"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return output tensors for full SAM2 model."""
        return {
            "masks": {0: "batch_size"},
            "iou_scores": {0: "batch_size"},
        }


# -----------------------------------------------------------------------------
# Mask generation export (SAM2MaskGeneration wrapper)
# -----------------------------------------------------------------------------
@register_onnx_overwrite("sam2", "mask-generation", library_name="transformers")
@register_onnx_overwrite("sam2_video", "mask-generation", library_name="transformers")
class Sam2MaskGenerationIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for SAM2MaskGeneration (decoder with raw FPN inputs).

    Model: facebook/sam2-hiera-small (decoder wrapper)
    Uses SAM2MaskGeneration nn.Module which takes raw 256-channel FPN features
    and applies conv_s0/conv_s1 projections internally.

    Inputs:
        - input_points:       {0: "batch_size"} [B, 1, N, 2]
        - input_labels:       {0: "batch_size"} [B, 1, N]
        - image_embeddings:   {0: "batch_size"} [B, 256, 64, 64]
        - high_res_features0: {0: "batch_size"} [B, 256, 256, 256]
        - high_res_features1: {0: "batch_size"} [B, 256, 128, 128]
        - mask_input:         {0: "batch_size"} [B, 1, 256, 256]
        - use_mask_input:     {0: "batch_size"} [B]

    Outputs:
        - masks:          {0: "batch_size"} [B, 3, 1024, 1024]
        - iou_scores:     {0: "batch_size"} [B, 3]
        - low_res_masks:  {0: "batch_size"} [B, 3, 256, 256]
    """

    NORMALIZED_CONFIG_CLASS = Sam2NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        Sam2PointsInputGenerator,
        Sam2EmbeddingsInputGenerator,
        Sam2MaskInputGenerator,
    )
    _MODEL_PATCHER = Sam2ModelPatcher

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return input tensors for SAM2 mask generation."""
        return {
            "input_points": {0: "batch_size"},
            "input_labels": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"},
            "high_res_features0": {0: "batch_size"},
            "high_res_features1": {0: "batch_size"},
            "mask_input": {0: "batch_size"},
            "use_mask_input": {0: "batch_size"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return output tensors for SAM2 mask generation."""
        return {
            "masks": {0: "batch_size"},
            "iou_scores": {0: "batch_size"},
            "low_res_masks": {0: "batch_size"},
        }


# =============================================================================
# SAM v1 Custom Dummy Input Generators
# =============================================================================
class SamEmbeddingsInputGenerator(DummyInputGenerator):  # type: ignore[misc]  # optimum base is untyped
    """Embeddings input generator for SAM v1 mask generation decoder.

    Generates:
        - image_embeddings: [B, 256, 64, 64] - From vision encoder
    """

    SUPPORTED_INPUT_NAMES = ("image_embeddings",)

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.batch_size = batch_size

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        # SAM v1 decoder export expects the canonical embedding shape from the
        # vision encoder output; this mirrors the existing SAM2 generator path.
        shape = [self.batch_size, 256, 64, 64]
        # optimum's DummyInputGenerator is untyped, so random_float_tensor returns Any.
        return cast(
            "torch.Tensor", self.random_float_tensor(shape, framework=framework, dtype=float_dtype)
        )


# =============================================================================
# SAM v1 Optimum ONNX Export Config Registration
# =============================================================================


# -----------------------------------------------------------------------------
# Mask generation export (SAMMaskGeneration wrapper) - SAM v1
# -----------------------------------------------------------------------------
@register_onnx_overwrite("sam", "mask-generation", library_name="transformers")
class SamMaskGenerationIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for SAMMaskGeneration (SAM v1 decoder).

    Model: facebook/sam-vit-huge, facebook/sam-vit-large, facebook/sam-vit-base
    Uses SAMMaskGeneration nn.Module which takes image_embeddings from the
    vision encoder and runs prompt encoding + mask decoding.

    Inputs:
        - input_points:     {0: "batch_size"} [B, 1, N, 2]
        - input_labels:     {0: "batch_size"} [B, 1, N]
        - image_embeddings: {0: "batch_size"} [B, 256, 64, 64]
        - mask_input:       {0: "batch_size"} [B, 1, 256, 256]
        - use_mask_input:   {0: "batch_size"} [B]

    Outputs:
        - masks:          {0: "batch_size"} [B, 3, 1024, 1024]
        - iou_scores:     {0: "batch_size"} [B, 3]
        - low_res_masks:  {0: "batch_size"} [B, 3, 256, 256]
    """

    # SAM v1 also uses 1024x1024 default image size, so this normalized config
    # is intentionally shared across SAM v1 and SAM2 export configs.
    NORMALIZED_CONFIG_CLASS = Sam2NormalizedVisionConfig
    # SAM v1 reuses SAM2-named generators because prompt/mask tensor shapes
    # are identical for this export path.
    DUMMY_INPUT_GENERATOR_CLASSES = (
        Sam2PointsInputGenerator,
        SamEmbeddingsInputGenerator,
        Sam2MaskInputGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Return input tensors for SAM v1 mask generation."""
        return {
            "input_points": {0: "batch_size"},
            "input_labels": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"},
            "mask_input": {0: "batch_size"},
            "use_mask_input": {0: "batch_size"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return output tensors for SAM v1 mask generation."""
        return {
            "masks": {0: "batch_size"},
            "iou_scores": {0: "batch_size"},
            "low_res_masks": {0: "batch_size"},
        }


__all__ = [
    "MODEL_CLASS_MAPPING",
    "SAM2MaskGeneration",
    "SAMMaskGeneration",
    "Sam2IOConfig",
    "Sam2ImageEncoderIOConfig",
    "Sam2MaskGenerationIOConfig",
    "Sam2ModelPatcher",
    "Sam2NormalizedVisionConfig",
    "SamMaskGenerationIOConfig",
    "_patched_sam2_multiscale_block_forward",
    "_patched_sam2_prompt_encoder_forward",
]
