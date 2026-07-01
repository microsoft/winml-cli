# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unlimited-OCR (DeepSeek-OCR family) HuggingFace Model Configuration.

``baidu/Unlimited-OCR`` (model_type ``unlimited-ocr``, architecture
``UnlimitedOCRForCausalLM``) is a ``trust_remote_code`` vision-language OCR
model. Its full ``forward`` is a generative pipeline: a dual vision encoder
(SAM ViT-B + CLIP-L-14) feeds an MLP projector, whose image embeddings are
spliced into a DeepSeek-V2 (MoE + MLA) causal LM via data-dependent tiling and
``masked_scatter_`` control flow. That generative decoder half is out of scope
for ONNX export (no vendor OnnxConfig exists; Optimum ships ``deepseek_v3``,
not ``deepseek_v2``).

Optimum has NO ``unlimited-ocr`` OnnxConfig — this module writes the
vision-tower export contract from scratch. It exposes ONLY the pure-tensor
vision sub-graph (SAM -> CLIP -> projector) under the ``feature-extraction``
task. That sub-graph is architecture-isolated by a thin wrapper whose
``forward`` bypasses the tiling/crop/scatter control flow, exporting just the
image-embedding computation the downstream LM consumes.

This is an Effort-L1 contribution per the `adding-model-support` skill: a
from-scratch OnnxConfig plus a sub-graph wrapper, no changes to the export
engine itself. The generative decoder remains unexported by design.
"""

from __future__ import annotations

from typing import Any

import torch
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyVisionInputGenerator
from torch import nn
from transformers import AutoModel

from ...export import register_onnx_overwrite


# Vision tower operates on fixed 1024x1024 RGB tiles (SAM ViT-B input geometry).
_VISION_NUM_CHANNELS = 3
_VISION_IMAGE_SIZE = 1024


# =============================================================================
# Vision-tower sub-graph wrapper
# =============================================================================


class UnlimitedOCRVisionTowerWrapper(nn.Module):
    """Export-only wrapper exposing the SAM -> CLIP -> projector sub-graph.

    The full ``UnlimitedOCRModel.forward`` performs data-dependent tiling and
    ``masked_scatter_`` splicing that is not ONNX-traceable. This wrapper
    isolates the deterministic vision path that produces image embeddings:

        sam_feat  = sam_model(pixel_values)
        clip_feat = vision_model(pixel_values, sam_feat)
        fused     = cat(clip_feat[:, 1:], sam_feat.flatten(2).permute(0, 2, 1))
        return      projector(fused)
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.sam_model = base.sam_model
        self.vision_model = base.vision_model
        self.projector = base.projector

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs: Any
    ) -> UnlimitedOCRVisionTowerWrapper:
        """Load the full model via ``AutoModel``, then wrap its vision tower.

        ``trust_remote_code`` is forwarded by the loader through ``kwargs``;
        the model's ``auto_map`` registers ``UnlimitedOCRForCausalLM`` under
        ``AutoModel``. ``get_model()`` returns the base module owning the
        ``sam_model`` / ``vision_model`` / ``projector`` attributes.
        """
        full_model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        base = full_model.get_model() if hasattr(full_model, "get_model") else full_model.model
        wrapper = cls(base)
        wrapper.eval()
        return wrapper

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return image embeddings ``[batch, 256, 1280]`` for the LM to consume."""
        sam_feat = self.sam_model(pixel_values)
        clip_feat = self.vision_model(pixel_values, sam_feat)
        fused = torch.cat(
            (clip_feat[:, 1:], sam_feat.flatten(2).permute(0, 2, 1)), dim=-1
        )
        return self.projector(fused)


# =============================================================================
# ONNX export config
# =============================================================================


@register_onnx_overwrite("unlimited-ocr", "feature-extraction", library_name="transformers")
class UnlimitedOCRVisionIOConfig(OnnxConfig):
    """From-scratch ONNX config for the Unlimited-OCR vision tower.

    Input geometry is pinned to 1024x1024 (the SAM encoder's fixed input
    size); only the batch dimension is dynamic. The single output is the
    projected image-embedding sequence consumed by the (unexported) LM.
    """

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(allow_new=True)
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)
    DEFAULT_ONNX_OPSET = 17

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Single ``pixel_values`` input; only batch is dynamic (H,W pinned)."""
        return {
            "pixel_values": {0: "batch_size"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Projected image-embedding sequence consumed by the LM."""
        return {
            "image_embeds": {0: "batch_size"},
        }

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs: Any):  # type: ignore[override]
        """Emit a fixed ``[1, 3, 1024, 1024]`` ``pixel_values`` tensor.

        The vision tower has no data-dependent control flow, so a zero tensor
        of the correct geometry traces the full sub-graph. Geometry is pinned
        here rather than derived from the nested ``vision_config`` to keep the
        export contract explicit and architecture-agnostic.
        """
        return {
            "pixel_values": torch.zeros(
                1, _VISION_NUM_CHANNELS, _VISION_IMAGE_SIZE, _VISION_IMAGE_SIZE
            )
        }


# =============================================================================
# Model Class Mapping
# =============================================================================

# (model_type, task) -> export wrapper. Binds the ``feature-extraction`` task
# on ``unlimited-ocr`` to the vision-tower sub-graph wrapper so the loader
# exports image embeddings instead of attempting the generative full model.
MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("unlimited-ocr", "feature-extraction"): UnlimitedOCRVisionTowerWrapper,
}
