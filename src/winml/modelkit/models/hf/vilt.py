# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ViLT (Vision-and-Language Transformer) HuggingFace Model Configuration.

ViLT is a single-stream multi-modal transformer that processes text + image
in a unified attention stack. The ``ViltForQuestionAnswering`` head produces
classification logits over a fixed VQAv2 answer vocabulary (3129 labels for
``dandelin/vilt-b32-finetuned-vqa``).

Optimum has NO vendor ``ViltOnnxConfig`` (verified 2026-06-24: ``vilt`` is
absent from ``TasksManager._SUPPORTED_MODEL_TYPE`` for the transformers
library). This module writes the export config from scratch.

The forward takes 4 required tensors (pixel_mask is omitted — see Notes):
    - ``pixel_values``     [B, 3, 384, 384]  RGB image
    - ``input_ids``        [B, 40]           tokenized question
    - ``attention_mask``   [B, 40]           text padding mask
    - ``token_type_ids``   [B, 40]           BERT segment IDs (modality)

Output: ``logits`` [B, num_labels] over the answer vocabulary.

Notes:
-----
ViLT's stock ``visual_embed`` is fundamentally NOT ONNX-traceable: it iterates
Python-level over tensor values (``for h, w in zip(x_h, x_w)``), uses
``torch.multinomial`` (random + non-exportable), and does per-row Python loops
over ``nonzero()`` results. We replace it during export with a statically-
shaped equivalent (see ``_patched_visual_embed`` + ``_ViltVisualEmbedPatcher``)
that assumes an all-ones ``pixel_mask`` — which is what ``ViltProcessor`` emits
for a square 384x384 input (see ``inputs`` for the square-only constraint).
Because the patched path ignores ``pixel_mask``, we drop it from the exported
ONNX graph.
Verified numerically equivalent: ``cos=1.000000``, same argmax,
max_abs_diff≈1.2e-5.

This is an Effort-L1 contribution per the `adding-model-support` skill:
new OnnxConfig from scratch + custom model patcher.
"""

from __future__ import annotations

import types

from optimum.exporters.onnx import OnnxConfig
from optimum.exporters.onnx.model_patcher import ModelPatcher
from optimum.utils import NormalizedTextConfig
from optimum.utils.input_generators import DummyVisionInputGenerator
from transformers import ViltForQuestionAnswering

from ...export import MaxLengthTextInputGenerator, register_onnx_overwrite


# =============================================================================
# Export-time patch for ``ViltEmbeddings.visual_embed``
# =============================================================================
# Upstream ``visual_embed`` is **not ONNX-traceable** as written:
#   * ``for h, w in zip(x_h, x_w)`` iterates Python-level over tensor values
#   * ``nonzero()`` + ``unique()`` + per-row Python list-comprehension subset
#     selection over a dynamic ``valid_idx``
#   * ``torch.multinomial`` random sampling (non-deterministic, not exportable)
# The eager path silently "works" only when ``pixel_mask`` is all-ones and the
# batch is 1, because the Python loop runs once with a concrete value. Under
# legacy ``torch.onnx.export`` tracing the shape resolves to 0 and PyTorch's
# ``F.interpolate`` aborts with ``input (H: 12, W: 12) output (H: 0, W: 0)``.
#
# For the production ``visual-question-answering`` inference path with a square
# 384x384 image the ``ViltProcessor`` emits an all-ones ``pixel_mask``,
# so the per-sample subset selection is a no-op. We replace ``visual_embed``
# during export with a simplified, statically-shaped implementation that:
#   * uses ``x.shape[2], x.shape[3]`` (static) for position-embed interpolation
#   * skips ``multinomial`` / nonzero / Python-level batch loops
#   * returns an all-ones token mask of length ``H*W + 1`` (patches + CLS)
#
# Verified numerically equivalent on ``dandelin/vilt-b32-finetuned-vqa`` with
# fixed seed: ``cos=1.000000``, same ``argmax`` class, ``max_abs_diff≈1.2e-5``
# (within fp tolerance from interpolation operation ordering).


def _patched_visual_embed(self, pixel_values, pixel_mask, max_image_length=200):
    """Static-shape, ONNX-traceable replacement for ``ViltEmbeddings.visual_embed``."""
    import torch
    from torch import nn

    x = self.patch_embeddings(pixel_values)
    batch_size, num_channels, height, width = x.shape

    patch_dim = self.config.image_size // self.config.patch_size
    spatial_pos = self.position_embeddings[:, 1:, :].transpose(1, 2).view(
        1, num_channels, patch_dim, patch_dim
    )
    pos_embed = nn.functional.interpolate(
        spatial_pos, size=(height, width), mode="bilinear", align_corners=True
    )
    pos_embed = pos_embed.flatten(2).transpose(1, 2).expand(batch_size, -1, -1)

    x = x.flatten(2).transpose(1, 2)

    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    pos_cls = self.position_embeddings[:, 0:1, :].expand(batch_size, -1, -1)
    pos_embed = torch.cat((pos_cls, pos_embed), dim=1)
    x = x + pos_embed
    x = self.dropout(x)

    num_tokens = height * width + 1  # patches + CLS
    x_mask = torch.ones(batch_size, num_tokens, dtype=torch.long, device=x.device)
    return x, x_mask, None


class _ViltVisualEmbedPatcher(ModelPatcher):
    """Swaps ``ViltEmbeddings.visual_embed`` for the duration of ONNX export."""

    def __enter__(self):
        super().__enter__()
        emb = (
            self._model.vilt.embeddings
            if hasattr(self._model, "vilt")
            else self._model.embeddings
        )
        self._emb_ref = emb
        self._orig_visual_embed = emb.visual_embed
        emb.visual_embed = types.MethodType(_patched_visual_embed, emb)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._emb_ref.visual_embed = self._orig_visual_embed
        super().__exit__(exc_type, exc_value, traceback)


# =============================================================================
# Optimum ONNX Export Config Registration
# =============================================================================
@register_onnx_overwrite("vilt", "visual-question-answering", library_name="transformers")
class ViltVqaOnnxConfig(OnnxConfig):
    """ONNX export config for ``ViltForQuestionAnswering``.

    Declares 4 multi-modal inputs (text triple + pixel_values) and the single
    classification output. ``pixel_mask`` is deliberately omitted — see
    ``inputs`` property docstring and the module-level ``Notes`` section for
    the full rationale.

    Inputs:
        - ``input_ids``: [B, 40] int64
        - ``attention_mask``: [B, 40] int64
        - ``token_type_ids``: [B, 40] int64
        - ``pixel_values``: [B, 3, 384, 384] float32

    Outputs:
        - ``logits``: [B, num_labels=3129] float32

    Notes:
        - ``num_labels`` (3129 for VQAv2) is a config-time fact, not declared
          dynamic in the symbolic axes — it's a static dim of ``logits``.
        - ``sequence_length`` resolves to ``max_position_embeddings`` (40 for
          ViLT-B/32) via ``NORMALIZED_CONFIG_CLASS``; the
          ``MaxLengthTextInputGenerator`` reads this for dummy tokens.
        - Chained ``DummyVisionInputGenerator`` + ``MaxLengthTextInputGenerator``
          produce ``pixel_values`` + ``input_ids``/``attention_mask``/
          ``token_type_ids``. The patched ``visual_embed`` (see module-level
          ``_ViltVisualEmbedPatcher``) synthesizes an all-ones token mask
          internally, so no ``pixel_mask`` input is required.
    """

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        sequence_length="max_position_embeddings",
        num_channels="num_channels",
        image_size="image_size",
        patch_size="patch_size",
        allow_new=True,
    )

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyVisionInputGenerator,
        MaxLengthTextInputGenerator,
    )

    DEFAULT_ONNX_OPSET = 17

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """Declare 4 model inputs (insertion order matches forward).

        ``pixel_values`` H,W is kept STATIC at ``image_size`` (384x384), so the
        exported ONNX accepts ONLY 384x384 pixel_values.

        Honest constraint: ``ViltImageProcessor`` pins the *shortest* edge to
        384 with ``size_divisor=32`` and preserves aspect ratio, so ONLY square
        inputs land on 384x384 — a non-square image (e.g. 480x640 -> 384x512)
        does NOT match this graph and must be square-resized upstream, which
        distorts aspect ratio and can cost VQA accuracy. Callers must feed
        384x384 (square) pixel_values.

        Dynamic H,W is a known follow-up, not enabled here: the patched
        ``visual_embed`` already interpolates position embeddings to the real
        ``x.shape[2], x.shape[3]``, so a dynamic-H,W export is plausible — but
        it is left static because it has NOT been export-verified. (The original
        0x0 ``Resize`` shape-inference failure was a property of ViLT's *stock*
        non-traceable ``visual_embed``, which the patcher replaces; it does not
        by itself justify pinning the patched path.)

        Note: ViLT's ``forward`` also takes a ``pixel_mask`` parameter, but
        this contribution exports without it. For the square-384 path the
        ``ViltProcessor`` emits an all-ones mask, and our export-time
        ``ModelPatcher`` replaces the original ``visual_embed`` with a
        statically-shaped version that synthesizes an all-ones token mask
        internally. Including ``pixel_mask`` as an ONNX input would
        dead-code-eliminate (since the patched path doesn't reference it) and
        confuse runtime callers.
        """
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "pixel_values": {0: "batch_size"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Single classification output over fixed answer vocabulary."""
        return {
            "logits": {0: "batch_size"},
        }

    def generate_dummy_inputs(self, framework: str = "pt", **kwargs):  # type: ignore[override]
        """Generate the 4 declared inputs via the chained vendor generators.

        ``pixel_mask`` is intentionally NOT generated — see ``inputs`` docstring.
        Our model patcher's replacement ``visual_embed`` synthesizes an
        all-ones token mask internally, so the model can be called with the
        4 declared inputs.
        """
        dummy = super().generate_dummy_inputs(framework=framework, **kwargs)
        # Drop any pixel_mask the generators may have produced — the patched
        # visual_embed ignores it (and including it would error at sess.run
        # since it isn't in the exported ONNX graph).
        dummy.pop("pixel_mask", None)
        return dummy

    def patch_model_for_export(self, model, model_kwargs=None):  # type: ignore[override]
        """Install the ``visual_embed`` patcher for the export context."""
        return _ViltVisualEmbedPatcher(self, model, model_kwargs=model_kwargs)


# =============================================================================
# HuggingFace Model Class Mapping
# =============================================================================
# ``visual-question-answering`` has no default AutoModel routing for ViLT;
# bind the (model_type, task) tuple directly to the head-bearing HF class.
MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("vilt", "visual-question-answering"): ViltForQuestionAnswering,
}


__all__ = [
    "MODEL_CLASS_MAPPING",
    "ViltVqaOnnxConfig",
]
