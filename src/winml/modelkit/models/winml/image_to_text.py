# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinMLModelForImageToText.

Thin wrapper for image-to-text (captioning / OCR) inference via ONNX Runtime.
Supports BLIP, VisionEncoderDecoder (TrOCR, Donut, ViT-GPT2, Nougat, etc.),
and non-generative models (MGP-STR).

All encoder-decoder models export as a monolithic ONNX graph with uniform I/O:
    Inputs:  pixel_values, input_ids, attention_mask
    Outputs: logits (batch, seq_len, vocab_size)

Generation (greedy, beam search, sampling) is handled by HF's GenerationMixin.
This class only provides forward() to run the ONNX model plus three
GenerationMixin hooks:
  - _prepare_model_inputs()  — keeps pixel_values in the decode loop, seeds input_ids
  - prepare_inputs_for_generation() — passes pixel_values + full input_ids per step
  - can_generate()           — tells HF pipeline to call generate()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from transformers.generation.utils import GenerationMixin

from .base import WinMLPreTrainedModel


if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class ImageToTextModelOutput:
    """Output container for image-to-text forward pass.

    Implements ``__contains__`` and ``__getitem__`` for GenerationMixin compat.
    GenerationMixin checks ``"past_key_values" in outputs`` and accesses it
    via ``outputs["past_key_values"]``.  Since we have no KV cache, the field
    is always ``None`` and ``__contains__`` returns ``False`` for it, cleanly
    skipping the cache update path.
    """

    logits: torch.Tensor
    past_key_values: None = None

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and getattr(self, key) is not None

    def __getitem__(self, key: str):
        return getattr(self, key)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class WinMLModelForImageToText(WinMLPreTrainedModel, GenerationMixin):
    """WinML model for image-to-text tasks (captioning, OCR).

    Wraps a monolithic ONNX model (pixel_values + input_ids → logits).
    All decoding logic (greedy, beam search, sampling) is provided by
    HF's ``GenerationMixin`` — this class only implements ``forward()``
    and the hooks that bridge the ONNX/HF gap.
    """

    main_input_name = "pixel_values"
    _is_stateful = False
    _supports_cache_class = False

    # -----------------------------------------------------------------
    # Properties required by GenerationMixin / pipeline
    # -----------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """``torch.device`` for GenerationMixin compatibility.

        GenerationMixin accesses ``self.device.type`` and passes
        ``self.device`` to ``torch.tensor(..., device=)``.  Both require a
        ``torch.device``, not a plain string.
        The base class returns a string (``"auto"``, ``"npu"``, …).

        This does NOT affect the ONNX inference device — that is controlled
        by the EP policy in ``WinMLSession``.
        """
        return torch.device("cpu")

    def can_generate(self) -> bool:
        """Tell HF pipeline this model supports ``generate()``."""
        return True

    @property
    def generation_config(self):
        """Lazy ``GenerationConfig`` built from the model's HF config."""
        if not hasattr(self, "_generation_config"):
            from transformers import GenerationConfig

            gc_kwargs: dict[str, Any] = {}
            if self.config is not None:
                for attr in (
                    "decoder_start_token_id",
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                ):
                    val = self._resolve_config_attr(attr)
                    if val is not None:
                        gc_kwargs[attr] = val
            gc_kwargs.setdefault("max_new_tokens", 20)
            self._generation_config = GenerationConfig(**gc_kwargs)
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value):
        self._generation_config = value

    # -----------------------------------------------------------------
    # GenerationMixin hooks
    # -----------------------------------------------------------------

    def _prepare_model_inputs(
        self,
        inputs: torch.Tensor | None = None,
        bos_token_id: torch.Tensor | None = None,
        model_kwargs: dict | None = None,
    ):
        """Keep ``pixel_values`` in the decode loop and seed ``input_ids``.

        In HF's PyTorch models (e.g. ``VisionEncoderDecoderModel``), the
        encoder runs once at the start of ``generate()`` and the result is
        stored as ``encoder_outputs`` in ``model_kwargs`` — raw
        ``pixel_values`` are consumed and discarded.  Our monolithic ONNX
        model has no separate encoder: the full graph (encoder + decoder)
        runs every decode step, so ``pixel_values`` must be available at
        every call to ``forward()``.

        Problems solved:
        1. GenerationMixin pops ``main_input_name`` (``pixel_values``) from
           ``model_kwargs`` and never passes it back to the decode loop.
           We re-inject it so ``prepare_inputs_for_generation()`` receives it.
        2. HF's image-to-text pipeline does not send ``input_ids``.  We seed
           the decoder with ``[decoder_start_token_id]`` (or ``[bos_token_id]``).
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Seed input_ids when the pipeline doesn't provide them
        if "input_ids" not in model_kwargs and inputs is not None:
            batch_size = inputs.shape[0]
            start_id = self._resolve_config_attr("decoder_start_token_id")
            if start_id is None:
                start_id = self._resolve_config_attr("bos_token_id")
            if start_id is None:
                start_id = 0
            model_kwargs["input_ids"] = torch.full(
                (batch_size, 1), int(start_id), dtype=torch.long, device=inputs.device
            )

        inputs_tensor, model_input_name, model_kwargs = super()._prepare_model_inputs(
            inputs, bos_token_id, model_kwargs
        )

        # Re-inject pixel_values so it survives into the decode loop
        if model_input_name == "pixel_values":
            model_kwargs["pixel_values"] = inputs_tensor
        return inputs_tensor, model_input_name, model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the input dict for each decode step.

        No KV-cache slicing — the monolithic ONNX model receives the full
        (growing) ``input_ids`` and the same ``pixel_values`` every step.
        """
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }

    # -----------------------------------------------------------------
    # ONNX forward
    # -----------------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor | np.ndarray | None = None,
        input_ids: torch.Tensor | np.ndarray | None = None,
        attention_mask: torch.Tensor | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ImageToTextModelOutput:
        """Run a single forward pass through the ONNX model.

        Handles static sequence length by padding ``input_ids`` and
        ``attention_mask`` to the ONNX model's expected size, then slices
        logits back to the real length so ``GenerationMixin`` reads
        ``logits[:, -1, :]`` at the correct position.
        """
        feed: dict[str, Any] = {}

        if pixel_values is not None:
            feed["pixel_values"] = pixel_values

        real_seq_len = 0
        expected_seq_len = self._get_expected_seq_len()

        if input_ids is not None:
            input_ids_t = (
                input_ids
                if isinstance(input_ids, torch.Tensor)
                else torch.tensor(input_ids, dtype=torch.long)
            )
            real_seq_len = input_ids_t.shape[-1]

            if expected_seq_len is not None and real_seq_len < expected_seq_len:
                pad_len = expected_seq_len - real_seq_len
                input_ids_t = torch.nn.functional.pad(input_ids_t, (0, pad_len), value=0)
                if attention_mask is None:
                    attention_mask = torch.cat(
                        [
                            torch.ones(input_ids_t.shape[0], real_seq_len, dtype=torch.long),
                            torch.zeros(input_ids_t.shape[0], pad_len, dtype=torch.long),
                        ],
                        dim=-1,
                    )

            feed["input_ids"] = input_ids_t

        if attention_mask is not None:
            mask_t = (
                attention_mask
                if isinstance(attention_mask, torch.Tensor)
                else torch.tensor(attention_mask, dtype=torch.long)
            )
            if expected_seq_len is not None and mask_t.shape[-1] < expected_seq_len:
                mask_t = torch.nn.functional.pad(
                    mask_t, (0, expected_seq_len - mask_t.shape[-1]), value=0
                )
            feed["attention_mask"] = mask_t

        inputs = self._format_inputs(**feed)
        outputs = self._run_inference(inputs)
        logits = outputs.get("logits", next(iter(outputs.values())))

        # Slice logits back to the real sequence length so GenerationMixin's
        # ``logits[:, -1, :]`` reads the correct (last real) position.
        if real_seq_len > 0 and expected_seq_len is not None and real_seq_len < expected_seq_len:
            logits = logits[:, :real_seq_len, :]

        return ImageToTextModelOutput(logits=logits)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _get_expected_seq_len(self) -> int | None:
        """Read static sequence length from the ONNX model's input shapes.

        Returns ``None`` for dynamic shapes (no padding needed).
        """
        io = self.io_config
        for name, shape in zip(io.get("input_names", []), io.get("input_shapes", []), strict=False):
            if (
                name == "input_ids"
                and len(shape) == 2
                and isinstance(shape[1], int)
                and shape[1] > 0
            ):
                return shape[1]
        return None

    def _resolve_config_attr(self, attr: str) -> Any:
        """Resolve a config attribute, checking nested ``text_config`` / ``decoder``.

        Different architectures store token IDs at different config levels:
        - ``config.bos_token_id``
        - ``config.decoder.bos_token_id``
        - ``config.text_config.bos_token_id``
        """
        if self.config is None:
            return None
        val = getattr(self.config, attr, None)
        if val is not None:
            return val
        for sub in ("decoder", "text_config"):
            sub_cfg = getattr(self.config, sub, None)
            if sub_cfg is not None:
                val = getattr(sub_cfg, attr, None)
                if val is not None:
                    return val
        return None
