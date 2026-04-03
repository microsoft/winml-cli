# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX generation via HF ``GenerationMixin`` — forward-only override.

HF drives the decode loop (greedy, sampling, beam search).  We override
only ``forward()`` to map HF-named args to ONNX names, pad to static
shapes, run the session, and slice padding from logits.

Public contract with ``GenerationMixin``:

- ``forward()`` — one decode step, receives ``decoder_input_ids`` and
  ``encoder_outputs``, returns an object with ``.logits``.
- ``get_encoder()`` — returns a callable whose ``forward()`` signature
  explicitly names the main input (e.g. ``pixel_values``).
- ``config.is_encoder_decoder`` — must be ``True``.
- ``main_input_name`` — what the encoder consumes.
- ``device``, ``dtype``, ``generation_config``, ``can_generate()``.

No private methods of ``GenerationMixin`` are overridden.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import ModelOutput

from .base import WinMLPreTrainedModel


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed ONNX input mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnnxGenerativeInputs:
    """Typed mapping from semantic decoder roles to ONNX input names.

    Attributes:
        decoder_input_ids: ONNX name for the decoder token-ID input.
        attention_mask: ONNX name for the decoder attention mask, if any.
        encoder_hidden_states: ONNX name for encoder hidden states
            (split architecture only).
        encoder_input: ONNX name for the raw encoder input that is fed to
            the fused graph every step (monolithic architecture only).
    """

    decoder_input_ids: str
    attention_mask: str | None = None
    encoder_hidden_states: str | None = None
    encoder_input: str | None = None


# ---------------------------------------------------------------------------
# Encoder wrappers
# ---------------------------------------------------------------------------

# HF inspects ``encoder.forward()`` signature to filter kwargs.
# Both classes must name accepted parameters explicitly.

class DummyEncoder:
    """Monolithic encoder — returns input unchanged.

    For fused encoder+decoder ONNX graphs, actual encoding happens
    inside the decoder's ``forward()``.  This wrapper satisfies HF's
    encoder contract without running a separate session.
    """

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> BaseModelOutput:
        """Return input unchanged as ``BaseModelOutput``."""
        value = pixel_values if pixel_values is not None else input_ids
        return BaseModelOutput(last_hidden_state=value)

    def __call__(self, **kwargs: Any) -> BaseModelOutput:
        """Delegate to ``forward()``."""
        return self.forward(**kwargs)


class OnnxEncoder(WinMLPreTrainedModel):
    """Split encoder — runs a separate ONNX encoder session.

    Inherits from ``WinMLPreTrainedModel`` to reuse session creation
    and inference.  Returns ``BaseModelOutput`` for HF.
    """

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> BaseModelOutput:
        """Run ONNX encoder and return ``BaseModelOutput``."""
        value = pixel_values if pixel_values is not None else input_ids
        inputs = self._format_inputs(value)
        outputs = self._run_inference(inputs)
        return BaseModelOutput(last_hidden_state=next(iter(outputs.values())))

    def __call__(self, **kwargs: Any) -> BaseModelOutput:
        """Delegate to ``forward()``."""
        return self.forward(**kwargs)


# ---------------------------------------------------------------------------
# Generation mixin
# ---------------------------------------------------------------------------


class WinMLGenerationMixin(WinMLPreTrainedModel, GenerationMixin):
    """ONNX generation bridge — overrides only ``forward()``.

    HF's ``GenerationMixin`` handles the decode loop, sampling, beam
    search, stopping criteria, etc.  We provide:

    1. ``forward()`` — build ONNX feed, pad, run, slice, return logits.
    2. ``get_encoder()`` — return the encoder wrapper.
    3. Config plumbing (``generation_config``, ``can_generate``).
    """

    onnx_inputs: OnnxGenerativeInputs
    encoder: OnnxEncoder | DummyEncoder

    base_model_prefix = ""
    _is_stateful = False
    _supports_cache_class = False

    # -- forward (the one method we override) ---------------------------------

    def forward(
        self,
        decoder_input_ids: torch.LongTensor | None = None,
        encoder_outputs: BaseModelOutput | tuple | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> ModelOutput:
        """One decode step: map HF args → ONNX feed → pad → run → slice.

        Parameters are named to match what HF sends for encoder-decoder
        models.  ``past_key_values``, ``cache_position``, ``use_cache``,
        and ``return_dict`` are accepted but unused (no KV cache).
        """
        feed = self._prepare_onnx_inputs(
            decoder_input_ids, encoder_outputs.last_hidden_state,
        )

        # Run ONNX and extract logits
        real_len = decoder_input_ids.shape[1]
        outputs = self._run_inference(feed)
        logits = outputs.get("logits", next(iter(outputs.values())))

        # Strip static-shape padding from logits
        if logits.dim() == 3 and logits.shape[1] > real_len:
            logits = logits[:, :real_len, :]

        return ModelOutput(logits=logits)

    # -- ONNX input preparation ------------------------------------------------

    def _prepare_onnx_inputs(
        self,
        decoder_input_ids: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """Build ONNX feed dict from HF args.

        Pads ``decoder_input_ids`` to the static sequence length,
        creates an attention mask, and routes the encoder output to
        the correct ONNX input name::

            decoder_input_ids=[30522]  static_len=512
                → {"input_ids": [30522, 0, 0, ...],       # padded
                   "attention_mask": [1, 0, 0, ...],
                   "pixel_values": <encoder hidden states>}
        """
        mapping = self.onnx_inputs
        real_len = decoder_input_ids.shape[1]
        static_len = self._static_seq_len
        feed: dict[str, np.ndarray] = {}

        # Decoder token IDs — pad to static length
        ids = torch.nn.functional.pad(decoder_input_ids, (0, static_len - real_len))
        feed[mapping.decoder_input_ids] = ids.numpy()

        # Attention mask — 1 for real tokens, 0 for padding
        if mapping.attention_mask is not None:
            mask = torch.zeros(ids.shape[0], static_len, dtype=torch.long)
            mask[:, :real_len] = 1
            feed[mapping.attention_mask] = mask.numpy()

        # Encoder output
        hidden_np = (
            encoder_hidden_states.numpy()
            if isinstance(encoder_hidden_states, torch.Tensor)
            else np.asarray(encoder_hidden_states)
        )
        if mapping.encoder_hidden_states is not None:
            feed[mapping.encoder_hidden_states] = hidden_np
        elif mapping.encoder_input is not None:
            feed[mapping.encoder_input] = hidden_np

        return feed

    # -- Public HF contract ---------------------------------------------------

    def get_encoder(self):
        """Return encoder callable for HF's generation flow."""
        return self.encoder

    def can_generate(self) -> bool:
        """Allow HF's ``generate()`` to be called on this model."""
        return True

    @property
    def device(self) -> torch.device:
        """CPU — ONNX sessions manage device placement internally.

        HF's ``generate()`` needs a ``torch.device`` to create tensors.
        The base class returns a ``str``, so we override here.
        """
        return torch.device("cpu")

    @property
    def generation_config(self):
        """Build ``GenerationConfig`` from HF model config on first access."""
        if not hasattr(self, "_generation_config"):
            from transformers import GenerationConfig

            config = GenerationConfig.from_model_config(self.config)
            config.use_cache = False
            self._generation_config = config
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value):
        self._generation_config = value

    # -- Internal helpers (not overriding any HF method) ----------------------

    @functools.cached_property
    def _static_seq_len(self) -> int:
        """Static decoder sequence length from ``io_config``."""
        names = self.io_config["input_names"]
        shapes = self.io_config["input_shapes"]
        return shapes[names.index(self.onnx_inputs.decoder_input_ids)][1]
