# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Image-to-text model — supports both monolithic and split ONNX architectures.

- **Monolithic** (no ``encoder_path``): single ONNX with fused encoder+decoder.
  ``DummyEncoder`` passes input through unchanged; actual encoding happens
  inside the fused ONNX graph when ``forward()`` runs the decoder session.
- **Split** (``encoder_path`` provided): separate encoder and decoder ONNX files.
  ``OnnxEncoder`` runs the encoder once; HF caches the output and feeds
  it to every decode step.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

from .generation_mixin import (
    DummyEncoder,
    OnnxEncoder,
    OnnxGenerativeInputs,
    WinMLGenerationMixin,
)


logger = logging.getLogger(__name__)


class WinMLModelForImageToText(WinMLGenerationMixin):
    """Image-to-text: monolithic or split, selected by ``encoder_path``."""

    main_input_name = "pixel_values"

    def __init__(
        self,
        onnx_path: str | Path,
        config: Any = None,
        device: str = "auto",
        *,
        encoder_path: str | Path | None = None,
        **kwargs: Any,
    ):
        super().__init__(onnx_path=onnx_path, config=config, device=device, **kwargs)

        if self.config is not None:
            self.config.is_encoder_decoder = True

        self.onnx_inputs = self._resolve_inputs(is_split=encoder_path is not None)
        self.encoder = (
            OnnxEncoder(onnx_path=encoder_path, device=device)
            if encoder_path
            else DummyEncoder()
        )
        self.encoder.main_input_name = self.main_input_name

    @property
    def generation_config(self):
        """Build generation config, reading token IDs from nested sub-configs.

        BLIP stores token IDs in ``config.text_config``, TrOCR in
        ``config.decoder``.  ``GenerationConfig.from_model_config()``
        only reads top-level attributes, so we fill in the gaps::

            config.text_config.bos_token_id = 30522
            config.decoder_start_token_id   = None
        """
        if not hasattr(self, "_generation_config"):
            config = super().generation_config
            for sub in ("text_config", "decoder"):
                sub_cfg = getattr(self.config, sub, None)
                if sub_cfg is None:
                    continue
                for attr in (
                    "decoder_start_token_id", "bos_token_id",
                    "eos_token_id", "pad_token_id",
                ):
                    if getattr(config, attr, None) is None:
                        val = getattr(sub_cfg, attr, None)
                        if val is not None:
                            setattr(config, attr, val)
            self._generation_config = config
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value):
        self._generation_config = value

    def _resolve_inputs(self, *, is_split: bool) -> OnnxGenerativeInputs:
        """Map ONNX input names to their semantic roles.

        Reads ``io_config["input_names"]`` and matches known names::

            io_config["input_names"] = ["pixel_values", "input_ids", "attention_mask"]
                                          ↓                ↓              ↓
            OnnxGenerativeInputs(
                encoder_input="pixel_values",        # monolithic only
                decoder_input_ids="input_ids",
                attention_mask="attention_mask",
                encoder_hidden_states=None,           # split only
            )
        """
        names = self.io_config["input_names"]

        def _find(*candidates: str) -> str | None:
            return next((n for n in candidates if n in names), None)

        decoder_input_ids = _find("decoder_input_ids", "input_ids")
        if decoder_input_ids is None:
            raise ValueError(f"No decoder token input found in {names}")

        return OnnxGenerativeInputs(
            decoder_input_ids=decoder_input_ids,
            attention_mask=_find("attention_mask", "decoder_attention_mask"),
            encoder_hidden_states=(
                _find("encoder_hidden_states") if is_split else None
            ),
            encoder_input=(
                _find("pixel_values") if not is_split else None
            ),
        )
