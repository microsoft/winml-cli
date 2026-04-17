# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Encoder-Decoder inference model and shared input generator.

Class hierarchy::

    WinMLCompositeModel                             — multi-component base
      └─ WinMLEncoderDecoderModel(GenerationMixin) — encoder-decoder inference
           ├─ WinMLT5Model (t5.py)                 — WinMLStaticCache
           └─ WinMLMu2Model (mu2.py)               — WinMLSlidingWindowCache

How ``forward()`` works:

1. Encoder runs once (via ``get_encoder()``), hidden states cached by
   GenerationMixin across decode steps.

2. Each decode step: ``_resolve_cache`` unwraps GenerationMixin's
   ``EncoderDecoderCache`` wrapper (or creates a fresh ``WinMLCache``
   on first call).  Cache type is determined by ``get_cache_class()``.

3. Feeds are built from ``model_kwargs`` (decoder_input_ids, attention_mask)
   plus generated inputs (encoder_hidden_states, decoder_attention_mask,
   position input, KV buffers).  ``pad_inputs`` filters to ONNX input
   names and pads undersized tensors.

4. After ONNX inference, ``cache.update_all_layers(outputs)`` writes
   present KV back and advances step — fully polymorphic, no isinstance.

Cache-type gotchas (lessons learned):

- **GenerationMixin wraps cache**: On the first decode call, GenerationMixin
  may pass an ``EncoderDecoderCache`` (not None).  ``_resolve_cache`` must
  unwrap it, and cache reset must check ``not isinstance(WinMLCache)``.

- **Causal mask with seq_len=1**: ``torch.tril(ones(1, N))`` only keeps
  column 0.  For single-token KV-cached decoding, the decoder_attention_mask
  alone is sufficient — no tril needed.

- **RoPE position vs buffer position**: With ``WinMLSlidingWindowCache``,
  the ONNX input is ``position_id`` (absolute sequence position for RoPE).
  With ``WinMLStaticCache``, it's ``cache_position`` (= buffer position =
  sequence position).

- **T5 cannot use sliding window**: ``T5Attention.compute_bias`` assumes
  ``buffer_position == sequence_position`` via ``arange(key_length)``.
  See ``WinMLT5Model.get_cache_class()`` for details.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from optimum.utils.input_generators import DummyInputGenerator
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from ...utils.data_utils import pad_inputs
from .composite_model import WinMLCompositeModel


if TYPE_CHECKING:
    from optimum.utils import NormalizedConfig
    from transformers import Cache, PretrainedConfig

logger = logging.getLogger(__name__)


# =============================================================================
# EncoderDecoderInputGenerator — shared dummy input generator
# =============================================================================


class EncoderDecoderInputGenerator(DummyInputGenerator):
    """Generates decoder base inputs for encoder-decoder models.

    Produces ``decoder_input_ids``, ``encoder_hidden_states``,
    ``attention_mask`` (encoder), ``decoder_attention_mask``, and
    ``cache_position``. Reads dimensions from ``NormalizedConfig``.
    """

    SUPPORTED_INPUT_NAMES = (
        "decoder_input_ids",
        "encoder_hidden_states",
        "attention_mask",
        "decoder_attention_mask",
        "cache_position",
        "position_id",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = 1,
        max_cache_len: int | None = None,
        sequence_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.batch_size = batch_size
        self.d_model = normalized_config.hidden_size
        self.enc_seq = sequence_length or getattr(normalized_config, "sequence_length", 16)
        self.max_cache_len = max_cache_len or normalized_config.max_cache_len
        self.vocab_size = normalized_config.vocab_size

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        """Generate a dummy tensor for the given input name."""
        if input_name == "decoder_input_ids":
            return self.random_int_tensor(
                (self.batch_size, 1),
                max_value=self.vocab_size,
                framework=framework,
                dtype=int_dtype,
            )
        if input_name == "encoder_hidden_states":
            return self.random_float_tensor(
                (self.batch_size, self.enc_seq, self.d_model),
                framework=framework,
                dtype=float_dtype,
            )
        if input_name == "attention_mask":
            return torch.ones(self.batch_size, self.enc_seq, dtype=torch.int64)
        if input_name == "decoder_attention_mask":
            return torch.ones(self.batch_size, self.max_cache_len, dtype=torch.int64)
        if input_name == "cache_position":
            return torch.tensor([5], dtype=torch.int64)  # arbitrary position for tracing
        if input_name == "position_id":
            return torch.tensor([5], dtype=torch.int64)  # absolute seq position for RoPE
        raise ValueError(f"Unknown input: {input_name}")


# =============================================================================
# WinMLEncoderDecoderModel — encoder-decoder with StaticCache
# =============================================================================


class WinMLEncoderDecoderModel(WinMLCompositeModel, GenerationMixin):
    """composite model with HF GenerationMixin support.

    Expects sub-components ``"encoder"`` and ``"decoder"`` in
    ``_SUB_MODEL_CONFIG``. Provides the full interface required by
    ``GenerationMixin.generate()`` for encoder-decoder models with
    static KV cache.

    Input/output names and shapes are read from ONNX I/O metadata — no
    model-specific names are assumed.
    """

    main_input_name = "input_ids"
    base_model_prefix = ""
    _is_stateful = False
    _supports_cache_class = False

    def __init__(
        self,
        sub_models: dict[str, Any],
        config: PretrainedConfig,
    ) -> None:
        super().__init__(sub_models, config)
        raw_encoder = sub_models["encoder"]
        self._decoder = sub_models["decoder"]

        # Build {name: shape} lookups from ONNX I/O metadata
        enc_io = raw_encoder.io_config
        enc_expected = dict(
            zip(enc_io.get("input_names", []), enc_io.get("input_shapes", []), strict=False)
        )
        # Wrap encoder with auto-padding so all callsites just use self._encoder(...)
        self._encoder = self._EncoderWithInputPadding(raw_encoder, enc_expected)

        dec_io = self._decoder.io_config
        self._dec_expected = dict(
            zip(dec_io.get("input_names", []), dec_io.get("input_shapes", []), strict=False)
        )

        # Max decode length and KV dtype from decoder ONNX metadata
        self._max_dec = self._dec_expected["past_0_key"][2]
        self._num_kv_layers = sum(
            1 for n in self._dec_expected if n.startswith("past_") and n.endswith("_key")
        )
        # Resolve KV cache dtype from ONNX input types (fp32 or fp16)
        dec_type_map = dict(
            zip(dec_io.get("input_names", []), dec_io.get("input_types", []), strict=False)
        )
        import numpy as np

        _np_dtype = dec_type_map.get("past_0_key", np.float32)
        self._kv_dtype = torch.from_numpy(np.zeros(1, dtype=_np_dtype)).dtype

    # ----- Encoder -----

    class _EncoderWithInputPadding(torch.nn.Module):
        """Wraps an encoder sub-model with auto-padding to ONNX expected shapes.

        Matches kwargs against ONNX input names, pads undersized tensors,
        and forwards to the underlying WinMLAutoModel. Used as both
        ``self._encoder`` (direct calls) and the return value of
        ``get_encoder()`` (GenerationMixin contract).
        """

        def __init__(self, encoder: Any, expected: dict[str, list[int]]) -> None:
            super().__init__()
            self._encoder = encoder
            self._expected = expected

        def forward(self, **kwargs: Any) -> BaseModelOutput:
            feeds = pad_inputs(kwargs, self._expected)
            return self._encoder(**feeds)

    def get_encoder(self) -> torch.nn.Module:
        """Return encoder for GenerationMixin (already wrapped with padding)."""
        return self._encoder

    def can_generate(self) -> bool:  # noqa: D102
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_outputs: BaseModelOutput | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build decoder inputs for each generate() step."""
        return {
            "decoder_input_ids": input_ids[:, -1:],
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    # ----- Cache management -----

    @classmethod
    def get_cache_class(cls) -> type:
        """Return the WinMLCache subclass. Subclasses must override."""
        raise NotImplementedError

    def _resolve_cache(self, past_key_values: Any) -> Any:
        """Unwrap or create the WinMLCache for this generation step.

        1. Unwrap EncoderDecoderCache wrapper (GenerationMixin may add it).
        2. If already a WinMLCache, return directly.
        3. Otherwise create a fresh one and reset it.
        """
        from .kv_cache import WinMLCache

        # (1) Unwrap EncoderDecoderCache
        if hasattr(past_key_values, "self_attention_cache"):
            past_key_values = past_key_values.self_attention_cache

        # (2) Already our cache — return as-is
        if isinstance(past_key_values, WinMLCache):
            return past_key_values

        # (3) Create fresh cache and reset
        kv_shape = self._dec_expected["past_0_key"]
        cache = self.get_cache_class().create(self.config, kv_shape, self._kv_dtype)
        cache.reset()
        return cache

    # ----- Forward (decoder via WinMLAutoModel + KV cache) -----

    def forward(
        self,
        *,
        encoder_outputs: BaseModelOutput | tuple | None = None,
        past_key_values: Cache | None = None,
        input_ids: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> Seq2SeqLMOutput:
        """Run decoder with static KV cache.

        Args:
            encoder_outputs: Pre-computed encoder hidden states.
            past_key_values: StaticCache (or wrapper) from previous step.
            input_ids: Fallback — run encoder if encoder_outputs is None.
            **model_kwargs: Remaining kwargs forwarded to the decoder ONNX
                (e.g., decoder_input_ids, attention_mask). Each tensor is
                auto-padded to match the ONNX model's expected input shape.
        """
        # Encoder hidden states
        if encoder_outputs is None and input_ids is not None:
            encoder_outputs = self._encoder(input_ids=input_ids, **model_kwargs)
        if encoder_outputs is None:
            raise ValueError("Either encoder_outputs or input_ids required")
        enc_h = encoder_outputs["last_hidden_state"]

        # Resolve or create cache (subclasses override _create_cache).
        cache = self._resolve_cache(past_key_values)

        fc = cache.step
        dec_mask = cache.build_decoder_mask(self._max_dec)

        feeds: dict[str, Any] = dict(model_kwargs)
        feeds.setdefault("encoder_hidden_states", enc_h.detach())
        feeds.setdefault("decoder_attention_mask", dec_mask)
        feeds.setdefault(cache.position_input_name, torch.tensor([fc], dtype=torch.int64))
        for i in range(self._num_kv_layers):
            feeds[f"past_{i}_key"] = cache.layers[i].keys.detach()
            feeds[f"past_{i}_value"] = cache.layers[i].values.detach()

        # Run decoder ONNX (pad_inputs filters to expected names + pads)
        outputs = self._decoder(**pad_inputs(feeds, self._dec_expected))

        # Write present KV back and advance step
        cache.update_all_layers(outputs)

        return Seq2SeqLMOutput(
            logits=outputs["logits"],
            past_key_values=cache,
        )
