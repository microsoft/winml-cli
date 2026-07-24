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

2. Each decode call: ``_resolve_cache`` unwraps GenerationMixin's
   ``EncoderDecoderCache`` wrapper (or creates a fresh ``WinMLCache``
   on first call). Multi-token prompts are prefetched token by token when
   the exported decoder has a single-token static input.

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

- **Position inputs, two roles**: ``forward`` seeds ``cache_position`` from
  ``cache.get_query_cache_position(...)`` (the query's *buffer index* — used by
  HF's causal mask ``kv_idx <= q_idx`` and by T5's ``compute_bias``) and
  ``position_id`` from the absolute sequence step (used by RoPE models).
  ``pad_inputs`` then filters to whatever the decoder ONNX actually declares,
  so T5 (consumes ``cache_position``) and Mu2 (consumes ``position_id``) share
  the same wrapper code.

- **T5 on sliding window**: Works without any ``compute_bias`` patch because
  ``WinMLSlidingWindowCache.get_query_cache_position`` returns
  ``[max_cache_len - 1]`` (the rightmost buffer slot).  With that value,
  ``memory_position - context_position = j - (W-1)`` yields the correct
  negative distances for all buffer slots, and the 2D right-aligned mask
  selects the filled region.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import torch
from optimum.utils.input_generators import DummyInputGenerator
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from ...utils.data_utils import pad_inputs
from .composite_model import WinMLCompositeModel


if TYPE_CHECKING:
    from optimum.utils import NormalizedConfig
    from transformers import Cache, PretrainedConfig

    from .kv_cache import WinMLCache

logger = logging.getLogger(__name__)


# =============================================================================
# EncoderDecoderInputGenerator — shared dummy input generator
# =============================================================================


class EncoderDecoderInputGenerator(DummyInputGenerator):  # type: ignore[misc]  # optimum/transformers base is untyped
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
        self.enc_seq: int = sequence_length or cast(
            "int", getattr(normalized_config, "sequence_length", 16)
        )
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
        # optimum's DummyInputGenerator is untyped, so random_*_tensor returns Any.
        if input_name == "decoder_input_ids":
            return cast(
                "torch.Tensor",
                self.random_int_tensor(
                    (self.batch_size, 1),
                    max_value=self.vocab_size,
                    framework=framework,
                    dtype=int_dtype,
                ),
            )
        if input_name == "encoder_hidden_states":
            return cast(
                "torch.Tensor",
                self.random_float_tensor(
                    (self.batch_size, self.enc_seq, self.d_model),
                    framework=framework,
                    dtype=float_dtype,
                ),
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
        device: str = "cpu",
    ) -> None:
        super().__init__(sub_models, config, device)
        raw_encoder = sub_models["encoder"]
        self._decoder = sub_models["decoder"]

        # Build {name: shape} lookups from ONNX I/O metadata
        enc_io = raw_encoder.io_config
        enc_expected = dict(
            zip(enc_io.get("input_names", []), enc_io.get("input_shapes", []), strict=False)
        )
        self._encoder_input_names = frozenset(enc_expected)
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

        if "past_0_key" not in dec_type_map:
            raise KeyError(
                "'past_0_key' is missing from the decoder ONNX input type map; "
                "cannot derive KV cache dtype. Verify the decoder ONNX was built with "
                "PastKeyValueInputGenerator."
            )
        _np_dtype = dec_type_map["past_0_key"]
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
            # self._encoder is a torch Module (untyped __call__ -> Any).
            return cast("BaseModelOutput", self._encoder(**feeds))

    def get_encoder(self) -> torch.nn.Module:
        """Return encoder for GenerationMixin (already wrapped with padding)."""
        return self._encoder

    def can_generate(self) -> bool:  # noqa: D102
        return True

    def _validate_model_kwargs(self, model_kwargs: dict[str, Any]) -> None:
        """Allow inputs declared by the encoder ONNX graph during generation."""
        remaining_kwargs = {
            name: value
            for name, value in model_kwargs.items()
            if name not in self._encoder_input_names
        }
        GenerationMixin._validate_model_kwargs(cast("Any", self), remaining_kwargs)

    def prepare_inputs_for_generation(  # type: ignore[override]  # GenerationMixin's base signature differs; static-cache flow
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_outputs: BaseModelOutput | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build decoder inputs for each generate() step."""
        from .kv_cache import WinMLCache

        active_cache = getattr(past_key_values, "self_attention_cache", past_key_values)
        if isinstance(active_cache, WinMLCache) and active_cache.get_seq_length() > 0:
            decoder_input_ids = input_ids[:, -1:]
        else:
            decoder_input_ids = input_ids
        prepared = {
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
        if decoder_attention_mask is not None:
            prepared["decoder_attention_mask"] = decoder_attention_mask
        return prepared

    # ----- Cache management -----

    @classmethod
    def get_cache_class(cls) -> type[WinMLCache]:
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

    def _run_decoder(
        self,
        feeds: dict[str, Any],
        cache: WinMLCache,
        num_new_tokens: int,
    ) -> dict[str, Any]:
        """Run one decoder chunk and advance its KV cache."""
        first_position = cache.step
        runtime_feeds = dict(feeds)
        cache_mask = cache.build_decoder_mask(self._max_dec, num_new_tokens)
        decoder_attention_mask = runtime_feeds.get("decoder_attention_mask")
        runtime_feeds["decoder_attention_mask"] = (
            self._align_decoder_attention_mask(cache_mask, decoder_attention_mask)
            if isinstance(decoder_attention_mask, torch.Tensor)
            else cache_mask
        )
        runtime_feeds.setdefault(
            "cache_position",
            cache.get_query_cache_position(self._max_dec, num_new_tokens).to(torch.int64),
        )
        runtime_feeds.setdefault(
            "position_id",
            torch.arange(
                first_position,
                first_position + num_new_tokens,
                dtype=torch.int64,
            ),
        )
        for i in range(self._num_kv_layers):
            layer = cache._layer(i)
            runtime_feeds[f"past_{i}_key"] = cast("torch.Tensor", layer.keys).detach()
            runtime_feeds[f"past_{i}_value"] = cast("torch.Tensor", layer.values).detach()

        outputs = self._decoder(
            **pad_inputs(
                runtime_feeds,
                self._dec_expected,
                mode="left",
            )
        )
        cache.update_all_layers(outputs)
        return cast("dict[str, Any]", outputs)

    @staticmethod
    def _align_decoder_attention_mask(
        cache_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project a sequence-length mask onto the cache's active buffer slots."""
        if cache_mask.ndim != 2 or sequence_mask.ndim != 2:
            raise ValueError("decoder_attention_mask must be a 2D tensor")
        if cache_mask.shape[0] != sequence_mask.shape[0]:
            raise ValueError(
                "decoder_attention_mask batch size does not match the decoder cache mask"
            )

        sequence_mask = sequence_mask.to(device=cache_mask.device, dtype=cache_mask.dtype)
        if sequence_mask.shape == cache_mask.shape:
            return cache_mask * sequence_mask

        aligned = cache_mask.clone()
        for batch_index in range(cache_mask.shape[0]):
            active_slots = torch.nonzero(
                cache_mask[batch_index],
                as_tuple=False,
            ).flatten()
            count = min(active_slots.numel(), sequence_mask.shape[1])
            if count:
                aligned[batch_index, active_slots[-count:]] = sequence_mask[batch_index, -count:]
        return aligned

    # ----- Forward (decoder via WinMLAutoModel + KV cache) -----

    def forward(
        self,
        *,
        encoder_outputs: BaseModelOutput | tuple | None = None,
        past_key_values: Cache | None = None,
        input_ids: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> Seq2SeqLMOutput:
        """Run decoder with a WinML KV cache.

        Uses ``WinMLStaticCache`` or ``WinMLSlidingWindowCache``, selected by
        the subclass via ``get_cache_class()``.

        Args:
            encoder_outputs: Pre-computed encoder hidden states.
            past_key_values: ``WinMLCache`` (or ``EncoderDecoderCache``
                wrapper) from previous step.
            input_ids: Fallback — run encoder if encoder_outputs is None.
            decoder_input_ids: Decoder prompt or next generated token.
            decoder_attention_mask: Optional mask over the decoder sequence.
            **model_kwargs: Remaining kwargs forwarded to the decoder ONNX
                (e.g., attention_mask). Each tensor is auto-padded to match
                the ONNX model's expected input shape.
        """
        # Encoder hidden states
        if encoder_outputs is None and input_ids is not None:
            encoder_outputs = self._encoder(input_ids=input_ids, **model_kwargs)
        if encoder_outputs is None:
            raise ValueError("Either encoder_outputs or input_ids required")
        # The encoder wrapper always returns a dict-like BaseModelOutput; the tuple
        # arm of the annotation exists only for GenerationMixin signature compat.
        enc_h = cast("BaseModelOutput", encoder_outputs)["last_hidden_state"]

        # Resolve or create cache (subclasses override get_cache_class).
        cache = self._resolve_cache(past_key_values)

        feeds: dict[str, Any] = dict(model_kwargs)
        if decoder_input_ids is not None:
            feeds["decoder_input_ids"] = decoder_input_ids
        if decoder_attention_mask is not None:
            feeds["decoder_attention_mask"] = decoder_attention_mask
        feeds.setdefault("encoder_hidden_states", enc_h.detach())

        num_new_tokens = decoder_input_ids.shape[1] if decoder_input_ids is not None else 1
        decoder_shape = self._dec_expected.get("decoder_input_ids")
        static_seq_len = (
            decoder_shape[1]
            if decoder_shape is not None
            and len(decoder_shape) > 1
            and isinstance(decoder_shape[1], int)
            else None
        )

        if num_new_tokens > 1 and static_seq_len == 1 and decoder_input_ids is not None:
            prompt_logits: list[torch.Tensor] = []
            outputs: dict[str, Any] = {}
            prompt_mask = feeds.get("decoder_attention_mask")
            for token_index, token_ids in enumerate(decoder_input_ids.split(1, dim=1)):
                token_feeds = {**feeds, "decoder_input_ids": token_ids}
                if isinstance(prompt_mask, torch.Tensor):
                    token_feeds["decoder_attention_mask"] = prompt_mask[:, : token_index + 1]
                outputs = self._run_decoder(token_feeds, cache, 1)
                logits = outputs["logits"]
                prompt_logits.append(
                    logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
                )
            outputs["logits"] = torch.cat(prompt_logits, dim=1)
        else:
            if static_seq_len is not None and num_new_tokens > static_seq_len:
                raise ValueError(
                    f"Decoder prompt length {num_new_tokens} exceeds its static input "
                    f"length {static_seq_len}."
                )
            outputs = self._run_decoder(feeds, cache, num_new_tokens)

        return Seq2SeqLMOutput(
            logits=outputs["logits"],
            past_key_values=cache,
        )
