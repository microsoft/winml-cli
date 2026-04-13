# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Encoder-Decoder inference model and shared input generator.

Provides ``WinMLEncoderDecoderModel`` — inference wrapper for encoder-decoder
pipelines (T5, mBART, etc.) with static KV cache, and
``EncoderDecoderInputGenerator`` — reusable ``DummyInputGenerator`` for
decoder base inputs shared across encoder-decoder architectures.

Class hierarchy::

    WinMLPipelineModel(PreTrainedModel)            — multi-component base
      └─ WinMLEncoderDecoderModel(GenerationMixin) — encoder-decoder with StaticCache
           └─ WinMLT5Model (in t5.py)              — T5 tasks + generation config

How it works:

1. Each pipeline model declares ``_SUB_MODEL_CONFIG = {"encoder": "feature-extraction",
   "decoder": "text2text-generation"}``. ``from_pretrained()`` builds each component
   via ``WinMLAutoModel`` (export → optimize → compile) independently.

2. The encoder is wrapped in ``_EncoderWithInputPadding`` which reads ONNX input
   names/shapes from ``io_config`` and zero-pads any undersized inputs.

3. ``forward()`` takes ``(*, encoder_outputs, past_key_values, input_ids, **model_kwargs)``
   where ``model_kwargs`` carries decoder inputs like ``decoder_input_ids`` and
   ``attention_mask``. Feeds are built from model_kwargs + generated inputs
   (encoder_hidden_states, decoder_attention_mask, cache_position, KV cache),
   filtered to decoder ONNX input names, and auto-padded.

4. KV cache uses HF ``StaticCache`` — same class for both export (``index_copy_``
   traces correctly in ``torch.onnx.export``) and inference (mutated in-place via
   ``cache.update()``). The ONNX decoder takes the full static buffer as input
   and outputs only the new token's KV ``[batch, heads, 1, d_kv]``.

Key findings from T5 KV cache study:

- HF's ``DynamicCache`` is stateful (same object, mutated in-place via ``cat``).
  ``GenerationMixin._update_model_kwargs_for_generation`` reads ``past_key_values``
  from the output and reassigns it in ``model_kwargs`` — but for stateful caches
  it's the same reference.
- ``StaticCache`` uses ``index_copy_`` at ``cache_position`` (traces correctly).
  ``StaticCache.get_seq_length()`` counts non-zero positions automatically.
- ``EncoderDecoderCache`` with empty cross-attn cache → ``is_updated`` dict is
  empty → cross-attention always recomputed from ``encoder_hidden_states`` →
  prevents constant-folding during ONNX export.
- ``GenerationMixin`` may wrap our ``StaticCache`` in an ``EncoderDecoderCache``
  before passing it back. ``forward()`` must unwrap to find the ``StaticCache``.
- ``TranslationPipeline`` passes its own ``generation_config`` with ``num_beams=4``
  to ``generate()``. Use ``num_beams=1`` at call time or override in subclass.

Design principles:

- NEVER guard config access with default values. Use ``self.config.param``
  directly and let AttributeError raise if the config is missing a field.
- ONNX I/O names and shapes are read from ``io_config``, never hardcoded.
- Inputs smaller than ONNX expected shape are zero-padded automatically.
  Inputs larger than expected are NOT truncated — let ORT raise the error.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from optimum.utils.input_generators import DummyInputGenerator
from transformers import Cache, StaticCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from ..winml.pipeline_model import WinMLPipelineModel


if TYPE_CHECKING:
    from optimum.utils import NormalizedConfig
    from transformers import PretrainedConfig

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
        raise ValueError(f"Unknown input: {input_name}")


# =============================================================================
# WinMLEncoderDecoderModel — encoder-decoder with StaticCache
# =============================================================================


class WinMLEncoderDecoderModel(WinMLPipelineModel, GenerationMixin):
    """Pipeline model with HF GenerationMixin support.

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

        # Max decode length from decoder ONNX KV input shape
        self._max_dec = self._dec_expected["past_0_key"][2]
        self._num_kv_layers = sum(
            1 for n in self._dec_expected if n.startswith("past_") and n.endswith("_key")
        )

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
            feeds = WinMLPipelineModel._pad_inputs(kwargs, self._expected)
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

        # Resolve the self-attention cache.
        # GenerationMixin may pass None, a StaticCache, or an
        # EncoderDecoderCache wrapping a DynamicCache (auto-created).
        cache = None
        if isinstance(past_key_values, StaticCache):
            cache = past_key_values
        elif hasattr(past_key_values, "self_attention_cache"):
            sa = past_key_values.self_attention_cache
            if isinstance(sa, StaticCache):
                cache = sa
        if cache is None:
            # Read KV geometry from ONNX metadata (architecture-agnostic)
            kv_shape = self._dec_expected["past_0_key"]  # [batch, heads, max_dec, head_dim]
            cache = StaticCache(self.config, max_cache_len=self._max_dec)
            cache.early_initialization(
                batch_size=1,
                num_heads=kv_shape[1],
                head_dim=kv_shape[3],
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

        # Determine write position from cache occupancy
        fc = cache.get_seq_length()
        dec_mask = torch.zeros(1, self._max_dec, dtype=torch.int64)
        dec_mask[0, : fc + 1] = 1

        # Build feeds: model_kwargs first, then fill in generated inputs
        feeds: dict[str, Any] = dict(model_kwargs)
        feeds.setdefault("encoder_hidden_states", enc_h.detach())
        feeds.setdefault("decoder_attention_mask", dec_mask)
        feeds.setdefault("cache_position", torch.tensor([fc], dtype=torch.int64))
        for i in range(self._num_kv_layers):
            layer = cache.layers[i]
            feeds[f"past_{i}_key"] = layer.keys.detach()
            feeds[f"past_{i}_value"] = layer.values.detach()

        # Filter to decoder ONNX inputs and pad any undersized tensors
        outputs = self._decoder(**self._pad_inputs(feeds, self._dec_expected))

        # Write new token's KV into the StaticCache in-place
        cache_kwargs = {"cache_position": torch.tensor([fc], dtype=torch.int64)}
        for i in range(self._num_kv_layers):
            cache.update(
                outputs[f"present_{i}_key"],
                outputs[f"present_{i}_value"],
                layer_idx=i,
                cache_kwargs=cache_kwargs,
            )

        return Seq2SeqLMOutput(
            logits=outputs["logits"],
            past_key_values=cache,
        )
