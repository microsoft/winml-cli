# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Decoder-Only Pipeline Model.

Class hierarchy::

    WinMLPipelineModel(PreTrainedModel)          — multi-component base
      └─ WinMLDecoderOnlyModel(GenerationMixin)  — prefill + gen with StaticCache
           └─ WinMLQwen3Model                    — Qwen3 tasks + generation config

How it works:

1. ``@register_pipeline_model("qwen3", "text-generation")`` hooks into
   ``winml config`` so that ``winml config -m Qwen/Qwen3-0.6B --task text-generation``
   generates ``qwen_decoder_prefill.json`` + ``qwen_decoder_gen.json``.

2. ``from_pretrained()`` builds each component via ``WinMLAutoModel``
   independently.  Sub-models are registered as ``WinMLModelForGenericTask``
   (via ``register_specialization``) so their raw ONNX outputs (logits + KV)
   are returned as-is — task-specific wrappers like
   ``WinMLModelForFeatureExtraction`` would discard the KV outputs.

3. ``forward()`` is called by ``GenerationMixin.generate()`` on each step:

   - **Prefill** (``input_ids`` has multiple tokens): chunks into
     ``prefill_seq_len`` pieces and runs the prefill ONNX model in a loop.
     Right-pads the last chunk; only writes real tokens' KV into the cache
     (padding positions are discarded).  Returns logits for ALL real
     positions ``[1, seq_len, vocab]`` — matches HF convention, enabling
     both generation (last-token selection) and perplexity evaluation
     (shifted cross-entropy over all positions).

   - **Generation** (``input_ids`` has 1 token): runs the gen ONNX model
     with the single token + full KV cache buffer as input.

4. KV cache uses HF ``StaticCache`` — same class as T5.  ``get_seq_length()``
   counts non-zero positions; ``cache.update()`` writes new KV via
   ``index_copy_``.  The cache persists across generate() steps via
   ``CausalLMOutputWithPast.past_key_values``.

5. ``prepare_inputs_for_generation()`` handles a subtle interaction with
   ``GenerationMixin``: on the FIRST call, GenerationMixin may pass an
   auto-created ``DynamicCache`` (empty).  We detect this (not a
   ``StaticCache`` or empty) and pass the full prompt through for prefill
   rather than trimming to the last token.  On subsequent calls with a
   populated ``StaticCache``, we trim to the last token as usual.

Design principles (same as pipeline_model.py):

- ONNX I/O names and shapes are read from ``io_config``, never hardcoded.
- Inputs smaller than ONNX expected shape are zero-padded via ``_pad_inputs``.
- ``_pad_inputs`` is reused from ``WinMLEncoderDecoderModel`` (static method).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from optimum.utils.input_generators import DummyInputGenerator
from transformers import Cache, StaticCache
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .pipeline_model import WinMLPipelineModel


_pad_inputs = WinMLPipelineModel._pad_inputs


if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


# =========================================================================
# DecoderOnlyInputGenerator — shared dummy input generator
# =========================================================================


class DecoderOnlyInputGenerator(DummyInputGenerator):
    """Generates base inputs for decoder-only models with static KV cache.

    Produces ``input_ids``, ``attention_mask``, ``position_ids``, and
    ``cache_position``.  Reads ``vocab_size``, ``max_cache_len``, and
    ``seq_len`` from the ``NormalizedConfig``.

    ``seq_len`` controls the input token count and is read from
    ``normalized_config.seq_len`` (falls back to ``_default_seq_len``).
    Subclasses override the default for prefill vs generation:

    - ``DecoderOnlyPrefillInputGenerator``: ``_default_seq_len = 64``
    - ``DecoderOnlyInputGenerator`` (base / gen): ``_default_seq_len = 1``

    To override at config time, set ``config.seq_len = N`` on the HF config.
    """

    SUPPORTED_INPUT_NAMES = (
        "input_ids",
        "attention_mask",
        "position_ids",
        "cache_position",
    )

    _default_seq_len: int = 1

    def __init__(
        self,
        task: str,
        normalized_config: Any,
        batch_size: int = 1,
        seq_len: int | None = None,
        max_cache_len: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.batch_size = batch_size
        self.vocab_size = normalized_config.vocab_size
        self.max_cache_len = max_cache_len or normalized_config.max_cache_len
        self.seq_len: int = seq_len or getattr(normalized_config, "seq_len", self._default_seq_len)

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        """Generate a dummy tensor for the given input name."""
        if input_name == "input_ids":
            return self.random_int_tensor(
                (self.batch_size, self.seq_len),
                max_value=self.vocab_size,
                framework=framework,
                dtype=int_dtype,
            )
        if input_name == "attention_mask":
            mask = torch.zeros(self.batch_size, self.max_cache_len, dtype=torch.int64)
            mask[:, : self.seq_len] = 1
            return mask
        if input_name == "position_ids":
            return torch.arange(self.seq_len, dtype=torch.int64).unsqueeze(0)
        if input_name == "cache_position":
            return torch.arange(self.seq_len, dtype=torch.int64)
        raise ValueError(f"Unknown input: {input_name}")


class DecoderOnlyPrefillInputGenerator(DecoderOnlyInputGenerator):
    """Prefill variant with ``_default_seq_len = 64``."""

    _default_seq_len: int = 64


# =========================================================================
# WinMLDecoderOnlyModel — prefill + gen with StaticCache
# =========================================================================


class WinMLDecoderOnlyModel(WinMLPipelineModel, GenerationMixin):
    """Decoder-only pipeline model with HF GenerationMixin support.

    Expects sub-components ``"decoder_prefill"`` and ``"decoder_gen"`` in
    ``_SUB_MODEL_CONFIG``.  Provides the full interface required by
    ``GenerationMixin.generate()`` for decoder-only models with static KV cache.

    Input/output names and shapes are read from ONNX I/O metadata.
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
        self._prefill_model = sub_models["decoder_prefill"]
        self._gen_model = sub_models["decoder_gen"]

        # Build {name: shape} lookups from ONNX I/O metadata
        prefill_io = self._prefill_model.io_config
        self._prefill_expected = dict(
            zip(
                prefill_io.get("input_names", []),
                prefill_io.get("input_shapes", []),
                strict=False,
            )
        )
        gen_io = self._gen_model.io_config
        self._gen_expected = dict(
            zip(gen_io.get("input_names", []), gen_io.get("input_shapes", []), strict=False)
        )

        # Cache geometry from gen model's KV input shape
        self._max_cache_len = self._gen_expected["past_0_key"][2]
        self._num_kv_heads = self._gen_expected["past_0_key"][1]
        self._head_dim = self._gen_expected["past_0_key"][3]
        self._num_kv_layers = sum(
            1 for n in self._gen_expected if n.startswith("past_") and n.endswith("_key")
        )

        # Prefill chunk size
        self._prefill_seq_len = self._prefill_expected["input_ids"][1]

    # ----- GenerationMixin interface -----

    def can_generate(self) -> bool:  # noqa: D102
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build inputs for each generate() step.

        GenerationMixin may pass a DynamicCache (auto-created, empty) on the
        first call.  Only trim to last token when we have a populated
        StaticCache (i.e., after prefill).
        """
        if isinstance(past_key_values, StaticCache) and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
        else:
            # First call or empty cache: pass full prompt for prefill
            past_key_values = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

    # ----- Forward -----

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Run prefill or gen with static KV cache.

        Called by ``GenerationMixin.generate()`` on each step:
        - First call: ``input_ids`` is the full prompt → prefill (chunked).
        - Subsequent calls: ``input_ids`` is 1 token → gen.

        Args:
            input_ids: Token IDs ``[batch, seq_len]``.
            past_key_values: StaticCache from previous step (None on first call).
            attention_mask: Not used directly — rebuilt from cache occupancy.
            **kwargs: Absorbed for GenerationMixin compatibility.

        Returns:
            CausalLMOutputWithPast with logits and updated StaticCache.
        """
        # Resolve or create StaticCache (same pattern as T5)
        cache = past_key_values if isinstance(past_key_values, StaticCache) else None
        if cache is None:
            cache = StaticCache(self.config, max_cache_len=self._max_cache_len)
            cache.early_initialization(
                batch_size=1,
                num_heads=self._num_kv_heads,
                head_dim=self._head_dim,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

        seq_len = input_ids.shape[1]
        if seq_len > 1:
            logits = self._run_prefill(input_ids, cache)
        else:
            logits = self._run_gen(input_ids, cache)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=cache,
        )

    # ----- Prefill (chunked) -----

    def _run_prefill(self, input_ids: torch.Tensor, cache: StaticCache) -> torch.Tensor:
        """Run prefill model in a loop over chunks of ``prefill_seq_len``.

        Returns logits for ALL real input positions ``[1, seq_len, vocab_size]``
        (same convention as HF CausalLM — enables perplexity evaluation).
        """
        seq_len = input_ids.shape[1]
        all_logits: list[torch.Tensor] = []

        for start in range(0, seq_len, self._prefill_seq_len):
            end = min(start + self._prefill_seq_len, seq_len)
            chunk_len = end - start

            # Pad chunk to prefill_seq_len (right-padding)
            padded_ids = torch.zeros(1, self._prefill_seq_len, dtype=input_ids.dtype)
            padded_ids[0, :chunk_len] = input_ids[0, start:end]

            position_ids = torch.arange(
                start, start + self._prefill_seq_len, dtype=torch.int64
            ).unsqueeze(0)
            cache_position = torch.arange(start, start + self._prefill_seq_len, dtype=torch.int64)

            # Attention mask: 1 for all real tokens so far
            attn_mask = torch.zeros(1, self._max_cache_len, dtype=torch.int64)
            attn_mask[0, : start + chunk_len] = 1

            feeds: dict[str, Any] = {
                "input_ids": padded_ids,
                "attention_mask": attn_mask,
                "position_ids": position_ids,
                "cache_position": cache_position,
            }
            for i in range(self._num_kv_layers):
                feeds[f"past_{i}_key"] = cache.layers[i].keys.detach()
                feeds[f"past_{i}_value"] = cache.layers[i].values.detach()

            outputs = self._prefill_model(**_pad_inputs(feeds, self._prefill_expected))

            # Write only real tokens' KV into cache (skip padding)
            real_positions = cache_position[:chunk_len]
            ck = {"cache_position": real_positions}
            for i in range(self._num_kv_layers):
                cache.update(
                    outputs[f"present_{i}_key"][:, :, :chunk_len, :],
                    outputs[f"present_{i}_value"][:, :, :chunk_len, :],
                    layer_idx=i,
                    cache_kwargs=ck,
                )

            # Keep logits for real tokens only (discard padding positions)
            all_logits.append(outputs["logits"][:, :chunk_len, :])

        return torch.cat(all_logits, dim=1)

    # ----- Generation (single token) -----

    def _run_gen(self, input_ids: torch.Tensor, cache: StaticCache) -> torch.Tensor:
        """Run gen model for a single token. Returns logits ``[1, 1, vocab_size]``."""
        fc = cache.get_seq_length()

        attn_mask = torch.zeros(1, self._max_cache_len, dtype=torch.int64)
        attn_mask[0, : fc + 1] = 1

        feeds: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "position_ids": torch.tensor([[fc]], dtype=torch.int64),
            "cache_position": torch.tensor([fc], dtype=torch.int64),
        }
        for i in range(self._num_kv_layers):
            feeds[f"past_{i}_key"] = cache.layers[i].keys.detach()
            feeds[f"past_{i}_value"] = cache.layers[i].values.detach()

        outputs = self._gen_model(**_pad_inputs(feeds, self._gen_expected))

        # Write new token's KV into cache
        ck = {"cache_position": torch.tensor([fc], dtype=torch.int64)}
        for i in range(self._num_kv_layers):
            cache.update(
                outputs[f"present_{i}_key"],
                outputs[f"present_{i}_value"],
                layer_idx=i,
                cache_kwargs=ck,
            )

        return outputs["logits"]
