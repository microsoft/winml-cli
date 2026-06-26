# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""BART HuggingFace Model Configuration.

Provides encoder/decoder export wrappers and OnnxConfig registrations for
BART encoder-decoder models with sliding-window KV cache (Slice+Concat;
no ScatterElements).

Export Strategy (split by task):
- BartEncoderWrapper + BartEncoderIOConfig: ``feature-extraction`` task
  → encoder-only ONNX (input_ids, attention_mask → encoder_hidden_states).
- BartDecoderWrapper + BartDecoderIOConfig: ``text2text-generation`` task
  → decoder ONNX with sliding-window buffer input + single-token KV output.

Why the compute_bias-free design works with sliding window:

BART's positional encoding is ``BartLearnedPositionalEmbedding`` — an
``nn.Embedding`` over ``max_position_embeddings`` rows (with a +2 offset
to skip BART's reserved slots) added to ``inputs_embeds`` *before* the
first attention layer.  Stock ``BartDecoder.forward`` (transformers
4.57.6 modeling_bart.py) feeds the decoder's ``cache_position`` into two
consumers that want different semantics under sliding window:

1. ``self.embed_positions(..., position_ids=cache_position)`` — indices
   into the learned embedding table.  Needs the token's *absolute
   sequence position*.
2. ``self._update_causal_mask(..., cache_position=...)`` →
   ``_prepare_4d_causal_attention_mask_with_cache_position`` — HF's
   ``kv_idx <= q_idx`` check over the KV buffer.  Needs the query's
   *buffer index*.

For static cache these coincide.  For sliding window, the query is
permanently pinned at the rightmost buffer slot (``max_cache_len - 1``),
while its absolute seq position grows unboundedly.  To serve both, we:

- Bake ``cache_position = [max_cache_len - 1]`` as a Constant in the
  graph — constant-folds the causal mask to all-True.
- Add ``position_id`` as an ONNX input carrying the absolute seq
  position.
- Patch ``BartLearnedPositionalEmbedding.forward`` via ``PATCHING_SPECS``
  so the embedding lookup reads the query's absolute seq position from a
  ``position_id`` tensor attribute set on the embedding module by the
  export wrapper — bypassing the kwarg-based plumbing entirely.  The
  ``+self.offset`` (BART's reserved-slot shift) is preserved inside the
  patch.

The patch is a no-op when ``position_id`` is not set — the embedding
forward then behaves exactly like stock HF (so the encoder's
``embed_positions``, which also uses ``BartLearnedPositionalEmbedding``
but never gets ``position_id`` set, is unaffected).

Cache type:

The default configuration uses ``WinMLSlidingWindowCache`` (FIFO
Slice+Concat) plus the ``BartLearnedPositionalEmbedding.forward`` patch
described above.  ``WinMLEncoderDecoderModel`` is cache-agnostic — mask
construction and cache updates are delegated to the cache class via
``build_decoder_mask``, ``get_query_cache_position``, and
``update_all_layers``.

Models: facebook/bart-large-cnn (summarization), facebook/bart-large,
facebook/bart-base, etc.

Usage:
    winml config -m facebook/bart-large-cnn --task feature-extraction       → encoder
    winml config -m facebook/bart-large-cnn --task text2text-generation     → decoder
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from optimum.exporters.onnx import OnnxConfig
from optimum.exporters.onnx.model_patcher import PatchingSpec
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyTextInputGenerator
from transformers import BartForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from ...config import WinMLBuildConfig
from ...export import register_onnx_overwrite
from ...optim import WinMLOptimizationConfig
from ..winml.composite_model import register_composite_model
from ..winml.encoder_decoder import EncoderDecoderInputGenerator, WinMLEncoderDecoderModel

# from ..winml.kv_cache import PastKeyValueInputGenerator, WinMLSlidingWindowCache
from ..winml.kv_cache import PastKeyValueInputGenerator, WinMLStaticCache


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig
    from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding

logger = logging.getLogger(__name__)


# =============================================================================
# Patch for sliding-window-compatible learned positional embedding
# =============================================================================
#
# Why this patch is *required* (not just preferred) for sliding-window export
# -----------------------------------------------------------------------------
# In transformers==4.57.6, ``BartDecoder.forward`` (modeling_bart.py) drives
# the learned positional embedding with:
#
#     positions = self.embed_positions(input, past_key_values_length,
#                                      position_ids=cache_position)
#     hidden_states = inputs_embeds + positions
#
# and ``BartLearnedPositionalEmbedding.forward`` indexes the learned embedding
# table directly with whatever ``position_ids`` it receives (plus the +2
# offset).  So the embedding lookup indices are ``cache_position`` values —
# nothing else.
#
# For sliding-window + single-token gen, our wrapper *must* bake
# ``cache_position = [max_cache_len - 1]`` (the rightmost buffer slot) so HF's
# causal mask ``kv_idx <= q_idx`` reduces to all-True and constant-folds.  But
# that same ``cache_position`` is then handed verbatim to the learned embedding
# table — which would make every step look up row ``W-1`` regardless of actual
# generation position.  That is wrong for every step.
#
# The two consumers of ``cache_position`` want different things (buffer index
# for the mask, absolute seq pos for the embedding table) and there is no
# parameter in stock ``BartDecoder.forward`` that splits them.  We therefore
# patch ``BartLearnedPositionalEmbedding.forward`` to read the absolute seq pos
# from a tensor attribute ``position_id`` set on the embedding module by the
# wrapper, ignoring the ``position_ids`` kwarg that HF passes in.  With the
# patch, ``cache_position`` serves only the causal mask, and the learned
# lookup is driven by our separately-fed ``position_id`` input.  Without the
# patch, there is no other HF hook to inject a different lookup index — this
# is the minimal intrusion that makes sliding-window BART correct.
#
# Encoder safety: ``BartEncoder`` also instantiates its own
# ``BartLearnedPositionalEmbedding``, but our export wrapper only sets the
# ``position_id`` attribute on the *decoder's* ``embed_positions``.  The
# patched forward falls back to stock behavior when the attribute is absent,
# so the encoder export is unaffected.


def _patched_bart_learned_forward(
    self: BartLearnedPositionalEmbedding,  # monkey-patched onto this HF module
    input_ids: torch.Tensor,
    past_key_values_length: int = 0,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Patched ``BartLearnedPositionalEmbedding.forward``.

    If the export wrapper has stored an absolute-seq-pos tensor on this
    module as the attribute ``position_id``, use it as the lookup index
    (with BART's +2 offset preserved) and ignore the explicit
    ``position_ids`` kwarg that HF's ``BartDecoder.forward`` would
    otherwise pass in (which, under sliding-window semantics, carries the
    query's *buffer index* — the baked ``[max_cache_len - 1]`` — not its
    sequence position).

    See the module header for why this override cannot be avoided under
    the stock transformers==4.57.6 ``BartDecoder.forward`` flow.

    Without ``position_id`` set, behavior is bit-identical to the
    original HF implementation — which keeps the patch safe for the
    encoder's own ``BartLearnedPositionalEmbedding`` (the wrapper never
    sets the attribute there) and for any other non-sliding-window
    export path.
    """
    abs_pos = getattr(self, "position_id", None)
    if abs_pos is not None:
        # Match HF's reshape: [seq_len] → [1, seq_len] before lookup.
        if abs_pos.dim() == 1:
            abs_pos = abs_pos.unsqueeze(0)
        return F.embedding(abs_pos + self.offset, self.weight)
    # Fallback: bit-identical to stock HF behavior.
    if position_ids is None:
        bsz, seq_len = input_ids.shape[:2]
        position_ids = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        ).expand(bsz, -1)
    else:
        position_ids = position_ids.unsqueeze(0)
    return F.embedding(position_ids + self.offset, self.weight)


def _build_bart_patching_specs() -> list[PatchingSpec]:
    """Return PatchingSpec list for BART, or [] if BartLearnedPositionalEmbedding is unavailable."""
    try:
        from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
    except ImportError:
        logger.debug("BartLearnedPositionalEmbedding not found; learned-embedding patch skipped.")
        return []
    return [
        PatchingSpec(
            o=BartLearnedPositionalEmbedding,
            name="forward",
            custom_op=_patched_bart_learned_forward,
        ),
    ]


# =============================================================================
# Wrapper nn.Modules (with from_pretrained, matching T5/Marian/Mu2 pattern)
# =============================================================================


class BartEncoderWrapper(nn.Module):
    """Wraps BART encoder for standalone ONNX export.

    Loads the full BartForConditionalGeneration and extracts the encoder.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> BartEncoderWrapper:
        """Load full BartForConditionalGeneration, extract encoder."""
        full_model = BartForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        wrapper = cls(full_model.get_encoder())
        wrapper.eval()
        return wrapper

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return encoder last hidden state."""
        # self.encoder is a torch submodule (untyped __call__ -> Any).
        return cast(
            "torch.Tensor",
            self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state,
        )


class BartDecoderWrapper(nn.Module):
    """Wraps ``BartForConditionalGeneration`` with sliding-window KV cache I/O.

    Input: full buffer ``[batch, heads, max_decode, d_kv]`` per layer.
    Output: only the new token's KV ``[batch, heads, 1, d_kv]`` per layer.

    Uses ``WinMLSlidingWindowCache`` (Slice+Concat eviction) wrapped in
    ``EncoderDecoderCache``.  Two design choices handle the two-consumer
    problem of BART's ``cache_position``:

    1. ``cache_position`` is NOT an ONNX input — it is pinned to
       ``[max_cache_len - 1]`` as a Constant inside ``forward``, matching
       the rightmost-slot invariant of sliding-window + single-token gen.
       HF's causal mask ``kv_idx <= q_idx`` then reduces to all-True and
       constant-folds away.
    2. ``position_id`` IS an ONNX input (absolute seq pos) and is threaded
       to ``BartLearnedPositionalEmbedding`` via the ``position_id``
       tensor attribute set on that module.  The patched embedding
       forward (registered in ``PATCHING_SPECS``) reads that attribute
       instead of HF's default position_ids derivation, giving dynamic
       learned-embedding lookup that tracks the actual generation step.
    """

    def __init__(self, model: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        # Expose config for OnnxConfig / NormalizedConfig access.
        # model is typed nn.Module, so torch's __getattr__ types .config as
        # Tensor | Module; it is really the model's PretrainedConfig.
        self.config: PretrainedConfig = cast("PretrainedConfig", model.config)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> BartDecoderWrapper:
        """Load full BartForConditionalGeneration, wrap with sliding-window cache."""
        full_model = BartForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        num_layers = full_model.config.decoder_layers
        wrapper = cls(full_model, num_layers)
        wrapper.eval()
        return wrapper

    def get_export_args(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Convert dict inputs to positional args for torch.onnx.export."""
        return tuple(inputs.values())

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run decoder with sliding-window KV cache.

        Positional args (order matches OnnxConfig.inputs):
            decoder_input_ids, encoder_hidden_states, attention_mask,
            decoder_attention_mask, position_id,
            past_0_key, past_0_value, past_1_key, past_1_value, ...

        Returns:
            (logits, present_0_key, present_0_value, ...) where each
            present KV is ``[batch, heads, 1, d_kv]`` — the new token only.
        """
        decoder_input_ids = args[0]
        encoder_hidden_states = args[1]
        attention_mask = args[2]
        decoder_attention_mask = args[3]
        # position_id = args[4]  # sliding-window cache
        cache_position = args[4]  # static cache: absolute seq pos, drives mask + learned pos
        kv_start = 5

        max_cache_len = args[kv_start].size(2)
        # self_attn_cache = WinMLSlidingWindowCache(self.config, max_cache_len=max_cache_len)
        self_attn_cache = WinMLStaticCache(self.config, max_cache_len=max_cache_len)
        self_attn_cache.early_initialization(
            batch_size=decoder_input_ids.size(0),
            num_heads=args[kv_start].size(1),
            head_dim=args[kv_start].size(3),
            dtype=args[kv_start].dtype,
            device=decoder_input_ids.device,
        )
        for i in range(self.num_layers):
            self_attn_cache.layers[i].keys = args[kv_start + i * 2]
            self_attn_cache.layers[i].values = args[kv_start + i * 2 + 1]

        # Thread absolute seq pos to the (patched) learned embedding via a
        # module attribute.  The patched forward reads this and uses it for
        # the lookup, ignoring the explicit position_ids kwarg that HF would
        # otherwise pass (which under sliding window is the buffer index).
        # self.model.get_decoder().embed_positions.position_id = position_id  # sliding only
        # Static cache: stock HF passes `cache_position` as `position_ids` into
        # BartLearnedPositionalEmbedding.forward (which applies +self.offset),
        # so it already indexes the learned table correctly — no hook needed.

        # Sliding window + single-token gen: the query is always at the
        # rightmost slot.  Constructing this constant inside forward traces
        # it as a Constant node — the causal-mask subgraph then constant-folds.
        # cache_position = torch.tensor(
        #     [max_cache_len - 1], dtype=torch.int64, device=decoder_input_ids.device
        # )  # sliding only; static cache uses the ONNX `cache_position` input directly

        # EncoderDecoderCache routes self-attention vs cross-attention to
        # separate caches.  DynamicCache for cross-attn is a no-op during
        # export (each layer computes fresh from encoder_hidden_states).
        cross_attn_cache = DynamicCache()
        cache = EncoderDecoderCache(self_attn_cache, cross_attn_cache)

        out = self.model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )

        # Return new-token KV directly from the capturing cache.
        result: list[torch.Tensor] = [out.logits]
        for i in range(self.num_layers):
            k, v = self_attn_cache.captured[i]
            result.extend([k, v])
        return tuple(result)


# =============================================================================
# OnnxConfig Registrations
# =============================================================================


@register_onnx_overwrite("bart", "feature-extraction", library_name="transformers")
class BartEncoderIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for BART encoder (feature-extraction task).

    Inputs:  input_ids, attention_mask
    Outputs: encoder_hidden_states
    """

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        vocab_size="vocab_size",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return {
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        }


class _BartDecoderNormalizedConfig(NormalizedConfig):  # type: ignore[misc]  # optimum base is untyped
    """NormalizedConfig for BART decoder-side export.

    Maps NormalizedConfig attributes to BartConfig's decoder-side attrs.
    ``head_dim`` is derived — BartConfig has no such attr natively.
    """

    VOCAB_SIZE = "vocab_size"
    HIDDEN_SIZE = "d_model"
    NUM_LAYERS = "decoder_layers"
    NUM_ATTENTION_HEADS = "decoder_attention_heads"
    MAX_CACHE_LEN = "max_position_embeddings"

    @property
    def head_dim(self) -> int:
        # hidden_size / num_attention_heads come from the untyped NormalizedConfig base.
        return cast("int", self.hidden_size // self.num_attention_heads)


@register_onnx_overwrite("bart", "text2text-generation", library_name="transformers")
class BartDecoderIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ONNX config for BART decoder with sliding-window KV cache.

    Inputs:  decoder_input_ids, encoder_hidden_states, attention_mask,
             decoder_attention_mask, position_id, past_{i}_key/value
    Outputs: logits, present_{i}_key/value

    ``cache_position`` is *not* an input: ``BartDecoderWrapper.forward``
    pins it to ``[max_cache_len - 1]`` (rightmost buffer slot) as a
    Constant.  ``position_id`` (absolute seq pos) drives the patched
    learned-embedding lookup — see the wrapper and
    ``_patched_bart_learned_forward`` for details.

    Input past KV: full buffer ``[batch, heads, max_decode, d_kv]``.
    Output present KV: new token only ``[batch, heads, 1, d_kv]``.
    """

    NORMALIZED_CONFIG_CLASS = _BartDecoderNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (
        EncoderDecoderInputGenerator,
        PastKeyValueInputGenerator,
    )
    # PATCHING_SPECS = _build_bart_patching_specs()  # sliding only; no-op under static

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        result: dict[str, dict[int, str]] = {
            "decoder_input_ids": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "decoder_attention_mask": {0: "batch_size"},
            # "position_id": {},  # sliding-window cache input name
            "cache_position": {},  # static cache input name (== absolute seq pos)
        }
        num_layers = self._normalized_config.num_layers
        for i in range(num_layers):
            result[f"past_{i}_key"] = {0: "batch_size"}
            result[f"past_{i}_value"] = {0: "batch_size"}
        return result

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        result: dict[str, dict[int, str]] = {
            "logits": {0: "batch_size"},
        }
        num_layers = self._normalized_config.num_layers
        for i in range(num_layers):
            result[f"present_{i}_key"] = {0: "batch_size"}
            result[f"present_{i}_value"] = {0: "batch_size"}
        return result


# =============================================================================
# Model Class Mapping + Build Config
# =============================================================================

MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("bart", "feature-extraction"): BartEncoderWrapper,
    ("bart", "text2text-generation"): BartDecoderWrapper,
}

BART_CONFIG = WinMLBuildConfig(
    optim=WinMLOptimizationConfig(
        gelu_fusion=True,
        matmul_add_fusion=True,
        clamp_constant_values=True,
        remove_isnan_in_attention_mask=True,
    ),
)


# =============================================================================
# WinMLBartModel — inference wrapper (registered as composite model)
# =============================================================================


@register_composite_model("bart", "summarization")
@register_composite_model("bart", "table-question-answering")
class WinMLBartModel(WinMLEncoderDecoderModel):
    """BART encoder-decoder model for summarization and table-question-answering.

    Declares BART sub-component tasks and generation-config defaults.
    All encoder-decoder forward/cache logic lives in
    ``WinMLEncoderDecoderModel``.  Uses ``WinMLSlidingWindowCache`` — see
    module docstring for the rationale.

    The ``table-question-answering`` registration covers TAPEX-style models
    (e.g., ``microsoft/tapex-base-finetuned-wikisql``).  TAPEX is plain BART
    fine-tuned for table QA — the table+query are serialized to text by the
    tokenizer (``TapexTokenizer``), then the encoder-decoder generates the
    answer.  The export and inference paths are identical to summarization
    (encoder=feature-extraction, decoder=text2text-generation sub-tasks).
    """

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }

    @classmethod
    def get_cache_class(cls) -> type:
        """BART defaults to ``WinMLSlidingWindowCache`` (Slice+Concat; no ScatterElements).

        The learned position embedding is fed the absolute seq pos via the
        ``position_id`` ONNX input (threaded through the patched
        ``BartLearnedPositionalEmbedding.forward``); the causal mask
        consumes a baked ``cache_position = [max_cache_len - 1]`` Constant.
        See the module docstring for the full reasoning.
        """
        # return WinMLSlidingWindowCache  # sliding-window (FIFO Slice+Concat)
        return WinMLStaticCache  # static cache (index_put_ → ScatterND)

    @property
    def generation_config(self) -> GenerationConfig:  # noqa: D102
        if not hasattr(self, "_generation_config"):
            from transformers import GenerationConfig

            gc_kw: dict[str, Any] = {}
            if self.config is not None:
                # forced_bos_token_id / forced_eos_token_id matter for
                # bart-large-cnn: the summarization fine-tune requires
                # forcing the first generated token to <s> (id=0) and the
                # last token before EOS to </s> (id=2).
                for attr in (
                    "decoder_start_token_id",
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                    "forced_bos_token_id",
                    "forced_eos_token_id",
                    "no_repeat_ngram_size",
                ):
                    val = getattr(self.config, attr, None)
                    if val is not None:
                        gc_kw[attr] = val
            gc_kw.setdefault("max_new_tokens", self._max_dec - 1)
            # Static batch=1 ONNX models don't support beam search
            gc_kw.setdefault("num_beams", 1)
            gc_kw.setdefault("do_sample", False)
            self._generation_config = GenerationConfig(**gc_kw)
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value: Any) -> None:
        self._generation_config = value


__all__ = [
    "BART_CONFIG",
    "MODEL_CLASS_MAPPING",
    "BartDecoderIOConfig",
    "BartDecoderWrapper",
    "BartEncoderIOConfig",
    "BartEncoderWrapper",
    "WinMLBartModel",
]
