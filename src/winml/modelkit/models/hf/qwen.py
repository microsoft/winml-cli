# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Qwen3 HuggingFace Model Configuration.

Provides decoder export wrappers and OnnxConfig registrations for
Qwen3 decoder-only models with KV cache, split into prefill and
generation sub-models.

Export Strategy (split by task):
- QwenDecoderWrapper + QwenPrefillIOConfig: ``feature-extraction`` task
  → prefill ONNX (input_ids [1, 64] → logits [1, 64, vocab] + KV [1, kv_heads, 64, head_dim])
- QwenDecoderWrapper + QwenGenIOConfig: ``text-generation`` task
  → generation ONNX (input_ids [1, 1] → logits [1, 1, vocab] + KV [1, kv_heads, 1, head_dim])

Both tasks share the same wrapper class; OnnxConfig determines static shapes.
The wrapper captures new-token KV directly as ONNX outputs, eliminating the
scatter→gather round-trip.

How it works:

1. ``QwenDecoderWrapper.forward()`` takes positional args (order matches
   OnnxConfig.inputs): input_ids, attention_mask, position_ids,
   past_0_key, past_0_value, ...  It builds a ``WinMLSlidingWindowCache``
   from the input KV buffers, computes right-aligned ``cache_position``
   internally, runs ``Qwen3ForCausalLM``, and returns logits + captured KV.

2. Decoder-only models need NO ``EncoderDecoderCache`` wrapping —
   ``StaticCache`` is passed directly as ``past_key_values``.  (Contrast with
   T5 where ``EncoderDecoderCache`` is required to route self-attention and
   cross-attention to separate caches.)

3. Logits are returned for ALL input positions (not just last token).
   This matches HF convention and enables both generation (last-token logits)
   and perplexity evaluation (all-position logits with shifted labels).

4. ``dynamo=True`` is required for Qwen3 ONNX export — the TorchScript
   exporter fails with an internal error.  Dynamo produces opset 18 models;
   opset 17 downconversion currently fails for these graphs.

Cache type:

The default configuration uses ``WinMLSlidingWindowCache`` (FIFO
Slice+Concat).  ``WinMLDecoderOnlyModel`` is cache-agnostic — padding,
mask construction, and cache updates are all delegated to the cache class
via ``prepare_prefill_chunk``, ``build_decoder_mask``, and
``update_all_layers``.  To switch to ``WinMLStaticCache`` (index_copy_):

1. **Export wrapper**: change ``QwenDecoderWrapper.forward()`` to use
   ``WinMLStaticCache``, take ``cache_position`` as an explicit ONNX
   input (instead of computing it internally), and set ``kv_start = 4``.
2. **OnnxConfig inputs**: add ``"cache_position": {}`` to
   ``_qwen_io_inputs`` (after ``position_ids``, before ``past_*``).
3. **Inference**: override ``get_cache_class()`` to return
   ``WinMLStaticCache``.  ``WinMLDecoderOnlyModel`` passes
   ``cache_position`` in feeds automatically when the ONNX model
   expects it.

Task name constraints (Optimum compatibility):

- Task names must exist in ``TasksManager.get_all_tasks()`` to pass
  validation in ``register_onnx_overwrite``.  Custom names like
  ``"causal-lm-prefill"`` require pre-registration in
  ``TasksManager._LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP``.
- ``"causal-lm"`` is a synonym for ``"text-generation"`` in Optimum's
  ``_SYNONYM_TASK_MAP`` — registering an OnnxConfig under ``"causal-lm"``
  silently resolves to ``"text-generation"`` at lookup time.
- ``"text-generation-with-past"`` requires the OnnxConfig to implement
  ``with_past`` support (raises ``ValueError`` otherwise).
- We use ``"feature-extraction"`` (prefill) and ``"text-generation"`` (gen)
  as they are standard tasks with no normalization surprises.

Model: Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B, etc.

Usage::

    # Generate both configs (pipeline mode)
    winml config -m Qwen/Qwen3-0.6B --task text-generation -o qwen.json

    # Build both sub-models
    from winml.modelkit.models.winml.decoder_only import WinMLQwen3Model
    model = WinMLQwen3Model.from_pretrained("Qwen/Qwen3-0.6B")

    # Or load pre-built ONNX directly (skip_build=True avoids re-optimization)
    from winml.modelkit.models.auto import WinMLAutoModel
    prefill = WinMLAutoModel.from_pretrained("prefill.onnx", skip_build=True)
    gen = WinMLAutoModel.from_pretrained("gen.onnx", skip_build=True)
    model = WinMLQwen3Model(sub_models={...}, config=hf_config)
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from transformers import AutoModelForCausalLM

from ...config import WinMLBuildConfig
from ...export import register_onnx_overwrite
from ...export.config import WinMLExportConfig
from ..winml import register_specialization
from ..winml.composite_model import register_composite_model
from ..winml.decoder_only import (
    DecoderOnlyInputGenerator,
    DecoderOnlyPrefillInputGenerator,
    WinMLDecoderOnlyModel,
)
from .kv_cache import PastKeyValueInputGenerator, WinMLSlidingWindowCache


# =============================================================================
# Wrapper nn.Module
# =============================================================================


class QwenDecoderWrapper(nn.Module):
    """Wraps Qwen3ForCausalLM with static KV cache I/O.

    Used for both prefill and generation ONNX export — same forward logic,
    different OnnxConfig determines the static input shapes.

    Input KV: full static buffer ``[batch, kv_heads, max_cache_len, head_dim]``.
    Output KV: new positions only ``[batch, kv_heads, seq_len, head_dim]``.
    Logits: last position only ``[batch, 1, vocab_size]`` (both prefill and gen).
    """

    def __init__(self, model: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.config = model.config

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> QwenDecoderWrapper:
        """Load Qwen3ForCausalLM and wrap for export."""
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        wrapper = cls(model, model.config.num_hidden_layers)
        wrapper.eval()
        return wrapper

    def get_export_args(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Convert dict inputs to positional args for torch.onnx.export."""
        return tuple(inputs.values())

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run decoder with static KV cache.

        Positional args (order matches OnnxConfig.inputs):
            input_ids, attention_mask, position_ids, position_id,
            past_0_key, past_0_value, past_1_key, past_1_value, ...

        Returns:
            (logits, present_0_key, present_0_value, ...) where:
            - logits is ``[batch, 1, vocab_size]`` (last position only)
            - present KV is ``[batch, kv_heads, seq_len, head_dim]``
        """
        input_ids = args[0]
        attention_mask = args[1]
        position_ids = args[2]
        kv_start = 3

        seq_len = input_ids.size(1)

        # Build WinMLSlidingWindowCache from input KV tensors.
        cache = WinMLSlidingWindowCache(self.config, max_cache_len=args[kv_start].size(2))
        cache.early_initialization(
            batch_size=input_ids.size(0),
            num_heads=args[kv_start].size(1),
            head_dim=args[kv_start].size(3),
            dtype=args[kv_start].dtype,
            device=input_ids.device,
        )
        max_cache_len = args[kv_start].size(2)
        for i in range(self.num_layers):
            cache.layers[i].keys = args[kv_start + i * 2]
            cache.layers[i].values = args[kv_start + i * 2 + 1]

        # Sliding window: tokens always append at the END of the buffer.
        # cache_position = buffer positions (right-aligned) so HF's
        # create_causal_mask builds correct kv_idx <= q_idx constraint.
        # position_ids (separate) handles RoPE with absolute positions.
        cache_position = torch.arange(
            max_cache_len - seq_len,
            max_cache_len,
            dtype=torch.int64,
            device=input_ids.device,
        )

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )

        # All logits + captured KV directly (no gather).
        # forward() selects the right position for padded prefill inputs.
        result: list[torch.Tensor] = [out.logits]
        for i in range(self.num_layers):
            k, v = cache.captured[i]
            result.extend([k, v])
        return tuple(result)


# Sub-models must use GenericTask (raw ONNX outputs) — task-specific
# wrappers like WinMLModelForFeatureExtraction would discard KV outputs.
register_specialization("qwen3", "feature-extraction", "WinMLModelForGenericTask")
register_specialization("qwen3", "text-generation", "WinMLModelForGenericTask")


# =============================================================================
# OnnxConfig Registrations (using standard Optimum task names)
# =============================================================================

_QWEN_NORMALIZED = NormalizedConfig.with_args(
    hidden_size="hidden_size",
    num_layers="num_hidden_layers",
    num_attention_heads="num_key_value_heads",  # KV cache uses GQA heads
    head_dim="head_dim",
    max_cache_len="max_position_embeddings",
    vocab_size="vocab_size",
    allow_new=True,
)


def _qwen_io_inputs(num_layers: int) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "position_ids": {0: "batch_size"},
    }
    for i in range(num_layers):
        result[f"past_{i}_key"] = {0: "batch_size"}
        result[f"past_{i}_value"] = {0: "batch_size"}
    return result


def _qwen_io_outputs(num_layers: int) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {"logits": {0: "batch_size"}}
    for i in range(num_layers):
        result[f"present_{i}_key"] = {0: "batch_size"}
        result[f"present_{i}_value"] = {0: "batch_size"}
    return result


@register_onnx_overwrite("qwen3", "feature-extraction", library_name="transformers")
class QwenPrefillIOConfig(OnnxConfig):
    """ONNX config for Qwen3 prefill (feature-extraction task).

    Inputs: input_ids [1, 64], attention_mask [1, 256], position_ids [1, 64],
            cache_position [64], past_{i}_key/value [1, 8, 256, 128]
    Outputs: logits [1, 1, vocab], present_{i}_key/value [1, 8, 64, 128]
    """

    NORMALIZED_CONFIG_CLASS = _QWEN_NORMALIZED
    DUMMY_INPUT_GENERATOR_CLASSES = (DecoderOnlyPrefillInputGenerator, PastKeyValueInputGenerator)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return _qwen_io_inputs(self._normalized_config.num_layers)

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return _qwen_io_outputs(self._normalized_config.num_layers)


@register_onnx_overwrite("qwen3", "text-generation", library_name="transformers")
class QwenGenIOConfig(OnnxConfig):
    """ONNX config for Qwen3 generation (text-generation task).

    Inputs: input_ids [1, 1], attention_mask [1, 256], position_ids [1, 1],
            cache_position [1], past_{i}_key/value [1, 8, 256, 128]
    Outputs: logits [1, 1, vocab], present_{i}_key/value [1, 8, 1, 128]
    """

    NORMALIZED_CONFIG_CLASS = _QWEN_NORMALIZED
    DUMMY_INPUT_GENERATOR_CLASSES = (DecoderOnlyInputGenerator, PastKeyValueInputGenerator)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return _qwen_io_inputs(self._normalized_config.num_layers)

    @property
    def outputs(self) -> dict[str, dict[int, str]]:  # noqa: D102
        return _qwen_io_outputs(self._normalized_config.num_layers)


# =============================================================================
# Build Config (dynamo=True required for Qwen3)
# =============================================================================

QWEN_CONFIG = WinMLBuildConfig(
    export=WinMLExportConfig(dynamo=True, opset_version=18),
)


# =============================================================================
# Model Class Mapping
# =============================================================================

MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("qwen3", "feature-extraction"): QwenDecoderWrapper,
    ("qwen3", "text-generation"): QwenDecoderWrapper,
}

# =============================================================================
# WinMLQwen3Model — inference wrapper (registered as pipeline model)
# =============================================================================


@register_composite_model("qwen3", "text-generation")
class WinMLQwen3Model(WinMLDecoderOnlyModel):
    """Qwen3 decoder-only model for text generation.

    Declares Qwen3 sub-component tasks and generation config defaults.
    All forward/cache logic lives in ``WinMLDecoderOnlyModel``.
    """

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "decoder_prefill": "feature-extraction",
        "decoder_gen": "text-generation",
    }

    @classmethod
    def get_cache_class(cls) -> type:  # noqa: D102
        return WinMLSlidingWindowCache

    @property
    def generation_config(self):  # noqa: D102
        if not hasattr(self, "_generation_config"):
            from transformers import GenerationConfig

            gc_kw: dict[str, Any] = {}
            for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
                val = getattr(self.config, attr, None)
                if val is not None:
                    gc_kw[attr] = val
            gc_kw.setdefault("max_new_tokens", self._max_cache_len - self._prefill_seq_len)
            gc_kw.setdefault("num_beams", 1)
            gc_kw.setdefault("do_sample", False)
            self._generation_config = GenerationConfig(**gc_kw)
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value: Any) -> None:
        self._generation_config = value


__all__ = [
    "MODEL_CLASS_MAPPING",
    "QWEN_CONFIG",
    "QwenDecoderWrapper",
    "QwenGenIOConfig",
    "QwenPrefillIOConfig",
    "WinMLQwen3Model",
]
