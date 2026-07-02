# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Transformer-only ``qwen3`` build variant, registered as a distinct model_type.

This module registers a self-contained build path under the model_type
``"qwen3_transformer_only"`` (distinct from the stock ``"qwen3"`` path in
``qwen.py``). Selecting it is explicit — pass ``model_type="qwen3_transformer_only"``
to ``WinMLAutoModel.from_pretrained(...)`` (or the underlying
``generate_hf_build_config(...)``). Both paths coexist; neither overrides the
other, and there is no import-ordering requirement.

The variant exports two transformer-only ONNX files (a prefill/context graph
and an iteration/decode graph) with this I/O:

  Inputs : past_keys_{i}, past_values_{i} (FP16, ``[1, kv_heads, max_cache, head_dim]``),
           input_hidden_states (FP32, ``[1, seq_len, hidden]``),
           past_seq_len (INT32, ``[1, 1]``), total_seq_len (INT32, ``[1]``)
  Outputs: output_hidden_states (FP32), present_keys_{i}, present_values_{i} (FP16)
  Ops    : ``com.microsoft::GroupQueryAttention`` (do_rotary=1),
           ``onnx::LpNormalization`` (RMSNorm), 1x1 ``Conv`` projections.

Registration happens at import time via decorators and module-level mappings,
mirroring ``qwen.py``. The aggregating ``models.hf`` package imports this
module so the entries land in ``MODEL_CLASS_MAPPING`` / ``MODEL_BUILD_CONFIGS``.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyInputGenerator
from transformers import AutoModelForCausalLM, PretrainedConfig

from ....config import WinMLBuildConfig
from ....export import register_onnx_overwrite
from ....export.config import WinMLExportConfig
from ....optim import WinMLOptimizationConfig
from ...winml import register_specialization
from ...winml.composite_model import WinMLCompositeModel, register_composite_model
from ...winml.kv_cache import WinMLSlidingWindowCache
from .qwen3_modeling import apply_transformer_only_export_prep


# Distinct model_type for this variant. The underscore form is what the
# exporter sees on ``model.config.model_type`` and what Optimum's TasksManager
# and ``register_specialization`` are keyed on; the hyphenated form is used for
# the ``MODEL_CLASS_MAPPING`` / ``MODEL_BUILD_CONFIGS`` lookups (those callers
# normalize ``_`` -> ``-``).
TRANSFORMER_ONLY_MODEL_TYPE = "qwen3_transformer_only"


# =============================================================================
# Wrapper module
# =============================================================================


class QwenTransformerOnlyDecoderWrapper(nn.Module):
    """Wraps ``Qwen3ForCausalLM`` for transformer-only export.

    The wrapper applies the export prep (LpNorm RMSNorm, GQA op, 1x1
    Conv projections) in ``__init__`` and exposes a positional ``forward``
    whose argument order matches :class:`QwenTransformerOnlyPrefillIOConfig.inputs`.
    Only ``self.model.model`` (the inner ``Qwen3Model``) is invoked at
    export time — embedding lookup and ``lm_head`` stay out of the graph.
    """

    def __init__(self, model: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.config: PretrainedConfig = cast("PretrainedConfig", model.config)
        apply_transformer_only_export_prep(model, matmul_to_conv=True)
        # Tag the config so the exporter resolves this variant's OnnxConfig
        # (registered under ``TRANSFORMER_ONLY_MODEL_TYPE``) rather than the
        # stock qwen3 one. Mirrors the CLIP/zoedepth sub-model precedent.
        self.config.model_type = TRANSFORMER_ONLY_MODEL_TYPE

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs: Any
    ) -> QwenTransformerOnlyDecoderWrapper:
        """Load the HF model and wrap it for transformer-only export."""
        kwargs.setdefault("torch_dtype", torch.float32)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        model.config._attn_implementation = "eager"
        wrapper = cls(model, model.config.num_hidden_layers)
        wrapper.eval()
        return wrapper

    def get_export_args(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Flatten the dummy-input dict into positional export args."""
        return tuple(inputs.values())

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the decoder stack on positional inputs (order matches OnnxConfig.inputs).

        Positional inputs are ``past_keys_0, past_values_0, ...,
        past_keys_{L-1}, past_values_{L-1}, input_hidden_states, past_seq_len,
        total_seq_len``. Returns ``(output_hidden_states, present_keys_0,
        present_values_0, ...)``.
        """
        kv_args = args[: 2 * self.num_layers]
        input_hidden_states = args[2 * self.num_layers]
        past_seq_len = args[2 * self.num_layers + 1]
        total_seq_len = args[2 * self.num_layers + 2]

        past_key_values = [(kv_args[2 * i], kv_args[2 * i + 1]) for i in range(self.num_layers)]

        hidden_states, present_kvs = cast("nn.Module", self.model.model)(
            inputs_embeds=input_hidden_states,
            past_key_values=past_key_values,
            past_seq_len=past_seq_len,
            total_seq_len=total_seq_len,
            use_cache=True,
        )

        out: list[torch.Tensor] = [hidden_states]
        for k, v in present_kvs:
            out.extend([k, v])
        return tuple(out)


# =============================================================================
# Dummy input generators (transformer-only I/O)
# =============================================================================


class _TransformerOnlyHiddenStateGenerator(DummyInputGenerator):  # type: ignore[misc]  # optimum base is untyped
    """Generates ``input_hidden_states`` (FP32, ``[1, seq_len, hidden]``)."""

    SUPPORTED_INPUT_NAMES = ("input_hidden_states",)

    _default_seq_len: ClassVar[int] = 1

    def __init__(
        self,
        task: str,
        normalized_config: Any,
        batch_size: int = 1,
        seq_len: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.batch_size = batch_size
        self.hidden_size = normalized_config.hidden_size
        self.seq_len = (
            int(seq_len)
            if seq_len
            else int(getattr(normalized_config, "seq_len", self._default_seq_len))
        )

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        if input_name == "input_hidden_states":
            return torch.randn(self.batch_size, self.seq_len, self.hidden_size, dtype=torch.float32)
        raise ValueError(f"Unknown input: {input_name}")


class _TransformerOnlyHiddenStatePrefillGenerator(_TransformerOnlyHiddenStateGenerator):
    _default_seq_len = 64


class _TransformerOnlySeqLenGenerator(DummyInputGenerator):  # type: ignore[misc]  # optimum base is untyped
    """Generates ``past_seq_len`` (INT32 ``[1,1]``) and ``total_seq_len`` (INT32 ``[1]``)."""

    SUPPORTED_INPUT_NAMES = ("past_seq_len", "total_seq_len")

    def __init__(self, task: str, normalized_config: Any, **kwargs: Any) -> None:
        self.max_cache_len = normalized_config.max_cache_len

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        if input_name == "past_seq_len":
            return torch.zeros((1, 1), dtype=torch.int32)
        if input_name == "total_seq_len":
            return torch.tensor([self.max_cache_len], dtype=torch.int32)
        raise ValueError(f"Unknown input: {input_name}")


class _TransformerOnlyKvCacheGenerator(DummyInputGenerator):  # type: ignore[misc]  # optimum base is untyped
    """Generates ``past_keys_{i}`` / ``past_values_{i}`` (FP16)."""

    SUPPORTED_INPUT_NAMES: tuple[str, ...] = ()  # built dynamically in __init__

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = 1,
        max_cache_len: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.batch_size = batch_size
        self.num_layers: int = normalized_config.num_layers
        self.num_heads: int = (
            normalized_config.num_attention_heads
        )  # KV heads (NormalizedConfig maps it)
        self.head_dim: int = normalized_config.head_dim
        self.max_cache_len: int = max_cache_len or normalized_config.max_cache_len
        self.SUPPORTED_INPUT_NAMES = tuple(
            name for i in range(self.num_layers) for name in (f"past_keys_{i}", f"past_values_{i}")
        )

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        shape = (self.batch_size, self.num_heads, self.max_cache_len, self.head_dim)
        return torch.zeros(shape, dtype=torch.float16)


# =============================================================================
# OnnxConfigs — transformer-only I/O layout
# =============================================================================


_QWEN_TRANSFORMER_ONLY_NORMALIZED = NormalizedConfig.with_args(
    hidden_size="hidden_size",
    num_layers="num_hidden_layers",
    num_attention_heads="num_key_value_heads",  # KV heads (GQA)
    head_dim="head_dim",
    max_cache_len="max_position_embeddings",
    vocab_size="vocab_size",
    allow_new=True,
)


def _transformer_only_inputs(
    num_layers: int, kv_seq_axis: str = "max_seq_len"
) -> dict[str, dict[int, str]]:
    """Input ordering: past KV pairs, then hidden states, then seq lens."""
    result: dict[str, dict[int, str]] = {}
    for i in range(num_layers):
        result[f"past_keys_{i}"] = {2: kv_seq_axis}
        result[f"past_values_{i}"] = {2: kv_seq_axis}
    result["input_hidden_states"] = {1: "seq_len"}
    result["past_seq_len"] = {}
    result["total_seq_len"] = {}
    return result


def _transformer_only_outputs(
    num_layers: int, kv_seq_axis: str = "max_seq_len"
) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {"output_hidden_states": {1: "seq_len"}}
    for i in range(num_layers):
        result[f"present_keys_{i}"] = {2: kv_seq_axis}
        result[f"present_values_{i}"] = {2: kv_seq_axis}
    return result


@register_onnx_overwrite(
    TRANSFORMER_ONLY_MODEL_TYPE, "feature-extraction", library_name="transformers"
)
class QwenTransformerOnlyPrefillIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """Prefill (seq=64) — transformer-only I/O."""

    NORMALIZED_CONFIG_CLASS = _QWEN_TRANSFORMER_ONLY_NORMALIZED
    DUMMY_INPUT_GENERATOR_CLASSES = (
        _TransformerOnlyKvCacheGenerator,
        _TransformerOnlyHiddenStatePrefillGenerator,
        _TransformerOnlySeqLenGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """ONNX input axes (past KV pairs, hidden states, seq lengths)."""
        return _transformer_only_inputs(self._normalized_config.num_layers)

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """ONNX output axes (hidden states then present KV pairs)."""
        return _transformer_only_outputs(self._normalized_config.num_layers)


@register_onnx_overwrite(
    TRANSFORMER_ONLY_MODEL_TYPE, "text2text-generation", library_name="transformers"
)
class QwenTransformerOnlyGenIOConfig(OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """Generation (seq=1) — transformer-only I/O."""

    NORMALIZED_CONFIG_CLASS = _QWEN_TRANSFORMER_ONLY_NORMALIZED
    DUMMY_INPUT_GENERATOR_CLASSES = (
        _TransformerOnlyKvCacheGenerator,
        _TransformerOnlyHiddenStateGenerator,
        _TransformerOnlySeqLenGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """ONNX input axes (past KV pairs, hidden states, seq lengths)."""
        return _transformer_only_inputs(self._normalized_config.num_layers)

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """ONNX output axes (hidden states then present KV pairs)."""
        return _transformer_only_outputs(self._normalized_config.num_layers)


# =============================================================================
# Build config — TorchScript exporter required for the custom autograd ops
# =============================================================================


QWEN_TRANSFORMER_ONLY_CONFIG = WinMLBuildConfig(
    export=WinMLExportConfig(dynamo=False, opset_version=18),
    # Pure graph (no post-export RMSNorm fusion / matmul-add fusion): the default
    # WinMLOptimizationConfig() leaves every fusion flag off.
    optim=WinMLOptimizationConfig(),
)


# =============================================================================
# Composite handle for the transformer-only build.
#
# This variant is EXPORT-ONLY: it produces the prefill + decode transformer
# ONNX for downstream quantization. It extends ``WinMLCompositeModel`` (not
# ``WinMLDecoderOnlyModel``) on purpose — the decoder-only base wires a full
# generation runtime from the eager KV I/O signature (``past_0_key`` etc.),
# which does not match the transformer-only graph (``past_keys_0`` + symbolic
# axes). Inheriting it would make ``from_pretrained`` crash while constructing
# the handle, even though both sub-model ONNX built fine. The plain composite
# base just stores the built sub-models, so ``from_pretrained`` returns a usable
# handle exposing ``.sub_models[name].onnx_path``.
# =============================================================================


@register_composite_model(TRANSFORMER_ONLY_MODEL_TYPE, "text-generation")
class WinMLQwen3TransformerOnlyModel(WinMLCompositeModel):
    """Export-only composite handle for the transformer-only Qwen3 build.

    ``from_pretrained`` builds both sub-models (``decoder_prefill`` seq=64,
    ``decoder_gen`` seq=1) and returns this handle; the built ONNX paths are
    available via ``self.sub_models[name].onnx_path``.

    ``generate()`` is **not** functional with this build path — the inference
    feeds and KV update logic target the eager I/O signature. Use the eager
    :class:`WinMLQwen3Model` for generation; use this class to produce the
    transformer-only ONNX for downstream quantization.
    """

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "decoder_prefill": "feature-extraction",
        "decoder_gen": "text2text-generation",
    }

    @classmethod
    def get_cache_class(cls) -> type:
        """KV-cache class the decode model targets at runtime (sliding window)."""
        return WinMLSlidingWindowCache

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        task: str = "text-generation",
        *,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> WinMLCompositeModel:
        """Build both transformer-only sub-models and return the composite handle.

        Forces ``model_type="qwen3_transformer_only"`` for every sub-model so the
        composite builds the transformer-only variant instead of silently falling
        back to the native (full) ``qwen3`` architecture when the caller omits it.
        """
        return super().from_pretrained(
            model_id,
            task,
            model_type=model_type or TRANSFORMER_ONLY_MODEL_TYPE,
            **kwargs,
        )


# =============================================================================
# Declarative registration (import-time)
# =============================================================================

# Wrapper-class lookup keyed by (model_type, task). Keys use the hyphenated
# model_type form because ``_get_custom_model_class`` normalizes ``_`` -> ``-``
# before lookup. Merged into the aggregate mapping by ``models.hf.__init__``.
MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("qwen3-transformer-only", "feature-extraction"): QwenTransformerOnlyDecoderWrapper,
    ("qwen3-transformer-only", "text2text-generation"): QwenTransformerOnlyDecoderWrapper,
}

# Inference specialization (GenericTask — the wrapper returns raw hidden states / KV).
register_specialization(
    TRANSFORMER_ONLY_MODEL_TYPE, "feature-extraction", "WinMLModelForGenericTask"
)
register_specialization(
    TRANSFORMER_ONLY_MODEL_TYPE, "text2text-generation", "WinMLModelForGenericTask"
)


__all__ = [
    "MODEL_CLASS_MAPPING",
    "QWEN_TRANSFORMER_ONLY_CONFIG",
    "TRANSFORMER_ONLY_MODEL_TYPE",
    "QwenTransformerOnlyDecoderWrapper",
    "QwenTransformerOnlyGenIOConfig",
    "QwenTransformerOnlyPrefillIOConfig",
    "WinMLQwen3TransformerOnlyModel",
]
