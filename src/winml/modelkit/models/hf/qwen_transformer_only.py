# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Parallel ``qwen3`` build path that produces a transformer-only ONNX.

Opt-in via ``install()`` — calling it hot-patches the WinML registries so
that the next ``WinMLAutoModel.from_pretrained("Qwen/Qwen3-*", task="text-generation")``
exports two transformer-only ONNX files (a prefill/context graph and an
iteration/decode graph) with this I/O:

  Inputs : past_keys_{i}, past_values_{i} (FP16, ``[1, kv_heads, max_cache, head_dim]``),
           input_hidden_states (FP32, ``[1, seq_len, hidden]``),
           past_seq_len (INT32, ``[1, 1]``), total_seq_len (INT32, ``[1]``)
  Outputs: output_hidden_states (FP32), present_keys_{i}, present_values_{i} (FP16)
  Ops    : ``com.microsoft::GroupQueryAttention`` (do_rotary=1),
           ``onnx::LpNormalization`` (RMSNorm), 1x1 ``Conv`` projections.

The original eager-export path in ``qwen.py`` is left intact — only the
qwen3 entries in the registries are replaced. ``install()`` is idempotent.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyInputGenerator
from transformers import AutoModelForCausalLM

from ...config import WinMLBuildConfig
from ...export import register_onnx_overwrite
from ...export.config import WinMLExportConfig
from ..winml import register_specialization
from ..winml.decoder_only import WinMLDecoderOnlyModel
from ..winml.kv_cache import WinMLSlidingWindowCache
from .qwen3_export_ops import apply_transformer_only_export_prep


logger = logging.getLogger(__name__)


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
        self.config = model.config
        apply_transformer_only_export_prep(model, matmul_to_conv=True)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> QwenTransformerOnlyDecoderWrapper:
        kwargs.setdefault("torch_dtype", torch.float32)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        model.config._attn_implementation = "eager"
        wrapper = cls(model, model.config.num_hidden_layers)
        wrapper.eval()
        return wrapper

    def get_export_args(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return tuple(inputs.values())

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Positional inputs (order matches OnnxConfig.inputs):

            past_keys_0, past_values_0, ..., past_keys_{L-1}, past_values_{L-1},
            input_hidden_states, past_seq_len, total_seq_len

        Returns ``(output_hidden_states, present_keys_0, present_values_0, ...)``.
        """
        kv_args = args[: 2 * self.num_layers]
        input_hidden_states = args[2 * self.num_layers]
        past_seq_len = args[2 * self.num_layers + 1]
        total_seq_len = args[2 * self.num_layers + 2]

        past_key_values = [
            (kv_args[2 * i], kv_args[2 * i + 1]) for i in range(self.num_layers)
        ]

        hidden_states, present_kvs = self.model.model(
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


class _TransformerOnlyHiddenStateGenerator(DummyInputGenerator):
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
        self.seq_len = seq_len or getattr(normalized_config, "seq_len", self._default_seq_len)

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32") -> torch.Tensor:  # noqa: ARG002
        if input_name == "input_hidden_states":
            return torch.randn(self.batch_size, self.seq_len, self.hidden_size, dtype=torch.float32)
        raise ValueError(f"Unknown input: {input_name}")


class _TransformerOnlyHiddenStatePrefillGenerator(_TransformerOnlyHiddenStateGenerator):
    _default_seq_len = 64


class _TransformerOnlySeqLenGenerator(DummyInputGenerator):
    """Generates ``past_seq_len`` (INT32 ``[1,1]``) and ``total_seq_len`` (INT32 ``[1]``)."""

    SUPPORTED_INPUT_NAMES = ("past_seq_len", "total_seq_len")

    def __init__(self, task: str, normalized_config: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.max_cache_len = normalized_config.max_cache_len

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32") -> torch.Tensor:  # noqa: ARG002
        if input_name == "past_seq_len":
            return torch.zeros((1, 1), dtype=torch.int32)
        if input_name == "total_seq_len":
            return torch.tensor([self.max_cache_len], dtype=torch.int32)
        raise ValueError(f"Unknown input: {input_name}")


class _TransformerOnlyKvCacheGenerator(DummyInputGenerator):
    """Generates ``past_keys_{i}`` / ``past_values_{i}`` (FP16)."""

    SUPPORTED_INPUT_NAMES = ()  # built dynamically in __init__

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
        self.num_heads: int = normalized_config.num_attention_heads  # KV heads (NormalizedConfig maps it)
        self.head_dim: int = normalized_config.head_dim
        self.max_cache_len: int = max_cache_len or normalized_config.max_cache_len
        self.SUPPORTED_INPUT_NAMES = tuple(
            name for i in range(self.num_layers) for name in (f"past_keys_{i}", f"past_values_{i}")
        )

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32") -> torch.Tensor:  # noqa: ARG002
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


def _transformer_only_inputs(num_layers: int, kv_seq_axis: str = "max_seq_len") -> dict[str, dict[int, str]]:
    """Input ordering: past KV pairs, then hidden states, then seq lens."""
    result: dict[str, dict[int, str]] = {}
    for i in range(num_layers):
        result[f"past_keys_{i}"] = {2: kv_seq_axis}
        result[f"past_values_{i}"] = {2: kv_seq_axis}
    result["input_hidden_states"] = {1: "seq_len"}
    result["past_seq_len"] = {}
    result["total_seq_len"] = {}
    return result


def _transformer_only_outputs(num_layers: int, kv_seq_axis: str = "max_seq_len") -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {"output_hidden_states": {1: "seq_len"}}
    for i in range(num_layers):
        result[f"present_keys_{i}"] = {2: kv_seq_axis}
        result[f"present_values_{i}"] = {2: kv_seq_axis}
    return result


class QwenTransformerOnlyPrefillIOConfig(OnnxConfig):
    """Prefill (seq=64) — transformer-only I/O."""

    NORMALIZED_CONFIG_CLASS = _QWEN_TRANSFORMER_ONLY_NORMALIZED
    DUMMY_INPUT_GENERATOR_CLASSES = (
        _TransformerOnlyKvCacheGenerator,
        _TransformerOnlyHiddenStatePrefillGenerator,
        _TransformerOnlySeqLenGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return _transformer_only_inputs(self._normalized_config.num_layers)

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return _transformer_only_outputs(self._normalized_config.num_layers)


class QwenTransformerOnlyGenIOConfig(OnnxConfig):
    """Generation (seq=1) — transformer-only I/O."""

    NORMALIZED_CONFIG_CLASS = _QWEN_TRANSFORMER_ONLY_NORMALIZED
    DUMMY_INPUT_GENERATOR_CLASSES = (
        _TransformerOnlyKvCacheGenerator,
        _TransformerOnlyHiddenStateGenerator,
        _TransformerOnlySeqLenGenerator,
    )

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        return _transformer_only_inputs(self._normalized_config.num_layers)

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return _transformer_only_outputs(self._normalized_config.num_layers)


# =============================================================================
# Build config — TorchScript exporter required for the custom autograd ops
# =============================================================================


QWEN_TRANSFORMER_ONLY_CONFIG = WinMLBuildConfig(
    export=WinMLExportConfig(dynamo=False, opset_version=18),
    # Pure graph (no post-export RMSNorm fusion / matmul-add fusion).
    optim=None,
)


# =============================================================================
# Composite inference wrapper (placeholder so the build pipeline finds a
# composite class — generation isn't yet wired for the transformer-only
# I/O signature).
# =============================================================================


class WinMLQwen3TransformerOnlyModel(WinMLDecoderOnlyModel):
    """Composite handle for the transformer-only Qwen3 build (export only).

    ``generate()`` is **not** functional with this build path — the inference
    feeds and KV update logic still target the eager I/O signature. Use the
    eager :class:`WinMLQwen3Model` for generation; use this class to produce
    the transformer-only ONNX for downstream quantization.
    """

    _SUB_MODEL_CONFIG: ClassVar[dict[str, str]] = {
        "decoder_prefill": "feature-extraction",
        "decoder_gen": "text2text-generation",
    }

    @classmethod
    def get_cache_class(cls) -> type:
        return WinMLSlidingWindowCache


# =============================================================================
# install() — hot-patch the registries
# =============================================================================


_INSTALLED = False


def install() -> None:
    """Replace the qwen3 entries in WinML registries with the transformer-only variants.

    Idempotent. After this call, building any qwen3 model via
    :class:`~winml.modelkit.models.winml.composite_model.WinMLCompositeModel`
    or :class:`~winml.modelkit.models.auto.WinMLAutoModel` produces
    transformer-only ONNX files.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    # 1) Per-model build config + wrapper-class lookup live on the parent
    #    ``models.hf`` package as module-level dicts; mutating them is the
    #    documented hook for adding/overriding a model_type.
    from .. import hf as _hf_pkg  # noqa: PLC0415

    _hf_pkg.MODEL_BUILD_CONFIGS["qwen3"] = QWEN_TRANSFORMER_ONLY_CONFIG
    _hf_pkg.MODEL_CLASS_MAPPING[("qwen3", "feature-extraction")] = QwenTransformerOnlyDecoderWrapper
    _hf_pkg.MODEL_CLASS_MAPPING[("qwen3", "text2text-generation")] = QwenTransformerOnlyDecoderWrapper

    # 2) Optimum OnnxConfig (overwrites existing registration).
    register_onnx_overwrite("qwen3", "feature-extraction", library_name="transformers")(QwenTransformerOnlyPrefillIOConfig)
    register_onnx_overwrite("qwen3", "text2text-generation", library_name="transformers")(QwenTransformerOnlyGenIOConfig)

    # 3) Inference specialization (still GenericTask — wrapper returns raw KV).
    register_specialization("qwen3", "feature-extraction", "WinMLModelForGenericTask")
    register_specialization("qwen3", "text2text-generation", "WinMLModelForGenericTask")

    # 4) Composite registry — swap to the transformer-only handle.
    from ..winml.composite_model import COMPOSITE_MODEL_REGISTRY

    COMPOSITE_MODEL_REGISTRY[("qwen3", "text-generation")] = WinMLQwen3TransformerOnlyModel

    _INSTALLED = True
    logger.info("qwen_transformer_only: transformer-only export path installed for qwen3.")


__all__ = [
    "QWEN_TRANSFORMER_ONLY_CONFIG",
    "QwenTransformerOnlyDecoderWrapper",
    "QwenTransformerOnlyGenIOConfig",
    "QwenTransformerOnlyPrefillIOConfig",
    "WinMLQwen3TransformerOnlyModel",
    "install",
]
