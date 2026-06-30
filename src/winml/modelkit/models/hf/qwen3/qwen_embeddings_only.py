# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Embeddings-only ``qwen3`` build variant, registered as a distinct model_type.

This module registers a self-contained build path under the model_type
``"qwen3_embeddings_only"`` (a sibling of ``"qwen3_transformer_only"`` and
``"qwen3_lm_head_only"``). Selecting it is explicit — pass
``model_type="qwen3_embeddings_only"`` to ``WinMLAutoModel.from_pretrained(...)``.

The variant exports the input-embedding lookup as a standalone ONNX file:

  Inputs : input_ids (INT, ``[1, seq_len]``)
  Outputs: input_hidden_states (FP32, ``[1, seq_len, hidden]``)
  Ops    : ``onnx::Gather`` (embedding table lookup).

Embeddings are deliberately left in float — they are **not** quantized. Build
it with a float precision (e.g. ``precision="fp32"``) so the device/precision
policy leaves ``config.quant=None`` and no QDQ / RTN pass runs.

The output name ``input_hidden_states`` matches the transformer-only graph's
``input_hidden_states`` input, so the two ONNX chain together at runtime.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
from optimum.exporters.onnx import OnnxConfig
from optimum.utils import NormalizedConfig
from optimum.utils.input_generators import DummyInputGenerator
from transformers import AutoModelForCausalLM

from ....config import WinMLBuildConfig
from ....export import register_onnx_overwrite
from ....export.config import WinMLExportConfig
from ...winml import register_specialization


# Distinct model_type for this variant. The underscore form is what the
# exporter sees on ``model.config.model_type``; the hyphenated form is used for
# the ``MODEL_CLASS_MAPPING`` / ``MODEL_BUILD_CONFIGS`` lookups (those callers
# normalize ``_`` -> ``-``).
EMBEDDINGS_ONLY_MODEL_TYPE = "qwen3_embeddings_only"


# =============================================================================
# Wrapper module
# =============================================================================


class QwenEmbeddingsOnlyWrapper(nn.Module):
    """Wraps the ``Qwen3ForCausalLM`` input embedding for standalone export.

    Only ``get_input_embeddings()`` (the embedding table) is exported; the
    transformer stack and ``lm_head`` stay out of the graph.
    """

    def __init__(self, embed_tokens: nn.Module, config: Any) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        # The embedding submodule has no ``config``; carry the parent's so the
        # exporter resolves this variant's OnnxConfig and the build pipeline can
        # read ``model.config.model_type`` for quant-policy dispatch.
        self.config = config
        self.config.model_type = EMBEDDINGS_ONLY_MODEL_TYPE

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs: Any
    ) -> QwenEmbeddingsOnlyWrapper:
        """Load the HF model and wrap its input embedding for export."""
        kwargs.setdefault("torch_dtype", torch.float32)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        wrapper = cls(model.get_input_embeddings(), model.config)
        wrapper.eval()
        return wrapper

    def get_export_args(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Flatten the dummy-input dict into positional export args."""
        return (inputs["input_ids"],)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed ``input_ids`` -> ``input_hidden_states`` (FP32).

        The embedding table is loaded in float32, so its output is already
        FP32 — no explicit cast is added (keeps the graph a single ``Gather``).
        """
        return self.embed_tokens(input_ids)


# =============================================================================
# Dummy input generator (embeddings I/O)
# =============================================================================


class _EmbeddingsInputIdsGenerator(DummyInputGenerator):
    """Generates ``input_ids`` (INT, ``[1, seq_len]``)."""

    SUPPORTED_INPUT_NAMES = ("input_ids",)

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
        self.vocab_size = normalized_config.vocab_size
        self.seq_len = seq_len or getattr(normalized_config, "seq_len", self._default_seq_len)

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        if input_name == "input_ids":
            return torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), dtype=torch.int64)
        raise ValueError(f"Unknown input: {input_name}")


# =============================================================================
# OnnxConfig — embeddings I/O layout
# =============================================================================


_QWEN_EMBEDDINGS_NORMALIZED = NormalizedConfig.with_args(
    hidden_size="hidden_size",
    vocab_size="vocab_size",
    allow_new=True,
)


@register_onnx_overwrite(
    EMBEDDINGS_ONLY_MODEL_TYPE, "feature-extraction", library_name="transformers"
)
class QwenEmbeddingsOnlyIOConfig(OnnxConfig):
    """Embeddings lookup — ``input_ids`` -> ``input_hidden_states``."""

    NORMALIZED_CONFIG_CLASS = _QWEN_EMBEDDINGS_NORMALIZED
    DUMMY_INPUT_GENERATOR_CLASSES = (_EmbeddingsInputIdsGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """ONNX input axes (token ids)."""
        return {"input_ids": {1: "seq_len"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """ONNX output axes (hidden states)."""
        return {"input_hidden_states": {1: "seq_len"}}


# =============================================================================
# Build config — TorchScript exporter (matches the transformer-only variant).
# =============================================================================


QWEN_EMBEDDINGS_ONLY_CONFIG = WinMLBuildConfig(
    export=WinMLExportConfig(dynamo=False, opset_version=18),
    # Pure graph (no post-export fusion).
    optim=None,
)


# =============================================================================
# Declarative registration (import-time)
# =============================================================================

# Wrapper-class lookup keyed by (model_type, task). Keys use the hyphenated
# model_type form because ``_get_custom_model_class`` normalizes ``_`` -> ``-``
# before lookup. Merged into the aggregate mapping by ``models.hf.__init__``.
MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("qwen3-embeddings-only", "feature-extraction"): QwenEmbeddingsOnlyWrapper,
}

# Inference specialization (GenericTask — the wrapper returns raw hidden states).
register_specialization(
    EMBEDDINGS_ONLY_MODEL_TYPE, "feature-extraction", "WinMLModelForGenericTask"
)


__all__ = [
    "EMBEDDINGS_ONLY_MODEL_TYPE",
    "MODEL_CLASS_MAPPING",
    "QWEN_EMBEDDINGS_ONLY_CONFIG",
    "QwenEmbeddingsOnlyIOConfig",
    "QwenEmbeddingsOnlyWrapper",
]
