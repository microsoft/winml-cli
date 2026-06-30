# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""LM-head-only ``qwen3`` build variant, registered as a distinct model_type.

This module registers a self-contained build path under the model_type
``"qwen3_lm_head_only"`` (a sibling of ``"qwen3_transformer_only"`` and
``"qwen3_embeddings_only"``). Selecting it is explicit — pass
``model_type="qwen3_lm_head_only"`` to ``WinMLAutoModel.from_pretrained(...)``.

The variant exports the final projection (``lm_head``) as a standalone ONNX file:

  Inputs : output_hidden_states (FP32, ``[1, seq_len, hidden]``)
  Outputs: logits (FP32, ``[1, seq_len, vocab]``)
  Ops    : ``onnx::MatMul`` (vocab projection).

The LM head is quantized weight-only to 4 bits (MatMulNBits / RTN, symmetric
int4 weights, block_size=32). Build it with a weight-only precision
(``precision="w4a32"`` or ``precision="int4"``) so the device/precision policy
creates an RTN quant config; the registered :class:`Qwen3LMHeadOnlyQuantFinalizer`
then pins the block size / symmetry / accuracy level.

The input name ``output_hidden_states`` matches the transformer-only graph's
``output_hidden_states`` output, so the two ONNX chain together at runtime.
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
LM_HEAD_ONLY_MODEL_TYPE = "qwen3_lm_head_only"


# =============================================================================
# Wrapper module
# =============================================================================


class QwenLMHeadOnlyWrapper(nn.Module):
    """Wraps the ``Qwen3ForCausalLM`` ``lm_head`` for standalone export.

    Only ``get_output_embeddings()`` (the vocab projection) is exported; the
    embedding table and transformer stack stay out of the graph.
    """

    def __init__(self, lm_head: nn.Module, config: Any) -> None:
        super().__init__()
        self.lm_head = lm_head
        # The ``lm_head`` submodule has no ``config``; carry the parent's so the
        # exporter resolves this variant's OnnxConfig and the build pipeline can
        # read ``model.config.model_type`` for quant-policy dispatch.
        self.config = config
        self.config.model_type = LM_HEAD_ONLY_MODEL_TYPE

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any) -> QwenLMHeadOnlyWrapper:
        """Load the HF model and wrap its ``lm_head`` for export."""
        kwargs.setdefault("torch_dtype", torch.float32)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        wrapper = cls(model.get_output_embeddings(), model.config)
        wrapper.eval()
        return wrapper

    def get_export_args(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Flatten the dummy-input dict into positional export args."""
        return (inputs["output_hidden_states"],)

    def forward(self, output_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states -> ``logits`` (FP32).

        The ``lm_head`` is loaded in float32, so its output is already FP32 —
        no explicit cast is added (keeps the graph a single projection).
        """
        return self.lm_head(output_hidden_states)


# =============================================================================
# Dummy input generator (lm_head I/O)
# =============================================================================


class _LMHeadHiddenStateGenerator(DummyInputGenerator):
    """Generates ``output_hidden_states`` (FP32, ``[1, seq_len, hidden]``)."""

    SUPPORTED_INPUT_NAMES = ("output_hidden_states",)

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

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        if input_name == "output_hidden_states":
            return torch.randn(self.batch_size, self.seq_len, self.hidden_size, dtype=torch.float32)
        raise ValueError(f"Unknown input: {input_name}")


# =============================================================================
# OnnxConfig — lm_head I/O layout
# =============================================================================


_QWEN_LM_HEAD_NORMALIZED = NormalizedConfig.with_args(
    hidden_size="hidden_size",
    vocab_size="vocab_size",
    allow_new=True,
)


@register_onnx_overwrite(
    LM_HEAD_ONLY_MODEL_TYPE, "feature-extraction", library_name="transformers"
)
class QwenLMHeadOnlyIOConfig(OnnxConfig):
    """LM head — ``output_hidden_states`` -> ``logits``."""

    NORMALIZED_CONFIG_CLASS = _QWEN_LM_HEAD_NORMALIZED
    DUMMY_INPUT_GENERATOR_CLASSES = (_LMHeadHiddenStateGenerator,)

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        """ONNX input axes (hidden states)."""
        return {"output_hidden_states": {1: "seq_len"}}

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """ONNX output axes (logits)."""
        return {"logits": {1: "seq_len"}}


# =============================================================================
# Build config — TorchScript exporter (matches the transformer-only variant).
# =============================================================================


QWEN_LM_HEAD_ONLY_CONFIG = WinMLBuildConfig(
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
    ("qwen3-lm-head-only", "feature-extraction"): QwenLMHeadOnlyWrapper,
}

# Inference specialization (GenericTask — the wrapper returns raw logits).
register_specialization(
    LM_HEAD_ONLY_MODEL_TYPE, "feature-extraction", "WinMLModelForGenericTask"
)


__all__ = [
    "LM_HEAD_ONLY_MODEL_TYPE",
    "MODEL_CLASS_MAPPING",
    "QWEN_LM_HEAD_ONLY_CONFIG",
    "QwenLMHeadOnlyIOConfig",
    "QwenLMHeadOnlyWrapper",
]
