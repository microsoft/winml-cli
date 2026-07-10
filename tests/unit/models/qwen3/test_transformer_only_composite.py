# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the Qwen3 transformer-only composite handle.

The transformer-only build is EXPORT-ONLY. Its composite handle must:

1. Extend the plain ``WinMLCompositeModel`` (NOT ``WinMLDecoderOnlyModel``) so
   ``from_pretrained`` can return after building the sub-models. The decoder-only
   base wires a generation runtime from the eager KV name ``past_0_key`` in
   ``__init__``, which the transformer-only graph (``past_keys_0``) lacks — and
   would crash handle construction even though both ONNX built fine.
2. Inject ``model_type="qwen3_transformer_only"`` for every sub-model so the
   composite builds the transformer-only variant rather than the native (full)
   ``qwen3`` architecture when the caller omits ``model_type``.
"""

from __future__ import annotations

from unittest.mock import patch

from winml.modelkit.models.hf.qwen3.qwen_transformer_only import (
    WinMLQwen3TransformerOnlyModel,
)
from winml.modelkit.models.winml import WinMLCompositeModel
from winml.modelkit.models.winml.composite_model import COMPOSITE_MODEL_REGISTRY
from winml.modelkit.models.winml.decoder_only import WinMLDecoderOnlyModel


class TestTransformerOnlyCompositeHandle:
    def test_registered_for_text_generation(self) -> None:
        assert (
            COMPOSITE_MODEL_REGISTRY.get(("qwen3_transformer_only", "text-generation"))
            is WinMLQwen3TransformerOnlyModel
        )

    def test_is_plain_composite_not_decoder_runtime(self) -> None:
        # Export-only: must not inherit the decoder-only generation runtime whose
        # __init__ assumes the eager KV signature and crashes on this graph.
        assert issubclass(WinMLQwen3TransformerOnlyModel, WinMLCompositeModel)
        assert not issubclass(WinMLQwen3TransformerOnlyModel, WinMLDecoderOnlyModel)

    def test_sub_model_config(self) -> None:
        assert WinMLQwen3TransformerOnlyModel._SUB_MODEL_CONFIG == {
            "decoder_prefill": "feature-extraction",
            "decoder_gen": "text2text-generation",
        }

    def test_from_pretrained_injects_transformer_only_model_type(self) -> None:
        recorded: dict[str, object] = {}

        def _fake(cls, model_id, task="text-generation", **kwargs):
            recorded["model_id"] = model_id
            recorded["task"] = task
            recorded["model_type"] = kwargs.get("model_type")
            return "SENTINEL"

        with patch.object(WinMLCompositeModel, "from_pretrained", classmethod(_fake)):
            result = WinMLQwen3TransformerOnlyModel.from_pretrained("Qwen/Qwen3-0.6B")

        assert result == "SENTINEL"
        assert recorded["model_id"] == "Qwen/Qwen3-0.6B"
        assert recorded["model_type"] == "qwen3_transformer_only"

    def test_from_pretrained_preserves_explicit_model_type(self) -> None:
        recorded: dict[str, object] = {}

        def _fake(cls, model_id, task="text-generation", **kwargs):
            recorded["model_type"] = kwargs.get("model_type")
            return "SENTINEL"

        with patch.object(WinMLCompositeModel, "from_pretrained", classmethod(_fake)):
            WinMLQwen3TransformerOnlyModel.from_pretrained(
                "Qwen/Qwen3-0.6B", model_type="custom-variant"
            )

        assert recorded["model_type"] == "custom-variant"
