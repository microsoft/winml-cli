# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""LayoutLM HuggingFace Model Configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from optimum.exporters.onnx.model_configs import LayoutLMOnnxConfig
from optimum.utils import NormalizedTextConfig
from optimum.utils.input_generators import DummyBboxInputGenerator, DummyVisionInputGenerator

from ...export import MaxLengthTextInputGenerator, register_onnx_overwrite


if TYPE_CHECKING:
    import torch


class ZeroTokenTypeLayoutLMTextInputGenerator(MaxLengthTextInputGenerator):
    """LayoutLM text dummy generator that keeps token_type_ids within type_vocab_size=1."""

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        """Generate LayoutLM text inputs, replacing token_type_ids with zeros."""
        tensor = cast(
            "torch.Tensor",
            super().generate(
                input_name,
                framework=framework,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
            ),
        )
        if input_name == "token_type_ids":
            return tensor.new_zeros(tensor.shape)
        return tensor


@register_onnx_overwrite("layoutlm", "question-answering", library_name="transformers")
class LayoutLMQAIOConfig(LayoutLMOnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """LayoutLM question-answering OnnxConfig with bbox and safe token type IDs."""

    # sequence_length is bound to the model's max_position_embeddings so
    # MaxLengthTextInputGenerator emits full-length text inputs instead of
    # Optimum's default of 16 (allow_new=True permits adding this mapping).
    # We deliberately do NOT map max_2d_position_embeddings here: Optimum's
    # DummyBboxInputGenerator hardcodes its coordinate range (its
    # normalized_config.max_2d_position_embeddings read is commented out
    # upstream), so such a mapping is inert and never becomes a sequence
    # length. bbox coordinate bounds for the shipped recipe come from the
    # recipe's `value_range` instead.
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        sequence_length="max_position_embeddings",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES: tuple[type[Any], ...] = (
        ZeroTokenTypeLayoutLMTextInputGenerator,
        DummyVisionInputGenerator,
        DummyBboxInputGenerator,
    )
