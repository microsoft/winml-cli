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
from transformers import LayoutLMForQuestionAnswering

from ...export import MaxLengthTextInputGenerator, register_onnx_overwrite


if TYPE_CHECKING:
    import torch


MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("layoutlm", "question-answering"): LayoutLMForQuestionAnswering,
}


class _LayoutLMNormalizedConfig(NormalizedTextConfig):  # type: ignore[misc]
    """LayoutLM text config with a metadata-derived usable sequence length."""

    @property
    def sequence_length(self) -> int:
        """Account for the padding offset used by RoBERTa-style checkpoints."""
        max_positions = int(self.config.max_position_embeddings)
        padding_idx = getattr(self.config, "pad_token_id", None)
        if padding_idx:
            return max_positions - int(padding_idx) - 1
        return max_positions


class ZeroTokenTypeLayoutLMTextInputGenerator(MaxLengthTextInputGenerator):
    """LayoutLM text dummy generator that bounds token_type_ids by type_vocab_size."""

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        sequence_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Cap generic shape inference to LayoutLM's usable position range."""
        self._type_vocab_size = max(1, int(normalized_config.config.type_vocab_size))
        usable_length = normalized_config.sequence_length
        if sequence_length is None or sequence_length > usable_length:
            sequence_length = usable_length
        super().__init__(
            task,
            normalized_config,
            sequence_length=sequence_length,
            **kwargs,
        )

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        """Generate LayoutLM text inputs with metadata-bounded token type IDs."""
        if input_name == "token_type_ids":
            return cast(
                "torch.Tensor",
                self.random_int_tensor(
                    [self.batch_size, self.sequence_length],
                    max_value=self._type_vocab_size,
                    framework=framework,
                    dtype=int_dtype,
                ),
            )
        return cast(
            "torch.Tensor",
            super().generate(
                input_name,
                framework=framework,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
            ),
        )


class UsableLengthLayoutLMBboxInputGenerator(DummyBboxInputGenerator):  # type: ignore[misc]
    """LayoutLM bbox generator aligned with the usable text sequence length."""

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        sequence_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Cap generic shape inference to LayoutLM's usable position range."""
        usable_length = normalized_config.sequence_length
        if sequence_length is None or sequence_length > usable_length:
            sequence_length = usable_length
        super().__init__(
            task,
            normalized_config,
            sequence_length=sequence_length,
            **kwargs,
        )


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
    NORMALIZED_CONFIG_CLASS = _LayoutLMNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES: tuple[type[Any], ...] = (
        ZeroTokenTypeLayoutLMTextInputGenerator,
        DummyVisionInputGenerator,
        UsableLengthLayoutLMBboxInputGenerator,
    )
