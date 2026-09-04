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

    @property
    def bbox_coordinate_range(self) -> int:
        """Return the high-exclusive normalized bbox coordinate bound."""
        max_2d_positions = int(self.config.max_2d_position_embeddings)
        coordinate_range = min(1001, max_2d_positions)
        if coordinate_range < 2:
            raise ValueError("LayoutLM requires at least two 2D embedding positions")
        return coordinate_range


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
    """Generate positive-area normalized boxes aligned with the text sequence."""

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        sequence_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Derive safe sequence and coordinate bounds from model metadata."""
        usable_length = normalized_config.sequence_length
        if sequence_length is None or sequence_length > usable_length:
            sequence_length = usable_length
        self.coordinate_range = normalized_config.bbox_coordinate_range
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
    ) -> Any:
        """Generate ``[x1, y1, x2, y2]`` with positive, in-bounds width and height."""
        del input_name, float_dtype
        coordinates = self.random_int_tensor(
            [self.batch_size, self.sequence_length, 4],
            max_value=self.coordinate_range,
            framework=framework,
            dtype=int_dtype,
        )
        max_coordinate = self.coordinate_range - 1
        if framework == "pt":
            import torch

            x1 = torch.minimum(coordinates[..., 0], coordinates[..., 2])
            y1 = torch.minimum(coordinates[..., 1], coordinates[..., 3])
            x2 = torch.maximum(coordinates[..., 0], coordinates[..., 2])
            y2 = torch.maximum(coordinates[..., 1], coordinates[..., 3])

            # Equal corners are expanded inward so width/height embedding indexes
            # stay in [1, coordinate_range) without exceeding the coordinate table.
            x_equal = x1 == x2
            y_equal = y1 == y2
            x1 = torch.where(x_equal & (x2 == max_coordinate), x1 - 1, x1)
            y1 = torch.where(y_equal & (y2 == max_coordinate), y1 - 1, y1)
            x2 = torch.where(x_equal & (x2 < max_coordinate), x2 + 1, x2)
            y2 = torch.where(y_equal & (y2 < max_coordinate), y2 + 1, y2)
            return torch.stack((x1, y1, x2, y2), dim=-1)
        if framework == "np":
            import numpy as np

            np_x1 = np.minimum(coordinates[..., 0], coordinates[..., 2])
            np_y1 = np.minimum(coordinates[..., 1], coordinates[..., 3])
            np_x2 = np.maximum(coordinates[..., 0], coordinates[..., 2])
            np_y2 = np.maximum(coordinates[..., 1], coordinates[..., 3])

            np_x_equal = np_x1 == np_x2
            np_y_equal = np_y1 == np_y2
            np_x1 = np.where(
                np_x_equal & (np_x2 == max_coordinate), np_x1 - 1, np_x1
            )
            np_y1 = np.where(
                np_y_equal & (np_y2 == max_coordinate), np_y1 - 1, np_y1
            )
            np_x2 = np.where(
                np_x_equal & (np_x2 < max_coordinate), np_x2 + 1, np_x2
            )
            np_y2 = np.where(
                np_y_equal & (np_y2 < max_coordinate), np_y2 + 1, np_y2
            )
            return np.stack((np_x1, np_y1, np_x2, np_y2), axis=-1)
        raise ValueError(
            f"LayoutLM bbox generation supports only 'pt' and 'np', got {framework!r}"
        )


@register_onnx_overwrite("layoutlm", "question-answering", library_name="transformers")
class LayoutLMQAIOConfig(LayoutLMOnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """LayoutLM question-answering OnnxConfig with bbox and safe token type IDs."""

    # sequence_length is bound to the model's max_position_embeddings so
    # MaxLengthTextInputGenerator emits full-length text inputs instead of
    # Optimum's default of 16 (allow_new=True permits adding this mapping).
    # The custom bbox generator separately derives normalized coordinate bounds
    # from max_2d_position_embeddings and emits ordered, positive-area boxes.
    NORMALIZED_CONFIG_CLASS = _LayoutLMNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES: tuple[type[Any], ...] = (
        ZeroTokenTypeLayoutLMTextInputGenerator,
        DummyVisionInputGenerator,
        UsableLengthLayoutLMBboxInputGenerator,
    )
