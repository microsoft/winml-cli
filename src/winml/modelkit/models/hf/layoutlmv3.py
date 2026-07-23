# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""LayoutLMv3 HuggingFace Model Configuration."""

from __future__ import annotations

from typing import Any

from optimum.exporters.onnx.model_configs import LayoutLMv3OnnxConfig
from optimum.utils import NormalizedTextConfig
from optimum.utils.input_generators import DummyBboxInputGenerator, DummyVisionInputGenerator

from ...config import WinMLBuildConfig
from ...export import MaxLengthTextInputGenerator, register_onnx_overwrite
from ...export.config import WinMLExportConfig
from ...optim import WinMLOptimizationConfig
from .roberta import _adjust_position_embeddings


LAYOUTLMV3_TASKS = (
    "feature-extraction",
    "question-answering",
    "text-classification",
    "token-classification",
)


LAYOUTLMV3_CONFIG = WinMLBuildConfig(
    export=WinMLExportConfig(dynamo=False),
    optim=WinMLOptimizationConfig(
        clamp_constant_values=True,
    ),
)


@register_onnx_overwrite("layoutlmv3", *LAYOUTLMV3_TASKS, library_name="transformers")
class LayoutLMv3IOConfig(LayoutLMv3OnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """LayoutLMv3 OnnxConfig with usable text length and TorchScript export.

    LayoutLMv3 checkpoints follow the RoBERTa position-offset convention where
    max_position_embeddings includes padding offset slots. Export dummy inputs
    must use the usable sequence length (for example 512 from 514), otherwise
    tracing can index past the text position embedding table.
    """

    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        sequence_length="max_position_embeddings",
        MAX_2D_POSITION_EMBEDDINGS="max_2d_position_embeddings",
        image_size="input_size",
        allow_new=True,
    )
    DUMMY_INPUT_GENERATOR_CLASSES: tuple[type[Any], ...] = (
        MaxLengthTextInputGenerator,
        DummyVisionInputGenerator,
        DummyBboxInputGenerator,
    )

    def __init__(self, config: Any, task: str, **kwargs: Any) -> None:
        _adjust_position_embeddings(config)
        super().__init__(config, task, **kwargs)
