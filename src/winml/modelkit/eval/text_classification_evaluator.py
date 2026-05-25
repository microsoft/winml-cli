# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Text classification evaluator with tokenizer padding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from transformers.pipelines.base import Pipeline


class WinMLTextClassificationEvaluator(WinMLEvaluator):
    """Evaluator for text/sequence classification tasks.

    Configures tokenizer padding to match the ONNX model's fixed sequence length.
    """

    def prepare_pipeline(self) -> Pipeline:
        """Create pipeline and set tokenizer padding for fixed-shape ONNX."""
        pipe = super().prepare_pipeline()

        if pipe.tokenizer is not None:
            io_config = getattr(self.model, "io_config", None) or {}
            shapes = io_config.get("input_shapes", [[]])
            if shapes and len(shapes[0]) > 1 and isinstance(shapes[0][1], int):
                pipe._preprocess_params.setdefault("padding", "max_length")
                pipe._preprocess_params.setdefault("max_length", shapes[0][1])
                pipe._preprocess_params.setdefault("truncation", True)

        return pipe
