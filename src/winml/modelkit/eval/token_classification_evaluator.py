# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Token classification evaluator with tokenizer padding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from transformers.pipelines.base import Pipeline


class WinMLTokenClassificationEvaluator(WinMLEvaluator):
    """Evaluator for token classification tasks (e.g., NER).

    Configures tokenizer padding to match the ONNX model's fixed sequence length.
    """

    def prepare_pipeline(self) -> Pipeline:
        """Create pipeline and set tokenizer padding for fixed-shape ONNX."""
        pipe = super().prepare_pipeline()

        if pipe.tokenizer is not None:
            io_config = getattr(self.model, "io_config", None) or {}
            shapes = io_config.get("input_shapes", [[]])
            if shapes and len(shapes[0]) > 1 and isinstance(shapes[0][1], int):
                max_length = shapes[0][1]

                pipe._preprocess_params.setdefault("tokenizer_params", {})
                tok_params = pipe._preprocess_params["tokenizer_params"]
                tok_params.setdefault("padding", "max_length")
                tok_params.setdefault("max_length", max_length)

                pipe._preprocess_params.setdefault("truncation", True)
                pipe.tokenizer.model_max_length = max_length

        return pipe
