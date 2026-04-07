# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Question answering evaluator with tokenizer padding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from transformers.pipelines.base import Pipeline


class WinMLQuestionAnsweringEvaluator(WinMLEvaluator):
    """Evaluator for extractive question answering tasks.

    Uses HF QuestionAnsweringEvaluator with SQuAD metrics (exact_match, f1).
    Configures tokenizer max length to match the ONNX model's fixed sequence length.
    """

    @classmethod
    def schema_info(cls) -> list:
        """Return expected dataset schema for question answering."""
        from .config import SchemaColumn

        return [
            SchemaColumn(
                "question", "Value(string)", "question_column", description="question text"
            ),
            SchemaColumn(
                "context", "Value(string)", "context_column", description="context passage"
            ),
            SchemaColumn(
                "id", "Value(string)", "id_column", description="unique question-answer pair ID"
            ),
            SchemaColumn(
                "answers",
                "dict(text: list[str], answer_start: list[int])",
                "label_column",
                description="answers with text spans and start positions",
            ),
        ]

    def prepare_pipeline(self) -> Pipeline:
        """Create pipeline and set tokenizer padding for fixed-shape ONNX."""
        pipe = super().prepare_pipeline()

        if pipe.tokenizer is not None:
            io_config = getattr(self.model, "io_config", None) or {}
            shapes = io_config.get("input_shapes", [[]])
            if shapes and len(shapes[0]) > 1 and isinstance(shapes[0][1], int):
                max_length = shapes[0][1]
                pipe.tokenizer.model_max_length = max_length
                pipe._preprocess_params.setdefault("padding", "max_length")
                pipe._preprocess_params.setdefault("max_seq_len", max_length)

        return pipe
