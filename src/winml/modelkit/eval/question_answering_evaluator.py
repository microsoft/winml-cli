# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Question answering evaluator with tokenizer padding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base_evaluator import WinMLEvaluator


if TYPE_CHECKING:
    from transformers.pipelines.base import Pipeline

logger = logging.getLogger(__name__)


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

    def compute(self) -> dict[str, Any]:
        """Run QA evaluation with automatic SQuAD v2 detection.

        Detects whether the dataset has unanswerable questions (SQuAD v2
        format) and passes the correct metric and format flag to the HF
        QuestionAnsweringEvaluator.
        """
        from evaluate import evaluator

        logger.info("Running evaluation...")
        task_evaluator = evaluator(self.config.task)

        label_col = self.config.dataset.columns_mapping.get(
            "label_column", "answers"
        )
        squad_v2 = task_evaluator.is_squad_v2_format(
            self.data, label_column=label_col
        )

        kwargs: dict[str, Any] = {
            "model_or_pipeline": self.pipe,
            "data": self.data,
            "metric": "squad_v2" if squad_v2 else "squad",
            "squad_v2_format": squad_v2,
            **self.config.dataset.columns_mapping,
        }

        return task_evaluator.compute(**kwargs)
