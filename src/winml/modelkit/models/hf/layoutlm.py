# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""LayoutLM v1 document-question-answering support.

LayoutLM v1 uses an extractive question-answering head for document QA: the
model consumes token IDs plus OCR-derived bounding boxes and emits start/end
logits. Optimum does not register this model-type/task pair, and it cannot
infer the task from ``LayoutLMForQuestionAnswering``.

This module supplies the missing task metadata, model-class mapping, and ONNX
configuration for the whole LayoutLM v1 family. It does not implement OCR,
word/box alignment, or answer decoding; those remain external preprocessing
and postprocessing steps.
"""

from __future__ import annotations

import logging
from typing import Any

from optimum.exporters.onnx.model_configs import LayoutLMOnnxConfig
from optimum.utils.input_generators import DummyTextInputGenerator
from transformers import AutoModelForDocumentQuestionAnswering

from ...export import register_onnx_overwrite


logger = logging.getLogger(__name__)


# The task is derived from checkpoint architecture metadata instead of a model
# ID. The resolver consults this registry before Optimum task inference.
ARCHITECTURE_TASK_MAPPING: dict[tuple[str, str], str] = {
    ("layoutlm", "LayoutLMForQuestionAnswering"): "document-question-answering",
}


# Optimum has no model-class entry for document-question-answering. Transformers
# does: its document-QA auto class maps LayoutLMConfig to
# LayoutLMForQuestionAnswering.
MODEL_CLASS_MAPPING: dict[tuple[str, str], type] = {
    ("layoutlm", "document-question-answering"): AutoModelForDocumentQuestionAnswering,
}


def _adjust_roberta_position_embeddings(config: Any) -> None:
    """Use the non-padding sequence length for RoBERTa-tokenized checkpoints.

    RoBERTa tokenizers reserve ``pad_token_id + 1`` position slots, so their
    configs commonly encode ``max_position_embeddings`` as usable length plus
    that offset (for example, 514 for a usable length of 512). Only configs
    that explicitly declare a RoBERTa tokenizer are adjusted; a LayoutLM model
    with a positive padding ID but another position convention is untouched.
    """
    if getattr(config, "_position_offset_applied", False):
        return

    tokenizer_class = getattr(config, "tokenizer_class", "") or ""
    if "roberta" not in tokenizer_class.lower():
        return

    max_positions = getattr(config, "max_position_embeddings", None)
    pad_token_id = getattr(config, "pad_token_id", 0) or 0
    if max_positions is None or pad_token_id <= 0:
        return

    adjusted = max_positions - pad_token_id - 1
    if adjusted <= 0:
        raise ValueError(
            "Position offset adjustment would produce non-positive "
            f"max_position_embeddings={adjusted} "
            f"(original={max_positions}, pad_token_id={pad_token_id})"
        )

    config.max_position_embeddings = adjusted
    config._position_offset_applied = True
    logger.debug(
        "Adjusted LayoutLM max_position_embeddings: %d -> %d (pad_token_id=%d)",
        max_positions,
        adjusted,
        pad_token_id,
    )


class LayoutLMTextInputGenerator(DummyTextInputGenerator):  # type: ignore[misc]
    """Generate token types within the checkpoint's configured vocabulary.

    Optimum's generic text generator samples token-type IDs from ``[0, 2)``.
    LayoutLM checkpoints commonly set ``type_vocab_size=1``, where sampling 1
    causes an embedding Gather out of bounds. Derive the exclusive upper bound
    from config metadata instead.
    """

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> Any:
        """Generate a safe tensor for ``input_name``."""
        if input_name != "token_type_ids":
            return super().generate(input_name, framework, int_dtype, float_dtype)

        type_vocab_size = max(1, int(self.normalized_config.type_vocab_size))
        shape = [self.batch_size, self.sequence_length]
        return self.random_int_tensor(
            shape,
            max_value=type_vocab_size,
            min_value=0,
            framework=framework,
            dtype=int_dtype,
        )


@register_onnx_overwrite("layoutlm", "document-question-answering", library_name="transformers")
class LayoutLMDocumentQAOnnxConfig(LayoutLMOnnxConfig):  # type: ignore[misc]
    """ONNX config for the LayoutLM v1 extractive document-QA head.

    Inputs remain ``input_ids``, ``bbox``, ``attention_mask`` and
    ``token_type_ids`` from Optimum's LayoutLM config. Outputs are the two span
    tensors returned by ``LayoutLMForQuestionAnswering``.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (
        LayoutLMTextInputGenerator,
        *LayoutLMOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES[1:],
    )

    def __init__(self, config: Any, task: str, **kwargs: Any) -> None:
        _adjust_roberta_position_embeddings(config)
        super().__init__(config, task, **kwargs)

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        """Return the extractive span-head output names and dynamic axes."""
        return {
            "start_logits": {0: "batch_size", 1: "sequence_length"},
            "end_logits": {0: "batch_size", 1: "sequence_length"},
        }


__all__ = [
    "ARCHITECTURE_TASK_MAPPING",
    "MODEL_CLASS_MAPPING",
    "LayoutLMDocumentQAOnnxConfig",
    "LayoutLMTextInputGenerator",
]
