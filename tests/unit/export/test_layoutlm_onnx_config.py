# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for LayoutLM v1 document-question-answering export support."""

from __future__ import annotations

import pytest
from optimum.exporters.tasks import TasksManager
from transformers import LayoutLMConfig

from winml.modelkit.export import generate_dummy_inputs, resolve_io_specs
from winml.modelkit.models.hf.layoutlm import (
    LayoutLMDocumentQAOnnxConfig,
    _adjust_roberta_position_embeddings,
)


@pytest.fixture
def layoutlm_config() -> LayoutLMConfig:
    config = LayoutLMConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=514,
        pad_token_id=1,
        type_vocab_size=1,
    )
    config.tokenizer_class = "RobertaTokenizer"
    config.architectures = ["LayoutLMForQuestionAnswering"]
    return config


def test_document_qa_config_registered() -> None:
    constructor = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model_type="layoutlm",
        task="document-question-answering",
        library_name="transformers",
    )
    assert constructor.func is LayoutLMDocumentQAOnnxConfig


def test_document_qa_io_specs(layoutlm_config: LayoutLMConfig) -> None:
    specs = resolve_io_specs("layoutlm", "document-question-answering", layoutlm_config)
    assert specs["input_names"] == [
        "input_ids",
        "bbox",
        "attention_mask",
        "token_type_ids",
    ]
    assert specs["output_names"] == ["start_logits", "end_logits"]


def test_dummy_token_types_respect_type_vocab_size(layoutlm_config: LayoutLMConfig) -> None:
    inputs = generate_dummy_inputs(
        "layoutlm",
        "document-question-answering",
        layoutlm_config,
        sequence_length=8,
    )
    assert inputs["token_type_ids"].shape == (1, 8)
    assert inputs["token_type_ids"].unique().tolist() == [0]


def test_roberta_position_offset_is_adjusted_once(layoutlm_config: LayoutLMConfig) -> None:
    _adjust_roberta_position_embeddings(layoutlm_config)
    assert layoutlm_config.max_position_embeddings == 512

    _adjust_roberta_position_embeddings(layoutlm_config)
    assert layoutlm_config.max_position_embeddings == 512


def test_non_roberta_position_convention_is_untouched() -> None:
    config = LayoutLMConfig(max_position_embeddings=512, pad_token_id=1)
    config.tokenizer_class = "BertTokenizer"

    _adjust_roberta_position_embeddings(config)

    assert config.max_position_embeddings == 512


def test_roberta_position_offset_rejects_non_positive_length(
    layoutlm_config: LayoutLMConfig,
) -> None:
    layoutlm_config.max_position_embeddings = 2
    with pytest.raises(ValueError, match="non-positive"):
        _adjust_roberta_position_embeddings(layoutlm_config)
