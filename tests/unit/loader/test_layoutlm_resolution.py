# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Task and model-class resolution for LayoutLM v1 document QA."""

from __future__ import annotations

from transformers import LayoutLMConfig

from winml.modelkit.loader.resolution import TaskSource, resolve_task


def _document_qa_config() -> LayoutLMConfig:
    config = LayoutLMConfig()
    config.architectures = ["LayoutLMForQuestionAnswering"]
    return config


def test_layoutlm_document_qa_resolves_from_architecture() -> None:
    resolution = resolve_task(_document_qa_config())

    assert resolution.task == "document-question-answering"
    assert resolution.optimum_task == "document-question-answering"
    assert resolution.model_class.__name__ == "AutoModelForDocumentQuestionAnswering"
    assert resolution.source == TaskSource.ARCHITECTURE_MAPPING


def test_layoutlm_document_qa_explicit_task_uses_document_auto_class() -> None:
    resolution = resolve_task(
        _document_qa_config(),
        task="document-question-answering",
    )

    assert resolution.task == "document-question-answering"
    assert resolution.model_class.__name__ == "AutoModelForDocumentQuestionAnswering"
    assert resolution.source == TaskSource.USER_TASK


def test_layoutlm_other_head_is_not_forced_to_document_qa() -> None:
    config = LayoutLMConfig()
    config.architectures = ["LayoutLMForTokenClassification"]

    resolution = resolve_task(config)

    assert resolution.task == "token-classification"
    assert resolution.source == TaskSource.TASKS_MANAGER
