# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for ``composite_pipeline_tasks`` — registry-driven, offline."""

import pytest

from winml.modelkit.loader import composite_pipeline_tasks


def test_bart_serves_summarization_and_table_qa_sorted():
    # Sorted -> deterministic & model-id-independent: TAPEX and a plain bart
    # summarizer are config-identical, so the order must not imply which pipeline
    # a given checkpoint is.
    assert composite_pipeline_tasks("bart") == ["summarization", "table-question-answering"]


def test_marian_serves_translation():
    assert composite_pipeline_tasks("marian") == ["translation"]


def test_qwen3_serves_text_generation():
    assert composite_pipeline_tasks("qwen3") == ["text-generation"]


def test_non_composite_model_types_return_empty():
    assert composite_pipeline_tasks("bert") == []
    assert composite_pipeline_tasks("resnet") == []


def test_registry_accessor_raises_loudly_when_empty(monkeypatch):
    # The registry is populated as an import side effect; if registrations ever
    # move/rename and it comes up empty, the shared accessor must fail loudly rather
    # than let every reader silently return []/None (composites disabled unnoticed).
    import winml.modelkit.models.hf  # noqa: F401 — ensure real registrations land first

    monkeypatch.setattr(
        "winml.modelkit.models.winml.composite_model.COMPOSITE_MODEL_REGISTRY", {}
    )
    from winml.modelkit.loader.resolution import _composite_registry

    with pytest.raises(RuntimeError, match="COMPOSITE_MODEL_REGISTRY is empty"):
        _composite_registry()
