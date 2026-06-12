# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from winml.modelkit.loader.resolution import _composite_components_for_task, resolve_composite


def test_explicit_pipeline_task_returns_components():
    assert resolve_composite("bart", "summarization") == {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }
    assert resolve_composite("bart", "table-question-answering") == {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }


def test_explicit_text2text_generation_is_not_a_composite_key():
    # Asymmetry: explicit granular task -> single model, NOT auto-expanded.
    assert resolve_composite("bart", "text2text-generation") is None


def test_non_composite_returns_none():
    assert resolve_composite("bert", "text-classification") is None


def test_bridge_maps_detected_seq2seq_to_composite():
    assert _composite_components_for_task("bart", "text2text-generation") == {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }


def test_bridge_maps_registration_task_directly():
    assert _composite_components_for_task("qwen3", "text-generation") is not None
