# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Composite resolution for `winml config` without an explicit ``--task`` (#850).

An encoder-decoder model built without ``--task`` must export the full
encoder+decoder composite, not a decoder-only half whose ``encoder_hidden_states``
input has no producer. The no-task path detects the task with the *same*
``detect_task`` that ``inspect`` uses, then expands to the registered composite when
the detected task is one that composite serves (so no-task routing matches explicit
``--task`` for the same model). A checkpoint whose detected task isn't a composite
task (sequence-classification BART, a T5 encoder, a CLIP) stays a single model,
consistent with ``inspect``.

The routing logic is exercised as a pure ``(model_type, detected_task)`` helper
(deterministic, offline); the full resolver is exercised with crafted real
architectures so detection runs for real without a network download.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from transformers import BartConfig, Qwen3Config, T5Config

from winml.modelkit.commands.config import (
    _resolve_composite_model_components as _resolve,
)
from winml.modelkit.loader.resolution import (
    _composite_components_for_task as _serve,
)


# =============================================================================
# Pure routing: which (model_type, detected_task) pairs expand to a composite
# =============================================================================


def test_t5_text2text_generation_expands_to_composite() -> None:
    components = _serve("t5", "text2text-generation")
    assert components is not None
    assert "encoder" in components and "decoder" in components


def test_marian_text2text_generation_expands_to_composite() -> None:
    assert _serve("marian", "text2text-generation") == {
        "encoder": "feature-extraction",
        "decoder": "text2text-generation",
    }


def test_blip_image_to_text_expands_to_composite() -> None:
    components = _serve("blip", "image-to-text")
    assert components is not None
    assert "encoder" in components and "decoder" in components


def test_bart_text_classification_stays_single() -> None:
    """facebook/bart-large-mnli (BartForSequenceClassification) detects
    text-classification; it must NOT expand to a seq2seq composite -- consistent
    with inspect."""
    assert _serve("bart", "text-classification") is None


def test_t5_feature_extraction_stays_single() -> None:
    """A T5 encoder (feature-extraction) must not be force-routed to encoder+decoder."""
    assert _serve("t5", "feature-extraction") is None


def test_bart_fill_mask_defers_until_detection_fix() -> None:
    """On main, BartForConditionalGeneration detects fill-mask; not served, so bart
    stays single until the #851 detection fix flips it to text2text-generation
    (at which point it expands -- see the text2text-generation case above)."""
    assert _serve("bart", "fill-mask") is None


def test_decoder_only_composite_expands() -> None:
    """qwen3 (decoder-only) also has a registered composite; no-task must route to it,
    matching explicit ``--task text-generation`` (the prefill + gen split)."""
    components = _serve("qwen3", "text-generation")
    assert components is not None
    assert "decoder_prefill" in components and "decoder_gen" in components


def test_dual_encoder_composite_task_mismatch_stays_single() -> None:
    """CLIP has a zero-shot-image-classification composite, but a no-task CLIP detects
    feature-extraction; that mismatch keeps it single (not the zero-shot composite)."""
    assert _serve("clip", "feature-extraction") is None


def test_non_composite_model_returns_none() -> None:
    assert _serve("bert", "fill-mask") is None


@pytest.mark.parametrize("task", ["text2text-generation", "translation", "summarization"])
def test_seq2seq_pipeline_and_export_tasks_resolve_identically(task: str) -> None:
    """t5's pipeline tasks (translation/summarization) and the detected export task
    (text2text-generation) all map to the same single composite export."""
    assert _serve("t5", task) == _serve("t5", "text2text-generation")


# =============================================================================
# Full resolver: detects like inspect, then expands only served seq2seq tasks
# =============================================================================


def test_resolver_expands_generation_checkpoint() -> None:
    cfg = T5Config(architectures=["T5ForConditionalGeneration"])
    with patch("transformers.AutoConfig.from_pretrained", return_value=cfg):
        components = _resolve("some/t5-checkpoint", None, None)
    assert components is not None
    assert "encoder" in components and "decoder" in components


def test_resolver_keeps_classification_checkpoint_single() -> None:
    """The bug this guards: a sequence-classification BART must stay
    text-classification (resolve to no composite) -- matching inspect."""
    cfg = BartConfig(architectures=["BartForSequenceClassification"])
    with patch("transformers.AutoConfig.from_pretrained", return_value=cfg):
        assert _resolve("facebook/bart-large-mnli", None, None) is None


def test_resolver_expands_decoder_only_checkpoint() -> None:
    """A no-task decoder-only checkpoint (qwen3) routes to its prefill+gen composite,
    matching explicit ``--task text-generation``."""
    cfg = Qwen3Config(architectures=["Qwen3ForCausalLM"])
    with patch("transformers.AutoConfig.from_pretrained", return_value=cfg):
        components = _resolve("some/qwen3-checkpoint", None, None)
    assert components is not None
    assert "decoder_prefill" in components and "decoder_gen" in components


def test_explicit_task_resolves_composite_without_detection() -> None:
    """model_type given + explicit task -> direct registry lookup, no config load."""
    components = _resolve(None, "t5", "translation")
    assert components is not None
    assert "encoder" in components and "decoder" in components


def test_explicit_task_resolves_decoder_only_composite() -> None:
    """The explicit path must still serve non-encoder-decoder composites
    (qwen3 decoder-only), which the no-task gate intentionally excludes."""
    components = _resolve(None, "qwen3", "text-generation")
    assert components is not None
    assert "decoder_prefill" in components and "decoder_gen" in components
