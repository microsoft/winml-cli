# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Cross-command task-detection consistency.

The whole point of the modality-aware ``resolve_task`` refactor: the same model
must resolve to the same WinMLTask through every detection entry point. ``inspect``
now consumes the unified ``resolve_task`` directly, so a single resolver is the
source of truth. These tests load real HF configs (small JSON, cached) and assert
the resolved WinMLTask, with modality-aware results for vision feature models.
"""

from __future__ import annotations

import pytest
from transformers import AutoConfig

from winml.modelkit.loader.resolution import resolve_task


# Every test here calls AutoConfig.from_pretrained, which hits the network unless
# the HF config is already cached; mark the module so offline runs can skip via
# ``-m 'not network'`` instead of failing with a confusing ConnectionError.
pytestmark = pytest.mark.network


# (model_id, expected WinMLTask) — exercises the D2 modality upgrade (DINOv2),
# a top-level-nested vision config that must NOT upgrade (CLIP), and a text control.
CASES = [
    ("facebook/dinov2-small", "image-feature-extraction"),
    ("openai/clip-vit-base-patch32", "feature-extraction"),  # image_size nested -> no D2
    ("bert-base-uncased", "fill-mask"),
    # bart maps to >1 task (feature-extraction + text2text-generation) in
    # MODEL_CLASS_MAPPING; detect_task must consult the architecture head instead
    # of short-circuiting to the first key, so a sequence-classification BART
    # resolves to text-classification rather than the lossy feature-extraction.
    ("facebook/bart-large-mnli", "text-classification"),
    # sam has a (sam, None) default-class sentinel plus a single real task; the
    # sentinel must not make detection fall through — it resolves to mask-generation.
    ("facebook/sam-vit-base", "mask-generation"),
    # optimum mislabels BartForConditionalGeneration as fill-mask; an encoder-decoder
    # reported as fill-mask is a seq2seq generator -> text2text-generation.
    ("facebook/bart-large-cnn", "text2text-generation"),
]


@pytest.mark.parametrize("model_id, expected", CASES)
def test_resolve_task_agrees_across_resolvers(model_id: str, expected: str) -> None:
    cfg = AutoConfig.from_pretrained(model_id)

    # resolve_task is the single source of truth consumed by every entry point;
    # calling it twice must be deterministic (same task + provenance).
    first = resolve_task(cfg)
    second = resolve_task(cfg)

    assert first.task == expected, f"{model_id}: resolve_task gave {first.task!r}"
    assert (second.task, second.source) == (first.task, first.source)


def test_dinov2_is_modality_aware_not_lossy() -> None:
    """Regression for #778: DINOv2 must NOT resolve to the lossy feature-extraction."""
    cfg = AutoConfig.from_pretrained("facebook/dinov2-small")
    task = resolve_task(cfg).task
    assert task == "image-feature-extraction"
    assert task != "feature-extraction"
