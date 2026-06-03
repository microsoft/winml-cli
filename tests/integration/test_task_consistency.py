# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Cross-command task-detection consistency.

The whole point of the modality-aware ``detect_task`` refactor: the same model
must resolve to the same WinMLTask through every detection entry point. These
tests load real HF configs (small JSON, cached) and assert the loader detector
and the inspect resolver (which delegates to it) agree, with modality-aware
results for vision feature models.
"""

from __future__ import annotations

import pytest
from transformers import AutoConfig

from winml.modelkit.inspect.resolver import detect_task as inspect_detect_task
from winml.modelkit.loader import detect_task


# (model_id, expected WinMLTask) — exercises the D2 modality upgrade (DINOv2),
# a top-level-nested vision config that must NOT upgrade (CLIP), and a text control.
CASES = [
    ("facebook/dinov2-small", "image-feature-extraction"),
    ("openai/clip-vit-base-patch32", "feature-extraction"),  # image_size nested -> no D2
    ("bert-base-uncased", "fill-mask"),
]


@pytest.mark.parametrize("model_id, expected", CASES)
def test_detect_task_agrees_across_resolvers(model_id: str, expected: str) -> None:
    cfg = AutoConfig.from_pretrained(model_id)

    loader_task, loader_source = detect_task(cfg)
    inspect_task, inspect_source = inspect_detect_task(cfg)

    assert loader_task == expected, f"{model_id}: loader detect_task gave {loader_task!r}"
    # The inspect resolver must agree (it delegates to the loader detector).
    assert (inspect_task, inspect_source) == (loader_task, loader_source)


def test_dinov2_is_modality_aware_not_lossy() -> None:
    """Regression for #778: DINOv2 must NOT resolve to the lossy feature-extraction."""
    cfg = AutoConfig.from_pretrained("facebook/dinov2-small")
    task, _ = detect_task(cfg)
    assert task == "image-feature-extraction"
    assert task != "feature-extraction"
