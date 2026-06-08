# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for ``to_optimum_task`` — the single WinML -> Optimum collapse boundary.

``to_optimum_task`` is the relocated/renamed ``export.io.map_task_synonym``. It must
reproduce that behavior verbatim: WinML extensions short-circuit before Optimum, and
everything else passes through ``TasksManager.map_from_synonym`` (which collapses
modality-aware names like ``image-feature-extraction`` to ``feature-extraction``).
"""

from __future__ import annotations

import pytest

from winml.modelkit.loader import to_optimum_task


@pytest.mark.parametrize(
    "task, expected",
    [
        # Optimum collapses modality (image-feature-extraction -> feature-extraction).
        ("image-feature-extraction", "feature-extraction"),
        # WinML extension: routed to its Optimum-canonical target.
        ("next-sentence-prediction", "text-classification"),
        # WinML extension preserved as-is (Optimum would mis-map it otherwise).
        ("mask-generation", "mask-generation"),
        # Already-canonical task passes through unchanged.
        ("text-classification", "text-classification"),
    ],
)
def test_to_optimum_task(task: str, expected: str) -> None:
    assert to_optimum_task(task) == expected
