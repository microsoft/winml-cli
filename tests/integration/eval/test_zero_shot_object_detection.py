# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Pinned-dataset integration coverage for zero-shot object detection Eval."""

from __future__ import annotations

import os

import pytest


_HF_HUB_AVAILABLE = os.environ.get("WINML_TEST_OFFLINE", "0") != "1"
_COCO_REVISION = "cf0b22332314a937e9dc8a1957b21725430bb41d"
_EXPECTED_IMAGE_IDS = [
    441247,
    435081,
    190236,
    254814,
    16228,
    435206,
    231831,
    357888,
    366884,
    542625,
    327769,
    561958,
    217285,
    87476,
    59386,
    463802,
    524850,
    286182,
    184324,
    213255,
    239857,
    540502,
    178028,
    229997,
]


@pytest.mark.skipif(not _HF_HUB_AVAILABLE, reason="HF Hub disabled in offline mode")
def test_pinned_coco_selection_reproduces_all_category_image_ids() -> None:
    """The frozen full-split greedy selection remains deterministic."""
    from winml.modelkit.eval import DatasetConfig, WinMLEvaluationConfig
    from winml.modelkit.eval.zero_shot_object_detection_evaluator import (
        WinMLZeroShotObjectDetectionEvaluator,
    )

    evaluator = object.__new__(WinMLZeroShotObjectDetectionEvaluator)
    evaluator.config = WinMLEvaluationConfig(
        task="zero-shot-object-detection",
        dataset=DatasetConfig(
            path="detection-datasets/coco",
            revision=_COCO_REVISION,
            split="val",
            samples=32,
            max_queries=None,
            shuffle=False,
            streaming=True,
        ),
    )
    evaluator._image_col = "image"
    evaluator._annotation_col = "objects"
    evaluator._bbox_key = "bbox"
    evaluator._category_key = "category"
    evaluator._image_id_col = "image_id"

    data = evaluator.prepare_data()

    assert list(data["image_id"]) == _EXPECTED_IMAGE_IDS
    assert evaluator._selection_accounting == {
        "requested": 32,
        "requested_cap": 32,
        "source_rows_scanned": 4952,
        "selected": 24,
        "category_count": 80,
        "categories_covered": 80,
        "query_requested": 80,
        "query_available": 80,
        "query_used": 80,
        "query_truncated": 0,
    }
    assert len(evaluator._vocabulary) == 80
