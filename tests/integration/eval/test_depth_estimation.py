# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""End-to-end integration test for depth-estimation evaluation.

Downloads small depth-estimation checkpoints and runs the evaluator
against a few NYU validation samples. Skipped by default via
``pytest -m "not slow"``.
"""

from __future__ import annotations

import pytest

from winml.modelkit.datasets.config import DatasetConfig
from winml.modelkit.eval import WinMLDepthEstimationEvaluator
from winml.modelkit.eval.config import WinMLEvaluationConfig


# Representative checkpoints across the three families listed in issue #326.
_MODEL_IDS = [
    "depth-anything/Depth-Anything-V2-Small-hf",
    "Intel/zoedepth-nyu-kitti",
    "Intel/dpt-hybrid-midas",
]


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.integration
@pytest.mark.parametrize("model_id", _MODEL_IDS)
def test_depth_estimation_end_to_end(model_id: str) -> None:
    from transformers import AutoModelForDepthEstimation

    model = AutoModelForDepthEstimation.from_pretrained(model_id)

    config = WinMLEvaluationConfig(
        model_id=model_id,
        task="depth-estimation",
        dataset=DatasetConfig(
            path="sayakpaul/nyu_depth_v2",
            split="validation",
            samples=3,
            shuffle=False,
            columns_mapping={
                "input_column": "image",
                "depth_column": "depth_map",
            },
        ),
    )

    results = WinMLDepthEstimationEvaluator(config, model).compute()

    assert "abs_rel" in results
    assert "rmse" in results
    assert "delta1" in results
    assert results["num_images"] == 3
    assert results["abs_rel"] >= 0.0
    assert results["rmse"] >= 0.0
    assert 0.0 <= results["delta1"] <= 1.0
