# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
from pathlib import Path

import pytest


ROOT = Path.cwd()

recipes = [
    {
        "path": ROOT
        / "examples"
        / "recipes"
        / "facebook_dinov2-base"
        / "qnn"
        / "image-feature-extraction_w8a16_opset21_bias-softmax_config.json",
        "loader_task": "image-feature-extraction",
        "optim_key": "bias_softmax_fusion",
        "optim_value": True,
        "quant_expected": {"weight_type": "uint8", "activation_type": "uint16"},
    },
    {
        "path": ROOT
        / "examples"
        / "recipes"
        / "facebook_dinov2-small"
        / "qnn"
        / "image-feature-extraction_w8a16_opset21_bias-softmax_config.json",
        "loader_task": "image-feature-extraction",
        "optim_key": "bias_softmax_fusion",
        "optim_value": True,
        "quant_expected": {"weight_type": "uint8", "activation_type": "uint16"},
    },
]

swin = {
    "path": ROOT
    / "examples"
    / "recipes"
    / "microsoft_swinv2-tiny-patch4-window16-256"
    / "qnn"
    / "image-classification_fp16_opset21_matmul-transpose_config.json",
    "loader_task": "image-classification",
    "optim_key": "matmul_transpose_fusion",
    "optim_value": True,
    "quant_expected": None,
}


@pytest.mark.parametrize("rec", recipes)
def test_dinov2_qnn_recipes(rec):
    path: Path = rec["path"]
    assert path.exists(), f"Recipe file missing: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))

    # export.opset_version == 21
    assert "export" in data and data["export"].get("opset_version") == 21

    # loader.task exact
    assert "loader" in data and data["loader"].get("task") == rec["loader_task"]

    # optim key/value
    assert "optim" in data and rec["optim_key"] in data["optim"]
    assert data["optim"][rec["optim_key"]] == rec["optim_value"]

    # quant presence and types
    assert "quant" in data and isinstance(data["quant"], dict)
    quant = data["quant"]
    assert quant.get("weight_type") == rec["quant_expected"]["weight_type"]
    assert quant.get("activation_type") == rec["quant_expected"]["activation_type"]


def test_swinv2_qnn_recipe():
    path: Path = swin["path"]
    assert path.exists(), f"Recipe file missing: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))

    # export.opset_version == 21
    assert "export" in data and data["export"].get("opset_version") == 21

    # loader.task exact
    assert "loader" in data and data["loader"].get("task") == swin["loader_task"]

    # optim key/value
    assert "optim" in data and swin["optim_key"] in data["optim"]
    assert data["optim"][swin["optim_key"]] == swin["optim_value"]

    # quant is None
    assert "quant" in data and data["quant"] is None
