# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
from pathlib import Path

import pytest

from winml.modelkit.config import WinMLBuildConfig


REPO_ROOT = Path(__file__).resolve().parents[3]

recipes = [
    {
        "path": REPO_ROOT
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
        "path": REPO_ROOT
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
    {
        "path": REPO_ROOT
        / "examples"
        / "recipes"
        / "microsoft_swinv2-tiny-patch4-window16-256"
        / "qnn"
        / "image-classification_fp16_opset21_matmul-transpose_config.json",
        "loader_task": "image-classification",
        "optim_key": "matmul_transpose_fusion",
        "optim_value": True,
        "quant_expected": None,
    },
]


@pytest.mark.parametrize("rec", recipes, ids=["dinov2-base", "dinov2-small", "swinv2-tiny"])
def test_qnn_recipes(rec):
    path: Path = rec["path"]
    assert path.exists(), f"Recipe file missing: {path}"

    data = json.loads(path.read_text(encoding="utf-8"))

    # Construct the validated config from the recipe dict
    config = WinMLBuildConfig.from_dict(data)

    # export.opset_version == 21
    assert config.export is not None
    assert config.export.opset_version == 21

    # loader.task exact
    assert config.loader.task == rec["loader_task"]

    # optim key/value
    # config.optim supports dict-like access in the approved API
    assert rec["optim_key"] in config.optim
    assert config.optim[rec["optim_key"]] == rec["optim_value"]

    # quant expectations
    if rec["quant_expected"] is None:
        assert config.quant is None
    else:
        assert config.quant is not None
        assert config.quant.weight_type == rec["quant_expected"]["weight_type"]
        assert config.quant.activation_type == rec["quant_expected"]["activation_type"]
