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
        / "audeering_wav2vec2-large-robust-12-ft-emotion-msp-dim"
        / "cpu"
        / "cpu"
        / "audio-classification_fp32_config.json",
        "loader_task": "audio-classification",
        "loader_model_class": "EmotionModel",
        "loader_model_type": "wav2vec2_emotion_regression",
        "opset_version": 17,
        "quant_mode": None,
    },
    {
        "path": REPO_ROOT
        / "examples"
        / "recipes"
        / "audeering_wav2vec2-large-robust-12-ft-emotion-msp-dim"
        / "cpu"
        / "cpu"
        / "audio-classification_fp16_config.json",
        "loader_task": "audio-classification",
        "loader_model_class": "EmotionModel",
        "loader_model_type": "wav2vec2_emotion_regression",
        "opset_version": 17,
        "quant_mode": "fp16",
    },
]


@pytest.mark.parametrize(
    "rec",
    recipes,
    ids=["audeering-wav2vec2-emotion-fp32", "audeering-wav2vec2-emotion-fp16"],
)
def test_cpu_recipes(rec):
    path: Path = rec["path"]
    assert path.exists(), f"Recipe file missing: {path}"

    # EP/device is encoded by folder layout: <model>/<ep>/<device>/<recipe>.json
    assert path.parent.name == "cpu"  # device
    assert path.parent.parent.name == "cpu"  # ep

    data = json.loads(path.read_text(encoding="utf-8"))

    # Construct the validated config from the recipe dict
    config = WinMLBuildConfig.from_dict(data)

    # export.opset_version exact
    assert config.export is not None
    assert config.export.opset_version == rec["opset_version"]

    # loader routes to the emotion-regression head
    assert config.loader.task == rec["loader_task"]
    assert config.loader.model_class == rec["loader_model_class"]
    assert config.loader.model_type == rec["loader_model_type"]

    if rec["quant_mode"] is None:
        assert config.quant is None
    else:
        assert config.quant is not None
        assert config.quant.mode == rec["quant_mode"]


@pytest.mark.parametrize(
    ("precision", "opset_version", "dynamo", "dtype", "quant_mode"),
    [
        ("fp32", 17, False, "float32", None),
        ("fp16", 18, True, "float16", "fp16"),
    ],
)
def test_birefnet_cpu_recipes_preserve_mixed_export_contract(
    precision: str,
    opset_version: int,
    dynamo: bool,
    dtype: str,
    quant_mode: str | None,
) -> None:
    path = (
        REPO_ROOT
        / "examples"
        / "recipes"
        / "ZhengPeng7_BiRefNet"
        / "cpu"
        / "cpu"
        / f"image-segmentation_{precision}_config.json"
    )
    config = WinMLBuildConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))

    assert config.export is not None
    assert config.export.opset_version == opset_version
    assert config.export.dynamo is dynamo
    assert config.export.enable_hierarchy_tags is True
    assert len(config.export.input_tensors) == 1
    assert config.export.input_tensors[0].name == "x"
    assert config.export.input_tensors[0].dtype == dtype
    assert config.export.input_tensors[0].shape == (1, 3, 1024, 1024)
    assert [tensor.name for tensor in config.export.output_tensors] == ["logits"]
    assert config.loader.task == "image-segmentation"
    assert config.loader.model_class == "AutoModelForImageSegmentation"
    assert config.loader.model_type == "SegformerForSemanticSegmentation"
    assert config.loader.trust_remote_code is True

    if quant_mode is None:
        assert config.quant is None
    else:
        assert config.quant is not None
        assert config.quant.mode == quant_mode
